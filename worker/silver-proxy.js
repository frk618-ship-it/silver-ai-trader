/**
 * 白銀 AI 資料中繼（Cloudflare Worker）
 *
 * 為什麼需要：瀏覽器不允許網頁直接讀取 Yahoo／Stooq 的回應（沒有 CORS 標頭），
 * 而網路上的免費 CORS proxy 已經停用或要求付費。這個 Worker 在伺服器端代為抓取，
 * 加上瀏覽器 User-Agent 與 CORS 標頭後回傳。
 *
 * 用法：https://<你的-worker>.workers.dev/?url=<經 encodeURIComponent 的目標網址>
 *
 * 安全：
 *   · 只轉發 ALLOWED_HOSTS 裡的資料來源（不是開放 proxy）
 *   · 只接受 ALLOWED_ORIGINS 的網頁呼叫（其他網站無法盜用你的額度）
 *   · 只允許 GET
 * 額度：Cloudflare 免費方案每天 10 萬次請求，本系統一次完整載入約 20–30 次。
 */

// 允許呼叫此中繼的網頁來源（可在 Worker 設定的環境變數 ALLOWED_ORIGINS 覆寫，逗號分隔）
const DEFAULT_ORIGINS = [
  'https://frk618-ship-it.github.io',
  'http://localhost', 'http://127.0.0.1',
  'null',                        // 直接開啟本機 HTML 檔時，瀏覽器送出的 Origin 是 "null"
];

// 只轉發這些資料來源
const ALLOWED_HOSTS = new Set([
  'query1.finance.yahoo.com', 'query2.finance.yahoo.com',
  'stooq.com',
  'publicreporting.cftc.gov', 'www.cftc.gov',
  'home.treasury.gov',
  'production.dataviz.cnn.io',
  'api.stocktwits.com',
  'trends.google.com',
]);

const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';

// 快取秒數：即時報價短、日線歷史長，減少對資料來源的請求次數
function ttlFor(target) {
  if (/range=1d|interval=1m|\/q\/l\//.test(target.href)) return 30;
  if (target.hostname === 'api.stocktwits.com') return 600;
  return 3600;
}

function originAllowed(origin, env) {
  const list = (env && env.ALLOWED_ORIGINS ? env.ALLOWED_ORIGINS.split(',') : DEFAULT_ORIGINS).map(s => s.trim()).filter(Boolean);
  if (!origin) return false;
  return list.some(o => origin === o || (o.startsWith('http://') && origin.startsWith(o + ':')));
}

function cors(origin) {
  return {
    'Access-Control-Allow-Origin': origin,
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Access-Control-Allow-Headers': 'Accept',
    'Access-Control-Max-Age': '86400',
    'Vary': 'Origin',
  };
}

const json = (status, obj, headers = {}) =>
  new Response(JSON.stringify(obj), { status, headers: { 'Content-Type': 'application/json; charset=utf-8', ...headers } });

export default {
  async fetch(request, env, ctx) {
    const origin = request.headers.get('Origin');
    const okOrigin = originAllowed(origin, env);
    const ch = okOrigin ? cors(origin) : {};

    if (request.method === 'OPTIONS') return new Response(null, { status: okOrigin ? 204 : 403, headers: ch });
    if (request.method !== 'GET') return json(405, { error: '只允許 GET' }, ch);

    const self = new URL(request.url);
    const raw = self.searchParams.get('url');
    if (!raw) return json(200, { ok: true, service: 'silver-proxy', hint: '用法：/?url=<encoded url>' }, ch);
    if (!okOrigin) return json(403, { error: `不允許的來源：${origin || '（無 Origin）'}` });

    let target;
    try { target = new URL(raw); } catch (_) { return json(400, { error: 'url 參數不是合法網址' }, ch); }
    if (target.protocol !== 'https:' || !ALLOWED_HOSTS.has(target.hostname)) {
      return json(403, { error: `不允許轉發到 ${target.hostname}` }, ch);
    }

    // Cloudflare 邊緣快取（在本機測試環境沒有 caches 時略過）
    const cache = typeof caches !== 'undefined' ? caches.default : null;
    const cacheKey = new Request(target.href, { method: 'GET' });
    if (cache) {
      const hit = await cache.match(cacheKey);
      if (hit) {
        const h = new Headers(hit.headers);
        Object.entries(ch).forEach(([k, v]) => h.set(k, v));
        h.set('X-Proxy-Cache', 'HIT');
        return new Response(hit.body, { status: hit.status, headers: h });
      }
    }

    let upstream;
    try {
      upstream = await fetch(target.href, {
        headers: { 'User-Agent': UA, 'Accept': '*/*', 'Accept-Language': 'en-US,en;q=0.9' },
        redirect: 'follow',
      });
    } catch (e) {
      return json(502, { error: `無法連線到 ${target.hostname}：${e.message}` }, ch);
    }

    const body = await upstream.arrayBuffer();
    const headers = new Headers({
      'Content-Type': upstream.headers.get('Content-Type') || 'text/plain; charset=utf-8',
      'Cache-Control': `public, max-age=${ttlFor(target)}`,
      'X-Proxy-Upstream-Status': String(upstream.status),
      'X-Proxy-Cache': 'MISS',
    });
    const res = new Response(body, { status: upstream.status, headers });
    if (cache && upstream.ok && ctx && ctx.waitUntil) ctx.waitUntil(cache.put(cacheKey, res.clone()));
    Object.entries(ch).forEach(([k, v]) => res.headers.set(k, v));
    return res;
  },
};

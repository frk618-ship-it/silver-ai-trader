// 本機測試 Worker（會真的連到 Yahoo 與 CFTC）：node test.mjs
// 在需要 proxy 的環境：NODE_USE_ENV_PROXY=1 node test.mjs
import worker from './silver-proxy.js';

const BASE = 'https://silver-proxy.example.workers.dev/';
const SITE = 'https://frk618-ship-it.github.io';
const call = (target, origin = SITE, method = 'GET') =>
  worker.fetch(new Request(BASE + (target ? '?url=' + encodeURIComponent(target) : ''),
    { method, headers: origin ? { Origin: origin } : {} }), {}, { waitUntil() {} });
let fail = 0;
const ok = (c, m) => { console.log(`${c ? 'PASS' : 'FAIL'} ${m}`); if (!c) fail++; };

const Q = 'https://query1.finance.yahoo.com/v8/finance/chart/SI=F?range=1d&interval=1m';
let r = await call(Q);
const j = await r.json().catch(() => null);
const meta = j?.chart?.result?.[0]?.meta;
ok(r.status === 200 && meta?.regularMarketPrice > 0, `Yahoo 即時報價 SI=F ${meta?.regularMarketPrice}（${meta ? new Date(meta.regularMarketTime * 1000).toISOString() : '—'}）`);
ok(r.headers.get('access-control-allow-origin') === SITE, '回應帶 CORS 標頭');
r = await call('https://query2.finance.yahoo.com/v8/finance/chart/GC=F?range=5y&interval=1d');
const h = await r.json().catch(() => null);
ok((h?.chart?.result?.[0]?.timestamp?.length || 0) > 1000, `Yahoo 5 年日線 ${h?.chart?.result?.[0]?.timestamp?.length || 0} 筆`);
r = await call('https://publicreporting.cftc.gov/resource/72hh-3qpy.json?$limit=1');
ok(r.status === 200, 'CFTC COT');
ok((await call(Q, 'https://evil.example.com')).status === 403, '其他網站呼叫 → 403');
ok((await call('https://example.com/')).status === 403, '非白名單主機 → 403');
ok((await call(Q, null)).status === 403, '沒有 Origin → 403');
ok((await call(Q, 'null')).status === 200, '本機 HTML 檔（Origin: null）可用');
ok((await call(Q, 'http://localhost:8080')).status === 200, 'localhost 任意埠可用');
r = await call(Q, SITE, 'OPTIONS');
ok(r.status === 204 && r.headers.get('access-control-allow-methods')?.includes('GET'), 'CORS 預檢');
ok((await call(null)).status === 200, '健康檢查');
process.exit(fail ? 1 : 0);

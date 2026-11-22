// 驗證「必須取得白銀期貨即時報價才進入模型設定畫面」的閘門
// 以 Playwright 攔截網路，模擬 Stooq / Yahoo 的各種回應情境。
// 用法：node live-gate.js（需 bench/node_modules 內有 playwright、react、babel、tfjs 等，見 README）
const { chromium } = require('playwright');
const fs = require('fs'), path = require('path');
const HTML = fs.readFileSync(path.join(__dirname, '..', 'index.html'));
const NM = path.join(__dirname, 'node_modules');
const LIBS = [
  [/tfjs-backend-wasm@[^/]+\/dist\/(tfjs-backend-wasm[^/]*\.wasm)/, m => path.join(NM, '@tensorflow/tfjs-backend-wasm/dist', m[1]), 'application/wasm'],
  [/tfjs-backend-wasm/, () => path.join(NM, '@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.min.js')],
  [/@tensorflow\/tfjs@/, () => path.join(NM, '@tensorflow/tfjs/dist/tf.min.js')],
  [/react-dom@/, () => path.join(NM, 'react-dom/umd/react-dom.production.min.js')],
  [/unpkg\.com\/react@/, () => path.join(NM, 'react/umd/react.production.min.js')],
  [/babel/, () => path.join(NM, '@babel/standalone/babel.min.js')],
  [/cdn\.plot\.ly/, () => path.join(NM, 'plotly.js-dist-min/plotly.min.js')],
  [/unpkg\.com\/lucide/, () => path.join(NM, 'lucide/dist/umd/lucide.min.js')],
  [/cdn\.tailwindcss\.com/, () => null],
];
const DAY = 86400e3;
const histCSV = (endMs) => {
  let v = 30, out = 'Date,Open,High,Low,Close,Volume\n';
  const rows = [];
  for (let i = 900; i >= 0; i--) { v *= 1 + Math.sin(i / 7) * 0.004; rows.push(`${new Date(endMs - i * DAY).toISOString().slice(0, 10)},${v},${v},${v},${v.toFixed(4)},1000`); }
  return out + rows.reverse().join('\n');   // Stooq: newest-first
};
const quoteCSV = (ms, price) => {
  const d = new Date(ms + 3600e3).toISOString();   // 華沙時間 ≈ UTC+1
  return `Symbol,Date,Time,Open,High,Low,Close,Volume\nSI.F,${d.slice(0, 10)},${d.slice(11, 19)},${price},${price},${price},${price},100\n`;
};
const yahooQuote = (ms, price) => JSON.stringify({ chart: { result: [{ meta: { regularMarketPrice: price, regularMarketTime: Math.floor(ms / 1000) }, timestamp: [], indicators: { quote: [{ close: [] }] } }] } });

async function scenario(name, { history, stooqQuote, yahoo, snapshotFirst = false, ctx: givenCtx }) {
  const br = givenCtx ? null : await chromium.launch();
  const ctx = givenCtx || await br.newContext();
  const pg = await ctx.newPage();
  const errs = []; pg.on('pageerror', e => errs.push(e.message));
  await pg.route('**/*', async r => {
    const u = r.request().url();
    if (u.startsWith('http://app.local/')) return r.fulfill({ body: HTML, contentType: 'text/html' });
    for (const [re, f, ct] of LIBS) { const m = u.match(re); if (m) { const p = f(m); return r.fulfill({ body: p ? fs.readFileSync(p) : '', contentType: ct || 'application/javascript' }); } }
    if (/stooq\.com\/q\/d\/l\/.*s=si(\.|%2E)f/i.test(u) && history) return r.fulfill({ body: history(), contentType: 'text/csv', headers: { 'access-control-allow-origin': '*' } });
    if (/stooq\.com\/q\/l\/.*si\.f/i.test(u) && !/proxy|allorigins|codetabs|thingproxy/.test(u) && stooqQuote) return r.fulfill({ body: stooqQuote(), contentType: 'text/csv', headers: { 'access-control-allow-origin': '*' } });
    if (/^https:\/\/query1\.finance\.yahoo\.com\/v8\/finance\/chart\/SI=F\?range=1d/.test(u) && yahoo) return r.fulfill({ body: yahoo(), contentType: 'application/json', headers: { 'access-control-allow-origin': '*' } });
    return r.abort();
  });
  await pg.goto('http://app.local/index.html');
  const startBtn = snapshotFirst ? 'button:has-text("立即開始")' : 'button:has-text("讀取歷史資料")';
  const fallbackBtn = 'button:has-text("等待最新資料再開始")';
  const btn = pg.locator(startBtn).first();
  if (await btn.count() === 0 && !snapshotFirst) await pg.locator(fallbackBtn).first().click({ timeout: 60000 });
  else await btn.click({ timeout: 60000 });
  const outcome = await Promise.race([
    pg.waitForSelector('text=無法取得白銀期貨即時價格', { timeout: 180000 }).then(() => 'blocked'),
    pg.waitForSelector('text=模型與訓練參數', { timeout: 180000 }).then(() => 'entered'),
  ]).catch(() => 'timeout');
  let detail = '';
  if (outcome === 'blocked') detail = await pg.locator('p.font-mono').first().innerText().catch(() => '');
  if (outcome === 'entered') detail = await pg.evaluate(() => { const el = [...document.querySelectorAll('span')].find(s => s.textContent.includes('期貨即時')); return el ? el.parentElement.innerText.replace(/\s+/g, ' ') : '(沒有即時價格欄位)'; });
  await pg.close();
  if (br) await br.close();
  return { name, outcome, detail: detail.slice(0, 200), errs };
}

(async () => {
  const now = Date.now();
  const ok = (r, want) => { const pass = r.outcome === want && r.errs.length === 0; console.log(`${pass ? 'PASS' : 'FAIL'} ${r.name}: ${r.outcome}（預期 ${want}）${r.detail ? ' — ' + r.detail : ''}${r.errs.length ? ' errors=' + r.errs.join(' | ') : ''}`); return pass; };
  let all = true;
  all &= ok(await scenario('全部來源失敗', {}), 'blocked');
  all &= ok(await scenario('有歷史、無即時報價', { history: () => histCSV(now - DAY) }), 'blocked');
  all &= ok(await scenario('有即時報價、無歷史', { stooqQuote: () => quoteCSV(now - 600e3, 30.5) }), 'blocked');
  all &= ok(await scenario('即時報價過舊（10 天前）', { history: () => histCSV(now - DAY), stooqQuote: () => quoteCSV(now - 10 * DAY, 30.5) }), 'blocked');
  all &= ok(await scenario('即時報價與收盤差距過大', { history: () => histCSV(now - DAY), yahoo: () => yahooQuote(now - 60e3, 300) }), 'blocked');
  all &= ok(await scenario('Stooq 即時報價正常', { history: () => histCSV(now - DAY), stooqQuote: () => quoteCSV(now - 600e3, 30.5) }), 'entered');
  all &= ok(await scenario('Yahoo 即時報價正常', { history: () => histCSV(now - DAY), yahoo: () => yahooQuote(now - 60e3, 30.7) }), 'entered');
  // 快照路徑：先成功載入一次留下快照，再以「立即開始」重進
  const br = await chromium.launch(); const ctx = await br.newContext();
  await scenario('建立快照', { ctx, history: () => histCSV(now - DAY), yahoo: () => yahooQuote(now - 60e3, 30.7) });
  all &= ok(await scenario('快照 + 無即時報價', { ctx, snapshotFirst: true }), 'blocked');
  all &= ok(await scenario('快照 + 即時報價正常', { ctx, snapshotFirst: true, yahoo: () => yahooQuote(now - 60e3, 30.9) }), 'entered');
  await br.close();
  process.exit(all ? 0 : 1);
})();

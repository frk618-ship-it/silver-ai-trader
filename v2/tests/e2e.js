// v2 端到端測試（Playwright）：攔截網路、提供假行情，驗證閘門與主要流程，並輸出截圖。
// 用法：NODE_MODULES=/path/to/node_modules SHOTS=/path/to/out node e2e.js
// node_modules 需含：playwright、@tensorflow/tfjs@4.22.0、@tensorflow/tfjs-backend-wasm@4.22.0、lightweight-charts@4.2.0
const { chromium, devices } = require('playwright');
const fs = require('fs'), path = require('path');
const ROOT = path.resolve(__dirname, '..');
const NM = process.env.NODE_MODULES || path.join(__dirname, 'node_modules');
const SHOTS = process.env.SHOTS || path.join(__dirname, 'shots');
fs.mkdirSync(SHOTS, { recursive: true });

const LIBS = [
  [/tfjs-backend-wasm@[^/]+\/dist\/(tfjs-backend-wasm[^/]*\.wasm)/, m => `${NM}/@tensorflow/tfjs-backend-wasm/dist/${m[1]}`, 'application/wasm'],
  [/tfjs-backend-wasm@[^/]+\/dist\/tf-backend-wasm/, () => `${NM}/@tensorflow/tfjs-backend-wasm/dist/tf-backend-wasm.min.js`],
  [/@tensorflow\/tfjs@/, () => `${NM}/@tensorflow/tfjs/dist/tf.min.js`],
  [/lightweight-charts@/, () => `${NM}/lightweight-charts/dist/lightweight-charts.standalone.production.js`],
];

// ── 假行情 ──
const DAY = 86400e3;
function history(base, seed) {
  const rows = []; let v = base, k = 0;
  for (let t = Date.now() - 1300 * DAY; t < Date.now() - DAY; t += DAY) {
    const d = new Date(t); if (d.getUTCDay() % 6 === 0) continue;
    k++; v *= 1 + Math.sin(k / 9 + seed) * 0.006 + Math.sin(k * 12.9898 + seed) * 0.008;
    rows.push(`${d.toISOString().slice(0, 10)},${v},${v},${v},${v.toFixed(4)},${1e6 + (k % 50) * 1e4}`);
  }
  return 'Date,Open,High,Low,Close,Volume\n' + rows.reverse().join('\n');
}
const BASE = { 'si=f': 30, 'slv.us': 27, 'gc=f': 2400, 'cl=f': 75, 'dx-y.nyb': 104, '^vix': 16, '^tnx': 4.2, 'hg=f': 4.3 };
const HIST = Object.fromEntries(Object.entries(BASE).map(([s, b], i) => [s, history(b, i)]));
const lastSI = () => +HIST['si=f'].split('\n')[1].split(',')[4];

async function setup(page, { quote = true, quoteAgeH = 0.1 } = {}) {
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/*', async r => {
    const u = r.request().url();
    if (u.startsWith('http://app.local/')) {
      const p = path.join(ROOT, decodeURIComponent(new URL(u).pathname).replace(/^\/+/, '') || 'index.html');
      const f = fs.existsSync(p) && fs.statSync(p).isFile() ? p : path.join(ROOT, 'index.html');
      const ct = f.endsWith('.css') ? 'text/css' : f.endsWith('.js') ? 'application/javascript' : 'text/html';
      return r.fulfill({ body: fs.readFileSync(f), contentType: ct });
    }
    for (const [re, fn, ct] of LIBS) { const m = u.match(re); if (m) return r.fulfill({ body: fs.readFileSync(fn(m)), contentType: ct || 'application/javascript' }); }
    const cors = { 'access-control-allow-origin': '*' };
    const hm = u.match(/^https:\/\/stooq\.com\/q\/d\/l\/\?s=([^&]+)/);
    if (hm) { const s = decodeURIComponent(hm[1]); if (HIST[s]) return r.fulfill({ body: HIST[s], contentType: 'text/csv', headers: cors }); }
    if (quote && /^https:\/\/query1\.finance\.yahoo\.com\/v8\/finance\/chart\/SI=F\?range=1d/.test(u)) {
      return r.fulfill({ contentType: 'application/json', headers: cors, body: JSON.stringify({ chart: { result: [{ meta: {
        regularMarketPrice: +(lastSI() * 1.004).toFixed(3), regularMarketTime: Math.floor((Date.now() - quoteAgeH * 3600e3) / 1000) } }] } }) });
    }
    if (/publicreporting\.cftc\.gov\/resource\/72hh-3qpy/.test(u) && !/corsproxy|allorigins|codetabs|thingproxy/.test(u)) {
      const rows = []; for (let i = 0; i < 200; i++) rows.push({ as_of_date_form_yyyy_mm_dd: new Date(Date.now() - i * 7 * DAY).toISOString().slice(0, 10) + 'T00:00:00.000',
        m_money_positions_long_all: String(40000 + Math.round(9000 * Math.sin(i / 6))), m_money_positions_short_all: '21000', open_interest_all: '150000', market_and_exchange_names: 'SILVER - COMMODITY EXCHANGE INC.' });
      return r.fulfill({ contentType: 'application/json', headers: cors, body: JSON.stringify(rows) });
    }
    return r.abort();
  });
  return errors;
}

const ok = (c, m) => { console.log(`${c ? 'PASS' : 'FAIL'} ${m}`); if (!c) process.exitCode = 1; };

(async () => {
  const browser = await chromium.launch();

  // 1) 沒有即時報價 → 停在閘門
  {
    const ctx = await browser.newContext({ viewport: { width: 1280, height: 860 } });
    const page = await ctx.newPage();
    const errs = await setup(page, { quote: false });
    await page.goto('http://app.local/');
    await page.getByText('無法取得白銀期貨即時價格').waitFor({ timeout: 90000 });
    ok(await page.locator('.shell').count() === 0, '無即時報價時不進入主畫面');
    ok(await page.getByRole('button', { name: /重新嘗試/ }).isVisible(), '顯示重新嘗試按鈕');
    await page.screenshot({ path: `${SHOTS}/0-gate-blocked.png` });
    ok(!errs.length, '閘門頁沒有 JS 錯誤 ' + errs.join(' | '));
    await ctx.close();
  }
  // 2) 報價過舊 → 擋下
  {
    const ctx = await browser.newContext();
    const page = await ctx.newPage();
    await setup(page, { quoteAgeH: 200 });
    await page.goto('http://app.local/');
    await page.getByText('無法取得白銀期貨即時價格').waitFor({ timeout: 90000 });
    ok(/過舊/.test(await page.locator('.alert.bad').innerText()), '報價過舊時擋下並說明原因');
    await ctx.close();
  }
  // 3) 主流程（桌機）
  const ctx = await browser.newContext({ viewport: { width: 1360, height: 900 }, deviceScaleFactor: 1, colorScheme: 'dark' });
  const page = await ctx.newPage();
  const errs = await setup(page);
  await page.goto('http://app.local/');
  await page.locator('.shell').waitFor({ timeout: 120000 });
  await page.waitForTimeout(800);
  ok(/\$\d/.test(await page.locator('#tk-px').innerText()), '頂部顯示即時價');
  await page.screenshot({ path: `${SHOTS}/1-overview-empty.png` });

  await page.goto('http://app.local/#/studio');
  await page.getByRole('radio', { name: /LSTM/ }).click();
  await page.getByRole('radio', { name: /雙頭/ }).click();
  await page.getByRole('button', { name: '精簡 10 個' }).click();
  await page.getByRole('radio', { name: /快速/ }).click();
  await page.evaluate(() => { window.__silver.params.epochs = 40; window.__silver.params.backend = 'wasm'; });
  ok((await page.locator('#fcount').innerText()).startsWith('11'), '因子預設「精簡」= 11 個（含價格）');
  await page.screenshot({ path: `${SHOTS}/2-studio.png`, fullPage: true });
  await page.getByRole('button', { name: '開始訓練' }).click();
  await page.locator('#runpanel.show').waitFor();
  await page.waitForFunction(() => /\d+\/\d+/.test(document.querySelector('#rp-ep')?.textContent || ''), null, { timeout: 120000 });
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `${SHOTS}/3-training.png` });
  await page.getByRole('link', { name: '查看判讀' }).waitFor({ timeout: 300000 });
  await page.getByRole('link', { name: '查看判讀' }).click();
  await page.waitForTimeout(1200);
  const word = await page.locator('.signal .word').innerText();
  ok(word === '看漲' || word === '看跌', '總覽顯示判讀：' + word);
  ok(await page.locator('.conf').count() === 1, '雙頭模型顯示信心條');
  ok(await page.locator('.day').count() === 3, '顯示 T+1…T+3');
  await page.screenshot({ path: `${SHOTS}/4-overview.png`, fullPage: true });

  await page.goto('http://app.local/#/backtest');
  await page.getByRole('button', { name: '執行回測' }).click();
  await page.locator('#eqc canvas').first().waitFor({ timeout: 60000 });
  await page.waitForTimeout(800);
  ok(await page.locator('#bt-out .metric').count() === 4, '回測 KPI 顯示');
  await page.screenshot({ path: `${SHOTS}/5-backtest.png`, fullPage: true });

  await page.goto('http://app.local/#/factors');
  await page.waitForTimeout(800);
  ok(/COT/.test(await page.locator('.card').first().innerText()), '因子頁顯示資料來源狀態');
  await page.screenshot({ path: `${SHOTS}/6-factors.png`, fullPage: true });

  await page.goto('http://app.local/#/journal');
  await page.getByText('預測日誌').waitFor();
  await page.waitForTimeout(600);
  ok(await page.locator('tbody tr').count() >= 3, '預測日誌有紀錄');
  await page.screenshot({ path: `${SHOTS}/7-journal.png`, fullPage: true });

  // 主題切換
  await page.goto('http://app.local/#/overview');
  const th0 = await page.evaluate(() => document.documentElement.dataset.theme);
  await page.locator('#tb-theme').click();
  await page.waitForTimeout(700);
  ok(await page.evaluate(() => document.documentElement.dataset.theme) !== th0, '主題切換：' + th0 + ' → 另一個');
  await page.screenshot({ path: `${SHOTS}/8-overview-light.png` });  // 主流程用深色，這張為淺色
  await page.locator('#tb-theme').click();

  // 4) 重新開啟：快照路徑（先確認報價、立即進入）
  const t0 = Date.now();
  await page.reload();
  await page.locator('.shell').waitFor({ timeout: 60000 });
  ok(Date.now() - t0 < 15000, `有快照時快速進入（${((Date.now() - t0) / 1000).toFixed(1)} 秒）`);
  ok(await page.locator('.signal').count() === 1, '重新開啟後保留上次判讀');
  ok(!errs.length, '主流程沒有 JS 錯誤 ' + errs.join(' | '));

  // 5) 手機版面
  const m = await browser.newContext({ ...devices['iPhone 13'], colorScheme: 'dark' });
  const mp = await m.newPage();
  const merrs = await setup(mp);
  await mp.goto('http://app.local/');
  await mp.locator('.shell').waitFor({ timeout: 120000 });
  await mp.waitForTimeout(800);
  ok(await mp.locator('.tabbar').isVisible(), '手機顯示底部分頁列');
  ok(await mp.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), '手機沒有水平捲動');
  await mp.screenshot({ path: `${SHOTS}/9-mobile-home.png` });
  await mp.goto('http://app.local/#/studio');
  await mp.waitForTimeout(600);
  ok(await mp.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), '手機模型工作室沒有水平捲動');
  await mp.screenshot({ path: `${SHOTS}/10-mobile-studio.png`, fullPage: true });
  ok(!merrs.length, '手機沒有 JS 錯誤 ' + merrs.join(' | '));
  await browser.close();
})().catch(e => { console.error('FAIL', e); process.exit(1); });

/* Silver AI Trader v2 — 介面 */
(function () {
'use strict';
const C = window.SV_CONFIG, SE = window.SE, SP = window.SP, SC = window.SC;

// ══════════════════════════════════════════════════════════════
// 小工具
// ══════════════════════════════════════════════════════════════
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fin = Number.isFinite;
const px = (v, d = 3) => fin(v) ? '$' + v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) : '—';
const pct = (v, d = 2, sign = false) => fin(v) ? (sign && v >= 0 ? '+' : '') + v.toFixed(d) + '%' : '—';
const num = (v, d = 2) => fin(v) ? Math.abs(v) >= 1e5 ? v.toLocaleString('en-US', { maximumFractionDigits: 0 }) : v.toFixed(d) : '—';
const ico = (id) => `<svg class="ico" aria-hidden="true"><use href="#i-${id}"/></svg>`;
const tip = (t) => `<span class="tip" tabindex="0" role="img" aria-label="${esc(t)}" data-tip="${esc(t)}">?</span>`;
const ls = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (_) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (_) {} },
};
const ago = (iso) => {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return '剛剛'; if (s < 3600) return `${Math.floor(s / 60)} 分鐘前`;
  if (s < 86400) return `${Math.floor(s / 3600)} 小時前`; return `${Math.floor(s / 86400)} 天前`;
};
const fmtDate = (d) => { const [y, m, dd] = String(d).split('-'); return `${+m}/${+dd}`; };
const wd = (d) => '日一二三四五六'[new Date(d + 'T00:00:00Z').getUTCDay()];

function toast(msg, ms = 3200) {
  const el = document.createElement('div');
  el.className = 'toast'; el.textContent = msg;
  $('#toasts').appendChild(el);
  setTimeout(() => el.remove(), ms);
}
// ── 自有資料中繼（Cloudflare Worker）設定 ──
const RELAY_KEY = 'silver-proxy-url';
const GUIDE_URL = 'https://github.com/frk618-ship-it/silver-ai-trader/blob/main/worker/README.md';
const relayUrl = () => { try { return localStorage.getItem(RELAY_KEY) || ''; } catch (_) { return ''; } };
async function testRelay(u) {
  u = (u || '').trim().replace(/\/+$/, '');
  if (!/^https:\/\/[^\s/]+/.test(u)) throw new Error('網址需以 https:// 開頭');
  const q = 'https://query1.finance.yahoo.com/v8/finance/chart/SI=F?range=1d&interval=1m';
  const r = await fetch(`${u}/?url=${encodeURIComponent(q)}`, { signal: AbortSignal.timeout(15000) });
  const t = await r.text();
  let j = null; try { j = JSON.parse(t); } catch (_) {}
  if (!r.ok) throw new Error((j && j.error) || `HTTP ${r.status}（Yahoo 回應 ${r.headers.get('x-proxy-upstream-status') || '—'}）`);
  const m = j && j.chart && j.chart.result && j.chart.result[0] && j.chart.result[0].meta;
  if (!m || !m.regularMarketPrice) throw new Error('中繼有回應，但不是 Yahoo 的資料');
  return { url: u, price: m.regularMarketPrice };
}
function relayForm(prefix, compact = false) {
  const cur = relayUrl();
  return `<div class="field"><label for="${prefix}-url">資料中繼網址 ${tip('你自己的 Cloudflare Worker 網址。瀏覽器不能直接讀取 Yahoo／Stooq，免費的公開中繼又已停用，所以需要自己的中繼。')}</label>
    <div class="row" style="flex-wrap:nowrap"><input class="input mono" id="${prefix}-url" placeholder="https://silver-proxy.你的名稱.workers.dev" value="${esc(cur)}" autocomplete="off" spellcheck="false">
    <button class="btn" id="${prefix}-test">測試並儲存</button></div>
    <span class="help" id="${prefix}-msg">${cur ? '目前已設定。' : '尚未設定。'}${compact ? '' : ` 還沒有中繼？<a href="${GUIDE_URL}" target="_blank" rel="noopener">5 分鐘建立教學</a>（免費）`}</span></div>`;
}
function bindRelay(prefix, onSaved) {
  const btn = $(`#${prefix}-test`), msg = $(`#${prefix}-msg`), inp = $(`#${prefix}-url`);
  if (!btn) return;
  btn.onclick = async () => {
    const v = inp.value.trim();
    if (!v) { try { localStorage.removeItem(RELAY_KEY); } catch (_) {} msg.textContent = '已清除中繼設定。'; return; }
    btn.disabled = true; msg.textContent = '測試中…';
    try {
      const r = await testRelay(v);
      try { localStorage.setItem(RELAY_KEY, r.url); } catch (_) {}
      msg.innerHTML = `<span class="up">✓ 連線成功，白銀期貨 ${px(r.price)}。已儲存。</span>`;
      onSaved && onSaved();
    } catch (e) {
      const why = e.name === 'TypeError' ? '連不到這個網址（網址打錯、Worker 還沒部署，或 Worker 不允許這個網站呼叫）'
        : e.name === 'TimeoutError' ? '等候逾時（15 秒）' : e.message;
      msg.innerHTML = `<span class="down">✕ ${esc(why)}</span>`;
    } finally { btn.disabled = false; }
  };
}

function confirmDialog(title, body, okText = '確定', danger = false) {
  return new Promise((res) => {
    const d = document.createElement('dialog');
    d.innerHTML = `<h3 style="margin:0 0 8px">${esc(title)}</h3><p class="muted" style="margin:0 0 18px">${esc(body)}</p>
      <div class="row" style="justify-content:flex-end"><button class="btn" value="no">取消</button>
      <button class="btn ${danger ? 'danger' : 'primary'}" value="yes">${esc(okText)}</button></div>`;
    document.body.appendChild(d);
    d.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) d.close(b.value); });
    d.addEventListener('close', () => { res(d.returnValue === 'yes'); d.remove(); });
    d.showModal();
  });
}

// ══════════════════════════════════════════════════════════════
// 狀態
// ══════════════════════════════════════════════════════════════
const S = {
  raw: null,
  params: { ...C.DEFAULT_PARAMS, ...ls.get('silver-v2-params', {}) },
  factors: { ...C.defaultFactors(), ...ls.get('silver-v2-factors', {}) },
  trainPreset: ls.get('silver-v2-preset', 'standard'),
  result: ls.get('silver-v2-result', null),
  run: null,              // { kind, signal, ... } 執行中的工作
  bt: null, btCfg: { testRatio: 0.3, initialCapital: 100000, leverage: 1, positionPct: 1, feeBps: 2, slippageBps: 2, threshold: 0, allowShort: true },
  trials: null,
  range: ls.get('silver-v2-range', 130),
  refreshing: false,
};
const saveSettings = () => { ls.set('silver-v2-params', S.params); ls.set('silver-v2-factors', S.factors); ls.set('silver-v2-preset', S.trainPreset); };
const nFactors = () => Object.values(S.factors).filter(Boolean).length + 1;
const lastClose = () => S.raw ? S.raw.prices[S.raw.prices.length - 1] : NaN;
const quote = () => S.raw && S.raw.latestValues && S.raw.latestValues.liveSilver;

// COMEX 白銀交易時段（美東）：週日 18:00 – 週五 17:00，每日 17:00–18:00 休息
function marketOpen(now = new Date()) {
  const p = Object.fromEntries(new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', weekday: 'short', hour: 'numeric', hourCycle: 'h23' })
    .formatToParts(now).map(x => [x.type, x.value]));
  const h = +p.hour, d = p.weekday;
  if (d === 'Sat') return false;
  if (d === 'Sun') return h >= 18;
  if (d === 'Fri') return h < 17;
  return !(h === 17);
}

// ══════════════════════════════════════════════════════════════
// 🔴 閘門：沒有白銀期貨即時報價就不進入
// ══════════════════════════════════════════════════════════════
const GATE_STEPS = [
  { k: 'quote', t: '白銀期貨即時報價', d: '必要 · Yahoo Finance／Stooq' },
  { k: 'history', t: '白銀期貨歷史價格', d: '必要 · 近 5 年日線' },
  { k: 'aux', t: '24 項輔助因子', d: '黃金、美元、利率、COT、情緒…抓不到會以中性值補位' },
];

function renderGate(state = {}, error = null, hasSnap = false) {
  $('#app').innerHTML = `<div class="gate"><div class="gate-card card fade-in">
    <div class="brand"><span class="ag">Ag</span><div><b>白銀期貨 AI 預測</b><small>COMEX SI=F · v2</small></div></div>
    <h1>${error ? '無法取得白銀期貨即時價格' : '正在連線市場資料'}</h1>
    <p class="lead">${error ? '必須先取得即時報價與真實歷史價格才會進入；系統不會用模擬資料頂替。'
      : hasSnap ? '先確認即時報價，就用上次的資料立即開始，最新行情在背景更新。' : '第一次開啟需要下載約 5 年的歷史資料，通常 10–40 秒。'}</p>
    <ul class="checks">${GATE_STEPS.map(s => {
      const st = state[s.k] || 'wait';
      const mark = { ok: '✓', fail: '✕', warn: '!', run: '', wait: '' }[st];
      return `<li data-s="${st}"><span class="st">${mark}</span><div><b>${s.t}</b><small>${esc(state[s.k + 'Msg'] || s.d)}</small></div></li>`;
    }).join('')}</ul>
    ${error ? `<div class="alert bad" style="margin-top:18px"><b>原因</b>${esc(error)}</div>
      <div class="row" style="margin-top:16px"><button class="btn primary lg" id="g-retry">${ico('refresh')} 重新嘗試</button>
      <button class="btn ghost" id="g-clear">清除快取後重試</button></div>
      <div class="divider"></div>
      <h3 style="margin:0 0 4px;font-size:15px">${relayUrl() ? '檢查資料中繼' : '還沒設定資料中繼？'}</h3>
      <p class="faint" style="font-size:12.5px;margin:0 0 12px">瀏覽器不能直接讀取 Yahoo／Stooq 的資料，必須透過中繼。網路上免費的公開中繼已陸續停用，
        建議用 Cloudflare 免費建立自己的中繼（約 5 分鐘），網址貼在下面即可。</p>
      ${relayForm('g')}` : ''}
  </div></div>`;
  if (error) {
    $('#g-retry').onclick = () => boot();
    $('#g-clear').onclick = () => { SE.MarketDataCache.clear(); boot(true); };
    bindRelay('g', () => setTimeout(() => boot(), 600));
  }
}

async function boot(skipSnapshot = false) {
  const st = {};
  let snap = null;
  if (!skipSnapshot) { try { snap = await SE.AgentStore.get('snapshot', 'latest'); } catch (_) {} }
  const hasSnap = !!(snap && snap.data && snap.data.prices && snap.data.prices.length);
  let gateErr = null;   // 一旦顯示錯誤，之後才到的進度更新不可把錯誤畫面蓋掉
  const set = (k, s, msg) => {
    if (gateErr) return;
    st[k] = s; if (msg) st[k + 'Msg'] = msg;
    renderGate(st, null, hasSnap);
  };

  if (hasSnap) {
    set('quote', 'run');
    try {
      const q = await SE.fetchLiveSilverQuote();
      const last = snap.data.prices[snap.data.prices.length - 1];
      if (Math.abs(q.price / last - 1) > 0.3) throw new Error(`即時報價 $${q.price} 與資料最新收盤 $${last.toFixed(3)} 差距過大`);
      set('quote', 'ok', `${px(q.price)} · ${q.source}`);
      set('history', 'ok', `使用 ${new Date(snap.ts).toLocaleString('zh-TW')} 的資料，背景更新中`);
      set('aux', 'ok', '背景更新中');
      S.raw = { ...snap.data, latestValues: { ...snap.data.latestValues, liveSilver: q } };
      startApp();
      refreshData(true);
      return;
    } catch (e) {
      renderGate({ quote: 'fail' }, e.message, hasSnap);
      return;
    }
  }

  set('quote', 'run'); set('history', 'run'); set('aux', 'run');
  try {
    const raw = await SE.fetchMarketData(null, (k, info) => {
      if (k === 'quote') set('quote', 'ok', `${px(info.price)} · ${info.source}`);
      if (k === 'quote-fail') set('quote', 'fail', '取得失敗');
      if (k === 'history') set('history', 'ok', `${info.n.toLocaleString()} 個交易日 · ${info.sources[0]}`);
      if (k === 'aux') set('aux', 'ok', '完成');
    });
    S.raw = raw;
    try { await SE.AgentStore.put('snapshot', { key: 'latest', ts: new Date().toISOString(), data: raw }); } catch (_) {}
    startApp();
  } catch (e) {
    for (const s of GATE_STEPS) if (st[s.k] === 'run') st[s.k] = s.k === 'aux' ? 'wait' : 'fail';
    gateErr = e.message;
    renderGate(st, gateErr, hasSnap);
  }
}

/** 背景重新抓取全部資料（主畫面已顯示時）*/
async function refreshData(silent = false) {
  if (S.refreshing) return;
  S.refreshing = true; updateTopbar();
  try {
    const raw = await SE.fetchMarketData(null, null);
    try { await SE.AgentStore.put('snapshot', { key: 'latest', ts: new Date().toISOString(), data: raw }); } catch (_) {}
    if (S.run) { S.pendingRaw = raw; }
    else { S.raw = raw; render(); }
    if (!silent) toast('已更新到最新行情');
    else toast('行情已在背景更新完成');
  } catch (e) {
    toast('背景更新失敗，繼續使用目前資料：' + e.message.slice(0, 60), 5000);
  } finally { S.refreshing = false; updateTopbar(); }
}

// ══════════════════════════════════════════════════════════════
// 外殼
// ══════════════════════════════════════════════════════════════
const ROUTES = [
  { k: 'overview', t: '總覽', i: 'home' },
  { k: 'studio', t: '模型工作室', short: '模型', i: 'studio' },
  { k: 'factors', t: '因子與資料', short: '因子', i: 'factors' },
  { k: 'backtest', t: '回測', i: 'backtest' },
  { k: 'journal', t: '預測紀錄', short: '紀錄', i: 'journal' },
];
const route = () => { const r = location.hash.replace('#/', ''); return ROUTES.some(x => x.k === r) ? r : 'overview'; };

function startApp() {
  $('#app').innerHTML = `<div class="shell">
    <aside class="side">
      <div class="brand"><span class="ag">Ag</span><div><b>白銀期貨 AI</b><small>COMEX SI=F · v2</small></div></div>
      <nav class="nav" aria-label="主選單">${ROUTES.map(r => `<a href="#/${r.k}" data-r="${r.k}">${ico(r.i)}<span>${r.t}</span></a>`).join('')}</nav>
      <div class="side-foot">
        <button class="btn ghost sm" id="theme-btn" style="justify-content:flex-start">${ico('sun')}<span>切換主題</span></button>
        <span>模型在你的瀏覽器內訓練，資料不會上傳。</span>
        <span>僅供研究參考，不構成投資建議。</span>
      </div>
    </aside>
    <div class="main">
      <header class="topbar" id="topbar"></header>
      <main class="content" id="view" tabindex="-1"></main>
    </div>
  </div>
  <nav class="tabbar" aria-label="主選單">${ROUTES.map(r => `<a href="#/${r.k}" data-r="${r.k}">${ico(r.i)}<span>${r.short || r.t}</span></a>`).join('')}</nav>
  <section class="runpanel" id="runpanel" aria-live="polite"></section>`;
  $('#theme-btn').onclick = toggleTheme;
  window.addEventListener('hashchange', render);
  render();
  setInterval(refreshQuote, 60000);
  setInterval(updateTopbar, 30000);
}

function toggleTheme() {
  const t = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
  document.documentElement.dataset.theme = t;
  ls.set('silver-v2-theme', t); try { localStorage.setItem('silver-v2-theme', t); } catch (_) {}
  render();
}

function updateTopbar(flash) {
  const tb = $('#topbar'); if (!tb) return;
  const q = quote(), last = lastClose();
  const chg = q ? (q.price / last - 1) * 100 : NaN;
  const open = marketOpen();
  tb.innerHTML = `<a class="brand" href="#/overview" aria-label="首頁"><span class="ag">Ag</span><div></div></a>
    <div class="ticker">
      <div><div class="lbl"><span class="live-dot ${open ? '' : 'closed'}"></span>白銀期貨 · ${open ? '交易中' : '休市'}</div>
        <span class="px ${flash || ''}" id="tk-px">${q ? px(q.price) : '—'}</span></div>
      <span class="chg ${chg >= 0 ? 'up' : 'down'}" title="相對最新收盤 ${px(last)}">${pct(chg, 2, true)}</span>
      <span class="faint hide-m" style="font-size:12px">${q ? `${esc(q.source)} · ${new Date(q.time).toLocaleTimeString('zh-TW', { hour: '2-digit', minute: '2-digit' })}` : ''}</span>
    </div>
    <div class="top-actions">
      ${S.refreshing ? '<span class="pill acc hide-m">更新行情中…</span>' : ''}
      <button class="btn ghost icon" id="tb-refresh" title="重新抓取全部資料" aria-label="重新抓取全部資料" ${S.refreshing || S.run ? 'disabled' : ''}>${ico('refresh')}</button>
      <button class="btn ghost icon" id="tb-theme" title="切換主題" aria-label="切換主題">${ico(document.documentElement.dataset.theme === 'light' ? 'moon' : 'sun')}</button>
    </div>`;
  $('#tb-refresh').onclick = () => refreshData(false);
  $('#tb-theme').onclick = toggleTheme;
}

async function refreshQuote() {
  if (document.hidden || !S.raw) return;
  try {
    const prev = quote();
    const q = await SE.fetchLiveSilverQuote();
    S.raw.latestValues = { ...S.raw.latestValues, liveSilver: q };
    updateTopbar(prev && q.price !== prev.price ? (q.price > prev.price ? 'flash-up' : 'flash-down') : '');
    setTimeout(() => { const e = $('#tk-px'); if (e) e.className = 'px'; }, 1200);
  } catch (e) { console.warn('[quote]', e.message); }
}

function render() {
  const r = route();
  $$('[data-r]').forEach(a => a.toggleAttribute('aria-current', false));
  $$(`[data-r="${r}"]`).forEach(a => a.setAttribute('aria-current', 'page'));
  updateTopbar();
  const v = $('#view');
  v.innerHTML = '';
  v.className = 'content fade-in';
  ({ overview, studio, factors, backtest, journal })[r](v);
  document.title = `${ROUTES.find(x => x.k === r).t} · 白銀期貨 AI`;
}

// ══════════════════════════════════════════════════════════════
// 總覽
// ══════════════════════════════════════════════════════════════
function overview(v) {
  const r = S.result;
  const q = quote();
  if (!r) {
    const hasModel = !!(SP.readMeta().champion || SP.readMeta().best);
    v.innerHTML = `<div class="page-h"><div><h1>今天想怎麼開始？</h1><p>選一種方式產生明日的漲跌判讀。</p></div></div>
      <div class="grid g-3">
        <button class="action featured" data-act="auto"><span class="ai">✨</span><b>一鍵智慧預測</b>
          <span>AI 先試幾組模型與因子組合，挑出驗證表現最好的，再完整訓練並預測。第一次建議選這個（約 3–6 分鐘）。</span></button>
        <button class="action" data-act="predict" ${hasModel ? '' : 'disabled'}><span class="ai">⚡</span><b>用冠軍模型預測</b>
          <span>${hasModel ? '沿用目前最好的模型，幾秒鐘就有結果。' : '還沒有訓練過模型。'}</span></button>
        <a class="action" href="#/studio" style="text-decoration:none;color:inherit"><span class="ai">🛠</span><b>自己挑模型與因子</b>
          <span>到模型工作室選 4 種模型、4 種預測目標與 25 個因子。</span></a>
      </div>
      <div class="card" style="margin-top:16px"><div class="card-h"><div><h3>白銀期貨走勢</h3>
        <p class="sub" style="margin:0">${S.raw.dates.length.toLocaleString()} 個交易日 · 最新收盤 ${px(lastClose())}</p></div>${rangeSeg()}</div>
        <div class="chart" id="pc"></div></div>`;
    SC.priceChart($('#pc'), { dates: S.raw.dates, actual: S.raw.prices, live: q && q.price, range: S.range });
    bindOverview(v, S.raw.dates.length);
    return;
  }

  const up = r.futurePrices[0] >= r.lastClose;
  const ch = (r.futurePrices[0] / r.lastClose - 1) * 100;
  const conf = r.futureProbs ? Math.abs(r.futureProbs[0] - 0.5) * 200 : null;
  const stale = r.lastDate < S.raw.dates[S.raw.dates.length - 1];
  const hd = r.challenger && r.challenger.holdoutDA;
  const champDA = r.champion && r.champion.holdoutDA;
  const tgt = C.TARGETS.find(t => t.key === r.params.predTarget);
  let prev = r.lastClose;
  const days = r.futurePrices.map((p, i) => {
    const c = (p / prev - 1) * 100; prev = p;
    const pr = r.futureProbs ? Math.abs(r.futureProbs[i] - 0.5) * 200 : null;
    return `<div class="day"><div class="t"><span>T+${i + 1} · ${fmtDate(r.futureDates[i])}（${wd(r.futureDates[i])}）</span>${pr != null ? `<span>${pr.toFixed(0)}%</span>` : ''}</div>
      <div class="p">${px(p)}</div><div class="c ${c >= 0 ? 'up' : 'down'}">${c >= 0 ? '▲' : '▼'} ${pct(c, 2, true)}</div></div>`;
  }).join('');

  v.innerHTML = `<div class="page-h"><div><h1>明日判讀</h1>
      <p>以 ${fmtDate(r.lastDate)} 收盤 ${px(r.lastClose)} 為基準 · ${esc(r.params.algorithm)} · ${esc(tgt ? tgt.name : '')} · ${r.factors ? Object.values(r.factors).filter(Boolean).length + 1 : '—'} 個因子</p></div>
      <div class="row"><button class="btn" data-act="predict">${ico('refresh')} 用最新資料重新預測</button>
      <a class="btn primary" href="#/studio">調整模型</a></div></div>
    ${stale ? `<div class="alert warn" style="margin-bottom:16px"><b>有更新的收盤價了</b>這份判讀是用 ${fmtDate(r.lastDate)} 的資料做的，最新資料到 ${fmtDate(S.raw.dates[S.raw.dates.length - 1])}。按「用最新資料重新預測」更新。</div>` : ''}
    <div class="hero">
      <section class="signal ${up ? 'up' : 'down'}" aria-label="明日判讀">
        <div class="eyebrow"><span>${fmtDate(r.futureDates[0])}（週${wd(r.futureDates[0])}）· T+1</span>
          <span class="pill ${r.verdict && r.verdict.promote ? 'ok' : ''}">${esc(r.mode === 'predict' ? r.modelSource : (r.verdict && r.verdict.promote ? '🏆 新冠軍' : '挑戰者'))}</span></div>
        <div class="dir"><span class="arrow">${up ? '▲' : '▼'}</span><div><div class="word">${up ? '看漲' : '看跌'}</div>
          <div class="pct ${up ? 'up' : 'down'}">${pct(ch, 2, true)}</div></div></div>
        <div class="target"><span class="px">${px(r.futurePrices[0])}</span><span class="faint">預測收盤</span></div>
        ${conf != null ? `<div class="conf"><div class="row" style="justify-content:space-between;font-size:12.5px"><span class="muted">模型信心 ${tip('方向機率離 50% 越遠，信心越高。50% 代表模型無法判斷。')}</span><b class="mono">${conf.toFixed(0)}%</b></div>
          <div class="bar"><i style="width:${Math.min(conf, 100)}%;background:var(--${up ? 'up' : 'down'})"></i></div></div>` : ''}
        <div class="foot"><span>產生於 ${ago(r.at)}</span>·<span>${r.seconds.toFixed(1)} 秒</span>${q ? `·<span>即時價 ${px(q.price)}</span>` : ''}</div>
      </section>
      <div class="stack">
        <div class="strip">${days}</div>
        <div class="grid g-4">
          <div class="metric hl"><div class="k">保留集方向準確率 ${tip('最後約 15% 的資料完全沒參與訓練，用來檢驗模型。這是最值得參考的數字；50% 等於擲硬幣。')}</div>
            <div class="v ${fin(hd) && hd >= 55 ? 'up' : ''}">${pct(hd, 1)}</div>
            <div class="d">${fin(champDA) ? `冠軍 ${pct(champDA, 1)}` : `${r.challenger ? r.challenger.holdoutN : 0} 個交易日`}</div></div>
          <div class="metric"><div class="k">全期方向準確率 ${tip('包含訓練資料，通常比保留集樂觀。')}</div><div class="v">${pct(r.da, 1)}</div><div class="d">${r.dates.length} 個交易日</div></div>
          <div class="metric"><div class="k">平均誤差 MAPE ${tip('預測價格與實際價格的平均百分比誤差。')}</div><div class="v">${pct(r.mape, 2)}</div><div class="d">${r.params.predTarget === 'direction' ? '方向模型不輸出價格' : '價格偏離程度'}</div></div>
          <div class="metric"><div class="k">${r.cv ? '交叉驗證 DA' : '訓練'} ${tip(r.cv ? '時序前推交叉驗證的平均方向準確率。' : '本次訓練的 epoch 數與速度。')}</div>
            <div class="v">${r.cv ? pct(r.cv.da, 1) : r.train ? r.train.epochs : '—'}</div>
            <div class="d">${r.cv ? `${r.cv.folds.length} 折` : r.train ? `${r.train.eps.toFixed(1)} epochs/s · ${r.train.backend.toUpperCase()}` : '只預測'}</div></div>
        </div>
        ${r.verdict && r.mode !== 'predict' ? `<div class="alert ${r.verdict.promote ? 'info' : ''}" style="${r.verdict.promote ? '' : 'border-color:var(--border);background:var(--surface)'}">
          <b>${r.verdict.promote ? '🏆 晉升為冠軍模型' : '🛡️ 冠軍維持不變'}</b>${esc(r.verdict.reason)}</div>` : ''}
      </div>
    </div>
    <div class="card" style="margin-top:16px">
      <div class="card-h"><div class="legend"><span><i style="background:var(--text)"></i>實際收盤</span><span><i style="background:var(--accent-2)"></i>模型擬合</span>
        <span><i style="background:var(--${up ? 'up' : 'down'});height:4px"></i>預測</span>${q ? '<span><i style="background:var(--down);height:1px"></i>即時價</span>' : ''}</div>${rangeSeg()}</div>
      <div class="chart" id="pc"></div>
    </div>
    <p class="faint" style="font-size:12.5px;margin-top:14px">模型輸出僅供研究參考，不構成投資建議。過去的準確率不代表未來表現。</p>`;
  SC.priceChart($('#pc'), { dates: r.dates, actual: r.actual, fitted: r.fitted, futureDates: r.futureDates, futurePrices: r.futurePrices,
    live: q && q.price, holdStart: r.holdStart, range: S.range });
  bindOverview(v, r.dates.length + r.futureDates.length);
}

function rangeSeg() {
  const opts = [[22, '1 月'], [66, '3 月'], [130, '6 月'], [252, '1 年'], [0, '全部']];
  return `<div class="seg" role="group" aria-label="顯示範圍">${opts.map(([n, t]) => `<button data-range="${n}" aria-pressed="${S.range === n}">${t}</button>`).join('')}</div>`;
}
function bindOverview(v, total) {
  $$('[data-range]', v).forEach(b => b.onclick = () => {
    S.range = +b.dataset.range; ls.set('silver-v2-range', S.range);
    $$('[data-range]', v).forEach(x => x.setAttribute('aria-pressed', x === b));
    SC.setRange($('#pc'), S.range, total);
  });
  $$('[data-act]', v).forEach(b => b.onclick = () => {
    if (b.dataset.act === 'auto') startAuto({ trials: 8, epochs: 40, strategy: 'random' });
    if (b.dataset.act === 'predict') startRun('predict');
  });
}

// ══════════════════════════════════════════════════════════════
// 模型工作室
// ══════════════════════════════════════════════════════════════
function studio(v) {
  const p = S.params;
  const eps = ls.get('silver-v2-eps', {});
  const est = eps[p.algorithm] ? Math.round(p.epochs / eps[p.algorithm] + 8) : null;
  const running = !!S.run;
  const field = (k, label, min, max, step, help, fmt = (x) => x) => `<div class="field"><label for="f-${k}">${label}<output id="o-${k}">${fmt(p[k])}</output></label>
    <input type="range" id="f-${k}" data-p="${k}" min="${min}" max="${max}" step="${step}" value="${p[k]}">${help ? `<span class="help">${help}</span>` : ''}</div>`;

  v.innerHTML = `<div class="page-h"><div><h1>模型工作室</h1><p>四步驟設定：模型 → 預測目標 → 因子 → 訓練。設定會自動記住。</p></div></div>
  <div class="studio">
    <div class="stack">
      <section class="card"><div class="step"><span class="n">1</span><div><h2>選擇模型</h2><p>四種架構，與 v20.3 相同。</p></div></div>
        <div class="choices" role="radiogroup" aria-label="模型">${C.MODELS.map(m => `<button class="choice" role="radio" aria-checked="${p.algorithm === m.key}" data-algo="${m.key}">
          <span class="nm"><span style="font-size:18px">${m.icon}</span>${m.name}<span class="badge">${m.tag}</span></span><p>${m.desc}</p>
          <span class="row faint" style="font-size:12px;gap:14px"><span class="row" style="gap:6px">速度 <span class="meter">${[1, 2, 3, 4, 5].map(i => `<i class="${i <= m.speed ? 'on' : ''}"></i>`).join('')}</span></span>
          <span class="row" style="gap:6px">表達力 <span class="meter">${[1, 2, 3, 4, 5].map(i => `<i class="${i <= m.depth ? 'on' : ''}"></i>`).join('')}</span></span></span></button>`).join('')}</div>
      </section>

      <section class="card"><div class="step"><span class="n">2</span><div><h2>預測目標</h2><p>模型要學什麼。</p></div></div>
        <div class="choices" role="radiogroup" aria-label="預測目標">${C.TARGETS.map(t => `<button class="choice" role="radio" aria-checked="${p.predTarget === t.key}" data-target="${t.key}">
          <span class="nm">${t.name}</span><p>${t.desc}</p></button>`).join('')}</div>
      </section>

      <section class="card"><div class="step"><span class="n">3</span><div><h2>因子 <span class="pill acc" id="fcount">${nFactors()} / 25</span></h2><p>白銀期貨價格固定納入。點一下切換。</p></div></div>
        <div class="row" style="margin-bottom:16px">${C.FACTOR_PRESETS.map(fp => `<button class="btn sm" data-fpreset="${fp.key}">${fp.name}</button>`).join('')}</div>
        ${Object.entries(C.GROUPS).map(([g, info]) => {
          const fs = C.FACTORS.filter(f => f.group === g);
          return `<div class="fgroup" style="--g:${info.color}"><div class="gh"><span class="sw"></span>${info.name}<span class="faint" style="font-weight:400">${fs.length}</span>
            <span class="spacer"></span><button class="btn ghost sm" data-gtoggle="${g}">全選／全不選</button></div>
            <div class="chips">${fs.map(f => f.fixed ? `<span class="chip fixed" aria-pressed="true" title="固定納入"><span class="dot"></span>${f.name} 🔒</span>`
              : `<button class="chip" aria-pressed="${!!S.factors[f.key]}" data-factor="${f.key}" title="來源：${esc(f.src)}${f.hint ? '。' + esc(f.hint) : ''}"><span class="dot"></span>${f.name}</button>`).join('')}</div></div>`;
        }).join('')}
      </section>

      <section class="card"><div class="step"><span class="n">4</span><div><h2>訓練</h2><p>先選強度；細節在「進階參數」。</p></div></div>
        <div class="grid g-3" style="margin-bottom:16px">${C.TRAIN_PRESETS.map(tp => `<button class="choice" role="radio" aria-checked="${S.trainPreset === tp.key}" data-tpreset="${tp.key}">
          <span class="nm">${tp.name}</span><p>${tp.desc}</p><span class="faint" style="font-size:12px">${tp.p.epochs} epochs${tp.p.earlyStopPatience ? ` · 早停 ${tp.p.earlyStopPatience}` : ''}</span></button>`).join('')}</div>
        <details class="adv"><summary>進階參數</summary><div class="body">
          <div class="grid g-2" style="gap:18px 24px">
            ${field('lookback', 'Lookback（回看天數）', 5, 60, 1, '每個樣本看過去幾個交易日')}
            ${field('layers', '隱藏單元數', 16, 128, 16, '越大越能學複雜型態，也越慢、越容易過度擬合')}
            ${field('epochs', '訓練週期 Epochs', 10, 1500, 10, '完整看過訓練資料幾次')}
            ${field('batchSize', 'Batch Size', 8, 128, 8, '越大每 epoch 越快（32→128 約快 1.7 倍）')}
            ${field('lr', '學習率', 0.0002, 0.005, 0.0001, '', (x) => (+x).toFixed(4))}
            ${field('futureDays', '預測天數 T+N', 1, 10, 1, '')}
            ${field('directionalLambda', '方向懲罰 α', 0, 1, 0.05, '只用於「價格迴歸」：越高越重視漲跌方向', (x) => (+x).toFixed(2))}
            ${field('huberDelta', 'Huber δ', 0.5, 2, 0.5, '對極端值的容忍度', (x) => (+x).toFixed(1))}
            ${field('historyYears', '使用資料年數', 1, 5, 0.5, '', (x) => (+x).toFixed(1) + ' 年')}
            ${field('earlyStopPatience', '早停 Patience', 0, 200, 5, '連續 N 個 epoch 沒進步就停止；0 = 關閉', (x) => +x ? x : '關閉')}
            <div class="field"><span class="lab">驗證方式</span><div class="seg" role="group">${[['none', '保留集'], ['kfold', '＋時序交叉驗證']].map(([k, t]) =>
              `<button data-val="${k}" aria-pressed="${p.validationMode === k}">${t}</button>`).join('')}</div>
              <span class="help">交叉驗證只用過去資料訓練、測下一段，較可信但較慢</span></div>
            ${p.validationMode === 'kfold' ? field('kFolds', '交叉驗證折數', 2, 8, 1, '') : '<div></div>'}
            <div class="field"><label for="f-backend">運算後端</label><select class="input" id="f-backend" data-p="backend">
              ${[['auto', '自動測速（建議）'], ['webgl', 'WebGL（GPU）'], ['wasm', 'WASM（SIMD）'], ['cpu', 'CPU']].map(([k, t]) => `<option value="${k}" ${p.backend === k ? 'selected' : ''}>${t}</option>`).join('')}</select></div>
            <div class="field"><span class="lab">加速</span><label class="switch"><input type="checkbox" id="f-fast" ${p.fastRnn !== false ? 'checked' : ''}>融合 RNN 核心（BiGRU／LSTM 快 1.3–2 倍）</label></div>
          </div>
          <div class="row" style="margin-top:14px"><button class="btn ghost sm" id="reset-p">還原 v20.3 預設值</button></div>
        </div></details>
      </section>

      <section class="card" id="auto-card"><div class="card-h"><div><h3>✨ AI 智慧流程</h3><p class="sub" style="margin:0">自動嘗試多組模型、參數與因子組合 → 用驗證段排名 → 套用最佳組合完整訓練 → 預測。</p></div></div>
        <div class="grid g-3">
          <div class="field"><label for="a-trials">嘗試組合數<output id="o-trials">8</output></label><input type="range" id="a-trials" min="3" max="40" value="8"></div>
          <div class="field"><label for="a-ep">每組 Epochs<output id="o-aep">40</output></label><input type="range" id="a-ep" min="10" max="150" step="10" value="40"></div>
          <div class="field"><label for="a-st">搜尋策略</label><select class="input" id="a-st"><option value="random">隨機</option><option value="bayesian">簡化貝葉斯</option><option value="factors">只換因子組合</option></select></div>
        </div>
        <div class="row" style="margin-top:14px"><button class="btn" id="a-go" ${running ? 'disabled' : ''}>開始智慧流程</button>
          <span class="faint" style="font-size:12.5px">完成後會自動套用最佳設定到上面的步驟。</span></div>
        ${S.trials && S.trials.length ? trialsTable(S.trials) : ''}
      </section>
    </div>

    <aside class="summary card">
      <h3>本次設定</h3><p class="sub">確認後開始訓練</p>
      <dl>
        <dt>模型</dt><dd>${esc(C.MODELS.find(m => m.key === p.algorithm).name)}</dd>
        <dt>預測目標</dt><dd>${esc(C.TARGETS.find(t => t.key === p.predTarget).name)}</dd>
        <dt>因子</dt><dd>${nFactors()} 個</dd>
        <dt>Lookback</dt><dd>${p.lookback} 天</dd>
        <dt>Epochs</dt><dd>${p.epochs}${p.earlyStopPatience ? `（早停 ${p.earlyStopPatience}）` : ''}</dd>
        <dt>預測</dt><dd>T+1 … T+${p.futureDays}</dd>
        <dt>預估時間</dt><dd>${est ? `約 ${est < 90 ? est + ' 秒' : Math.round(est / 60) + ' 分鐘'}` : '<span class="faint" style="font-weight:400">首次會先測速</span>'}</dd>
      </dl>
      <div class="seg" role="group" aria-label="訓練方式" style="width:100%;margin-bottom:12px">
        <button data-mode="new" aria-pressed="${(S.mode || 'new') === 'new'}" style="flex:1">從零訓練</button>
        <button data-mode="resume" aria-pressed="${S.mode === 'resume'}" style="flex:1">接續上次</button></div>
      <button class="btn primary lg block" id="go" ${running ? 'disabled' : ''}>${running ? '執行中…' : '開始訓練'}</button>
      <button class="btn ghost block" id="go-predict" style="margin-top:8px" ${running || !(SP.readMeta().champion || SP.readMeta().best) ? 'disabled' : ''}>只用冠軍模型預測</button>
      <p class="faint" style="font-size:12px;margin:12px 0 0">訓練在你的瀏覽器裡進行；切到其他分頁也會繼續（改用 WASM 後端）。</p>
    </aside>
  </div>`;
  bindStudio(v);
}

function trialsTable(trials) {
  const rows = [...trials].sort((a, b) => b.da - a.da).slice(0, 10);
  return `<div class="table-wrap" style="margin-top:16px"><table><thead><tr><th>#</th><th>模型</th><th class="r">Lookback</th><th class="r">單元</th><th class="r">因子</th><th class="r">驗證 DA</th><th class="r">Loss</th></tr></thead>
    <tbody>${rows.map((t, i) => `<tr><td>${i === 0 ? '🏆' : t.idx}</td><td>${esc(t.params.algorithm)}</td><td class="r mono">${t.params.lookback}</td><td class="r mono">${t.params.layers}</td>
    <td class="r mono">${t.nFactors}</td><td class="r mono ${t.da >= 55 ? 'up' : ''}">${pct(t.da, 1)}</td><td class="r mono">${t.loss.toFixed(4)}</td></tr>`).join('')}</tbody></table></div>`;
}

function bindStudio(v) {
  const p = S.params;
  const rerender = () => { saveSettings(); const y = window.scrollY; studio(v); window.scrollTo(0, y); };
  $$('[data-algo]', v).forEach(b => b.onclick = () => { p.algorithm = b.dataset.algo; rerender(); });
  $$('[data-target]', v).forEach(b => b.onclick = () => { p.predTarget = b.dataset.target; rerender(); });
  $$('[data-factor]', v).forEach(b => b.onclick = () => {
    S.factors[b.dataset.factor] = !S.factors[b.dataset.factor];
    b.setAttribute('aria-pressed', S.factors[b.dataset.factor]);
    $('#fcount').textContent = `${nFactors()} / 25`; saveSettings();
    const dd = $$('.summary dd', v)[2]; if (dd) dd.textContent = `${nFactors()} 個`;
  });
  $$('[data-gtoggle]', v).forEach(b => b.onclick = () => {
    const ks = C.FACTORS.filter(f => f.group === b.dataset.gtoggle && !f.fixed).map(f => f.key);
    const on = !ks.every(k => S.factors[k]);
    ks.forEach(k => S.factors[k] = on); rerender();
  });
  $$('[data-fpreset]', v).forEach(b => b.onclick = () => { S.factors = C.FACTOR_PRESETS.find(x => x.key === b.dataset.fpreset).make(); rerender(); toast(`已套用「${b.textContent}」`); });
  $$('[data-tpreset]', v).forEach(b => b.onclick = () => { S.trainPreset = b.dataset.tpreset; Object.assign(p, C.TRAIN_PRESETS.find(x => x.key === b.dataset.tpreset).p); rerender(); });
  $$('input[type="range"][data-p]', v).forEach(inp => {
    inp.oninput = () => {
      const k = inp.dataset.p; p[k] = +inp.value;
      const o = $('#o-' + k); if (o) o.textContent = k === 'lr' ? (+inp.value).toFixed(4) : k === 'directionalLambda' ? (+inp.value).toFixed(2)
        : k === 'huberDelta' ? (+inp.value).toFixed(1) : k === 'historyYears' ? (+inp.value).toFixed(1) + ' 年' : k === 'earlyStopPatience' ? (+inp.value || '關閉') : inp.value;
    };
    inp.onchange = () => { S.trainPreset = 'custom'; rerender(); };
  });
  $$('[data-val]', v).forEach(b => b.onclick = () => { p.validationMode = b.dataset.val; rerender(); });
  $$('[data-mode]', v).forEach(b => b.onclick = () => { S.mode = b.dataset.mode; rerender(); });
  $('#f-backend').onchange = (e) => { p.backend = e.target.value; saveSettings(); };
  $('#f-fast').onchange = (e) => { p.fastRnn = e.target.checked; saveSettings(); };
  $('#reset-p').onclick = () => { S.params = { ...C.DEFAULT_PARAMS }; S.trainPreset = 'standard'; rerender(); toast('已還原 v20.3 預設值'); };
  $('#a-trials').oninput = (e) => $('#o-trials').textContent = e.target.value;
  $('#a-ep').oninput = (e) => $('#o-aep').textContent = e.target.value;
  $('#a-go').onclick = () => startAuto({ trials: +$('#a-trials').value, epochs: +$('#a-ep').value, strategy: $('#a-st').value });
  $('#go').onclick = () => startRun(S.mode || 'new');
  $('#go-predict').onclick = () => startRun('predict');
}

// ══════════════════════════════════════════════════════════════
// 執行：訓練／預測／智慧流程（浮動面板）
// ══════════════════════════════════════════════════════════════
let wakeLock = null;
async function lockScreen(on) {
  try {
    if (on && 'wakeLock' in navigator) wakeLock = await navigator.wakeLock.request('screen');
    if (!on && wakeLock) { await wakeLock.release(); wakeLock = null; }
  } catch (_) {}
}
document.addEventListener('visibilitychange', async () => {
  const SB = window.SilverBackend;
  if (!S.run || !SB) return;
  if (document.hidden && tf.getBackend() === 'webgl') { if (await SB.use(SB.bgBackend())) S.bgMode = true; }
  else if (!document.hidden && S.bgMode) { await SB.restore(); S.bgMode = false; if (wakeLock === null) lockScreen(true); }
});

function panel(html) {
  const el = $('#runpanel');
  el.innerHTML = html;
  el.classList.add('show');
  return el;
}

function openRunPanel(title, sub) {
  const el = panel(`<div class="rp-head"><div class="ring" id="rp-ring" style="--p:0"><span id="rp-pct">0%</span></div>
      <div style="min-width:0;flex:1"><b id="rp-title">${esc(title)}</b><small id="rp-sub">${esc(sub)}</small></div>
      <button class="btn ghost sm" id="rp-min" aria-label="縮小">－</button><button class="btn danger sm" id="rp-stop">停止</button></div>
    <div class="rp-body"><div class="rp-stats">
      <div><small>Epoch</small><b id="rp-ep">—</b></div><div><small>Loss</small><b id="rp-loss">—</b></div>
      <div><small>速度</small><b id="rp-eps">—</b></div><div><small>剩餘</small><b id="rp-eta">—</b></div></div>
      <canvas class="lossplot" id="rp-plot"></canvas><div class="rp-log" id="rp-log"></div><div id="rp-extra"></div></div>`);
  $('#rp-min').onclick = () => el.classList.toggle('min');
  $('#rp-stop').onclick = () => { if (S.run) { S.run.signal.stop = true; $('#rp-stop').disabled = true; $('#rp-sub').textContent = '正在停止…'; } };
}
const rpLog = (m) => { const l = $('#rp-log'); if (l) { const d = document.createElement('div'); d.textContent = m; l.prepend(d); } };
function rpProgress(pr, base = 0, span = 100) {
  if (pr.phase === 'backend') $('#rp-sub').textContent = '挑選最快的運算後端…';
  if (pr.phase === 'cv') $('#rp-sub').textContent = '時序交叉驗證中…';
  if (pr.phase === 'predict') { $('#rp-sub').textContent = '產生預測與因子分析…'; setRing(base + span * 0.97); }
  if (pr.phase === 'train' && pr.epochs) {
    const f = pr.epoch / pr.epochs;
    setRing(base + span * 0.95 * f);
    $('#rp-sub').textContent = `訓練中 · ${(pr.backend || '').toUpperCase()}`;
    $('#rp-ep').textContent = `${pr.epoch}/${pr.epochs}`;
    if (fin(pr.loss)) $('#rp-loss').textContent = pr.loss.toExponential(2);
    if (fin(pr.eps)) $('#rp-eps').textContent = pr.eps.toFixed(1) + '/s';
    if (fin(pr.eta)) $('#rp-eta').textContent = pr.eta < 90 ? Math.ceil(pr.eta) + ' 秒' : Math.ceil(pr.eta / 60) + ' 分';
    SC.lossPlot($('#rp-plot'), pr.losses, pr.bestEpoch);
  }
}
const setRing = (p) => { const r = $('#rp-ring'); if (r) { r.style.setProperty('--p', p.toFixed(0)); $('#rp-pct').textContent = p.toFixed(0) + '%'; } };

function finishPanel(res, err) {
  const ok = !!res;
  setRing(100);
  $('#rp-stop').outerHTML = `<button class="btn ghost sm" id="rp-close" aria-label="關閉">✕</button>`;
  $('#rp-close').onclick = () => $('#runpanel').classList.remove('show');
  if (!ok) {
    $('#rp-title').textContent = '執行失敗';
    $('#rp-sub').textContent = err;
    return;
  }
  const up = res.futurePrices[0] >= res.lastClose;
  $('#rp-title').innerHTML = `完成 · T+1 <span class="${up ? 'up' : 'down'}">${up ? '▲ 看漲' : '▼ 看跌'} ${pct((res.futurePrices[0] / res.lastClose - 1) * 100, 2, true)}</span>`;
  $('#rp-sub').textContent = res.verdict && res.mode !== 'predict' ? res.verdict.reason : res.modelSource;
  $('#rp-extra').innerHTML = `<div class="row" style="margin-top:12px"><a class="btn primary" href="#/overview" id="rp-view">查看判讀</a></div>`;
  $('#rp-view').onclick = () => setTimeout(() => $('#runpanel').classList.remove('show'), 300);
}

function notifyDone(title, body) {
  try { if (document.hidden && 'Notification' in window && Notification.permission === 'granted') new Notification(title, { body }); } catch (_) {}
}

async function guardRun(kind, fn) {
  if (S.run) { toast('已有工作在執行中'); return null; }
  S.run = { kind, signal: { stop: false } };
  lockScreen(true);
  try { if ('Notification' in window && Notification.permission === 'default') Notification.requestPermission(); } catch (_) {}
  render();
  try { return await fn(S.run.signal); }
  finally {
    S.run = null; lockScreen(false);
    if (S.pendingRaw) { S.raw = S.pendingRaw; S.pendingRaw = null; }
    render();
  }
}

function storeResult(res) {
  S.result = res;
  ls.set('silver-v2-result', res);
  if (res.train && res.train.eps) { const e = ls.get('silver-v2-eps', {}); e[res.params.algorithm] = res.train.eps; ls.set('silver-v2-eps', e); }
}

async function startRun(mode) {
  const label = mode === 'predict' ? '用冠軍模型預測' : mode === 'resume' ? '接續訓練' : '訓練模型';
  openRunPanel(label, '準備資料…');
  let res = null, err = null;
  await guardRun(mode, async (signal) => {
    try {
      res = await SP.run({ raw: S.raw, params: S.params, factors: S.factors, mode, signal,
        onProgress: (pr) => rpProgress(pr), onLog: rpLog });
      storeResult(res);
      notifyDone('白銀 AI：' + label + '完成', `T+1 ${px(res.futurePrices[0])}（${pct((res.futurePrices[0] / res.lastClose - 1) * 100, 2, true)}）`);
    } catch (e) { err = e.message; console.error(e); }
  });
  finishPanel(res, err);
  if (res) toast(`${label}完成`);
}

async function startAuto(cfg) {
  openRunPanel('AI 智慧流程', `搜尋 ${cfg.trials} 組組合…`);
  let res = null, err = null;
  await guardRun('auto', async (signal) => {
    try {
      $('#rp-ep').textContent = `0/${cfg.trials}`;
      const trials = await SP.search(S.raw, { ...S.params }, S.factors, cfg, (t, rec, all) => {
        setRing((t + 1) / cfg.trials * 45);
        $('#rp-ep').textContent = `${t + 1}/${cfg.trials}`;
        $('#rp-sub').textContent = `搜尋第 ${t + 1}/${cfg.trials} 組`;
        if (rec) {
          rpLog(`#${rec.idx} ${rec.params.algorithm} LB${rec.params.lookback} · 驗證 DA ${pct(rec.da, 1)}`);
          const best = [...all].sort((a, b) => b.da - a.da)[0];
          $('#rp-loss').textContent = pct(best.da, 1);
          $('#rp-eps').textContent = best.params.algorithm;
        }
      }, signal);
      S.trials = trials;
      if (!trials.length) throw new Error('沒有任何有效的組合（資料可能不足）');
      if (signal.stop) throw new Error('已停止');
      const best = [...trials].sort((a, b) => b.da - a.da)[0];
      S.params = { ...S.params, ...best.params, epochs: S.params.epochs, earlyStopPatience: S.params.earlyStopPatience || 40,
        futureDays: S.params.futureDays, historyYears: S.params.historyYears };
      S.factors = { ...best.factors };
      saveSettings();
      rpLog(`🏆 最佳組合：${best.params.algorithm} · lookback ${best.params.lookback} · 驗證 DA ${pct(best.da, 1)}，開始完整訓練`);
      $('#rp-title').textContent = 'AI 智慧流程 · 完整訓練';
      res = await SP.run({ raw: S.raw, params: S.params, factors: S.factors, mode: 'new', signal,
        onProgress: (pr) => rpProgress(pr, 45, 55), onLog: rpLog });
      storeResult(res);
      notifyDone('白銀 AI 智慧流程完成', `T+1 ${px(res.futurePrices[0])}`);
    } catch (e) { err = e.message; console.error(e); }
  });
  finishPanel(res, err);
  if (res) toast('智慧流程完成，已套用最佳設定');
}

// ══════════════════════════════════════════════════════════════
// 因子與資料
// ══════════════════════════════════════════════════════════════
function factors(v) {
  const raw = S.raw, lv = raw.latestValues || {}, src = lv.sources || {};
  const symName = { 'SI=F': '白銀期貨', 'SLV': 'SLV ETF', 'GC=F': '黃金期貨', 'CL=F': '原油期貨', 'DX-Y.NYB': '美元指數', '^VIX': 'VIX', '^TNX': '10 年期殖利率', 'HG=F': '銅期貨' };
  const rows = [
    ['白銀期貨即時報價', quote() ? 'real' : 'fallback', quote() ? `${quote().source} · ${new Date(quote().time).toLocaleString('zh-TW')}` : ''],
    ...Object.entries(src.symbols || {}).map(([s, v_]) => [symName[s] || s, /補位/.test(v_) ? 'fallback' : 'real', v_]),
    ['COT 籌碼（CFTC）', src.cot, src.cotRecords ? `${src.cotRecords} 份週報` : '以 0 補位'],
    ['TIPS 實質利率', src.tips, src.tips === 'real' ? '美國財政部' : '以 0.5% 補位'],
    ['CNN 恐懼與貪婪', src.fearGreed, lv.fearGreedCurrent != null ? `目前 ${num(lv.fearGreedCurrent, 0)} · ${esc(lv.fearGreedRating || '')}` : ''],
    ['StockTwits 情緒', src.stocktwits, lv.stTotalTagged ? `${lv.stTotalTagged} 則標記訊息 · 多方 ${pct(lv.stBullishPct, 0)}` : '以動量代理'],
    ['Google 搜尋熱度', src.gtrend, src.gtrend === 'real' ? 'Google Trends' : '以 50 補位'],
  ].filter(r => r[1]);
  const realN = rows.filter(r => r[1] === 'real').length;
  const r = S.result;
  const unit = r && r.params.predTarget === 'direction' ? '%' : '$';

  v.innerHTML = `<div class="page-h"><div><h1>因子與資料</h1><p>每個因子的最新值與近 60 日走勢；以及每個資料來源是真實資料還是補位。</p></div>
      <span class="pill ${realN === rows.length ? 'ok' : 'warn'}">${realN}/${rows.length} 個來源為真實資料</span></div>
    <div class="grid g-2" style="align-items:start">
      <div class="card"><h3>資料來源狀態</h3><p class="sub">補位的因子仍可使用，但對預測的幫助有限。</p>
        <div class="stack" style="gap:0">${rows.map(([n, s, d]) => `<div class="row" style="padding:10px 0;border-bottom:1px solid var(--border)">
          <span class="pill ${s === 'real' ? 'ok' : 'warn'}" style="min-width:56px;justify-content:center">${s === 'real' ? '真實' : '補位'}</span>
          <b style="font-size:14px">${esc(n)}</b><span class="spacer"></span><span class="faint" style="font-size:12.5px;text-align:right">${esc(d)}</span></div>`).join('')}</div></div>
      <div class="card"><h3>因子影響力 ${tip('把最近一天的某個因子調高一點點（正規化後 +5%），看 T+1 預測價格變多少。正 = 推高預測，負 = 壓低預測。')}</h3>
        <p class="sub">${r ? `來自 ${fmtDate(r.lastDate)} 的預測（${esc(r.params.algorithm)}）` : '完成一次預測後會顯示'}</p>
        ${r ? SC.influenceBars(r.influence, unit) : `<div class="empty"><div class="art">🧭</div><p class="muted">還沒有預測。</p><a class="btn" href="#/overview">去產生判讀</a></div>`}</div>
    </div>
    <div class="card" style="margin-top:16px"><h3>資料中繼</h3>
      <p class="sub">所有 Yahoo／Stooq 等資料都會先經過這個中繼（你的 Cloudflare Worker）。更換後按右上角 ↻ 重新抓取。</p>${relayForm('f')}</div>
    ${Object.entries(C.GROUPS).map(([g, info]) => `<div class="card" style="margin-top:16px;--g:${info.color}">
      <div class="card-h"><h3 class="row" style="gap:10px"><span style="width:10px;height:10px;border-radius:3px;background:var(--g)"></span>${info.name}</h3></div>
      <div class="ftiles">${C.FACTORS.filter(f => f.group === g).map(f => {
        const arr = raw[f.key === 'price' ? 'prices' : f.key] || [];
        const last = arr[arr.length - 1], ref = arr[arr.length - 21];
        const c = fin(ref) && ref !== 0 ? (last / ref - 1) * 100 : NaN;
        const used = f.fixed || S.factors[f.key];
        return `<div class="ftile"><div class="top"><span>${f.name}</span><span class="pill ${used ? 'acc' : ''}" style="padding:1px 8px">${used ? '使用中' : '未使用'}</span></div>
          <div class="row" style="justify-content:space-between"><span class="val">${num(last, Math.abs(last) < 10 ? 3 : 2)}</span>
          <span class="mono ${c >= 0 ? 'up' : 'down'}" style="font-size:12.5px" title="近 20 個交易日變化">${fin(c) && Math.abs(c) < 1000 ? pct(c, 1, true) : ''}</span></div>
          ${SC.sparkline(arr.slice(-60), 'var(--g)')}<span class="faint" style="font-size:11.5px">來源：${esc(f.src)}</span></div>`;
      }).join('')}</div></div>`).join('')}`;
  bindRelay('f', () => toast('中繼已儲存，按右上角 ↻ 用新中繼重新抓取資料'));
}

// ══════════════════════════════════════════════════════════════
// 回測
// ══════════════════════════════════════════════════════════════
function backtest(v) {
  const meta = SP.readMeta();
  const has = !!(meta.champion || meta.best);
  const c = S.btCfg;
  const f = (k, label, min, max, step, fmt) => `<div class="field"><label for="b-${k}">${label}<output id="bo-${k}">${fmt(c[k])}</output></label>
    <input type="range" id="b-${k}" data-b="${k}" min="${min}" max="${max}" step="${step}" value="${c[k]}"></div>`;
  v.innerHTML = `<div class="page-h"><div><h1>回測</h1><p>用${meta.champion ? '冠軍模型' : '最近一次訓練的模型'}，對最後一段資料逐日做 1 步預測，模擬依訊號進出場。</p></div></div>
    ${!has ? `<div class="card empty"><div class="art">💹</div><h3>需要先有一個模型</h3><p class="muted">訓練一次後就能回測。</p><a class="btn primary" href="#/studio">前往模型工作室</a></div>` : `
    <div class="card"><div class="grid g-4" style="gap:18px 24px">
      ${f('testRatio', '測試區間（最後）', 0.1, 0.8, 0.05, x => Math.round(x * 100) + '%')}
      ${f('leverage', '槓桿', 0.5, 5, 0.5, x => (+x).toFixed(1) + '×')}
      ${f('positionPct', '每次動用資金', 0.1, 1, 0.1, x => Math.round(x * 100) + '%')}
      ${f('threshold', '進場門檻（預測漲跌幅）', 0, 0.01, 0.0005, x => (x * 100).toFixed(2) + '%')}
      ${f('feeBps', '單邊手續費', 0, 20, 0.5, x => (+x).toFixed(1) + ' bps')}
      ${f('slippageBps', '單邊滑價', 0, 20, 0.5, x => (+x).toFixed(1) + ' bps')}
      <div class="field"><label for="b-cap">初始資金</label><input class="input mono" id="b-cap" type="number" min="1000" step="1000" value="${c.initialCapital}"></div>
      <div class="field"><span class="lab">放空</span><label class="switch"><input type="checkbox" id="b-short" ${c.allowShort ? 'checked' : ''}>預測下跌時放空</label></div>
    </div><div class="row" style="margin-top:16px"><button class="btn primary" id="b-go">執行回測</button>
      <span class="faint" style="font-size:12.5px">測試區間若與訓練資料重疊，結果會偏樂觀；保留集（最後約 15%）才是模型沒看過的。</span></div></div>
    <div id="bt-out" style="margin-top:16px"></div>`}`;
  if (!has) return;
  $$('[data-b]', v).forEach(inp => inp.oninput = () => {
    const k = inp.dataset.b; c[k] = +inp.value;
    const fm = { testRatio: x => Math.round(x * 100) + '%', leverage: x => (+x).toFixed(1) + '×', positionPct: x => Math.round(x * 100) + '%',
      threshold: x => (x * 100).toFixed(2) + '%', feeBps: x => (+x).toFixed(1) + ' bps', slippageBps: x => (+x).toFixed(1) + ' bps' }[k];
    $('#bo-' + k).textContent = fm(c[k]);
  });
  $('#b-cap').onchange = (e) => c.initialCapital = Math.max(1000, +e.target.value || 100000);
  $('#b-short').onchange = (e) => c.allowShort = e.target.checked;
  $('#b-go').onclick = async () => {
    const b = $('#b-go'); b.disabled = true; b.textContent = '回測中…';
    $('#bt-out').innerHTML = '<div class="card"><div class="skel" style="height:90px"></div><div class="skel" style="height:300px;margin-top:14px"></div></div>';
    try { S.bt = await SP.backtest(S.raw, c); drawBacktest(); }
    catch (e) { $('#bt-out').innerHTML = `<div class="alert bad"><b>回測失敗</b>${esc(e.message)}</div>`; }
    finally { b.disabled = false; b.textContent = '執行回測'; }
  };
  if (S.bt) drawBacktest();
}

function drawBacktest() {
  const b = S.bt, out = $('#bt-out'); if (!out) return;
  const ex = (b.totalReturn - b.bhReturn) * 100;
  out.innerHTML = `<div class="grid g-4">
      <div class="metric hl"><div class="k">策略總報酬</div><div class="v ${b.totalReturn >= 0 ? 'up' : 'down'}">${pct(b.totalReturn * 100, 2, true)}</div>
        <div class="d ${ex >= 0 ? 'up' : 'down'}">${pct(ex, 2, true)} vs 買進持有</div></div>
      <div class="metric"><div class="k">買進持有</div><div class="v">${pct(b.bhReturn * 100, 2, true)}</div><div class="d">同期間白銀漲跌</div></div>
      <div class="metric"><div class="k">最大回撤 ${tip('權益從高點往下掉的最大幅度。')}</div><div class="v down">${pct(-b.maxDD * 100, 2)}</div><div class="d">Sharpe ${num(b.sharpe, 2)}</div></div>
      <div class="metric"><div class="k">交易次數</div><div class="v">${b.tradeCount}</div><div class="d">勝率 ${pct(b.winRate * 100, 1)}</div></div>
    </div>
    <div class="card" style="margin-top:16px"><div class="card-h"><div class="legend"><span><i style="background:var(--accent)"></i>策略權益</span><span><i style="background:var(--text-3)"></i>買進持有</span></div>
      <span class="faint" style="font-size:12.5px">${esc(b.modelLabel)} · ${esc(b.params.algorithm)}</span></div><div class="chart sm" id="eqc"></div></div>
    <div class="card" style="margin-top:16px"><div class="card-h"><h3>交易明細</h3><button class="btn sm" id="b-csv">下載 CSV</button></div>
      <div class="table-wrap" style="max-height:360px"><table><thead><tr><th>日期</th><th>方向</th><th class="r">進場</th><th class="r">出場</th><th class="r">預測漲跌</th><th class="r">損益</th><th class="r">權益</th></tr></thead>
      <tbody>${b.trades.slice(-200).reverse().map(t => `<tr><td>${t.date}</td><td><span class="pill ${t.dir > 0 ? 'ok' : 'bad'}">${t.dir > 0 ? '做多' : '放空'}</span></td>
        <td class="r mono">${num(t.entry, 3)}</td><td class="r mono">${num(t.exit, 3)}</td><td class="r mono">${pct(t.predRet * 100, 2, true)}</td>
        <td class="r mono ${t.pnl >= 0 ? 'up' : 'down'}">${t.pnl >= 0 ? '+' : ''}${Math.round(t.pnl).toLocaleString()}</td><td class="r mono">${Math.round(t.eq).toLocaleString()}</td></tr>`).join('')}</tbody></table></div></div>`;
  SC.equityChart($('#eqc'), b.curve, b.bh);
  $('#b-csv').onclick = () => {
    const csv = '﻿日期,方向,進場,出場,預測價,預測漲跌%,損益,權益\n' + b.trades.map(t =>
      [t.date, t.dir > 0 ? '做多' : '放空', t.entry, t.exit, t.pred, (t.predRet * 100).toFixed(3), t.pnl.toFixed(2), t.eq.toFixed(2)].join(',')).join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = 'silver-backtest.csv'; a.click();
  };
}

// ══════════════════════════════════════════════════════════════
// 預測紀錄
// ══════════════════════════════════════════════════════════════
async function journal(v) {
  v.innerHTML = '<div class="page-h"><div><h1>預測紀錄</h1><p>每次預測都會記下，等實際收盤價出來後自動對帳。</p></div></div><div class="skel" style="height:120px"></div>';
  try { await SP.resolveAll(S.raw); } catch (_) {}
  const [preds, models, runs] = await Promise.all(['predictions', 'models', 'runs'].map(s => SE.AgentStore.all(s).catch(() => [])));
  if (route() !== 'journal') return;
  const perf = SE.recentPerformance(preds, 20);
  const advice = SE.shouldRetrain(perf);
  const champ = models.filter(m => m.status === 'champion').pop();
  const recent = [...preds].sort((a, b) => (b.madeAt || '').localeCompare(a.madeAt || '') || a.horizon - b.horizon).slice(0, 120);
  const t1 = preds.filter(p => p.resolved && p.horizon === 1 && p.correct != null);
  const t1da = t1.length ? t1.filter(p => p.correct === 1).length / t1.length * 100 : NaN;
  v.innerHTML = `<div class="page-h"><div><h1>預測紀錄</h1><p>每次預測都會記下，等實際收盤價出來後自動對帳。這是模型在真實世界的成績單。</p></div>
      <button class="btn danger sm" id="j-clear">清除所有紀錄與模型</button></div>
    <div class="grid g-4">
      <div class="metric hl"><div class="k">實測方向準確率 ${tip('最近 20 筆已對帳預測的命中率（各個 T+N 都算）。')}</div><div class="v">${perf.n ? pct(perf.da, 1) : '—'}</div><div class="d">${perf.n} 筆已對帳</div></div>
      <div class="metric"><div class="k">T+1 實測準確率</div><div class="v">${pct(t1da, 1)}</div><div class="d">${t1.length} 筆</div></div>
      <div class="metric"><div class="k">實測平均誤差</div><div class="v">${perf.n ? pct(perf.mape, 2) : '—'}</div><div class="d">預測價 vs 實際價</div></div>
      <div class="metric"><div class="k">現任冠軍</div><div class="v" style="font-size:20px">${champ ? esc(champ.algorithm) : '—'}</div>
        <div class="d">${champ ? `保留集 DA ${pct(champ.holdoutDA, 1)} · ${ago(champ.ts)}` : '還沒有冠軍'}</div></div>
    </div>
    <div class="alert ${advice.retrain ? 'warn' : 'info'}" style="margin-top:16px"><b>${advice.retrain ? '🔁 建議重新訓練' : '🧭 代理判斷'}</b>${esc(advice.reason)}
      ${advice.retrain ? '<div style="margin-top:10px"><a class="btn sm" href="#/studio">前往模型工作室</a></div>' : ''}</div>
    <div class="card" style="margin-top:16px"><div class="card-h"><h3>預測日誌</h3><span class="faint" style="font-size:12.5px">最近 ${recent.length} 筆</span></div>
      ${recent.length ? `<div class="table-wrap" style="max-height:460px"><table><thead><tr><th>目標日</th><th>T+</th><th class="r">基準價</th><th class="r">預測</th><th class="r">實際</th><th>結果</th><th class="r">信心</th><th>模型</th><th>預測時間</th></tr></thead>
      <tbody>${recent.map(p => {
        const res = !p.resolved ? '<span class="pill">⏳ 待對帳</span>' : p.correct === 1 ? '<span class="pill ok">✓ 命中</span>' : '<span class="pill bad">✕ 未中</span>';
        const dir = p.predPrice >= p.basisPrice;
        return `<tr><td>${p.targetDate}</td><td class="mono">${p.horizon}</td><td class="r mono">${num(p.basisPrice, 3)}</td>
          <td class="r mono ${dir ? 'up' : 'down'}">${dir ? '▲' : '▼'} ${num(p.predPrice, 3)}</td><td class="r mono">${p.resolved ? num(p.actualPrice, 3) : '—'}</td>
          <td>${res}</td><td class="r mono">${p.confidence != null ? p.confidence.toFixed(0) + '%' : '—'}</td><td>${esc(p.algorithm)}</td><td class="faint">${ago(p.madeAt)}</td></tr>`;
      }).join('')}</tbody></table></div>` : `<div class="empty"><div class="art">📒</div><p class="muted">還沒有預測紀錄。</p><a class="btn" href="#/overview">產生第一個判讀</a></div>`}</div>
    <div class="grid g-2" style="margin-top:16px;align-items:start">
      <div class="card"><h3>模型註冊表</h3><p class="sub">每次訓練都在同一段保留集上和冠軍比較，贏了才晉升。</p>
        ${models.length ? `<div class="table-wrap" style="max-height:300px"><table><thead><tr><th>時間</th><th>模型</th><th class="r">保留集 DA</th><th>狀態</th></tr></thead><tbody>
        ${[...models].reverse().slice(0, 50).map(m => `<tr><td class="faint">${ago(m.ts)}</td><td>${esc(m.algorithm)} · LB${m.lookback}</td><td class="r mono">${pct(m.holdoutDA, 1)}</td>
          <td><span class="pill ${m.status === 'champion' ? 'ok' : ''}">${{ champion: '🏆 冠軍', challenger: '挑戰者', retired: '退役' }[m.status] || m.status}</span></td></tr>`).join('')}</tbody></table></div>` : '<p class="faint">尚無</p>'}</div>
      <div class="card"><h3>執行紀錄</h3><p class="sub">訓練與預測的歷程。</p>
        ${runs.length ? `<div class="table-wrap" style="max-height:300px"><table><thead><tr><th>時間</th><th>類型</th><th>模型</th><th class="r">DA</th><th class="r">秒</th></tr></thead><tbody>
        ${[...runs].reverse().slice(0, 50).map(r => `<tr><td class="faint">${ago(r.ts)}</td><td>${r.type === 'train' ? '訓練' : '預測'}${r.promoted ? ' 🏆' : ''}</td><td>${esc(r.algorithm)}</td>
          <td class="r mono">${pct(r.da, 1)}</td><td class="r mono">${r.seconds ? r.seconds.toFixed(0) : '—'}</td></tr>`).join('')}</tbody></table></div>` : '<p class="faint">尚無</p>'}</div>
    </div>`;
  $('#j-clear').onclick = async () => {
    if (!(await confirmDialog('清除所有紀錄與模型？', '預測日誌、模型註冊表、冠軍與最近一次訓練的模型都會刪除，無法復原。', '全部清除', true))) return;
    await SP.clearModels();
    S.result = null; ls.set('silver-v2-result', null);
    toast('已清除'); render();
  };
}

// ══════════════════════════════════════════════════════════════
// 啟動
// ══════════════════════════════════════════════════════════════
if (!window.tf || !window.SE) {
  $('#app').innerHTML = '<div class="gate"><div class="card gate-card"><h1>元件載入失敗</h1><p class="lead">無法載入 TensorFlow.js，請檢查網路後重新整理。</p></div></div>';
} else {
  boot();
}
window.__silver = S;   // 方便除錯
})();

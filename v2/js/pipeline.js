/* Silver AI Trader v2 — 流程層：訓練／預測／保留集裁決／回測／AI 搜尋
 * 不碰 DOM；透過 callback 回報進度，讓介面自由呈現。
 */
(function () {
'use strict';
const SE = window.SE;
const tf = SE.tf;

const BEST_URL = 'indexeddb://silver-v2-best';
const CHAMP_URL = 'indexeddb://silver-v2-champion';
const META_KEY = 'silver-v2-model-meta';     // { best: {...}, champion: {...} }

const readMeta = () => { try { return JSON.parse(localStorage.getItem(META_KEY)) || {}; } catch (_) { return {}; } };
const writeMeta = (patch) => {
  const m = { ...readMeta(), ...patch };
  try { localStorage.setItem(META_KEY, JSON.stringify(m)); } catch (_) {}
  return m;
};

const sig = (z) => 1 / (1 + Math.exp(-z));

// TF.js 的 WASM 後端沒有卷積的反向傳播 kernel（Conv2DBackpropFilter），Transformer 不能在 WASM 上訓練
const needsNoWasm = (algo) => algo === 'Transformer';
async function ensureBackendFor(algo) {
  const SB = window.SilverBackend;
  window._noWasm = needsNoWasm(algo);
  if (window._noWasm && tf.getBackend() === 'wasm') {
    if (!(await SB.use('webgl'))) await SB.use('cpu');
  }
}

/** 下 n 個交易日（略過週末） */
function nextTradingDays(lastDateStr, n) {
  const out = [];
  const d = new Date(lastDateStr + 'T00:00:00Z');
  while (out.length < n) {
    d.setUTCDate(d.getUTCDate() + 1);
    const wd = d.getUTCDay();
    if (wd !== 0 && wd !== 6) out.push(d.toISOString().slice(0, 10));
  }
  return out;
}

/** 把模型原始輸出轉成「預測價格」與（可能的）方向機率 */
function decode(raw, i, target, ds, prev) {
  const stride = target === 'dual' ? 2 : 1;
  if (target === 'logreturn') {
    const st = ds.minMax.logRet || { mean: 0, std: 1 };
    return { price: prev * Math.exp(raw[i] * st.std + st.mean), prob: null };
  }
  if (target === 'direction') return { price: prev * (raw[i] > 0.5 ? 1.0001 : 0.9999), prob: raw[i] };
  if (target === 'dual') return { price: SE.denormalizePrice(raw[i * stride], ds.minMax), prob: sig(raw[i * stride + 1]) };
  return { price: SE.denormalizePrice(raw[i], ds.minMax), prob: null };
}

/** 對資料集中 [start, end) 的樣本推論 */
async function predictRange(model, ds, start, end, target) {
  const xs = ds.X.slice(start, end);
  if (!xs.length) return { prices: [], probs: [] };
  const prices = [], probs = [];
  for (let s = 0; s < xs.length; s += 256) {
    const chunk = xs.slice(s, s + 256);
    const t = tf.tensor3d(chunk);
    const p = model.predict(t);
    const raw = await p.data();
    tf.dispose([t, p]);
    for (let i = 0; i < chunk.length; i++) {
      const d = decode(raw, i, target, ds, ds.prevPrices[start + s + i]);
      prices.push(d.price); probs.push(d.prob);
    }
  }
  return { prices, probs };
}

function scoreRange(ds, start, preds, probs, target) {
  const actual = [], prevs = [];
  for (let i = 0; i < preds.length; i++) {
    actual.push(SE.denormalizePrice(ds.y[start + i], ds.minMax));
    prevs.push(ds.prevPrices[start + i]);
  }
  if (target === 'direction' || target === 'dual') {
    let ok = 0;
    for (let i = 0; i < probs.length; i++) if ((probs[i] > 0.5) === (actual[i] > prevs[i])) ok++;
    const s = SE.scoreHoldout(actual, preds, prevs);
    return { ...s, holdoutDA: probs.length ? ok / probs.length * 100 : NaN,
             holdoutMAPE: target === 'direction' ? NaN : s.holdoutMAPE };
  }
  return SE.scoreHoldout(actual, preds, prevs);
}

/** 時序前推交叉驗證：第 i 折只用之前的資料訓練，測下一段（不拿未來訓練） */
async function walkForwardCV(ds, params, k, onLog, signal) {
  const n = ds.X.length, edges = [];
  for (let i = 0; i <= k + 1; i++) edges.push(Math.floor(n * i / (k + 1)));
  const folds = [];
  const spec = SE.resolveTargetSpec(params.predTarget, params);
  for (let i = 0; i < k; i++) {
    if (signal && signal.stop) break;
    const trEnd = edges[i + 1], teEnd = edges[i + 2];
    if (trEnd < 30 || teEnd - trEnd < 5) continue;
    onLog && onLog(`交叉驗證第 ${i + 1}/${k} 折：訓練 ${trEnd} 筆、測試 ${teEnd - trEnd} 筆`);
    const m = SE.createModel(params.algorithm, params.lookback, ds.numFeatures, params);
    m.compile({ optimizer: tf.train.adam(params.lr), loss: spec.loss });
    const x = tf.tensor3d(ds.X.slice(0, trEnd));
    const y = SE.buildTargetTensor(ds.y.slice(0, trEnd), ds.yDir.slice(0, trEnd), params.predTarget, ds.yLogRet.slice(0, trEnd));
    await m.fit(x, y, { epochs: Math.min(params.epochs, 150), batchSize: params.batchSize, yieldEvery: 'never', verbose: 0,
      callbacks: [{ onEpochEnd: async () => { if (signal && signal.stop) m.stopTraining = true; } }, SE.makeBatchYielder(200)] });
    tf.dispose([x, y]);
    const { prices, probs } = await predictRange(m, ds, trEnd, teEnd, params.predTarget);
    const s = scoreRange(ds, trEnd, prices, probs, params.predTarget);
    folds.push({ fold: i + 1, da: s.holdoutDA, mape: s.holdoutMAPE, nTrain: trEnd, nTest: teEnd - trEnd });
    try { m.optimizer && m.optimizer.dispose(); } catch (_) {}
    m.dispose();
  }
  const avg = (k_) => { const v = folds.map(f => f[k_]).filter(Number.isFinite); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN; };
  return { folds, da: avg('da'), mape: avg('mape') };
}

async function tryLoad(url, ds) {
  try {
    const m = await tf.loadLayersModel(url);
    const s = m.layers[0] && m.layers[0].batchInputShape;
    if (s && ds && (s[1] !== ds.X[0].length || s[2] !== ds.X[0][0].length)) { m.dispose(); return null; }
    return m;
  } catch (_) { return null; }
}

/**
 * 主流程
 * opts: { raw, params, factors, mode: 'new'|'resume'|'predict', onProgress, onLog, signal }
 */
async function run(opts) {
  const holder = { model: null };
  try { return await runInner(opts, holder); }
  finally { if (holder.model) { try { holder.model.dispose(); } catch (_) {} } }
}

async function runInner(opts, holder) {
  const { raw, mode = 'new', onProgress, onLog, signal = {} } = opts;
  let params = { ...opts.params };
  let factors = { ...opts.factors };
  const log = (m) => onLog && onLog(m);
  const t0 = performance.now();
  const meta = readMeta();
  let modelSource = '本次訓練';

  if (mode === 'predict') {
    const which = meta.champion ? 'champion' : (meta.best ? 'best' : null);
    if (!which) throw new Error('還沒有已存的模型，請先訓練一次。');
    const mm = meta[which];
    params = { ...params, ...mm.params, futureDays: params.futureDays, historyYears: params.historyYears };
    factors = { ...factors, ...mm.factors };
    modelSource = which === 'champion' ? '冠軍模型' : '最近一次訓練的模型';
  }

  const data = SE.filterDataByYears(raw, params.historyYears);
  const ds = SE.createMultivariateDataset(data, params.lookback, factors);
  if (!ds.X.length) throw new Error('資料量不足以建立時間窗，請縮短 lookback 或增加資料年數。');
  const hs = SE.splitHoldout(ds.X.length);
  const target = params.predTarget || 'price';
  log(`資料集 ${ds.X.length} 個樣本 · ${ds.numFeatures} 個特徵 · 保留集最後 ${hs.holdLen} 筆`);

  let model = null, train = null, cv = null;
  if (mode === 'predict') {
    model = await tryLoad(meta.champion ? CHAMP_URL : BEST_URL, ds);
    holder.model = model;
    if (!model && meta.champion) { model = await tryLoad(BEST_URL, ds); holder.model = model; modelSource = '最近一次訓練的模型'; }
    if (!model) throw new Error('已存模型與目前資料形狀不符，請重新訓練。');
    log(`使用${modelSource}（${params.algorithm}）`);
  } else {
    onProgress && onProgress({ phase: 'backend' });
    if (needsNoWasm(params.algorithm) && params.backend === 'wasm') {
      params = { ...params, backend: 'auto' };
      log('Transformer 無法在 WASM 上訓練，改為自動選擇後端');
    }
    window._noWasm = needsNoWasm(params.algorithm);
    await SE.prepareTrainingBackend(params, ds.numFeatures, ds.X,
      (n) => SE.buildTargetTensor(ds.y.slice(0, n), ds.yDir.slice(0, n), target, ds.yLogRet.slice(0, n)), log);

    if (params.validationMode === 'kfold') {
      onProgress && onProgress({ phase: 'cv' });
      cv = await walkForwardCV(ds, params, params.kFolds || 5, log, signal);
      if (cv.folds.length) log(`交叉驗證平均：DA ${cv.da.toFixed(2)}% · MAPE ${Number.isFinite(cv.mape) ? cv.mape.toFixed(2) + '%' : '—'}`);
    }

    if (mode === 'resume' && meta.best && meta.best.params.algorithm === params.algorithm &&
        meta.best.params.predTarget === target && meta.best.params.layers === params.layers) {
      model = await tryLoad(BEST_URL, ds);
      log(model ? '從上次最佳權重接續訓練' : '舊模型與目前設定不相容，改為從零開始');
    }
    if (!model) model = SE.createModel(params.algorithm, params.lookback, ds.numFeatures, params);
    holder.model = model;
    const spec = SE.resolveTargetSpec(target, params);
    model.compile({ optimizer: tf.train.adam(params.lr), loss: spec.loss });

    const trEnd = hs.holdLen > 0 ? hs.trainEnd : ds.X.length;
    const x = tf.tensor3d(ds.X.slice(0, trEnd));
    const y = SE.buildTargetTensor(ds.y.slice(0, trEnd), ds.yDir.slice(0, trEnd), target, ds.yLogRet.slice(0, trEnd));
    const losses = [];
    let best = Infinity, bestEpoch = -1, snap = null, since = 0, stoppedEarly = false;
    const patience = Math.max(0, params.earlyStopPatience | 0);
    const tick = SE.makeTicker(200), yieldTick = SE.makeTicker(100);
    const tt = performance.now();
    window._trainingActive = true;
    onProgress && onProgress({ phase: 'train', epoch: 0, epochs: params.epochs, losses });
    try {
      await model.fit(x, y, {
        epochs: params.epochs, batchSize: params.batchSize, yieldEvery: 'never', verbose: 0,
        callbacks: [{
          onEpochEnd: async (ep, logs) => {
            const L = logs.loss;
            losses.push(L);
            if (L < best - 1e-9) { best = L; bestEpoch = ep; snap = SE.snapshotWeights(model, snap); since = 0; }
            else if (patience && ++since >= patience) { model.stopTraining = true; stoppedEarly = true; }
            if (signal.stop) model.stopTraining = true;
            if (tick(ep === params.epochs - 1)) {
              const sec = (performance.now() - tt) / 1000;
              const eps = (ep + 1) / Math.max(sec, 1e-3);
              onProgress && onProgress({ phase: 'train', epoch: ep + 1, epochs: params.epochs, loss: L, best, bestEpoch,
                eps, eta: (params.epochs - ep - 1) / eps, losses, backend: tf.getBackend() });
            }
            if (yieldTick()) await SE.fastYield();
          },
        }, SE.makeBatchYielder(200)],
      });
    } finally {
      window._trainingActive = false;
      tf.dispose([x, y]);
    }
    if (snap) { model.setWeights(snap); SE.disposeSnapshot(snap); }
    try { model.optimizer && model.optimizer.dispose(); } catch (_) {}
    const sec = (performance.now() - tt) / 1000;
    train = { losses, best, bestEpoch, epochs: losses.length, seconds: sec, eps: losses.length / Math.max(sec, 1e-3),
              stoppedEarly, stoppedByUser: !!signal.stop, backend: tf.getBackend() };
    log(`訓練完成：${losses.length} epochs · ${sec.toFixed(1)} 秒（${train.eps.toFixed(1)} epochs/s，${train.backend.toUpperCase()}）` +
        (stoppedEarly ? ' · 早停' : '') + (signal.stop ? ' · 使用者中止' : ''));
    await model.save(BEST_URL);
    writeMeta({ best: { params, factors, ts: new Date().toISOString(), bestLoss: best } });
  }

  // ── 預測 ──
  onProgress && onProgress({ phase: 'predict' });
  const pr = await SE.runMultiStepPrediction(model, ds.X, ds.y, ds.minMax, ds.activeFeatures, ds.numFeatures,
    params.lookback, params.futureDays, log, target, ds.prevPrices);
  const { prices: fitted, probs: fittedProbs } = await predictRange(model, ds, 0, ds.X.length, target);
  const all = scoreRange(ds, 0, fitted, fittedProbs, target);
  const actual = ds.y.map(v => SE.denormalizePrice(v, ds.minMax));
  const lastDate = data.dates[data.dates.length - 1];
  const lastClose = data.prices[data.prices.length - 1];
  const futureDates = nextTradingDays(lastDate, params.futureDays);

  // ── 保留集裁決 ──
  let challenger = { holdoutDA: NaN, holdoutMAPE: NaN, holdoutN: 0 }, champion = null, verdict = { promote: false, reason: modelSource };
  if (hs.holdLen > 0) {
    const h = await predictRange(model, ds, hs.holdStart, ds.X.length, target);
    challenger = scoreRange(ds, hs.holdStart, h.prices, h.probs, target);
  }
  if (mode !== 'predict') {
    const cm = meta.champion;
    if (cm && hs.holdLen > 0 && (cm.params.predTarget || 'price') === target) {
      const champModel = await tryLoad(CHAMP_URL, ds);
      if (champModel) {
        const h = await predictRange(champModel, ds, hs.holdStart, ds.X.length, target);
        champion = scoreRange(ds, hs.holdStart, h.prices, h.probs, target);
        champModel.dispose();
      }
    }
    verdict = SE.shouldPromote(champion, challenger);
    log((verdict.promote ? '🏆 晉升為新冠軍：' : '🛡️ 維持現任冠軍：') + verdict.reason);
    if (verdict.promote) {
      await model.save(CHAMP_URL);
      writeMeta({ champion: { params, factors, ts: new Date().toISOString(), ...challenger } });
    }
    try {
      const models = await SE.AgentStore.all('models');
      if (verdict.promote) for (const m of models.filter(m => m.status === 'champion')) await SE.AgentStore.put('models', { ...m, status: 'retired' });
      await SE.AgentStore.add('models', { ts: new Date().toISOString(), algorithm: params.algorithm, predTarget: target,
        lookback: params.lookback, layers: params.layers, epochs: train ? train.epochs : 0, ...challenger,
        inSampleDA: all.holdoutDA, inSampleMAPE: all.holdoutMAPE, status: verdict.promote ? 'champion' : 'challenger', reason: verdict.reason });
    } catch (e) { console.warn('[Agent] models', e); }
  }

  // ── 紀錄：執行紀錄、預測日誌、對帳 ──
  try {
    await SE.AgentStore.add('runs', { ts: new Date().toISOString(), type: mode === 'predict' ? 'predict' : 'train',
      algorithm: params.algorithm, predTarget: target, da: all.holdoutDA, mape: all.holdoutMAPE,
      holdoutDA: challenger.holdoutDA, promoted: verdict.promote ? 1 : 0, note: verdict.reason,
      seconds: (performance.now() - t0) / 1000 });
    for (let i = 0; i < pr.futurePrices.length; i++) {
      await SE.AgentStore.add('predictions', { madeAt: new Date().toISOString(), basisDate: lastDate, basisPrice: lastClose,
        targetDate: futureDates[i], horizon: i + 1, predPrice: pr.futurePrices[i], predDir: pr.futurePrices[i] >= lastClose ? 1 : 0,
        confidence: pr.futureDirProbs && pr.futureDirProbs[i] != null ? Math.abs(pr.futureDirProbs[i] - 0.5) * 200 : null,
        algorithm: params.algorithm, predTarget: target, resolved: 0 });
    }
    await resolveAll(raw);
  } catch (e) { console.warn('[Agent] log', e); }

  return {
    mode, params, factors, modelSource, train, cv, challenger, champion, verdict,
    dates: ds.X.map((_, i) => data.dates[params.lookback + i] || data.dates[data.dates.length - ds.X.length + i]),
    actual, fitted, fittedProbs, holdStart: hs.holdLen > 0 ? hs.holdStart : null,
    futureDates, futurePrices: pr.futurePrices, futureProbs: pr.futureDirProbs && pr.futureDirProbs.length ? pr.futureDirProbs : null,
    influence: renameInfluence(pr.influenceScores, ds.activeFeatures), da: all.holdoutDA, mape: all.holdoutMAPE,
    lastClose, lastDate, seconds: (performance.now() - t0) / 1000, at: new Date().toISOString(),
  };
}

/** 因子敏感度的鍵改用 v2 介面上的因子名稱 */
function renameInfluence(infl, active) {
  const out = {};
  const byOld = Object.fromEntries((active || []).map(f => [f.name, f.key]));
  const nm = Object.fromEntries(window.SV_CONFIG.FACTORS.map(f => [f.key, f.name]));
  for (const [k, v] of Object.entries(infl || {})) out[nm[byOld[k]] || k] = v;
  return out;
}

/** 把所有未對帳的預測與實際收盤價比對 */
async function resolveAll(raw) {
  const pending = await SE.AgentStore.all('predictions');
  const byDate = {};
  raw.dates.forEach((d, i) => { byDate[d] = raw.prices[i]; });
  const done = SE.resolvePredictions(pending.filter(p => !p.resolved), byDate);
  for (const r of done) await SE.AgentStore.put('predictions', r);
  return done.length;
}

// ══════════════════════════════════════════════════════════════
// 回測
// ══════════════════════════════════════════════════════════════
async function backtest(raw, cfg) {
  const meta = readMeta();
  const which = meta.champion ? 'champion' : (meta.best ? 'best' : null);
  if (!which) throw new Error('需要先訓練一個模型才能回測。');
  const mm = meta[which];
  const params = mm.params;
  const data = SE.filterDataByYears(raw, params.historyYears);
  const ds = SE.createMultivariateDataset(data, params.lookback, mm.factors);
  const model = await tryLoad(which === 'champion' ? CHAMP_URL : BEST_URL, ds);
  if (!model) throw new Error('已存模型與目前資料不相容，請重新訓練。');
  const ratio = Math.min(Math.max(cfg.testRatio, 0.05), 0.8);
  const split = Math.floor(ds.X.length * (1 - ratio));
  const target = params.predTarget || 'price';
  const { prices: pred, probs } = await predictRange(model, ds, split, ds.X.length, target);
  model.dispose();
  const cost = (cfg.feeBps + cfg.slippageBps) / 10000;
  let eq = cfg.initialCapital, peak = eq, maxDD = 0, wins = 0;
  const curve = [], bh = [], rets = [], trades = [];
  const entry0 = ds.prevPrices[split];
  for (let i = 0; i < pred.length; i++) {
    const gi = split + i;
    const entry = ds.prevPrices[gi], exit = SE.denormalizePrice(ds.y[gi], ds.minMax);
    const date = data.dates[params.lookback + gi];
    const predRet = target === 'direction' ? (probs[i] - 0.5) * 0.02 : (pred[i] - entry) / entry;
    const dir = predRet > cfg.threshold ? 1 : (cfg.allowShort && predRet < -cfg.threshold ? -1 : 0);
    const qty = dir ? (eq * cfg.positionPct * cfg.leverage / entry) * dir : 0;
    const pnl = qty * (exit - entry) - Math.abs(qty) * (entry + exit) * cost;
    const prev = eq;
    eq += pnl;
    peak = Math.max(peak, eq);
    maxDD = Math.max(maxDD, peak > 0 ? (peak - eq) / peak : 0);
    rets.push(prev > 0 ? eq / prev - 1 : 0);
    curve.push({ date, value: eq });
    bh.push({ date, value: cfg.initialCapital * exit / entry0 });
    if (dir) { if (pnl > 0) wins++; trades.push({ date, dir, entry, exit, pred: pred[i], predRet, pnl, eq }); }
  }
  const mean = rets.reduce((a, b) => a + b, 0) / Math.max(rets.length, 1);
  const sd = Math.sqrt(rets.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(rets.length - 1, 1));
  return {
    curve, bh, trades, modelLabel: which === 'champion' ? '冠軍模型' : '最近一次訓練的模型', params,
    totalReturn: eq / cfg.initialCapital - 1, bhReturn: bh.length ? bh[bh.length - 1].value / cfg.initialCapital - 1 : 0,
    maxDD, sharpe: sd > 0 ? mean / sd * Math.sqrt(252) : 0, tradeCount: trades.length,
    winRate: trades.length ? wins / trades.length : 0, finalEquity: eq,
  };
}

// ══════════════════════════════════════════════════════════════
// AI 參數搜尋：每個試驗用前 80% 訓練、最後 20% 驗證排名（避免挑到背熟訓練資料的組合）
// ══════════════════════════════════════════════════════════════
const SPACE = {
  algorithm: ['BiGRU', 'LSTM', 'Transformer', 'TimesFM'],
  lookback: [5, 10, 15, 20, 30],
  layers: [16, 32, 48, 64],
  batchSize: [16, 32, 64],
  directionalLambda: [0.0, 0.1, 0.2, 0.3, 0.5],
  huberDelta: [0.5, 1.0, 1.5],
  lr: [0.0005, 0.001, 0.002],
};

async function search(raw, base, baseFactors, cfg, onTrial, signal = {}) {
  const pick = (a) => a[Math.floor(Math.random() * a.length)];
  const keys = window.SV_CONFIG.FACTORS.slice(1).map(f => f.key);
  const data = SE.filterDataByYears(raw, base.historyYears);
  const trials = [];
  let bestLossSoFar = Infinity;
  for (let t = 0; t < cfg.trials; t++) {
    if (signal.stop) break;
    let p, f;
    if (cfg.strategy === 'bayesian' && trials.length >= 4) {
      const b = [...trials].sort((a, c) => c.da - a.da)[0];
      p = { ...base }; for (const [k, v] of Object.entries(SPACE)) p[k] = Math.random() < 0.6 ? b.params[k] : pick(v);
      f = {}; for (const k of keys) f[k] = Math.random() < 0.7 ? b.factors[k] : Math.random() > 0.3;
    } else if (cfg.strategy === 'factors') {
      p = { ...base }; f = {}; for (const k of keys) f[k] = Math.random() > 0.3;
    } else {
      p = { ...base }; for (const [k, v] of Object.entries(SPACE)) p[k] = pick(v);
      f = {}; for (const k of keys) f[k] = Math.random() > 0.3;
    }
    p.epochs = cfg.epochs;
    await ensureBackendFor(p.algorithm);
    const t0 = performance.now();
    let model = null, x = null, y = null, rec = null;
    try {
      const ds = SE.createMultivariateDataset(data, p.lookback, f, 520);
      if (ds.X.length < 40) { onTrial && onTrial(t, null, trials); continue; }
      const cut = Math.floor(ds.X.length * 0.8);
      model = SE.createModel(p.algorithm, p.lookback, ds.numFeatures, p);
      const spec = SE.resolveTargetSpec(p.predTarget, p);
      model.compile({ optimizer: tf.train.adam(p.lr), loss: spec.loss });
      x = tf.tensor3d(ds.X.slice(0, cut));
      y = SE.buildTargetTensor(ds.y.slice(0, cut), ds.yDir.slice(0, cut), p.predTarget, ds.yLogRet.slice(0, cut));
      let bl = Infinity, since = 0, ep = 0, last = Infinity;
      await model.fit(x, y, { epochs: p.epochs, batchSize: Math.max(p.batchSize, 64), yieldEvery: 'never', verbose: 0,
        callbacks: [{ onEpochEnd: async (e, logs) => {
          ep = e + 1; last = logs.loss;
          if (last < bl - 1e-6) { bl = last; since = 0; } else if (++since >= 8) model.stopTraining = true;
          if (e >= 10 && bestLossSoFar < Infinity && last > bestLossSoFar * 4) model.stopTraining = true;
          if (signal.stop) model.stopTraining = true;
          if (e % 5 === 4) await SE.fastYield();
        } }, SE.makeBatchYielder(250)] });
      bestLossSoFar = Math.min(bestLossSoFar, bl);
      const v = await predictRange(model, ds, cut, ds.X.length, p.predTarget);
      const s = scoreRange(ds, cut, v.prices, v.probs, p.predTarget);
      rec = { idx: t + 1, params: p, factors: f, da: s.holdoutDA, mape: s.holdoutMAPE, loss: bl, epochs: ep,
              nFactors: Object.values(f).filter(Boolean).length + 1, seconds: (performance.now() - t0) / 1000 };
      trials.push(rec);
    } catch (e) {
      console.warn('[search] trial failed', e);
    } finally {
      if (x) tf.dispose(x); if (y) tf.dispose(y);
      if (model) { try { model.optimizer && model.optimizer.dispose(); } catch (_) {} model.dispose(); }
    }
    onTrial && onTrial(t, rec, trials);
    await SE.fastYield();
  }
  return trials;
}

async function clearModels() {
  for (const u of [BEST_URL, CHAMP_URL]) { try { await tf.io.removeModel(u); } catch (_) {} }
  localStorage.removeItem(META_KEY);
  for (const s of ['runs', 'models', 'predictions']) { try { await SE.AgentStore.clear(s); } catch (_) {} }
}

window.SP = { run, backtest, search, resolveAll, readMeta, clearModels, nextTradingDays };
})();

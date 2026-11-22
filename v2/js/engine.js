/* Silver AI Trader v2 — 引擎（資料、模型、訓練、預測）
 * 大部分程式碼取自 V20.4（已驗證），依 v2 介面需求整理。
 * 全部掛在 window.SE 命名空間，供 app.js 使用。
 */
(function () {
'use strict';
const tf = window.tf;
// ── ⚡ SilverBackend：後端偏好 + 自動測速 ──
//    preferred：前景使用的後端（'webgl' | 'wasm' | 'cpu'），由使用者選擇或 auto 測速決定
//    背景分頁：WebGL 會被暫停 → 改用 WASM（比 CPU 快約 4 倍），沒有 WASM 才退回 CPU
window.SilverBackend = {
    preferred: 'webgl',
    _benchCache: {},
    available: function () {
        const reg = (window.tf && tf.engine && tf.engine().registryFactory) ? Object.keys(tf.engine().registryFactory) : [];
        return ['webgl', 'wasm', 'cpu'].filter(b => reg.includes(b));
    },
    bgBackend: function () { return (!window._noWasm && this.available().includes('wasm')) ? 'wasm' : 'cpu'; },
    use: async function (name) {
        if (!name) return false;
        if (tf.getBackend() === name) return true;
        try { const ok = await tf.setBackend(name); await tf.ready(); return ok !== false && tf.getBackend() === name; }
        catch (e) { console.warn('[Backend] switch to', name, 'failed:', e && e.message); return false; }
    },
    restore: async function () {
        if (await this.use(this.preferred)) return this.preferred;
        for (const b of this.available()) { if (await this.use(b)) return b; }
        return tf.getBackend();
    },
    // 用「實際要訓練的模型結構」在每個可用後端跑幾個 batch，挑最快的。
    // buildModel: () => 已 compile 的 tf.LayersModel；xs/ys 為訓練用 tensor 的切片
    autoSelect: async function (key, buildModel, makeSample, log) {
        if (this._benchCache[key]) { this.preferred = this._benchCache[key]; await this.use(this.preferred); return this.preferred; }
        const original = tf.getBackend();
        const results = {};
        const avail = this.available();
        let cands = avail.length > 1 ? avail.filter(b => b !== 'cpu') : avail;
        if (window._noWasm) cands = cands.filter(b => b !== 'wasm');
        if (!cands.length) cands = ['cpu'];   // CPU 幾乎不可能最快，省下測速時間
        for (const b of cands) {
            if (!(await this.use(b))) continue;
            let m = null, x = null, y = null;
            try {
                [x, y] = makeSample();
                m = buildModel();
                const step = async () => { const h = await m.trainOnBatch(x, y); return h; };
                await step(); await step();                           // 暖機（shader 編譯 / wasm 初始化）
                const t0 = performance.now(); let n = 0;
                while (n < 3 || (performance.now() - t0 < 400 && n < 20)) { await step(); n++; }
                results[b] = (performance.now() - t0) / n;
            } catch (e) {
                console.warn('[Backend] benchmark failed on', b, e && e.message);
            } finally {
                try { if (m && m.optimizer) m.optimizer.dispose(); } catch (_) {}
                try { if (m) m.dispose(); } catch (_) {}
                try { if (x) x.dispose(); if (y) y.dispose(); } catch (_) {}
            }
        }
        const ranked = Object.entries(results).sort((a, c) => a[1] - c[1]);
        const best = ranked.length ? ranked[0][0] : original;
        if (log) log('⚡ 後端測速（每 batch）：' + ranked.map(([b, ms]) => `${b.toUpperCase()} ${ms.toFixed(1)}ms`).join(' / ') + ` → 使用 ${best.toUpperCase()}`);
        this._benchCache[key] = best;
        this.preferred = best;
        await this.use(best);
        return best;
    }
};

const denormalizePrice = (val, minMax) => val * (minMax.price.max - minMax.price.min) + minMax.price.min;

// ==========================================
// 自定義 Loss Function (MAPE + MSE) — 已改用 DirectionalHuberLoss
// 保留以備向後兼容
// ==========================================
const getCustomLoss = (lambda) => {
    return (yTrue, yPred) => {
        return tf.tidy(() => {
            const mape = tf.metrics.meanAbsolutePercentageError(yTrue, yPred);
            const mse = tf.metrics.meanSquaredError(yTrue, yPred);
            const weightedMape = tf.mul(mape, 1 - lambda);
            const weightedMse = tf.mul(mse, lambda);
            return tf.add(weightedMape, weightedMse);
        });
    };
};

// ==========================================
// 🎯 Directional Huber Loss — 核心 DA 提升 Loss
// 結合 Huber（穩健回歸）+ 方向懲罰（Direction Penalty）
// alpha: 方向懲罰權重 (0=純 Huber, 1=純方向懲罰)
// delta: Huber 分界點
// 原理：對於 batch 中的連續樣本 [i-1, i]，
//   若 yTrue 往上但 yPred < yTrue[i-1]，或 yTrue 往下但 yPred > yTrue[i-1]，
//   則施加額外懲罰，逼使模型預測正確方向
// ==========================================
const getDirectionalHuberLoss = (alpha = 0.3, delta = 1.0) => {
    return (yTrue, yPred) => {
        return tf.tidy(() => {
            // 基礎 Huber loss（對極端值穩健）
            const huber = tf.losses.huberLoss(yTrue, yPred, delta);

            const batchSize = yTrue.shape[0];
            if (batchSize <= 1 || alpha <= 0) return huber;

            // 取連續批次對：[0..n-2] vs [1..n-1]
            const truePrev  = tf.slice(yTrue, [0], [batchSize - 1]);   // y[i-1]
            const trueCurr  = tf.slice(yTrue, [1], [batchSize - 1]);   // y[i]   (actual)
            const predCurr  = tf.slice(yPred, [1], [batchSize - 1]);   // ŷ[i]   (predicted)

            // 實際方向向量（正=漲，負=跌）
            const actualDir   = tf.sub(trueCurr, truePrev);
            // 預測方向向量（相對於上一個真實值）
            const predictedDir = tf.sub(predCurr, truePrev);

            // 方向懲罰：relu(- actualDir * predictedDir)
            // 方向相同 → 乘積 > 0 → relu=0（無懲罰）
            // 方向相反 → 乘積 < 0 → relu>0（施加懲罰）
            const dotProduct  = tf.mul(actualDir, predictedDir);
            const dirPenalty  = tf.mean(tf.relu(tf.neg(dotProduct)));

            // 組合: (1-α) * Huber + α * DirectionPenalty
            const combined = tf.add(
                tf.mul(huber,      tf.scalar(1 - alpha)),
                tf.mul(dirPenalty, tf.scalar(alpha))
            );
            return combined;
        });
    };
};

// ==========================================
// Huber Loss（穩健回歸，對極端值較不敏感）
// ==========================================
const getHuberLoss = (delta = 1.0) => {
    return (yTrue, yPred) => {
        return tf.tidy(() => tf.losses.huberLoss(yTrue, yPred, delta));
    };
};

// ══════════════════════════════════════════════════════════════
// 雙頭損失（dual）：輸出 [price, dir_logit]
//   loss = Huber(price) + λ · BCE(sigmoid(dir_logit), dirLabel)
//   yTrue 形狀 [n,2]：欄0=正規化價格, 欄1=方向標籤(0/1)
// ══════════════════════════════════════════════════════════════
const getDualHeadLoss = (lambda = 0.5, delta = 1.0) => {
    return (yTrue, yPred) => tf.tidy(() => {
        const priceTrue = yTrue.slice([0, 0], [-1, 1]);
        const dirTrue   = yTrue.slice([0, 1], [-1, 1]);
        const pricepred = yPred.slice([0, 0], [-1, 1]);
        const dirLogit  = yPred.slice([0, 1], [-1, 1]);
        const priceLoss = tf.losses.huberLoss(priceTrue, pricepred, delta);
        const dirProb   = tf.sigmoid(dirLogit);
        const eps = tf.scalar(1e-7);
        const dirLoss = tf.neg(tf.mean(tf.add(
            tf.mul(dirTrue, tf.log(tf.add(dirProb, eps))),
            tf.mul(tf.sub(tf.scalar(1), dirTrue), tf.log(tf.add(tf.sub(tf.scalar(1), dirProb), eps)))
        )));
        return tf.add(priceLoss, tf.mul(tf.scalar(lambda), dirLoss));
    });
};

// 依預測目標回傳對應的 loss + 輸出單元數 + 輸出 activation
const resolveTargetSpec = (predTarget, params) => {
    const lambda = params.directionalLambda ?? 0.3;
    const delta  = params.huberDelta ?? 1.0;
    if (predTarget === 'direction') {
        return { units: 1, activation: 'sigmoid', loss: 'binaryCrossentropy', outputDim: 1 };
    }
    if (predTarget === 'dual') {
        return { units: 2, activation: 'linear', loss: getDualHeadLoss(lambda, delta), outputDim: 2 };
    }
    if (predTarget === 'logreturn') {
        // 目標本身已是報酬率 → 用純 Huber。
        // 不加方向懲罰項是刻意的：λ 的作用是拿價格精度換方向精度，
        // 而此模式的用途正是追求價格準確度。
        return { units: 1, activation: 'linear', loss: getHuberLoss(delta), outputDim: 1 };
    }
    // default: price
    return { units: 1, activation: 'linear', loss: getDirectionalHuberLoss(lambda, delta), outputDim: 1 };
};

// 依預測目標建立訓練目標張量
//   price     → [n,1] 正規化價格
//   direction → [n,1] 方向標籤 0/1
//   dual      → [n,2] [價格, 方向]
//   logreturn → [n,1] 標準化後的 ln(P_t/P_{t-1})
const buildTargetTensor = (yArr, yDirArr, predTarget, yLogRetArr) => {
    const n = yArr.length;
    if (predTarget === 'logreturn') {
        return tf.tensor(Float32Array.from((yLogRetArr || []).slice(0, n)), [n, 1]);
    }
    if (predTarget === 'direction') {
        return tf.tensor(Float32Array.from(yDirArr.slice(0, n)), [n, 1]);
    }
    if (predTarget === 'dual') {
        const flat = new Float32Array(n * 2);
        for (let i = 0; i < n; i++) { flat[i*2] = yArr[i]; flat[i*2+1] = yDirArr[i]; }
        return tf.tensor(flat, [n, 2]);
    }
    return tf.tensor(Float32Array.from(yArr.slice(0, n)), [n, 1]);
};



// ==========================================
// MAPE 指標計算函數
// ==========================================
const calculateMAPE = (actualPrices, predictedPrices) => {
    if (!actualPrices?.length || !predictedPrices?.length) return 0;
    const n = Math.min(actualPrices.length, predictedPrices.length);
    if (n === 0) return 0;
    let sumAbsPercentageError = 0;
    let count = 0;
    for (let i = 0; i < n; i++) {
        const actual = actualPrices[i];
        const predicted = predictedPrices[i];
        if (actual !== 0) {
            const absPercentageError = Math.abs((actual - predicted) / actual);
            sumAbsPercentageError += absPercentageError;
            count++;
        }
    }
    if (count === 0) return 0;
    return (sumAbsPercentageError / count) * 100;
};


// ==========================================
// 1. 核心指標計算 (RSI & ROC)
// ==========================================
const calculateRSI = (prices, period = 5) => { 
    if (prices.length < period) return new Array(prices.length).fill(50);
    let rsi = new Array(period).fill(50);
    let avgGain = 0, avgLoss = 0;
    for (let i = 1; i <= period; i++) {
        let change = prices[i] - prices[i-1];
        if (change >= 0) avgGain += change; else avgLoss -= change;
    }
    avgGain /= period; avgLoss /= period;
    if (avgLoss === 0) rsi[period] = 100; else rsi[period] = 100 - (100 / (1 + (avgGain / avgLoss)));
    for (let i = period + 1; i < prices.length; i++) {
        let change = prices[i] - prices[i-1];
        let currentGain = change > 0 ? change : 0;
        let currentLoss = change < 0 ? -change : 0;
        avgGain = (avgGain * (period - 1) + currentGain) / period;
        avgLoss = (avgLoss * (period - 1) + currentLoss) / period;
        if (avgLoss === 0) rsi.push(100); else rsi.push(100 - (100 / (1 + (avgGain / avgLoss))));
    }
    return prices.map((_, i) => i < period ? 50 : rsi[i]);
};

const calculateROC = (data, isDifference = false) => {
    const roc = [0]; 
    for (let i = 1; i < data.length; i++) {
        const prev = data[i-1];
        const curr = data[i];
        if (isDifference || prev === 0) {
            roc.push(curr - prev); 
        } else {
            const change = (curr - prev) / prev;
            if (!isFinite(change)) roc.push(0);
            else roc.push(change);
        }
    }
    return roc;
};

// ==========================================
// 計算移動平均線 (Simple Moving Average)
// ==========================================
const calculateMA = (prices, period) => {
    const ma = [];
    for (let i = 0; i < prices.length; i++) {
        if (i < period - 1) {
            // 資料不足時使用現有數據的平均
            const available = prices.slice(0, i + 1);
            ma.push(available.reduce((a, b) => a + b, 0) / available.length);
        } else {
            const window = prices.slice(i - period + 1, i + 1);
            ma.push(window.reduce((a, b) => a + b, 0) / period);
        }
    }
    return ma;
};

// ==========================================
// 計算 MACD 信號線差值 (fast - slow EMA normalized)
// ==========================================
const calculateEMA = (prices, period) => {
    const k = 2 / (period + 1);
    const ema = [prices[0]];
    for (let i = 1; i < prices.length; i++) {
        ema.push(prices[i] * k + ema[i-1] * (1 - k));
    }
    return ema;
};

// ==========================================
// 🔄 ProxyRotator — 全域 CORS Proxy 輪替器
//    - 維護多個備援 proxy，自動跳過近期失敗的
//    - wrapUrl(url) 依序嘗試：直連 → proxy 1 → proxy 2 → ...
// ==========================================
const ProxyRotator = (() => {
    // 每個 proxy 的包裝函式（直連 = null）
    const PROXIES = [
        null,   // 直連（Yahoo v7/COT Socrata 可能有 CORS，但試一下無妨）
        u => `https://corsproxy.io/?${encodeURIComponent(u)}`,
        u => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
        u => `https://api.codetabs.com/v1/proxy?quest=${encodeURIComponent(u)}`,
        u => `https://thingproxy.freeboard.io/fetch/${u}`,
        u => `https://corsproxy.dev/?${encodeURIComponent(u)}`,
    ];
    // 記錄各 proxy 最近失敗時間（ms）；COOLDOWN 內不再選用
    const failTs  = new Array(PROXIES.length).fill(0);
    const COOLDOWN = 90_000;   // 90 秒冷卻

    // 標記 proxy 失敗
    const markFail = (idx) => { failTs[idx] = Date.now(); };

    // 回傳「目前可用的 proxy 清單」（idx + wrapFn）
    const available = () => {
        const now = Date.now();
        return PROXIES
            .map((fn, i) => ({ i, fn }))
            .filter(({ i }) => now - failTs[i] > COOLDOWN);
    };

    // 競速取得：所有可用 proxy 錯開 STAGGER 毫秒依序「發動」，誰先回來就用誰。
    //
    // 為何不逐一等待：一個無法連線的 proxy 要等滿 timeout 才換下一個，
    // 三個 proxy 就是三倍的等待。實測「前兩個逾時、第三個成功」的典型情況，
    // 逐一要 30 秒，競速只要 2 秒；而第一個就成功時兩者一樣快，競速零代價。
    //
    // 錯開發動（而非全部同時）是為了：第一個若正常，後面根本不會發出去，
    // 不會平白增加流量或觸發對方限流。
    const STAGGER_MS = 250;

    const fetchAny = (url, opts = {}) => {
        const { timeoutMs = 7000, headers = {}, skipDirect = false } = opts;
        const pool = available().filter(({ i }) => !(skipDirect && i === 0));
        if (pool.length === 0) return Promise.reject(new Error('沒有可用的 proxy（全部在冷卻中）'));

        return new Promise((resolve, reject) => {
            let settled = false, failed = 0;
            const ctrls = [];
            const timers = [];

            // 注意：勝出者的 controller 必須排除在中止之外。
            // fetchAny 回傳的是 Response 物件，呼叫端之後才讀 body；
            // 若把贏家一起 abort，body 串流會被砍斷，
            // 呼叫端的 .text()/.json() 會拋出 "The user aborted a request"。
            const cleanup = (winner) => {
                timers.forEach(clearTimeout);
                ctrls.forEach(ct => { if (ct !== winner) { try { ct.abort(); } catch (_) {} } });
            };

            pool.forEach(({ i, fn }, pi) => {
                timers.push(setTimeout(async () => {
                    if (settled) return;            // 已有人勝出就不再發動
                    const wrapped = fn ? fn(url) : url;
                    const ctrl = new AbortController();
                    ctrls.push(ctrl);
                    const tid = setTimeout(() => { try { ctrl.abort(); } catch (_) {} }, timeoutMs);
                    timers.push(tid);
                    try {
                        const resp = await fetch(wrapped, { signal: ctrl.signal, headers });
                        clearTimeout(tid);
                        if (resp.status === 429 || resp.status === 503) {
                            markFail(i);
                            throw new Error(`HTTP ${resp.status} (rate limited)`);
                        }
                        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                        if (!settled) { settled = true; cleanup(ctrl); resolve(resp); }
                    } catch (e) {
                        clearTimeout(tid);
                        const msg = e?.message || '';
                        if (msg.includes('rate limit') || msg.includes('429') || msg.includes('503')) markFail(i);
                        if (++failed >= pool.length && !settled) {
                            settled = true; cleanup(null);
                            reject(new Error(`所有 proxy 都失敗: ${url.slice(0, 80)}`));
                        }
                    }
                }, pi * STAGGER_MS));
            });
        });
    };

    return { fetchAny, markFail, available, STAGGER_MS };
})();

// ==========================================
// 💾 MarketDataCache — 以 localStorage 快取市場數據
//    TTL：Yahoo EOD 數據 6 小時，其他 1 小時
// ==========================================
// ══════════════════════════════════════════════════════════════
// AgentStore —— 代理的持久化記憶（IndexedDB）
//
// 三個 object store：
//   runs        每次訓練/搜尋/預測的執行紀錄
//   models      模型註冊表，含 champion / challenger 狀態
//   predictions 預測日誌，事後與實際價格比對
//
// 用 IndexedDB 而非 localStorage：預測紀錄會逐日累積，
// localStorage 的 5MB 上限與同步阻塞都不適合。
// ══════════════════════════════════════════════════════════════
const AgentStore = (() => {
    const DB_NAME = 'silver-agent-v2';
    const VERSION = 2;   // v2 新增 snapshot（市場資料快照，供「立即顯示再背景更新」）
    let _db = null;

    const open = () => new Promise((resolve, reject) => {
        if (_db) return resolve(_db);
        if (typeof indexedDB === 'undefined') return reject(new Error('IndexedDB 不可用'));
        const req = indexedDB.open(DB_NAME, VERSION);
        req.onupgradeneeded = (e) => {
            const db = e.target.result;
            if (!db.objectStoreNames.contains('runs')) {
                const s = db.createObjectStore('runs', { keyPath: 'id', autoIncrement: true });
                s.createIndex('ts', 'ts');
            }
            if (!db.objectStoreNames.contains('models')) {
                const s = db.createObjectStore('models', { keyPath: 'id', autoIncrement: true });
                s.createIndex('status', 'status');
                s.createIndex('ts', 'ts');
            }
            if (!db.objectStoreNames.contains('predictions')) {
                const s = db.createObjectStore('predictions', { keyPath: 'id', autoIncrement: true });
                s.createIndex('targetDate', 'targetDate');
                s.createIndex('resolved', 'resolved');
            }
            // 市場資料快照：固定 key='latest'，只保留最新一份。
            // 用 IndexedDB 而非 localStorage —— 快照約 600KB，
            // localStorage 的同步 JSON 解析會卡住主執行緒，正好抵消「立即顯示」的意義。
            if (!db.objectStoreNames.contains('snapshot')) {
                db.createObjectStore('snapshot', { keyPath: 'key' });
            }
        };
        req.onsuccess = () => { _db = req.result; resolve(_db); };
        req.onerror = () => reject(req.error);
    });

    const tx = async (store, mode, fn) => {
        const db = await open();
        return new Promise((resolve, reject) => {
            const t = db.transaction(store, mode);
            const s = t.objectStore(store);
            let out;
            try { out = fn(s); } catch (e) { return reject(e); }
            // 注意：IDBRequest 在查無資料時 result 為 undefined。
            // 舊寫法會在這種情況回傳 IDBRequest 本身，讓「有沒有快照」的判斷永遠為真。
            t.oncomplete = () => resolve(
                (out && typeof out === 'object' && 'result' in out) ? out.result : out
            );
            t.onerror = () => reject(t.error);
        });
    };

    const get  = (store, key) => tx(store, 'readonly',  s => s.get(key));
    const add  = (store, obj) => tx(store, 'readwrite', s => s.add(obj));
    const put  = (store, obj) => tx(store, 'readwrite', s => s.put(obj));
    const all  = (store)      => tx(store, 'readonly',  s => s.getAll());
    const clear= (store)      => tx(store, 'readwrite', s => s.clear());

    return { open, get, add, put, all, clear, tx };
})();

// ══════════════════════════════════════════════════════════════
// 純函式決策邏輯 —— 與 IndexedDB 無關，可單獨驗證
// ══════════════════════════════════════════════════════════════

// 時序保留集切分：一律取「最後」一段，且永不打亂。
// 隨機切分會讓未來資料洩漏進訓練集，在時間序列上是嚴重錯誤。
function splitHoldout(n, frac = 0.15, minHold = 20) {
    if (!Number.isFinite(n) || n <= 0) return { trainEnd: 0, holdStart: 0, holdLen: 0 };
    let holdLen = Math.floor(n * frac);
    if (holdLen < minHold) holdLen = Math.min(minHold, Math.floor(n / 4));
    if (holdLen < 1) return { trainEnd: n, holdStart: n, holdLen: 0 };   // 資料太少就不切
    const trainEnd = n - holdLen;
    if (trainEnd < 30) return { trainEnd: n, holdStart: n, holdLen: 0 }; // 訓練集不足則放棄保留集
    return { trainEnd, holdStart: trainEnd, holdLen };
}

// 晉升規則：challenger 必須在「同一個保留集」上明確勝過 champion 才能上線。
// 沒有這道關卡，自動重訓會讓模型在無人察覺時逐步變差。
function shouldPromote(champion, challenger, opts = {}) {
    const daMargin   = opts.daMargin   ?? 1.0;   // 方向準確率至少要多 1 個百分點
    const mapeMargin = opts.mapeMargin ?? 0.0;
    const minHold    = opts.minHoldout ?? 20;

    if (!challenger || !Number.isFinite(challenger.holdoutDA)) {
        return { promote: false, reason: '挑戰者沒有有效的保留集分數' };
    }
    if ((challenger.holdoutN ?? 0) < minHold) {
        return { promote: false, reason: `保留集僅 ${challenger.holdoutN ?? 0} 筆，不足以判定（需 ${minHold} 筆）` };
    }
    if (!champion || !Number.isFinite(champion.holdoutDA)) {
        return { promote: true, reason: '目前沒有冠軍模型，挑戰者直接上任' };
    }

    const dDA = challenger.holdoutDA - champion.holdoutDA;
    if (dDA > daMargin) {
        return { promote: true, reason: `方向準確率 ${challenger.holdoutDA.toFixed(2)}% 勝過冠軍 ${champion.holdoutDA.toFixed(2)}%（+${dDA.toFixed(2)}）` };
    }
    // DA 打平時才看誤差，避免用價格誤差推翻方向表現
    if (Math.abs(dDA) <= daMargin) {
        const cM = champion.holdoutMAPE, gM = challenger.holdoutMAPE;
        if (Number.isFinite(cM) && Number.isFinite(gM) && (cM - gM) > mapeMargin) {
            return { promote: true, reason: `方向準確率相當，但誤差較低（${gM.toFixed(2)}% < ${cM.toFixed(2)}%）` };
        }
    }
    return { promote: false, reason: `未達晉升門檻（DA ${challenger.holdoutDA.toFixed(2)}% vs 冠軍 ${champion.holdoutDA.toFixed(2)}%）` };
}

// 由保留集的預測與實際值算出 DA / MAPE
function scoreHoldout(actualPrices, predPrices, prevPrices) {
    const n = Math.min(actualPrices.length, predPrices.length);
    if (n === 0) return { holdoutDA: NaN, holdoutMAPE: NaN, holdoutN: 0 };
    let correct = 0, valid = 0, mapeSum = 0, mapeN = 0;
    for (let i = 0; i < n; i++) {
        const a = actualPrices[i], p = predPrices[i];
        const prev = prevPrices ? prevPrices[i] : (i > 0 ? actualPrices[i - 1] : null);
        if (Number.isFinite(a) && Number.isFinite(p) && a !== 0) {
            mapeSum += Math.abs((a - p) / a) * 100; mapeN++;
        }
        if (Number.isFinite(prev) && Number.isFinite(a) && Number.isFinite(p)) {
            if ((a >= prev) === (p >= prev)) correct++;
            valid++;
        }
    }
    return {
        holdoutDA: valid ? (correct / valid) * 100 : NaN,
        holdoutMAPE: mapeN ? mapeSum / mapeN : NaN,
        holdoutN: n,
    };
}

// 把過去的預測與實際價格對帳。只結算目標日已有實際收盤價的紀錄。
function resolvePredictions(pending, priceByDate) {
    const resolved = [];
    for (const p of pending) {
        if (p.resolved) continue;
        const actual = priceByDate[p.targetDate];
        if (!Number.isFinite(actual)) continue;      // 目標日尚未到來或無資料
        const basis = p.basisPrice;
        const actualUp = Number.isFinite(basis) ? actual >= basis : null;
        const predUp   = Number.isFinite(basis) ? p.predPrice >= basis : (p.predDir === 1);
        resolved.push({
            ...p,
            resolved: 1,
            actualPrice: actual,
            actualUp: actualUp === null ? null : (actualUp ? 1 : 0),
            correct: actualUp === null ? null : ((actualUp === predUp) ? 1 : 0),
            absPctErr: actual !== 0 ? Math.abs((actual - p.predPrice) / actual) * 100 : null,
            resolvedAt: new Date().toISOString(),
        });
    }
    return resolved;
}

// 近期實測表現 —— 這是代理判斷「自己是否還準」的依據
function recentPerformance(resolvedPreds, window = 20) {
    const done = resolvedPreds
        .filter(p => p.resolved && p.correct !== null && p.correct !== undefined)
        .sort((a, b) => String(a.targetDate).localeCompare(String(b.targetDate)));
    const recent = done.slice(-window);
    if (recent.length === 0) return { n: 0, da: NaN, mape: NaN };
    const hits = recent.filter(p => p.correct === 1).length;
    const errs = recent.map(p => p.absPctErr).filter(Number.isFinite);
    return {
        n: recent.length,
        da: (hits / recent.length) * 100,
        mape: errs.length ? errs.reduce((a, b) => a + b, 0) / errs.length : NaN,
    };
}

// 依近期表現決定是否該重訓。門檻刻意保守：
// 樣本太少時不動作，避免拿雜訊當訊號。
function shouldRetrain(perf, opts = {}) {
    const minN = opts.minN ?? 10;
    const daFloor = opts.daFloor ?? 50;
    if (!perf || perf.n < minN) {
        return { retrain: false, reason: `實測樣本僅 ${perf?.n ?? 0} 筆，不足以判斷（需 ${minN} 筆）` };
    }
    if (Number.isFinite(perf.da) && perf.da < daFloor) {
        return { retrain: true, reason: `近 ${perf.n} 次方向準確率 ${perf.da.toFixed(1)}%，低於 ${daFloor}% 門檻` };
    }
    return { retrain: false, reason: `近 ${perf.n} 次方向準確率 ${perf.da.toFixed(1)}%，維持現任冠軍` };
}

const MarketDataCache = (() => {
    const PREFIX = 'slv_cache_v2_';

    // 計算「最近一次市場收盤時間」(UTC ms)
    // 規則：紐約收盤 = 16:00 ET = UTC 20:00（夏令，3~11月）/ UTC 21:00（冬令）
    // 保守取 UTC 21:00 作為每日收盤時間點（多算 1h 也不影響正確性）
    const lastCloseUTC = (nowMs = Date.now()) => {
        const d = new Date(nowMs);
        // 今日 UTC 21:00
        const todayClose = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate(), 21, 0, 0, 0);
        // 若現在 < 今日收盤 → 最近收盤是昨天 UTC 21:00
        return nowMs >= todayClose ? todayClose : todayClose - 86_400_000;
    };

    const get = (key) => {
        try {
            const raw = localStorage.getItem(PREFIX + key);
            if (!raw) return null;
            const obj = JSON.parse(raw);
            return obj; // { ts, data }
        } catch(e) { return null; }
    };

    const set = (key, data) => {
        try {
            localStorage.setItem(PREFIX + key, JSON.stringify({ ts: Date.now(), data }));
        } catch(e) { console.warn('[Cache] localStorage write failed:', e?.message); }
    };

    // isFresh: 每次啟動都重新抓取，永遠回傳 false
    // getData 仍保留，作為所有 API 均失敗時的最終備援
    const isFresh = (_key, _ttlMs) => false;

    const getData = (key) => { const e = get(key); return e ? e.data : null; };

    const clear = () => {
        Object.keys(localStorage)
            .filter(k => k.startsWith(PREFIX))
            .forEach(k => localStorage.removeItem(k));
        // 也清舊版 prefix（v1）
        Object.keys(localStorage)
            .filter(k => k.startsWith('slv_cache_v1_'))
            .forEach(k => localStorage.removeItem(k));
    };

    // 取得快取存入時間的日期字串（顯示用）
    const getCacheDate = (key) => {
        const e = get(key);
        if (!e) return null;
        return new Date(e.ts).toLocaleString();
    };

    return { set, isFresh, getData, clear, getCacheDate, lastCloseUTC };
})();

const fastYield = () => new Promise(r => {
    try { const ch = new MessageChannel(); ch.port1.onmessage = () => { ch.port1.close(); r(); }; ch.port2.postMessage(0); }
    catch (_) { setTimeout(r, 0); }
});
const safeYield = () => document.hidden
    ? fastYield()
    : tf.nextFrame();

// ⚡ V20.4 訓練節流工具
//   原本每個 epoch：React setState ×3 + Plotly 重繪 + localStorage 讀取 + rAF yield（≥16ms），
//   400 epochs 光等畫面就要 6 秒以上，且 loss 每創新低就把整個模型序列化寫進 localStorage。
//   現在：UI 每 ~250ms 才更新一次、yield 依時間觸發、最佳權重先存在記憶體，訓練結束才寫一次。
const makeTicker = (ms) => {
    let last = performance.now();
    return (force = false) => {
        const now = performance.now();
        if (force || now - last >= ms) { last = now; return true; }
        return false;
    };
};
// 最佳權重快照：在 GPU/WASM 記憶體內 clone，不經過序列化
const snapshotWeights = (model, prev) => {
    if (prev) tf.dispose(prev);
    return model.getWeights().map(w => tf.keep(w.clone()));
};
const disposeSnapshot = (snap) => { if (snap) tf.dispose(snap); };
// 把快照權重寫入 localStorage，不影響模型目前（訓練中）的權重
const persistSnapshot = async (model, snap, url) => {
    if (!snap) return;
    const cur = model.getWeights().map(w => w.clone());
    try { model.setWeights(snap); await model.save(url); }
    finally { model.setWeights(cur); tf.dispose(cur); }
};
// fit() 內的 batch 級讓出：只在距上次讓出超過 ms 才讓，避免每個 batch 都等一個畫面
//   （TF.js 預設 yieldEvery='auto' 用 requestAnimationFrame，背景分頁會整個停住）
const makeBatchYielder = (ms = 200) => {
    const tick = makeTicker(ms);
    return { onBatchEnd: async () => { if (tick()) await fastYield(); } };
};
// 依使用者設定挑選訓練後端；'auto' 會用實際模型測速並快取結果
const prepareTrainingBackend = async (params, finalNumFeatures, sampleX, buildTarget, logsUpdater) => {
    const SB = window.SilverBackend;
    const choice = params.backend || 'auto';
    if (choice !== 'auto') {
        SB.preferred = choice;
        const ok = await SB.use(choice);
        if (!ok) logsUpdater(`⚠️ 無法使用 ${choice.toUpperCase()} 後端，改用 ${(await SB.restore()).toUpperCase()}`);
        else logsUpdater(`⚡ 訓練後端：${choice.toUpperCase()}`);
        return tf.getBackend();
    }
    const bs = Math.max(1, Math.min(params.batchSize || 32, sampleX.length));
    const key = [params.algorithm, params.lookback, finalNumFeatures, params.layers, bs, params.predTarget || 'price', params.fastRnn !== false].join('|');
    const spec = resolveTargetSpec(params.predTarget || 'price', params);
    const rows = sampleX.slice(0, bs);
    try {
        return await SB.autoSelect(key, () => {
            const m = createModel(params.algorithm, params.lookback, finalNumFeatures, params);
            m.compile({ optimizer: tf.train.adam(params.lr), loss: spec.loss });
            return m;
        }, () => [tf.tensor3d(rows), buildTarget(bs)], logsUpdater);
    } catch (e) {
        console.warn('[Backend] auto select failed:', e && e.message);
        await SB.restore();
        return tf.getBackend();
    }
};

// raceText: 多個候選網址同時競速（錯開發動），取最先成功的文字內容。
// 給那些有自己 proxy 清單、沒走 ProxyRotator 的來源使用。
const raceText = (urls, { timeoutMs = 7000, stagger = 250 } = {}) => {
    if (!urls || urls.length === 0) return Promise.resolve(null);
    return new Promise((resolve) => {
        let settled = false, failed = 0;
        const ctrls = [], timers = [];
        const done = (val) => { if (settled) return; settled = true;
            timers.forEach(clearTimeout);
            ctrls.forEach(ct => { try { ct.abort(); } catch (_) {} });
            resolve(val); };
        urls.forEach((u, i) => {
            timers.push(setTimeout(async () => {
                if (settled) return;
                const ctrl = new AbortController(); ctrls.push(ctrl);
                const tid = setTimeout(() => { try { ctrl.abort(); } catch (_) {} }, timeoutMs);
                timers.push(tid);
                try {
                    const r = await fetch(u, { signal: ctrl.signal });
                    clearTimeout(tid);
                    if (!r.ok) throw new Error('HTTP ' + r.status);
                    done(await r.text());
                } catch (e) {
                    clearTimeout(tid);
                    if (++failed >= urls.length) done(null);   // 全失敗 → null，呼叫端決定備援
                }
            }, i * stagger));
        });
    });
};


// ══════════════════════════════════════════════════════════════
// CNN Fear & Greed Index 真實抓取
// 來源：CNN Business 非公開 DataViz API
// 回傳：{ values: number[], source: 'real'|'mock',
//         currentScore: number, currentRating: string }
//
// API 回傳 fear_and_greed_historical.data：
//   [{x: unixMs, y: score(0-100)}, ...]  每日一筆，數年歷史
// ==========================================
const fetchFearGreedIndex = async (dates) => {
    const CNN_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata";
    const CACHE_KEY = 'feargreed';
    const CACHE_TTL = Infinity;  // EOD — 以收盤時間點判斷，不用固定 TTL

    const alignToDates = (map, fallback = 50) => {
        const out = []; let last = fallback;
        for (const d of dates) { if (map.has(d)) last = map.get(d); out.push(Number.isFinite(last) ? last : fallback); }
        return out;
    };
    const mockFallback = (currentScore = 50) => {
        const out = []; let v = currentScore;
        for (let i = 0; i < dates.length; i++) { v = Math.max(5, Math.min(95, v + (Math.random() - 0.5) * 3)); out.push(v); }
        return { values: out, source: "mock", currentScore, currentRating: "Neutral" };
    };
    if (!Array.isArray(dates) || dates.length === 0) return mockFallback();

    // ── 快取命中 ──
    if (MarketDataCache.isFresh(CACHE_KEY, CACHE_TTL)) {
        const cached = MarketDataCache.getData(CACHE_KEY);
        if (cached) {
            console.log('[F&G] Cache hit');
            const fgMap = new Map(Object.entries(cached.map));
            return { values: alignToDates(fgMap, cached.score), source: "real", currentScore: cached.score, currentRating: cached.rating };
        }
    }

    try {
        const resp = await ProxyRotator.fetchAny(CNN_URL, { timeoutMs: 10000, headers: { Accept: 'application/json' } });
        const json = await resp.json();
        const currentScore  = parseFloat(json?.fear_and_greed?.score) || 50;
        const currentRating = json?.fear_and_greed?.rating || "Neutral";
        const historical = json?.fear_and_greed_historical?.data || [];
        if (historical.length === 0) throw new Error("Empty historical data");
        const fgMap = new Map();
        historical.forEach(pt => {
            const ms = parseInt(pt.x, 10); if (!Number.isFinite(ms)) return;
            const d = new Date(ms);
            const ymd = `${d.getUTCFullYear()}-${String(d.getUTCMonth()+1).padStart(2,'0')}-${String(d.getUTCDate()).padStart(2,'0')}`;
            const val = parseFloat(pt.y); if (Number.isFinite(val)) fgMap.set(ymd, val);
        });
        // 存快取
        const mapObj = {}; fgMap.forEach((v, k) => { mapObj[k] = v; });
        MarketDataCache.set(CACHE_KEY, { map: mapObj, score: currentScore, rating: currentRating });
        console.log(`[F&G] OK: ${fgMap.size} pts, score=${currentScore.toFixed(1)}`);
        return { values: alignToDates(fgMap, currentScore), source: "real", currentScore, currentRating };
    } catch (e) {
        console.warn("[F&G] All proxies failed:", e.message);
        return mockFallback();
    }
};

// ==========================================
// StockTwits 真實情緒抓取（強化版）
// 同時查詢 SLV + SI_F（白銀期貨），取加權平均
// 回傳：{ score: number(0-100), source: 'real'|'mock',
//         bullishPct: number, bearishPct: number,
//         totalTagged: number, symbolsQueried: string[] }
// ==========================================
const fetchStockTwitsSentimentReal = async () => {
    const symbols = ["SLV", "SI_F"];
    const CACHE_KEY = 'stocktwits';
    const CACHE_TTL = 4 * 3_600_000;  // 4 小時（StockTwits）

    // 快取命中
    if (MarketDataCache.isFresh(CACHE_KEY, CACHE_TTL)) {
        const cached = MarketDataCache.getData(CACHE_KEY);
        if (cached) { console.log('[ST] Cache hit'); return { ...cached, source: 'real' }; }
    }

    const querySymbol = async (sym) => {
        const url = `https://api.stocktwits.com/api/2/streams/symbol/${sym}.json`;
        try {
            const resp = await ProxyRotator.fetchAny(url, { timeoutMs: 8000, skipDirect: true });
            const data = await resp.json();
            if (data?.response?.status !== 200) throw new Error(`API status ${data?.response?.status}`);
            let bullish = 0, bearish = 0, tagged = 0;
            (data.messages || []).forEach(msg => {
                const s = msg?.entities?.sentiment?.basic;
                if (s === "Bullish") { bullish++; tagged++; }
                else if (s === "Bearish") { bearish++; tagged++; }
            });
            return { bullish, bearish, tagged, total: (data.messages || []).length };
        } catch (e) {
            console.warn(`[StockTwits] ${sym} failed:`, e.message);
            return null;
        }
    };

    // 並行查詢所有 symbol
    const results = await Promise.all(symbols.map(querySymbol));
    const valid = results.filter(r => r !== null && r.tagged > 0);

    if (valid.length === 0) {
        console.warn("[StockTwits] All symbols failed or no tagged messages.");
        return { score: 50, source: "mock", bullishPct: 50, bearishPct: 50, totalTagged: 0, symbolsQueried: symbols };
    }

    // 合計所有有效 symbol 的標記訊息
    const totalBullish = valid.reduce((s, r) => s + r.bullish, 0);
    const totalTagged  = valid.reduce((s, r) => s + r.tagged, 0);
    const bullishPct   = totalTagged > 0 ? (totalBullish / totalTagged) * 100 : 50;
    const bearishPct   = 100 - bullishPct;
    const score        = bullishPct;

    console.log(`[StockTwits] Real score: ${score.toFixed(1)}% bullish (${totalTagged} tagged msgs from ${valid.length}/${symbols.length} symbols)`);
    const result = { score, source: "real", bullishPct, bearishPct, totalTagged, symbolsQueried: symbols };
    MarketDataCache.set(CACHE_KEY, { score, bullishPct, bearishPct, totalTagged, symbolsQueried: symbols });
    return result;
};

/**
 * 🏦 fetchTIPSYield: 實質利率 (5Y TIPS Par Real Yield)
 * 來源①: US Treasury XML API — daily_treasury_real_yield_curve
 *   https://home.treasury.gov/.../xml?data=daily_treasury_real_yield_curve&field_tdr_date_value=YYYY
 *   XML 欄位: d:BC_5YEAR (5年期 TIPS 實質殖利率, %)
 * 來源②: corsproxy.io / allorigins.win 代理
 * 備援: 以 0.5% 填補 (TIPS 資料始於 2004)
 */
const fetchTIPSYield = async (dates) => {
    const TREASURY_BASE = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml';
    const CORS_PROXIES = [
        u => `https://corsproxy.io/?${encodeURIComponent(u)}`,
        u => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
        u => `https://api.codetabs.com/v1/proxy?quest=${encodeURIComponent(u)}`,
        u => `https://thingproxy.freeboard.io/fetch/${u}`,
        u => `https://corsproxy.dev/?${encodeURIComponent(u)}`,
    ];
    const parseXML = (xml) => {
        const map = new Map();
        const blocks = xml.match(/<m:properties>[\s\S]*?<\/m:properties>/g) || [];
        for (const b of blocks) {
            const dm = b.match(/<d:NEW_DATE[^>]*>(\d{4}-\d{2}-\d{2})/);
            const ym = b.match(/<d:BC_5YEAR[^>]*>([\d.\-]+)<\/d:BC_5YEAR>/);
            if (!dm || !ym) continue;
            const v = parseFloat(ym[1]);
            if (Number.isFinite(v)) map.set(dm[1], v);
        }
        return map;
    };
    const years = [...new Set(dates.map(d => d.slice(0, 4)))].sort();
    const tipsMap = new Map();
    // 各年度資料彼此獨立 → 平行抓取。原本 5~6 年逐年串行，
    // 每年最差要等到 proxy 逾時，是整個載入流程最長的單一環節。
    const perYear = await Promise.all(years.map(async (year) => {
        const baseUrl = TREASURY_BASE + '?data=daily_treasury_real_yield_curve&field_tdr_date_value=' + year;
        // 直連與各 proxy 一起競速，不再逐一等到逾時
        const xml = await raceText([baseUrl, ...CORS_PROXIES.map(p => p(baseUrl))], { timeoutMs: 7000 });
        if (xml) {
            try {
                const m = parseXML(xml);
                if (m.size > 0) { console.log('[TIPS] ' + year + ': ' + m.size + ' records'); return m; }
            } catch(e) { console.warn('[TIPS] ' + year + ' 解析失敗:', e?.message?.slice(0, 60)); }
        }
        return null;
    }));
    // 依年份順序合併，確保後續前向填補的結果與串行版本一致
    perYear.forEach(m => { if (m) m.forEach((v, k) => tipsMap.set(k, v)); });
    let lastKnown = null;
    const result = dates.map(d => {
        if (tipsMap.has(d)) lastKnown = tipsMap.get(d);
        return lastKnown !== null ? lastKnown : 0.5;
    });
    console.log('[TIPS] series length=' + result.length + ', latest=' + result[result.length-1]);
    result.source = tipsMap.size ? 'real' : 'fallback';
    return result;
};

/**
 * 📦 computeSLVHoldings: 白銀實物庫存代理
 * 方法: SLV_price / Silver_futures_price ≈ oz_per_share (實際約 0.93 oz/share)
 *   - 全球最大白銀 ETF，持倉約 3–5 億盎司實物 (LBMA 倫敦金庫)
 *   - oz_per_share 因 0.5%/年管理費緩慢下降
 *   - 比值短期偏離反映 premium/discount：正偏 = 市場需求 > NAV (創設壓力 = 庫存增)
 *   - 不需額外 API —— 全部基於既有 Yahoo Finance 真實數據
 * 回傳: oz_per_share × 1000 (數值約 920–1000，方便後續 Min-Max scale)
 */
const computeSLVHoldings = (slvPrices, silverFuturesPrices) => {
    return slvPrices.map((slv, i) => {
        const si = silverFuturesPrices[i];
        if (!si || si <= 0 || !slv || slv <= 0) return 930;
        return (slv / si) * 1000;
    });
};

/**
 * 🎯 NEW: 獲取 CFTC COT Disaggregated 數據
 * 來源: CFTC SODA API (Disaggregated Futures Only)
 * 篩選: Silver on COMEX
 */

/**
 * 🧩 更穩定版：獲取 CFTC COT Disaggregated Futures Only（Managed Money 淨部位）
 * - 資料源：CFTC publicreporting（Socrata / SODA）
 * - 主要鍵：as_of_date_form_yyyy_mm_dd（注意：不是 report_date_as_yyyy_mm_dd）
 * - 主要欄位：m_money_positions_long_all / m_money_positions_short_all / open_interest_all
 * - 先直連 Socrata（通常支援 CORS），失敗再走 proxy
 * - 具備：重試 / 指數退避 / 多種 market filter fallback / 輕量 $select / 偵錯輸出
 *
 * 欄位與日期格式參考：Socrata dataset 欄位定義（As_of_Date_Form_YYYY-MM-DD）。 
 */
/**
 * 📈 fetchGoogleTrends: Google Trends「silver」搜尋熱度 (0-100)
 *
 * 流程（兩步驟）：
 *  Step 1: POST /trends/api/explore → 取得 TIMESERIES widget token
 *  Step 2: GET  /trends/api/widgetdata/multiline?token=... → 取得週度時序資料
 *
 * 回應格式前置 ")]}'" 需剝除後再 JSON.parse
 * 回傳: number[] (長度 = dates.length, 以前向填補對齊每日)
 * 數值代表: 0=低熱度, 100=搜尋高峰（相對值）
 *
 * CORS: Google Trends 不允許直接 browser fetch → 必須透過 proxy
 *       若所有 proxy 均失敗 → 以 50 填充（中性值）
 */
const fetchGoogleTrends = async (dates) => {
    const KEYWORD   = 'silver';
    const GEO       = 'US';
    // 從最早日期到今天，Google Trends 最多 5 年
    const startDate = dates[0] ? dates[0].slice(0, 7) : '2020-01';
    const endDate   = new Date().toISOString().slice(0, 7);
    const TIME_STR  = `${startDate} ${endDate}`;

    const EXPLORE_URL = 'https://trends.google.com/trends/api/explore';
    const WIDGET_URL  = 'https://trends.google.com/trends/api/widgetdata/multiline';

    // Google Trends 的 proxy 清單（allorigins 對 text 較可靠）
    const GT_PROXIES = [
        u => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
        u => `https://corsproxy.io/?${encodeURIComponent(u)}`,
        u => `https://api.codetabs.com/v1/proxy?quest=${encodeURIComponent(u)}`,
    ];

    const stripPrefix = (text) => {
        // Google Trends API 回應前置 ")]}'"
        const idx = text.indexOf('{');
        return idx >= 0 ? text.slice(idx) : text;
    };

    // Google Trends 要兩段請求（取 token → 取資料），逐一試 proxy 會把等待加倍。
    // 改競速後最差情況從 3×14s×2 ≈ 84 秒降到約 13 秒。
    const fetchViaProxies = (url, ms = 7000) =>
        raceText(GT_PROXIES.map(fn => fn(url)), { timeoutMs: ms });

    // Step 1: explore → get token
    const exploreParams = new URLSearchParams({
        hl: 'en-US', tz: '-480',
        req: JSON.stringify({
            comparisonItem: [{ keyword: KEYWORD, geo: GEO, time: TIME_STR }],
            category: 0, property: ''
        })
    });
    const exploreUrl = EXPLORE_URL + '?' + exploreParams.toString();

    let token = null;
    try {
        const raw = await fetchViaProxies(exploreUrl);
        if (!raw) throw new Error('no response from explore');
        const json = JSON.parse(stripPrefix(raw));
        const widget = (json.widgets || []).find(w => w.id === 'TIMESERIES');
        if (!widget?.token) throw new Error('TIMESERIES widget not found');
        token = widget.token;
        console.log('[GTrend] token obtained:', token.slice(0, 20) + '...');
    } catch(e) {
        console.warn('[GTrend] explore step failed:', e?.message?.slice(0, 80));
        return new Array(dates.length).fill(50); // neutral fallback
    }

    // Step 2: widgetdata → get interest over time
    const widgetParams = new URLSearchParams({
        hl: 'en-US', tz: '-480',
        req: JSON.stringify({ time: TIME_STR, resolution: 'WEEK', locale: 'en-US',
            comparisonItem: [{ geo: { country: GEO }, complexKeywordsRestriction: { keyword: [{ type: 'BROAD', value: KEYWORD }] } }],
            requestOptions: { property: '', backend: 'IZG', category: 0 }
        }),
        token,
        tz: '-480'
    });
    const widgetUrl = WIDGET_URL + '?' + widgetParams.toString();

    try {
        const raw2 = await fetchViaProxies(widgetUrl);
        if (!raw2) throw new Error('no response from widgetdata');
        const json2 = JSON.parse(stripPrefix(raw2));
        const timeline = json2?.default?.timelineData || [];
        if (timeline.length === 0) throw new Error('empty timeline data');

        // Build Map<YYYY-MM-DD (week start), value>
        const trendMap = new Map();
        for (const pt of timeline) {
            const ms = parseInt(pt.time, 10) * 1000;
            if (!isFinite(ms)) continue;
            const d   = new Date(ms);
            const ymd = d.toISOString().slice(0, 10);
            const val = pt.value?.[0] ?? 50;
            // Fill all 7 days of this week
            for (let day = 0; day < 7; day++) {
                const dd = new Date(ms + day * 86_400_000);
                trendMap.set(dd.toISOString().slice(0, 10), val);
            }
        }

        // Align to dates (forward fill)
        let last = 50;
        const result = dates.map(d => {
            if (trendMap.has(d)) last = trendMap.get(d);
            return last;
        });
        console.log(`[GTrend] OK: ${timeline.length} weekly pts, latest=${result[result.length-1]}`);
        result.source = 'real';
        return result;
    } catch(e) {
        console.warn('[GTrend] widgetdata step failed:', e?.message?.slice(0, 80));
        return new Array(dates.length).fill(50);
    }
};

        const fetchCOTData = async (onStatus) => {
    // ==========================================
    // 3-tier COT Data Strategy:
    // 1) CFTC Disaggregated Futures API (Managed Money) - fixed URL encoding
    // 2) CFTC Legacy Futures API (Non-Commercial) - fallback
    // 3) CFTC HTML Report (current week only) - last resort
    // ==========================================
    const DISAG_BASE  = 'https://publicreporting.cftc.gov/resource/72hh-3qpy.json';
    const LEGACY_BASE = 'https://publicreporting.cftc.gov/resource/6dca-aqww.json';
    const HTML_URL    = 'https://www.cftc.gov/dea/futures/deacmxsf.htm';

    const CORS_PROXIES = [
        u => `https://corsproxy.io/?${encodeURIComponent(u)}`,
        u => `https://api.allorigins.win/raw?url=${encodeURIComponent(u)}`,
        u => `https://api.codetabs.com/v1/proxy?quest=${encodeURIComponent(u)}`,
        u => `https://thingproxy.freeboard.io/fetch/${u}`,
        u => `https://corsproxy.dev/?${encodeURIComponent(u)}`,
    ];

    const emitStatus = (patch) => {
        try { if (typeof onStatus === 'function') onStatus({ updatedAt: new Date().toISOString(), ...patch }); } catch(_) {}
    };
    const sleep = ms => new Promise(r => setTimeout(r, ms));

    const fetchT = (url, ms = 12000) => {
        const ctrl = new AbortController();
        const tid = setTimeout(() => ctrl.abort(), ms);
        return fetch(url, { signal: ctrl.signal }).finally(() => clearTimeout(tid));
    };

    const fetchJSON = async (url, retries = 2) => {
        for (let i = 0; i <= retries; i++) {
            try {
                const r = await fetchT(url);
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                return await r.json();
            } catch(e) {
                if (i < retries) await sleep(700 * (i + 1));
                else throw e;
            }
        }
    };

    // Build SoQL URL: $where value is encodeURIComponent'd (fixes wildcard % encoding)
    const buildUrl = (base, select, whereExpr, orderExpr, limit) =>
        `${base}?$select=${select}&$where=${encodeURIComponent(whereExpr)}&$order=${encodeURIComponent(orderExpr)}&$limit=${limit}`;

    const parseDisagg = (records) => {
        const map = new Map();
        if (!Array.isArray(records)) return map;
        for (const r of records) {
            const raw = r.as_of_date_form_yyyy_mm_dd || r.report_date_as_yyyy_mm_dd;
            if (!raw) continue;
            const d   = String(raw).split('T')[0];
            const lng = parseFloat(r.m_money_positions_long_all);
            const sht = parseFloat(r.m_money_positions_short_all);
            if (!Number.isFinite(lng) || !Number.isFinite(sht)) continue;
            map.set(d, { net: lng - sht, long: lng, short: sht, openInterest: parseFloat(r.open_interest_all) || null, market: r.market_and_exchange_names || '' });
        }
        return map;
    };

    const parseLegacy = (records) => {
        const map = new Map();
        if (!Array.isArray(records)) return map;
        for (const r of records) {
            const raw = r.report_date_as_yyyy_mm_dd || r.as_of_date_form_yyyy_mm_dd;
            if (!raw) continue;
            const d   = String(raw).split('T')[0];
            const lng = parseFloat(r.noncomm_positions_long_all);
            const sht = parseFloat(r.noncomm_positions_short_all);
            if (!Number.isFinite(lng) || !Number.isFinite(sht)) continue;
            map.set(d, { net: lng - sht, long: lng, short: sht, openInterest: parseFloat(r.open_interest_all) || null, market: r.market_and_exchange_names || '' });
        }
        return map;
    };

    const successReturn = (map, source, phase) => {
        const keys = [...map.keys()].sort();
        const latestDate = keys[keys.length - 1];
        const latest = map.get(latestDate);
        emitStatus({ state: 'ok', source, records: map.size, latestDate, latestNet: latest?.net ?? null, phase });
        console.log(`[COT] OK ${source}: ${map.size} records, latest=${latestDate}, net=${latest?.net}`);
        return map;
    };

    emitStatus({ state: 'loading', source: null, records: 0, latestDate: null, latestNet: null, message: null, phase: 'start' });

    const SILVER_WHERES = [
        "upper(market_and_exchange_names) like '%SILVER%COMMODITY%EXCHANGE%INC%'",
        "upper(market_and_exchange_names) like '%SILVER%EXCHANGE%INC%'",
        "upper(market_and_exchange_names) like '%SILVER%'",
    ];

    // ─── Source 1: Disaggregated Futures (Managed Money) ───
    const DISAG_SEL = 'as_of_date_form_yyyy_mm_dd,market_and_exchange_names,m_money_positions_long_all,m_money_positions_short_all,open_interest_all';
    for (let wi = 0; wi < SILVER_WHERES.length; wi++) {
        const base = buildUrl(DISAG_BASE, DISAG_SEL, SILVER_WHERES[wi], 'as_of_date_form_yyyy_mm_dd DESC', 300);
        // 直連與各 proxy 競速，不再逐一等到逾時（最差從 5×14s 降為一次逾時）
        emitStatus({ state: 'loading', phase: `disag_w${wi+1}` });
        const txt = await raceText([base, ...CORS_PROXIES.map(p => p(base))], { timeoutMs: 7000 });
        if (txt) {
            try {
                const map = parseDisagg(JSON.parse(txt));
                if (map.size > 0) return successReturn(map, 'Disagg', 'disag_ok');
            } catch(e) { console.warn(`[COT] Disagg w${wi+1} 解析失敗:`, e?.message?.slice(0,80)); }
        }
    }

    // ─── Source 2: Legacy Futures (Non-Commercial) ───
    const LEGACY_SEL = 'report_date_as_yyyy_mm_dd,market_and_exchange_names,noncomm_positions_long_all,noncomm_positions_short_all,open_interest_all';
    for (let wi = 0; wi < SILVER_WHERES.length; wi++) {
        const base = buildUrl(LEGACY_BASE, LEGACY_SEL, SILVER_WHERES[wi], 'report_date_as_yyyy_mm_dd DESC', 300);
        emitStatus({ state: 'loading', phase: `legacy_w${wi+1}` });
        const txt = await raceText([base, ...CORS_PROXIES.map(p => p(base))], { timeoutMs: 7000 });
        if (txt) {
            try {
                const map = parseLegacy(JSON.parse(txt));
                if (map.size > 0) return successReturn(map, 'Legacy', 'legacy_ok');
            } catch(e) { console.warn(`[COT] Legacy w${wi+1} 解析失敗:`, e?.message?.slice(0,80)); }
        }
    }

    // ─── Source 3: CFTC HTML Text Report (current week only) ───
    // URL: https://www.cftc.gov/dea/futures/deacmxsf.htm
    // Format: fixed-width text, NON-COMMERCIAL long/short in COMMITMENTS line
    emitStatus({ state: 'loading', phase: 'html', message: 'Parsing CFTC HTML report...' });
    const htmlUrls = [HTML_URL, ...CORS_PROXIES.map(p => p(HTML_URL))];
    for (let ai = 0; ai < 1; ai++) {
        try {
            const text = await raceText(htmlUrls, { timeoutMs: 7000 });
            if (!text) continue;
            const sidx = text.indexOf('SILVER - COMMODITY EXCHANGE INC');
            if (sidx === -1) { console.warn('[COT] HTML: SILVER section not found'); continue; }

            const sec = text.substring(sidx, sidx + 2000);

            // Date: "POSITIONS AS OF MM/DD/YY"
            const dm = sec.match(/POSITIONS AS OF\s+(\d{2})\/(\d{2})\/(\d{2,4})/);
            if (!dm) continue;
            const yr  = dm[3].length === 2 ? (parseInt(dm[3]) < 50 ? '20'+dm[3] : '19'+dm[3]) : dm[3];
            const dateStr = `${yr}-${dm[1]}-${dm[2]}`;

            // COMMITMENTS row: NON-COMMERCIAL LONG  SHORT  SPREADS ...
            const ci = sec.indexOf('COMMITMENTS');
            if (ci === -1) continue;
            const lm = sec.substring(ci + 12).match(/^\s*([\d,]+)\s+([\d,]+)/m);
            if (!lm) continue;

            const long  = parseInt(lm[1].replace(/,/g,''));
            const short = parseInt(lm[2].replace(/,/g,''));
            if (!Number.isFinite(long) || !Number.isFinite(short)) continue;

            const map = new Map();
            map.set(dateStr, { net: long - short, long, short, openInterest: null, market: 'SILVER - COMMODITY EXCHANGE INC.' });
            console.log(`[COT] HTML parsed: date=${dateStr}, net=${long-short}`);
            emitStatus({ state: 'ok', source: 'HTML Report (current week)', records: 1, latestDate: dateStr, latestNet: long - short, phase: 'html_ok', message: 'Only latest week available; history forward-filled' });
            return map;

        } catch(e) { console.warn(`[COT] HTML a${ai+1}:`, e?.message?.slice(0,80)); }
    }

    // ─── All failed ───
    console.warn('[COT] All sources failed.');
    emitStatus({ state: 'error', source: null, records: 0, latestDate: null, latestNet: null, message: 'All COT sources failed, using mock data', phase: 'failed_all' });
    return new Map();
};
// ══════════════════════════════════════════════════════════════
// 🔴 白銀期貨即時報價（SI=F，COMEX，美元/盎司）
//    進入模型設定畫面前「必須」成功取得，否則停在錯誤畫面，不會用模擬或舊資料頂替。
//    兩個來源同時發動，先回來且通過驗證者勝出：
//      A. Yahoo v8 chart（1 分鐘線）meta.regularMarketPrice
//      B. Stooq 報價 CSV（si.f），直連失敗再走 proxy
//    驗證：價格在合理範圍、報價時間在 LIVE_QUOTE_MAX_AGE_MS 以內（涵蓋週末與假日休市）
// ══════════════════════════════════════════════════════════════
const LIVE_QUOTE_MAX_AGE_MS = 4 * 24 * 3600 * 1000;
const validateLiveQuote = (q) => {
    if (!q || !Number.isFinite(q.price) || q.price < 3 || q.price > 1000) throw new Error('報價數值異常');
    if (!Number.isFinite(q.time)) throw new Error('報價缺少時間');
    const age = Date.now() - q.time;
    if (age > LIVE_QUOTE_MAX_AGE_MS) throw new Error(`報價過舊（${(age / 3600000).toFixed(1)} 小時前）`);
    return q;
};
const fetchLiveSilverQuote = async ({ timeoutMs = 12000 } = {}) => {
    const fromYahoo = async () => {
        const url = 'https://query1.finance.yahoo.com/v8/finance/chart/SI=F?range=1d&interval=1m';
        const resp = await ProxyRotator.fetchAny(url, { timeoutMs });
        const data = await resp.json();
        const r = data?.chart?.result?.[0];
        const meta = r?.meta || {};
        let price = Number(meta.regularMarketPrice);
        let tSec = Number(meta.regularMarketTime);
        // meta 缺值時，退回最後一根有收盤價的 1 分鐘 K 棒
        if (!Number.isFinite(price) || !Number.isFinite(tSec)) {
            const ts = r?.timestamp || [], cl = r?.indicators?.quote?.[0]?.close || [];
            for (let i = cl.length - 1; i >= 0; i--) {
                if (Number.isFinite(cl[i])) { price = cl[i]; tSec = ts[i]; break; }
            }
        }
        return validateLiveQuote({ price, time: tSec * 1000, source: 'Yahoo Finance', symbol: 'SI=F' });
    };
    const fromStooq = async () => {
        const url = 'https://stooq.com/q/l/?s=si.f&f=sd2t2ohlcv&h&e=csv';
        let text = null;
        try {
            const ctrl = new AbortController();
            const tid = setTimeout(() => ctrl.abort(), timeoutMs);
            const resp = await fetch(url, { signal: ctrl.signal });
            clearTimeout(tid);
            if (resp.ok) text = await resp.text();
        } catch (_) { /* 直連失敗 → 走 proxy */ }
        if (!text || !/\d/.test(text)) text = await (await ProxyRotator.fetchAny(url, { timeoutMs, skipDirect: true })).text();
        const lines = text.trim().split(/\r?\n/);
        if (lines.length < 2) throw new Error('Stooq 報價為空');
        const hdr = lines[0].toLowerCase().split(',');
        const row = lines[1].split(',');
        const get = (k) => row[hdr.indexOf(k)];
        const price = parseFloat(get('close'));
        const d = get('date'), t = get('time') || '00:00:00';
        // Stooq 時間為華沙當地時間；新鮮度判斷用，差一兩小時不影響
        const time = new Date(`${d}T${t}+01:00`).getTime();
        return validateLiveQuote({ price, time, source: 'Stooq', symbol: 'SI.F' });
    };
    const errors = [];
    const wrap = (name, fn) => fn().catch(e => { errors.push(`${name}: ${e?.message || e}`); throw e; });
    try {
        const q = await Promise.any([wrap('Yahoo', fromYahoo), wrap('Stooq', fromStooq)]);
        return { ...q, fetchedAt: Date.now() };
    } catch (_) {
        throw new Error(`無法取得白銀期貨（SI=F）即時報價 —— ${errors.join('；')}`);
    }
};

const fetchMarketData = async (cotStatusCb, onStep) => {
    const step = (k, info) => { try { onStep && onStep(k, info); } catch (_) {} };
    // 即時報價與歷史資料同時發動，最後才 await —— 不增加總等待時間
    const liveQuotePromise = fetchLiveSilverQuote();
    liveQuotePromise.then(q => step('quote', q), e => step('quote-fail', e));   // 避免在 await 前出現 unhandled rejection 警告
    const DATA_RANGE = "5y";
    const YAHOO_CACHE_TTL = Infinity;  // EOD — 以收盤時間點判斷，不用固定 TTL
    try { if (typeof cotStatusCb === "function") cotStatusCb({ state: "loading", source: null, records: 0, latestDate: null, latestNet: null, message: null, phase: "marketdata_start", updatedAt: new Date().toISOString() }); } catch (_) {}

    // ══════════════════════════════════════════════════════════
    // 市場數據三層取得策略
    //  Layer 1: Stooq.com CSV  — 無 proxy、無 rate limit、CORS = *
    //           URL: https://stooq.com/q/d/l/?s={symbol}&i=d
    //           回傳: Date,Open,High,Low,Close,Volume (CSV)
    //  Layer 2: Yahoo Finance v8 JSON (query1/query2)
    //           透過 ProxyRotator 輪替，proxy 失敗自動切換
    //  Layer 3: localStorage 快取（EOD 日期界限：收盤後首次開啟才更新，同日多次開啟使用快取）
    // ══════════════════════════════════════════════════════════

    // Stooq symbol 對照表
    const STOOQ_MAP = {
        'SI=F':     'si=f',
        'SLV':      'slv.us',
        'GC=F':     'gc=f',
        'CL=F':     'cl=f',
        'DX-Y.NYB': 'dx-y.nyb',
        '^VIX':     '^vix',
        '^TNX':     '^tnx',
        'HG=F':     'hg=f',
    };

    // Stooq CSV 解析 → { timestamps, close, volume }
    const parseStooqCSV = (csv, symbol) => {
        const lines = csv.trim().split('\n');
        if (lines.length < 2) throw new Error('Stooq: empty CSV');
        // header: Date,Open,High,Low,Close,Volume
        const hdr = lines[0].toLowerCase().split(',');
        const di = hdr.indexOf('date');
        const ci = hdr.indexOf('close');
        const vi = hdr.indexOf('volume');
        if (di < 0 || ci < 0) throw new Error('Stooq: missing columns');

        // Stooq returns newest-first → reverse to oldest-first
        const rows = lines.slice(1).reverse();
        const now = Date.now();
        const fiveYearsAgo = now - DATA_RANGE_MS;
        const timestamps = [], close = [], volume = [];
        for (const row of rows) {
            const cols = row.split(',');
            if (cols.length < 3) continue;
            // Date: YYYY-MM-DD
            const ts = new Date(cols[di] + 'T00:00:00Z').getTime() / 1000;
            if (!isFinite(ts) || ts * 1000 < fiveYearsAgo) continue;
            const c = parseFloat(cols[ci]);
            if (!isFinite(c) || c <= 0) continue;
            timestamps.push(ts);
            close.push(c);
            volume.push(vi >= 0 ? (parseFloat(cols[vi]) || 0) : 0);
        }
        if (timestamps.length < 30) throw new Error(`Stooq: too few rows (${timestamps.length})`);
        console.log(`[Stooq] ${symbol} OK: ${timestamps.length} pts`);
        return { timestamps, close, volume };
    };

    // Layer 1: Stooq direct (no proxy needed)
    const fetchStooq = async (symbol) => {
        const stooqSym = STOOQ_MAP[symbol];
        if (!stooqSym) return null;
        const url = `https://stooq.com/q/d/l/?s=${encodeURIComponent(stooqSym)}&i=d`;
        try {
            const ctrl = new AbortController();
            const tid = setTimeout(() => ctrl.abort(), 12000);
            const resp = await fetch(url, { signal: ctrl.signal });
            clearTimeout(tid);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const csv = await resp.text();
            return parseStooqCSV(csv, symbol);
        } catch (e) {
            console.warn(`[Stooq] ${symbol} failed:`, e.message?.slice(0, 80));
            return null;
        }
    };

    // Layer 2: Yahoo v8 via ProxyRotator
    const fetchYahooViaProxy = async (symbol) => {
        const urls = [
            `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?range=${DATA_RANGE}&interval=1d`,
            `https://query2.finance.yahoo.com/v8/finance/chart/${symbol}?range=${DATA_RANGE}&interval=1d`,
        ];
        for (const url of urls) {
            try {
                const response = await ProxyRotator.fetchAny(url, { timeoutMs: 15000 });
                const data = await response.json();
                if (!data.chart?.result?.length) continue;
                const result = data.chart.result[0];
                const ind = result.indicators.quote[0];
                const ts  = result.timestamp;
                if (!ind.close || !ts) continue;
                let vol = ind.volume;
                if (!vol || vol.length !== ts.length) vol = new Array(ind.close.length).fill(0);
                else vol = vol.map(v => v === null ? 0 : v);
                console.log(`[Yahoo] ${symbol} OK via proxy (${ts.length} pts)`);
                return { timestamps: ts, close: ind.close, volume: vol };
            } catch (e) {
                console.warn(`[Yahoo] ${symbol} proxy failed:`, e.message?.slice(0, 80));
            }
        }
        return null;
    };

    // ── 主要抓取流程：快取 → Stooq → Yahoo proxy ──
    const DATA_RANGE_MS = 5 * 365.25 * 24 * 3600 * 1000;

    // 產生佔位用的 mock 時序數據（當所有外部 API 均失敗時確保程式繼續運作）
    const makeMockSeries = (symbol, n = 1260) => {
        const SEEDS = { 'SI=F':28,'SLV':22,'GC=F':2000,'CL=F':75,'DX-Y.NYB':103,'^VIX':18,'^TNX':4.2,'HG=F':3.8 };
        let v = SEEDS[symbol] ?? 50;
        const now = Date.now() / 1000;
        const DAY = 86400;
        const timestamps = [], close = [], volume = [];
        for (let i = n - 1; i >= 0; i--) {
            v = Math.max(v * 0.5, v * (1 + (Math.random() - 0.499) * 0.012));
            timestamps.push(Math.round(now - i * DAY * 1.4));
            close.push(parseFloat(v.toFixed(4)));
            volume.push(Math.round(1e6 + Math.random() * 5e6));
        }
        console.warn(`[Mock] Using synthetic series for ${symbol}`);
        return { timestamps, close, volume, isMock: true };
    };

    // 快取命中或 Stooq 直連。Stooq 有 CORS 且無流量限制，
    // 因此這一段可以安全平行 —— 先前的逐一 + 150ms 是為了保護會限流的 proxy，
    // 套用在 Stooq 上只是白白多等。
    const fetchSymbolFast = async (symbol) => {
        const cacheKey = `sym_${symbol}`;
        if (MarketDataCache.isFresh(cacheKey, YAHOO_CACHE_TTL)) {
            const cached = MarketDataCache.getData(cacheKey);
            if (cached) { console.log(`[Cache] ${symbol} hit`); return cached; }
        }
        const result = await fetchStooq(symbol);
        if (result) MarketDataCache.set(cacheKey, result);
        return result;
    };

    const yahooSymbols = ["SI=F","SLV","GC=F","CL=F","DX-Y.NYB","^VIX","^TNX","HG=F"];

    // 第一輪：八個代碼同時抓，耗時 ≈ 最慢的那一個，而非八個相加
    const t0 = performance.now();
    const yahooResults = await Promise.all(yahooSymbols.map(s => fetchSymbolFast(s)));
    console.log(`[Fetch] Stooq 平行取得 ${yahooResults.filter(Boolean).length}/${yahooSymbols.length} 檔，耗時 ${((performance.now()-t0)/1000).toFixed(1)}s`);
    const symSrc = yahooResults.map(r => r ? 'Stooq' : null);

    // 第二輪：只有失敗的才走 proxy，這段仍逐一並保留間隔（proxy 會限流）
    for (let i = 0; i < yahooSymbols.length; i++) {
        if (yahooResults[i]) continue;
        const sym = yahooSymbols[i];
        let r = await fetchYahooViaProxy(sym);
        symSrc[i] = r ? 'Yahoo Finance' : null;
        if (!r) {
            const stale = MarketDataCache.getData(`sym_${sym}`);
            if (stale) { console.warn(`[Cache] ${sym} using stale cache`); r = stale; symSrc[i] = '本機快取'; }
        }
        if (r) MarketDataCache.set(`sym_${sym}`, r);
        yahooResults[i] = r;
        await new Promise(res => setTimeout(res, 150));
    }

    // 🔴 白銀期貨歷史是模型的預測目標，絕不能用模擬資料頂替
    if (!yahooResults[0]) {
        throw new Error('無法取得白銀期貨（SI=F）真實歷史價格（Stooq 與 Yahoo 均失敗，且無快取）');
    }
    step('history', { n: yahooResults[0].close.length, sources: symSrc });
    // 其餘因子若失敗則以 mock 補位（僅影響輔助特徵）
    const [silverFuturesData, slvEtfData, goldData, oilData, dxyData, vixData, tnxData, copperData]
        = yahooResults.map((r, i) => r ?? makeMockSeries(yahooSymbols[i]));

    // 四個外部來源彼此獨立 —— 在此同時「發動」，稍後才各自 await。
    // COT 在建立 dates 的迴圈中就要用到，TIPS/GTrend 則需要 dates，
    // 因此兩組的 await 時機不同，但網路請求是重疊進行的。
    const tAux = performance.now();
    const stockTwitsPromise = fetchStockTwitsSentimentReal();
    const cotPromise        = fetchCOTData(cotStatusCb);

    // 這兩個要等 dates 建好才能發，先留著函式參考
    let tipsPromise = null, gtrendPromise = null;

    const [stockTwitsResult, cotDataMap] = await Promise.all([stockTwitsPromise, cotPromise]);

    // 核心品種已由 makeMockSeries 補齊，此處只做警告
    const missingReal = [];
    if (silverFuturesData?.isMock) missingReal.push('SI=F');
    if (slvEtfData?.isMock)        missingReal.push('SLV');
    if (goldData?.isMock)           missingReal.push('GC=F');
    if (missingReal.length) console.warn('[Market] Using mock for:', missingReal.join(', '));

    const dates = [];
    const prices = []; 
    const slvPrices = []; 
    const slvVolume = []; 
    const goldPrices = [];
    const oilPrices = [];
    const dxyPrices = []; 
    const vixPrices = []; 
    const tnxPrices = []; 
    const copperPrices = []; 
    const silverInventory = [];
    const cotNetPositions = []; // <-- COT Array

    const createLookup = (data) => {
        const map = new Map();
        data.timestamps.forEach((t, i) => {
            if (data.close[i] !== null) {
                const dateStr = new Date(t * 1000).toISOString().split('T')[0];
                map.set(dateStr, { close: data.close[i], volume: data.volume[i] });
            }
        });
        return map;
    };

    const silverMap = createLookup(silverFuturesData);
    const slvMap = createLookup(slvEtfData); 
    const goldMap = createLookup(goldData);
    const oilMap = createLookup(oilData);
    const dxyMap = createLookup(dxyData); 
    const vixMap = createLookup(vixData); 
    const tnxMap = createLookup(tnxData); 
    const copperMap = createLookup(copperData); 

    let lastGold = 2000; let lastOil = 75; let lastDXY = 100; let lastVIX = 15; let lastTNX = 4.0; let lastCopper = 3.5; let lastPrice = 25; 
    let lastSlvPrice = 22; let lastSlvVolume = 5000000; 

    // COT data interpolation setup
    const cotDates = Array.from(cotDataMap.keys()).sort();
    let lastCotNet = 0; 
    if (cotDates.length > 0) {
         lastCotNet = cotDataMap.get(cotDates[0])?.net ?? 0;
    }
    let nextCotIndex = 0;

    silverFuturesData.timestamps.forEach((t, i) => {
        if (silverFuturesData.close[i] === null) return;

        const dateStr = new Date(t * 1000).toISOString().split('T')[0];
        const silverPrice = silverFuturesData.close[i];

        lastPrice = silverPrice;
        const gData = goldMap.get(dateStr)?.close; if (gData) lastGold = gData;
        const oData = oilMap.get(dateStr)?.close; if (oData) lastOil = oData;
        const dData = dxyMap.get(dateStr)?.close; if (dData) lastDXY = dData;
        const vData = vixMap.get(dateStr)?.close; if (vData) lastVIX = vData;
        const tData = tnxMap.get(dateStr)?.close; if (tData) lastTNX = tData;
        const cData = copperMap.get(dateStr)?.close; if (cData) lastCopper = cData;

        const slvDataPoint = slvMap.get(dateStr); 
        if (slvDataPoint) {
            lastSlvPrice = slvDataPoint.close;
            if (slvDataPoint.volume !== null && slvDataPoint.volume !== undefined) {
                lastSlvVolume = slvDataPoint.volume;
            }
        }

        // COT Data Interpolation
        while (nextCotIndex < cotDates.length && cotDates[nextCotIndex] <= dateStr) {
            const cotValue = cotDataMap.get(cotDates[nextCotIndex]);
            if (cotValue && Number.isFinite(cotValue.net)) {
                lastCotNet = cotValue.net;
            }
            nextCotIndex++;
        }
        cotNetPositions.push(lastCotNet); // 使用最新的 COT 報告值

        dates.push(dateStr);
        prices.push(silverPrice);
        slvPrices.push(lastSlvPrice);
        slvVolume.push(lastSlvVolume); 
        goldPrices.push(lastGold);
        oilPrices.push(lastOil);
        dxyPrices.push(lastDXY);
        vixPrices.push(lastVIX);
        tnxPrices.push(lastTNX); 
        copperPrices.push(lastCopper); 
    });

    const goldSilverRatio = goldPrices.map((g, i) => g / prices[i]);
    const rsiPrices = calculateRSI(prices, 5); 
    const rsiPrices12 = calculateRSI(prices, 12); 
    const vixRoc = calculateROC(vixPrices); 
    const slvVolumeRoc = calculateROC(slvVolume);
    // dates 已就緒 → TIPS 與 Google Trends 同時發出並一起等待
    tipsPromise   = fetchTIPSYield(dates);       // 🏦 TIPS 5Y 實質利率
    gtrendPromise = fetchGoogleTrends(dates);    // 📈 Google Trends「silver」搜尋熱度
    const fgPromise = fetchFearGreedIndex(dates); // 😨 CNN Fear & Greed（同樣只依賴 dates）

    // 三者都設上限：任何一個來源慢到超過預算就放棄它、用中性備援值，
    // 不讓單一來源卡住整個載入。這些是輔助因子，缺一個不影響核心流程。
    const AUX_BUDGET_MS = 20000;
    const withBudget = (promise, fallback, label) => Promise.race([
        promise,
        new Promise(res => setTimeout(() => {
            console.warn(`[Fetch] ${label} 超過 ${AUX_BUDGET_MS / 1000}s 預算，改用備援值`);
            res(fallback);
        }, AUX_BUDGET_MS))
    ]).catch(() => fallback);

    const [tipsYield, gtrend, fearGreedData] = await Promise.all([
        withBudget(tipsPromise,   dates.map(() => 1.5), 'TIPS'),
        withBudget(gtrendPromise, dates.map(() => 50),  'Google Trends'),
        withBudget(fgPromise,
            { values: dates.map(() => 50), source: 'mock', currentScore: 50, currentRating: 'Neutral' },
            'Fear & Greed'),
    ]);
    console.log(`[Fetch] 輔助因子完成，耗時 ${((performance.now()-tAux)/1000).toFixed(1)}s`);
    step('aux', {});

    // 📦 白銀庫存代理: SLV oz-per-share × 1000 (純本地計算，不需等待網路)
    const slvHoldings = computeSLVHoldings(slvPrices, prices);
    slvHoldings.forEach(v => silverInventory.push(v));

    // 🎯 新增動量/趨勢特徵（用於提升 DA）
    const priceROC    = calculateROC(prices);                       // 白銀日報酬率
    const ma5Silver   = calculateMA(prices, 5);                     // 5日均線
    const ma20Silver  = calculateMA(prices, 20);                    // 20日均線
    const maCross     = ma5Silver.map((v, i) =>                     // MA Cross: (MA5-MA20)/MA20
        ma20Silver[i] !== 0 ? (v - ma20Silver[i]) / ma20Silver[i] : 0
    );
    const ema12Silver = calculateEMA(prices, 12);
    const ema26Silver = calculateEMA(prices, 26);
    const macdLine    = ema12Silver.map((v, i) => v - ema26Silver[i]); // MACD 線

    // 🆕 CNN Fear & Greed Index（已在上方與 TIPS/Google Trends 一起平行取得）
    const fearGreedValues = fearGreedData.values;
    const fearGreedSource = fearGreedData.source;
    const fearGreedCurrent = fearGreedData.currentScore;
    const fearGreedRating  = fearGreedData.currentRating;
    console.log(`[F&G] fearGreedSource=${fearGreedSource}, current=${fearGreedCurrent}`);

    // 🆕 StockTwits 真實情緒序列建構
    // StockTwits 僅提供最新 ~30 則訊息，無歷史 API
    // 做法：
    //   - 最後 1 天：使用真實 StockTwits 評分（stockTwitsResult.score）
    //   - 歷史各天：以「價格動量」建構合理代理序列（趨勢吻合）
    //     然後對最後 N 天做線性插值收斂到真實分數，降低突變感
    const realSTScore = stockTwitsResult.score;        // 0–100
    const stSource    = stockTwitsResult.source;       // 'real' | 'mock'
    const socialSentiment = (() => {
        const n = prices.length;
        const arr = new Array(n);
        // 以價格動量建構歷史代理：動量正→偏多，動量負→偏空
        const momentum = prices.map((p, i) => i === 0 ? 0 : (p - prices[i-1]) / prices[i-1]);
        const smoothWin = 10;
        let base = 50;
        for (let i = 0; i < n; i++) {
            // 10日動量滑動平均（百分比） → 放大 200 倍映射到情緒
            const start = Math.max(0, i - smoothWin + 1);
            const slice = momentum.slice(start, i + 1);
            const avgMom = slice.reduce((a, b) => a + b, 0) / slice.length;
            base = 50 + avgMom * 200;
            base = Math.max(10, Math.min(90, base));
            arr[i] = base;
        }
        // 最後 30 天線性插值，平滑收斂到真實 StockTwits 分數
        const blendDays = Math.min(30, n);
        for (let i = 0; i < blendDays; i++) {
            const w = (i + 1) / blendDays;   // 0→1，越近現在權重越大
            arr[n - blendDays + i] = arr[n - blendDays + i] * (1 - w) + realSTScore * w;
        }
        return arr;
    })();

    // 🔴 必須取得即時報價，且與最新日收盤價差距合理（防止抓錯商品或單位）
    const liveSilver = await liveQuotePromise;
    const lastClose = prices[prices.length - 1];
    if (Number.isFinite(lastClose) && lastClose > 0 && Math.abs(liveSilver.price / lastClose - 1) > 0.3) {
        throw new Error(`即時報價 $${liveSilver.price} 與最新收盤 $${lastClose.toFixed(4)} 差距過大，資料可能有誤`);
    }

    // --- 提取最新的即時數據和來源標籤 ---
    const latestValues = {
        liveSilver,
        sources: {
            symbols: Object.fromEntries(yahooSymbols.map((s, i) => [s, symSrc[i] || '補位（模擬）'])),
            cot: cotDataMap.size ? 'real' : 'fallback', cotRecords: cotDataMap.size,
            tips: tipsYield.source === 'real' ? 'real' : 'fallback',
            gtrend: gtrend.source === 'real' ? 'real' : 'fallback',
            fearGreed: fearGreedSource === 'real' ? 'real' : 'fallback',
            stocktwits: stSource === 'real' ? 'real' : 'fallback',
        },
        dataSource: `Real (Stooq/Yahoo, TIPS:Treasury, SLV Holdings, StockTwits:${stSource}, CFTC, F&G:${fearGreedSource})`,
        dataFetchedAt: new Date().toISOString(),
        dataCacheDate: MarketDataCache.getCacheDate(`sym_SI=F`) || new Date().toLocaleString(),
        price: prices[prices.length - 1],
        slvPrices: slvPrices[slvPrices.length - 1],
        slvVolume: slvVolume[slvVolume.length - 1],
        tipsYield: tipsYield[tipsYield.length - 1],
        rsiPrices: rsiPrices[rsiPrices.length - 1],
        rsiPrices12: rsiPrices12[rsiPrices12.length - 1],
        vixRoc: vixRoc[vixRoc.length - 1],
        socialSentiment: socialSentiment[socialSentiment.length - 1],
        stSource,
        stBullishPct: stockTwitsResult.bullishPct,
        stBearishPct: stockTwitsResult.bearishPct,
        stTotalTagged: stockTwitsResult.totalTagged,
        fearGreed: fearGreedValues[fearGreedValues.length - 1],
        fearGreedSource,
        fearGreedCurrent,
        fearGreedRating,
        goldPrices: goldPrices[goldPrices.length - 1],
        oilPrices: oilPrices[oilPrices.length - 1],
        dxyPrices: dxyPrices[dxyPrices.length - 1],
        vixPrices: vixPrices[vixPrices.length - 1],
        tnxPrices: tnxPrices[tnxPrices.length - 1],
        copperPrices: copperPrices[copperPrices.length - 1],
        silverInventory: silverInventory[silverInventory.length - 1],
        goldSilverRatio: goldSilverRatio[goldSilverRatio.length - 1],
        cotNetPosition: cotNetPositions[cotNetPositions.length - 1],
        // 🎯 動量特徵
        priceROC: priceROC[priceROC.length - 1],
        ma5Silver: ma5Silver[ma5Silver.length - 1],
        ma20Silver: ma20Silver[ma20Silver.length - 1],
        maCross: maCross[maCross.length - 1],
        macdLine: macdLine[macdLine.length - 1],
        gtrend: gtrend[gtrend.length - 1],
    };

    return { 
        dates, prices, slvPrices, slvVolume, slvVolumeRoc, goldPrices, oilPrices, dxyPrices, vixPrices, tnxPrices, copperPrices, silverInventory, goldSilverRatio,
        socialSentiment, fearGreed: fearGreedValues,
        rsiPrices, rsiPrices12, vixRoc, tipsYield,
        cotNetPosition: cotNetPositions,
        // 🎯 新增動量特徵
        priceROC, ma5Silver, ma20Silver, maCross, macdLine,
        gtrend,
        latestValues
    };
};

const filterDataByYears = (data, years) => { 
    if (!data || !data.dates || data.dates.length === 0) return data;
    const daysToFilter = Math.floor(years * 365.25);
    const startIndex = Math.max(0, data.dates.length - daysToFilter);
    if (startIndex === 0 && data.dates.length <= 10) return data; 
    const filtered = {};
    Object.keys(data).forEach(key => {
        if (Array.isArray(data[key])) filtered[key] = data[key].slice(startIndex);
        else filtered[key] = data[key]; 
    });
    return filtered;
};

// ==========================================
// 3. TensorFlow 數據處理與模型構建
// ==========================================

// ══════════════════════════════════════════════════════════════
// 遮罩注意力層 (weightless)
//
// tfjs-layers 沒有內建 MultiHeadAttention。此層刻意「不帶權重」——
// Q/K/V 投影交給標準 tf.layers.dense（在 rank-3 上等同 TimeDistributed），
// 這層只負責注意力數學與因果遮罩。好處：
//   · 不需自行管理 addWeight，序列化/反序列化風險低
//   · 只用 rank-3 × rank-3 的 batched matMul，行為明確
// 「因果」= 第 t 個 patch 只能看到 ≤ t 的 patch，是 decoder-only 的核心，
// 也確保訓練時不會洩漏未來資訊。
// ══════════════════════════════════════════════════════════════
class MaskedAttention extends tf.layers.Layer {
    constructor(config) {
        super(config || {});
        this.dim = (config && config.dim) || 32;
        this._mask = null;
        this._maskP = -1;
    }
    computeOutputShape(inputShape) {
        const s = Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
        return [s[0], s[1], this.dim];
    }
    // 遮罩必須建立在 tidy 之外並以 tf.keep 保留，
    // 否則會被 model.predict/fit 內部的 tidy 回收掉。
    _ensureMask(P) {
        if (this._mask && this._maskP === P) return;
        if (this._mask) { try { this._mask.dispose(); } catch (_) {} }
        const m = [];
        for (let i = 0; i < P; i++) {
            const row = [];
            for (let j = 0; j < P; j++) row.push(j <= i ? 0 : -1e9);
            m.push(row);
        }
        this._mask = tf.keep(tf.tensor2d(m));
        this._maskP = P;
    }
    call(inputs) {
        const arr = Array.isArray(inputs) ? inputs : [inputs];
        const P = arr[0].shape[1];
        this._ensureMask(P);
        return tf.tidy(() => {
            const [q, k, v] = arr;
            let s = tf.matMul(q, k, false, true);        // [B,P,d]×[B,d,P] → [B,P,P]
            s = tf.div(s, Math.sqrt(this.dim));
            s = tf.add(s, this._mask);                    // 遮住未來
            return tf.matMul(tf.softmax(s, -1), v);       // [B,P,P]×[B,P,d] → [B,P,d]
        });
    }
    getConfig() { return Object.assign(super.getConfig(), { dim: this.dim }); }
    // 模型 dispose 時一併釋放遮罩（否則每建一次 TimesFM 就留下幾個 tensor）
    dispose() {
        if (this._mask) { try { this._mask.dispose(); } catch (_) {} this._mask = null; this._maskP = -1; }
        return super.dispose();
    }
    static get className() { return 'MaskedAttention'; }
}
try { tf.serialization.registerClass(MaskedAttention); } catch (_) {}

// 取序列最後一個 token —— decoder-only 的預測位置。
// 用它而非 globalAveragePooling，才符合「以最新 patch 預測下一步」的語意。
class LastToken extends tf.layers.Layer {
    constructor(config) { super(config || {}); }
    computeOutputShape(inputShape) {
        const s = Array.isArray(inputShape[0]) ? inputShape[0] : inputShape;
        return [s[0], s[2]];
    }
    call(inputs) {
        return tf.tidy(() => {
            const x = Array.isArray(inputs) ? inputs[0] : inputs;
            return x.slice([0, x.shape[1] - 1, 0], [-1, 1, -1]).squeeze([1]);
        });
    }
    static get className() { return 'LastToken'; }
}
try { tf.serialization.registerClass(LastToken); } catch (_) {}

// ══════════════════════════════════════════════════════════════
// ⚡ V20.4 融合 RNN 層（FusedBiGRU / FusedLSTM）
// 與 tf.layers.bidirectional(gru) / tf.layers.lstm 數學完全相同
// （含 TF.js 預設的 hardSigmoid 閘門），權重形狀與順序也相同，
// 但每個時間步的 kernel 呼叫數大幅減少：
//   · 所有時間步的輸入投影 X·W 一次 matMul 算完（原本每步一次）
//   · 遞迴權重切分移出迴圈、hardSigmoid 的 0.2/0.5 預先折進權重
//   · WebGL：雙向 GRU 兩個方向堆疊成一個 batched matMul
// 實測（BiGRU 32u, lookback 10, batch 32）：單步 kernel 數 6434 → 1768~3116，
//   WASM 訓練快約 30%，WebGL 快約 2 倍。數值誤差 < 1e-6（已與原生層逐一比對）。
// ══════════════════════════════════════════════════════════════
class FusedBiGRU extends tf.layers.Layer {
    constructor(cfg) {
        super(cfg);
        this.units = cfg.units;
        this.returnSequences = !!cfg.returnSequences;
        this.kernelInitializer = cfg.kernelInitializer || 'glorotUniform';
        this.recurrentInitializer = cfg.recurrentInitializer || 'glorotUniform';
    }
    static get className() { return 'FusedBiGRU'; }
    build(inputShape) {
        const F = inputShape[inputShape.length - 1], U = this.units;
        const ki = tf.initializers[this.kernelInitializer] ? tf.initializers[this.kernelInitializer]({}) : tf.initializers.glorotUniform({});
        const ri = tf.initializers[this.recurrentInitializer] ? tf.initializers[this.recurrentInitializer]({}) : tf.initializers.glorotUniform({});
        // 權重順序與 Bidirectional(GRU) 完全相同：fwd kernel/recurrent/bias, bwd kernel/recurrent/bias
        this.kf = this.addWeight('forward_kernel', [F, 3*U], 'float32', ki);
        this.rf = this.addWeight('forward_recurrent_kernel', [U, 3*U], 'float32', ri);
        this.bf = this.addWeight('forward_bias', [3*U], 'float32', tf.initializers.zeros());
        this.kb = this.addWeight('backward_kernel', [F, 3*U], 'float32', ki);
        this.rb = this.addWeight('backward_recurrent_kernel', [U, 3*U], 'float32', ri);
        this.bb = this.addWeight('backward_bias', [3*U], 'float32', tf.initializers.zeros());
        this.built = true;
    }
    computeOutputShape(s) { return this.returnSequences ? [s[0], s[1], 2*this.units] : [s[0], 2*this.units]; }
    call(inputs) {
        return tf.tidy(() => {
            const x = Array.isArray(inputs) ? inputs[0] : inputs;
            const [B, T, F] = x.shape; const U = this.units;
            // 1) 一次算完所有時間步、兩個方向的輸入投影
            const K  = tf.concat([this.kf.read(), this.kb.read()], 1);              // [F,6U]
            const bs = tf.concat([this.bf.read(), this.bb.read()], 0);              // [6U]
            const xp2 = tf.add(tf.matMul(tf.reshape(x, [B*T, F]), K), bs);            // [B*T,6U]
            // TF.js GRU 預設 recurrentActivation = hardSigmoid = clip(0.2a+0.5, 0, 1)
            // 把 0.2 / 0.5 預先折進輸入投影與遞迴權重 → 迴圈內只剩 add + clip
            if (tf.getBackend() === 'webgl') return this._callStacked(xp2, B, T, U);
            return this._callSplit(xp2, B, T, U);
        });
    }
    // WebGL：兩方向堆疊成 batched matMul（kernel 呼叫數減半，GPU dispatch 是瓶頸）
    _callStacked(xp2, B, T, U) {
        let xp = tf.transpose(tf.reshape(xp2, [B, T, 2, 3*U]), [1, 2, 0, 3]);       // [T,2,B,3U]
        const [xpF, xpBr] = tf.split(xp, 2, 1);
        xp = tf.concat([xpF, tf.reverse(xpBr, 0)], 1);                                // 反向分支時間倒序
        const [xzrAll, xhAll] = tf.split(xp, [2*U, U], 3);
        const xzr = tf.unstack(tf.add(tf.mul(xzrAll, 0.2), 0.5));                    // T × [2,B,2U]
        const xh  = tf.unstack(xhAll);                                                // T × [2,B,U]
        const R = tf.stack([this.rf.read(), this.rb.read()]);                        // [2,U,3U]
        const [R1raw, R2] = tf.split(R, [2*U, U], 2);
        const R1 = tf.mul(R1raw, 0.2);
        let h = tf.zeros([2, B, U]);
        const hs = [];
        for (let t = 0; t < T; t++) {
            const zr = tf.clipByValue(tf.add(xzr[t], tf.matMul(h, R1)), 0, 1);
            const [z, r] = tf.split(zr, 2, 2);
            const hh = tf.tanh(tf.add(xh[t], tf.matMul(tf.mul(r, h), R2)));
            h = tf.add(hh, tf.mul(z, tf.sub(h, hh)));                                // z*h + (1-z)*hh
            if (this.returnSequences) hs.push(h);
        }
        if (!this.returnSequences) return tf.reshape(tf.transpose(h, [1, 0, 2]), [B, 2*U]);  // concat(fwd,bwd)
        const H = tf.stack(hs);                                                       // [T,2,B,U]
        const [hF4, hB4] = tf.split(H, 2, 1);
        const hF = tf.squeeze(hF4, [1]);
        const hB = tf.reverse(tf.squeeze(hB4, [1]), 0);
        return tf.transpose(tf.concat([hF, hB], 2), [1, 0, 2]);                       // [B,T,2U]
    }
    // WASM/CPU：小矩陣 batched matMul 反而慢 → 兩個 2D matMul；時間倒序改用 JS 索引，省掉 reverse
    _callSplit(xp2, B, T, U) {
        const xp = tf.transpose(tf.reshape(xp2, [B, T, 6*U]), [1, 0, 2]);             // [T,B,6U]
        const [zrF, hxF, zrB, hxB] = tf.split(xp, [2*U, U, 2*U, U], 2);
        const X = [
            [tf.unstack(tf.add(tf.mul(zrF, 0.2), 0.5)), tf.unstack(hxF)],
            [tf.unstack(tf.add(tf.mul(zrB, 0.2), 0.5)), tf.unstack(hxB)],
        ];
        const W = [this.rf.read(), this.rb.read()].map(Rd => {
            const [a1, a2] = tf.split(Rd, [2*U, U], 1);
            return [tf.mul(a1, 0.2), a2];
        });
        const outs = [[], []];
        const last = [null, null];
        for (let d = 0; d < 2; d++) {
            let h = tf.zeros([B, U]);
            for (let s = 0; s < T; s++) {
                const t = d === 0 ? s : T - 1 - s;                                     // 反向分支由尾端讀起
                const zr = tf.clipByValue(tf.add(X[d][0][t], tf.matMul(h, W[d][0])), 0, 1);
                const [z, r] = tf.split(zr, 2, 1);
                const hh = tf.tanh(tf.add(X[d][1][t], tf.matMul(tf.mul(r, h), W[d][1])));
                h = tf.add(hh, tf.mul(z, tf.sub(h, hh)));
                if (this.returnSequences) outs[d][t] = h;                              // 依原始時間位置存放
            }
            last[d] = h;
        }
        if (!this.returnSequences) return tf.concat(last, 1);                          // [B,2U]
        return tf.concat([tf.stack(outs[0], 1), tf.stack(outs[1], 1)], 2);             // [B,T,2U]
    }
    getConfig() { return { ...super.getConfig(), units: this.units, returnSequences: this.returnSequences,
        kernelInitializer: this.kernelInitializer, recurrentInitializer: this.recurrentInitializer }; }
}
class FusedLSTM extends tf.layers.Layer {
    constructor(cfg) { super(cfg); this.units = cfg.units; this.returnSequences = !!cfg.returnSequences; }
    static get className() { return 'FusedLSTM'; }
    build(inputShape) {
        const F = inputShape[inputShape.length - 1], U = this.units;
        this.k = this.addWeight('kernel', [F, 4*U], 'float32', tf.initializers.glorotUniform({}));
        this.r = this.addWeight('recurrent_kernel', [U, 4*U], 'float32', tf.initializers.orthogonal({}));
        this.b = this.addWeight('bias', [4*U], 'float32', tf.initializers.zeros());
        this.built = true;
    }
    computeOutputShape(s) { return this.returnSequences ? [s[0], s[1], this.units] : [s[0], this.units]; }
    call(inputs) {
        return tf.tidy(() => {
            const x = Array.isArray(inputs) ? inputs[0] : inputs;
            const [B, T, F] = x.shape; const U = this.units;
            // 閘門 i,f,o 用 hardSigmoid（TF.js 預設）：把 0.2 / 0.5 折進權重；g 閘門維持原值（tanh）
            const one = Array(U).fill(1), sc = Array(U).fill(0.2), z0 = Array(U).fill(0), hf = Array(U).fill(0.5);
            const scale  = tf.tensor1d([...sc, ...sc, ...one, ...sc]);
            const offset = tf.tensor1d([...hf, ...hf, ...z0, ...hf]);
            let xp = tf.add(tf.matMul(tf.reshape(x, [B*T, F]), this.k.read()), this.b.read());
            xp = tf.add(tf.mul(xp, scale), offset);
            const xs = tf.unstack(tf.transpose(tf.reshape(xp, [B, T, 4*U]), [1, 0, 2]));  // T × [B,4U]
            const R = tf.mul(this.r.read(), scale);
            let h = tf.zeros([B, U]), c = tf.zeros([B, U]);
            const hs = [];
            for (let t = 0; t < T; t++) {
                const zz = tf.add(xs[t], tf.matMul(h, R));
                const s = tf.clipByValue(zz, 0, 1);            // i,f,o 一次算完 hardSigmoid
                const [i, f, , o] = tf.split(s, 4, 1);
                const g = tf.tanh(tf.split(zz, 4, 1)[2]);
                c = tf.add(tf.mul(f, c), tf.mul(i, g));
                h = tf.mul(o, tf.tanh(c));
                if (this.returnSequences) hs.push(h);
            }
            return this.returnSequences ? tf.stack(hs, 1) : h;
        });
    }
    getConfig() { return { ...super.getConfig(), units: this.units, returnSequences: this.returnSequences }; }
}
try { tf.serialization.registerClass(FusedBiGRU); } catch (_) {}
try { tf.serialization.registerClass(FusedLSTM); } catch (_) {}

// ══════════════════════════════════════════════════════════════
// TimesFM 架構 (patched decoder-only transformer)
//
// ⚠️ 這是 TimesFM 的「架構」，以你的資料從頭訓練，
//    不是 Google 發布的預訓練基礎模型權重
//    （那是 200M/500M 參數的 PyTorch/JAX 模型，無法在瀏覽器執行）。
//    因此沒有 zero-shot 能力，需與其他演算法一樣訓練。
//
// 論文核心要素，依瀏覽器算力縮小實作：
//   1. Input patching —— 時間軸切成不重疊 patch 作為 token，
//      比逐時間點當 token 更省算力，也更容易抓到區段型態
//   2. Residual block —— patch embedding 用帶線性跳接的 MLP
//   3. Stacked decoder layers —— 因果自注意力 + FFN
//   4. 取最後一個 patch 的表示解碼
// ══════════════════════════════════════════════════════════════
const createTimesFM = (inputTensor, lookback, finalNumFeatures, params) => {
    const dModel  = Math.min(params.layers, 64);   // shader uniform 安全上限
    const ffDim   = Math.min(dModel * 2, 128);
    const nLayers = 2;

    // patch 長度只取「能整除 lookback」的值，reshape 才合法；
    // 另要求至少切出 2 個 patch，否則整段變成單一 token，注意力就失去意義
    //（例如 lookback=7 若取 patchLen=7 只會得到 1 個 token）。
    // 都不滿足就退回 1：每個時間點自成一個 token，永遠合法。
    let patchLen = 1;
    for (let cand = Math.min(8, Math.floor(lookback / 2)); cand >= 2; cand--) {
        if (lookback % cand === 0) { patchLen = cand; break; }
    }
    const numPatches = lookback / patchLen;

    // [B, lookback, F] → [B, numPatches, patchLen×F]
    let x = tf.layers.reshape({ targetShape: [numPatches, patchLen * finalNumFeatures] }).apply(inputTensor);

    // ── 1. 輸入 residual block ──
    const inHidden = tf.layers.dense({ units: dModel, activation: 'relu' }).apply(x);
    const inProj   = tf.layers.dense({ units: dModel }).apply(inHidden);
    const inSkip   = tf.layers.dense({ units: dModel }).apply(x);
    x = tf.layers.add().apply([inProj, inSkip]);

    // ── 2. decoder layers ──
    for (let i = 0; i < nLayers; i++) {
        const q = tf.layers.dense({ units: dModel }).apply(x);
        const k = tf.layers.dense({ units: dModel }).apply(x);
        const v = tf.layers.dense({ units: dModel }).apply(x);
        const attn = new MaskedAttention({ dim: dModel }).apply([q, k, v]);
        x = tf.layers.add().apply([x, attn]);
        x = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(x);

        let ff = tf.layers.dense({ units: ffDim, activation: 'relu' }).apply(x);
        ff = tf.layers.dropout({ rate: 0.1 }).apply(ff);
        ff = tf.layers.dense({ units: dModel }).apply(ff);
        x = tf.layers.add().apply([x, ff]);
        x = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(x);
    }

    // ── 3. 取最後一個 patch ──
    x = new LastToken({}).apply(x);

    // ── 4. 輸出 residual block ──
    const outHidden = tf.layers.dense({ units: dModel, activation: 'relu' }).apply(x);
    const outProj   = tf.layers.dense({ units: dModel }).apply(outHidden);
    const outSkip   = tf.layers.dense({ units: dModel }).apply(x);
    return tf.layers.add().apply([outProj, outSkip]);
};

// Transformer 編碼器核心
const createTransformerEncoder = (inputTensor, lookback, finalNumFeatures, params) => {
    const numHeads = Math.min(8, Math.floor(params.layers / 16)); 
    // Cap ffDim: too large → MAX_FRAGMENT_UNIFORM_VECTORS exceeded → shader link failure
    const ffDim = Math.min(params.layers * 2, 128);  // was ×4, reduced to ×2, cap 128

    let x = tf.layers.dense({ units: params.layers, activation: 'relu' }).apply(inputTensor);

    let attentionOutput = tf.layers.conv1d({
        filters: params.layers,
        kernelSize: 3, 
        padding: 'same',
        activation: 'relu'
    }).apply(x);

    x = tf.layers.add().apply([x, attentionOutput]);
    x = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(x);

    let ffOutput = tf.layers.dense({ units: ffDim, activation: 'relu' }).apply(x);
    ffOutput = tf.layers.dropout({ rate: 0.1 }).apply(ffOutput);
    ffOutput = tf.layers.dense({ units: params.layers }).apply(ffOutput);

    x = tf.layers.add().apply([x, ffOutput]);
    x = tf.layers.layerNormalization({ epsilon: 1e-6 }).apply(x);

    let output = tf.layers.globalAveragePooling1d().apply(x); 

    return output; 
};


const createModel = (algorithm, lookback, finalNumFeatures, params) => {
    // 限制搜尋時的模型複雜度，防止 shader uniform 超限
    // WebGL MAX_FRAGMENT_UNIFORM_VECTORS = 1024 (mobile) ~ 4096 (desktop)
    // GRU/LSTM shader uniform 數 ≈ units² × 4 — 超過時 shader link 失敗
    const safeUnits = Math.min(params.layers, 64); // cap 64 for shader safety

    const inputShape = [lookback, finalNumFeatures];
    const inputTensor = tf.input({ shape: inputShape });
    let outputTensor;

    // ⚡ 融合 RNN（預設開啟）：數學與原生層相同，但 kernel 呼叫數大幅減少
    const useFused = params.fastRnn !== false;
    if (algorithm === 'BiGRU' && useFused) {
        outputTensor = new FusedBiGRU({ units: safeUnits, returnSequences: true, recurrentInitializer: 'glorotUniform' }).apply(inputTensor);
        outputTensor = tf.layers.dropout({ rate: 0.3 }).apply(outputTensor);
        outputTensor = new FusedBiGRU({ units: safeUnits, returnSequences: false, recurrentInitializer: 'glorotUniform' }).apply(outputTensor);
    } else if (algorithm === 'LSTM' && useFused) {
        outputTensor = new FusedLSTM({ units: safeUnits, returnSequences: true }).apply(inputTensor);
        outputTensor = tf.layers.dropout({ rate: 0.3 }).apply(outputTensor);
        outputTensor = new FusedLSTM({ units: safeUnits, returnSequences: false }).apply(outputTensor);
    } else if (algorithm === 'BiGRU') {
        outputTensor = tf.layers.bidirectional({
            layer: tf.layers.gru({ units: safeUnits, returnSequences: true, recurrentInitializer: 'glorotUniform' }),
            inputShape: inputShape
        }).apply(inputTensor);
        outputTensor = tf.layers.dropout({ rate: 0.3 }).apply(outputTensor);
        outputTensor = tf.layers.bidirectional({
            layer: tf.layers.gru({ units: safeUnits, returnSequences: false, recurrentInitializer: 'glorotUniform' })
        }).apply(outputTensor);
    } else if (algorithm === 'LSTM') {
        outputTensor = tf.layers.lstm({
            units: safeUnits,
            returnSequences: true,
            inputShape: inputShape
        }).apply(inputTensor);
        outputTensor = tf.layers.dropout({ rate: 0.3 }).apply(outputTensor);
        outputTensor = tf.layers.lstm({
            units: safeUnits,
            returnSequences: false
        }).apply(outputTensor);
    } else if (algorithm === 'Transformer') {
        // Transformer with safeUnits
        const tParams = { ...params, layers: safeUnits };
        outputTensor = createTransformerEncoder(inputTensor, lookback, finalNumFeatures, tParams);
    } else if (algorithm === 'TimesFM') {
        const fmParams = { ...params, layers: safeUnits };
        outputTensor = createTimesFM(inputTensor, lookback, finalNumFeatures, fmParams);
    } else {
        throw new Error(`未知的演算法: ${algorithm}`);
    }

    // 依預測目標決定輸出層（價格迴歸 / 方向分類 / 雙頭）
    const _spec = resolveTargetSpec(params.predTarget || 'price', params);
    outputTensor = tf.layers.dense({ units: _spec.units, activation: _spec.activation }).apply(outputTensor);
    const model = tf.model({ inputs: inputTensor, outputs: outputTensor });
    return model;
};

const createMultivariateDataset = (data, lookback, selectedFactors, maxSamples = null) => {
    const { prices, slvPrices, slvVolume, slvVolumeRoc, goldPrices, oilPrices, dxyPrices, vixPrices, tnxPrices, copperPrices, silverInventory, goldSilverRatio, socialSentiment, rsiPrices, rsiPrices12, vixRoc, tipsYield, cotNetPosition,
            priceROC, ma5Silver, ma20Silver, maCross, macdLine, fearGreed, gtrend } = data;

    const scale = (arr) => {
        const min = Math.min(...arr);
        const max = Math.max(...arr);
        const range = max - min;
        return { scaled: arr.map(v => range === 0 ? 0 : (v - min) / range), min, max };
    };

    const pObj = scale(prices);
    const slvPObj = scale(slvPrices); 
    const slvVObj = scale(slvVolume); 
    const slvVRocObj = scale(slvVolumeRoc);
    const tipsObj = scale(tipsYield);  // 🏦 TIPS 5Y Real Yield 
    const rsiObj = scale(rsiPrices); 
    const rsi12Obj = scale(rsiPrices12); 
    const vrocObj = scale(vixRoc);
    const sObj = scale(socialSentiment);
    const fgArr = fearGreed || prices.map(() => 50);
    const fgObj = scale(fgArr);
    const gObj = scale(goldPrices);
    const oObj = scale(oilPrices);
    const dObj = scale(dxyPrices);
    const vObj = scale(vixPrices); 
    const tObj = scale(tnxPrices); 
    const cObj = scale(copperPrices); 
    const invObj = scale(silverInventory); // 📦 SLV Holdings proxy 
    const gsrObj = scale(goldSilverRatio); 
    const cotObj = scale(cotNetPosition); 
    // 🎯 動量/趨勢特徵 scaling
    const priceROCArr  = priceROC  || prices.map(() => 0);
    const ma5Arr       = ma5Silver || calculateMA(prices, 5);
    const ma20Arr      = ma20Silver|| calculateMA(prices, 20);
    const maCrossArr   = maCross   || ma5Arr.map((v, i) => ma20Arr[i] !== 0 ? (v - ma20Arr[i]) / ma20Arr[i] : 0);
    const macdArr      = macdLine  || prices.map(() => 0);
    const pRocObj   = scale(priceROCArr);
    const ma5Obj    = scale(ma5Arr);
    const ma20Obj   = scale(ma20Arr);
    const maCrossObj= scale(maCrossArr);
    const macdObj   = scale(macdArr);
    const gtrendArr  = gtrend || prices.map(() => 50);
    const gtrendObj = scale(gtrendArr);


    const allFeatures = [
        { key: 'price', scaled: pObj.scaled, name: '白銀期貨價格 (Lag)', isExternal: false, minMax: pObj, isSelected: true }, 
        { key: 'slvPrices', scaled: slvPObj.scaled, name: 'SLV ETF 收盤價', isExternal: true, minMax: slvPObj, isSelected: selectedFactors.slvPrices }, 
        { key: 'slvVolume', scaled: slvVObj.scaled, name: 'SLV ETF 成交量', isExternal: true, minMax: slvVObj, isSelected: selectedFactors.slvVolume }, 
        { key: 'slvVolumeRoc', scaled: slvVRocObj.scaled, name: 'SLV ETF 成交量 ROC', isExternal: true, minMax: slvVRocObj, isSelected: selectedFactors.slvVolumeRoc }, 
        { key: 'tipsYield', scaled: tipsObj.scaled, name: '實質利率 TIPS 5Y Real Yield (%)', isExternal: true, minMax: tipsObj, isSelected: selectedFactors.tipsYield }, 
        { key: 'rsiPrices', scaled: rsiObj.scaled, name: '白銀 RSI 指標 (5)', isExternal: true, minMax: rsiObj, isSelected: selectedFactors.rsiPrices }, 
        { key: 'rsiPrices12', scaled: rsi12Obj.scaled, name: '白銀 RSI 指標 (12)', isExternal: true, minMax: rsi12Obj, isSelected: selectedFactors.rsiPrices12 },
        { key: 'vixRoc', scaled: vrocObj.scaled, name: 'VIX 變動率 (ROC)', isExternal: true, minMax: vrocObj, isSelected: selectedFactors.vixRoc }, 
        { key: 'fearGreed', scaled: fgObj.scaled, name: 'CNN Fear & Greed (0-100)', isExternal: true, minMax: fgObj, isSelected: selectedFactors.fearGreed },
        { key: 'socialSentiment', scaled: sObj.scaled, name: 'StockTwits 白銀情緒', isExternal: true, minMax: sObj, isSelected: selectedFactors.socialSentiment },
        { key: 'goldPrices', scaled: gObj.scaled, name: '黃金價格 (GC=F)', isExternal: true, minMax: gObj, isSelected: selectedFactors.goldPrices },
        { key: 'oilPrices', scaled: oObj.scaled, name: '原油價格 (CL=F)', isExternal: true, minMax: oObj, isSelected: selectedFactors.oilPrices },
        { key: 'dxyPrices', scaled: dObj.scaled, name: '美元指數 (DXY)', isExternal: true, minMax: dObj, isSelected: selectedFactors.dxyPrices },
        { key: 'vixPrices', scaled: vObj.scaled, name: 'VIX 恐慌指數 (^VIX)', isExternal: true, minMax: vObj, isSelected: selectedFactors.vixPrices }, 
        { key: 'tnxPrices', scaled: tObj.scaled, name: '10年期利率 (^TNX)', isExternal: true, minMax: tObj, isSelected: selectedFactors.tnxPrices },
        { key: 'copperPrices', scaled: cObj.scaled, name: '銅期貨價格 (HG=F)', isExternal: true, minMax: cObj, isSelected: selectedFactors.copperPrices }, 
        { key: 'silverInventory', scaled: invObj.scaled, name: '白銀庫存代理 (SLV oz/share×1k)', isExternal: true, minMax: invObj, isSelected: selectedFactors.silverInventory }, 
        { key: 'goldSilverRatio', scaled: gsrObj.scaled, name: '黃金/白銀比', isExternal: true, minMax: gsrObj, isSelected: selectedFactors.goldSilverRatio }, 
        { key: 'cotNetPosition', scaled: cotObj.scaled, name: 'COT Managed Money Net', isExternal: true, minMax: cotObj, isSelected: selectedFactors.cotNetPosition }, 
        // 🎯 動量/趨勢特徵（DA 提升關鍵）
        { key: 'priceROC',   scaled: pRocObj.scaled,    name: '白銀日報酬率 (ROC)',    isExternal: true, minMax: pRocObj,    isSelected: selectedFactors.priceROC   },
        { key: 'ma5Silver',  scaled: ma5Obj.scaled,     name: '白銀 MA5',              isExternal: true, minMax: ma5Obj,     isSelected: selectedFactors.ma5Silver  },
        { key: 'ma20Silver', scaled: ma20Obj.scaled,    name: '白銀 MA20',             isExternal: true, minMax: ma20Obj,    isSelected: selectedFactors.ma20Silver },
        { key: 'maCross',    scaled: maCrossObj.scaled,  name: 'MA Cross (MA5-MA20)',   isExternal: true, minMax: maCrossObj, isSelected: selectedFactors.maCross    },
        { key: 'macdLine',   scaled: macdObj.scaled,     name: 'MACD 線 (12-26 EMA)',   isExternal: true, minMax: macdObj,    isSelected: selectedFactors.macdLine   },
        { key: 'gtrend',     scaled: gtrendObj.scaled,   name: 'Google Trends「silver」(0-100)', isExternal: true, minMax: gtrendObj, isSelected: selectedFactors.gtrend },
    ];

    const activeFeatures = allFeatures.filter(f => f.isSelected);
    const numFeatures = activeFeatures.length; 

    const X = [];
    const y = [];
    const yDir = [];        // 方向標籤：1=上漲, 0=下跌（相對前一日）
    const yLogRet = [];     // 對數報酬率目標
    const prevPrices = [];  // 每個樣本對應的前一日實際價格（重建價格用）

    if (prices.length <= lookback) {
        console.error("數據點不足以建立 Lookback 序列。");
        return { X: [], y: [], yDir: [], yLogRet: [], prevPrices: [],
                 minMax: { price: pObj, logRet: { mean: 0, std: 1 } },
                 allMinMax: {}, activeFeatures, numFeatures: 0 };
    }

    // maxSamples: 只建最後 N 個視窗（搜尋用），省去建全量再 slice 的白工
    const startIdx = maxSamples ? Math.max(lookback, prices.length - maxSamples) : lookback;
    for(let i = startIdx; i < prices.length; i++) {
        const sequence = [];
        for(let j = 0; j < lookback; j++) {
            sequence.push(activeFeatures.map(f => f.scaled[i - lookback + j]));
        }
        X.push(sequence);
        y.push(pObj.scaled[i]);
        // 方向標籤：今日價格 vs 前一日（用未正規化價格判斷）
        yDir.push(prices[i] > prices[i - 1] ? 1 : 0);
        // 對數報酬率目標 r = ln(P_t / P_{t-1})，以及重建價格所需的前一日實際價格
        const pPrev = prices[i - 1], pNow = prices[i];
        yLogRet.push((pPrev > 0 && pNow > 0) ? Math.log(pNow / pPrev) : 0);
        prevPrices.push(pPrev);
    }

    // 對數報酬率用 z-score 標準化：報酬率以 0 為中心且量級小（~±0.02），
    // 用 min-max 會被極端值主導，標準化才能讓 Huber 在合理尺度上運作。
    const lrMean = yLogRet.length ? yLogRet.reduce((s, v) => s + v, 0) / yLogRet.length : 0;
    const lrVar  = yLogRet.length ? yLogRet.reduce((s, v) => s + (v - lrMean) ** 2, 0) / yLogRet.length : 0;
    const lrStd  = Math.sqrt(lrVar) || 1;   // 全為常數時避免除以 0
    const yLogRetScaled = yLogRet.map(v => (v - lrMean) / lrStd);

    const allMinMax = {};
    activeFeatures.forEach(f => { allMinMax[f.key] = f.minMax; });

    return { X, y, yDir, yLogRet: yLogRetScaled, prevPrices,
             minMax: { price: pObj, logRet: { mean: lrMean, std: lrStd } },
             allMinMax, activeFeatures, numFeatures };
};

// ... (calculateFactorInfluence, calculateDirectionalAccuracy, calculateKFoldDirectionalAccuracy, runMultiStepPrediction, exportModel remains the same) ...
const calculateFactorInfluence = async (model, lastInput, minMax, activeFeatures, finalNumFeatures) => {
    // ── 防護性校驗 ──
    if (!lastInput || !lastInput.length || !lastInput[0]?.length) {
        console.warn('[Influence] lastInput 無效，跳過分析');
        return {};
    }

    const actualNumFeatures = lastInput[0].length;
    const lookback          = lastInput.length;
    const perturbationDelta = 0.05;

    // activeFeatures 長度應與 actualNumFeatures 一致；若不一致則截斷或以 index label 補齊
    const safeName = (index) => {
        const f = Array.isArray(activeFeatures) ? activeFeatures[index] : undefined;
        return f?.name || `Feature_${index}`;
    };

    // 模型預期的輸入特徵數（從 model.layers[0] 取得）
    let modelExpectedFeatures = actualNumFeatures;
    try {
        const inputShape = model.layers[0].batchInputShape;
        if (inputShape && inputShape[2]) modelExpectedFeatures = inputShape[2];
    } catch(_) {}

    if (actualNumFeatures !== modelExpectedFeatures) {
        console.warn(`[Influence] 特徵數不符：X=${actualNumFeatures}, model=${modelExpectedFeatures}，跳過分析`);
        return null;
    }

    const influence = {};

    // ── 基準預測（帶 finally 確保 tensor 釋放）──
    let basePred = 0;
    {
        let bT, bP;
        try {
            bT = tf.tensor3d([lastInput], [1, lookback, actualNumFeatures]);
            bP = model.predict(bT);
            const bd = await bP.data();
            basePred = denormalizePrice(bd[0], minMax);
        } catch (e) {
            console.warn('[Influence] 基準預測失敗:', e?.message);
            return {};   // 基準都失敗就無從比較，早返回
        } finally {
            if (bT) tf.dispose(bT);
            if (bP) tf.dispose(bP);
        }
    }

    // ── 逐特徵擾動 ──
    for (let index = 1; index < actualNumFeatures; index++) {   // index 0 = price，跳過
        const featureName = safeName(index);
        let tP, pP, tM, pM;
        try {
            const origVal = lastInput[lookback - 1][index];

            // + delta
            const plus = lastInput.map((step, si) =>
                si === lookback - 1
                    ? step.map((v, fi) => fi === index ? Math.min(1.0, v + perturbationDelta) : v)
                    : [...step]
            );
            tP = tf.tensor3d([plus],  [1, lookback, actualNumFeatures]);
            pP = model.predict(tP);
            const dp = await pP.data();

            // - delta
            const minus = lastInput.map((step, si) =>
                si === lookback - 1
                    ? step.map((v, fi) => fi === index ? Math.max(0.0, v - perturbationDelta) : v)
                    : [...step]
            );
            tM = tf.tensor3d([minus], [1, lookback, actualNumFeatures]);
            pM = model.predict(tM);
            const dm = await pM.data();

            const predPlus  = denormalizePrice(dp[0], minMax);
            const predMinus = denormalizePrice(dm[0], minMax);
            influence[featureName] = (predPlus - predMinus) / 2;   // central difference

        } catch (e) {
            console.warn(`[Influence] 特徵 ${featureName} (idx=${index}) 擾動失敗:`, e?.message);
            influence[featureName] = 0;   // 失敗的特徵給 0，不中斷整個分析
        } finally {
            if (tP) tf.dispose(tP);
            if (pP) tf.dispose(pP);
            if (tM) tf.dispose(tM);
            if (pM) tf.dispose(pM);
        }

        // 每 8 個特徵讓 event loop 喘息，避免 UI 凍結
        if (index % 8 === 0) await new Promise(r => setTimeout(r, 0));
    }

    return influence;
};

const calculateDirectionalAccuracy = (fullPrices, lookback, predictedHistory) => {
    if (!predictedHistory || predictedHistory.length === 0) return 0;
    let correctDirectionCount = 0;
    let validCount = 0;
    for (let i = 0; i < predictedHistory.length; i++) {
        const prevIdx = lookback + i - 1;
        const curIdx  = lookback + i;
        if (prevIdx < 0 || curIdx >= fullPrices.length) continue;
        const actualPrevPrice    = fullPrices[prevIdx];
        const actualCurrentPrice = fullPrices[curIdx];
        const predictedCurrentPrice = predictedHistory[i];
        if (!Number.isFinite(actualPrevPrice) || !Number.isFinite(actualCurrentPrice) ||
            !Number.isFinite(predictedCurrentPrice)) continue;
        const actualDir    = (actualCurrentPrice - actualPrevPrice) >= 0;
        const predictedDir = (predictedCurrentPrice - actualPrevPrice) >= 0;
        if (actualDir === predictedDir) correctDirectionCount++;
        validCount++;
    }
    return validCount > 0 ? (correctDirectionCount / validCount) * 100 : 0;
};

const calculateKFoldDirectionalAccuracy = (testActuals, testPreds) => {
    if (testActuals.length <= 1 || testActuals.length !== testPreds.length) return 0;
    let correctDirectionCount = 0;
    for (let i = 1; i < testActuals.length; i++) {
        const actualChange = testActuals[i] - testActuals[i - 1];
        const predictedChange = testPreds[i] - testActuals[i - 1]; 

        const actualDir = actualChange >= 0; 
        const predictedDir = predictedChange >= 0; 
        if (actualDir === predictedDir) correctDirectionCount++;
    }
    return (correctDirectionCount / (testActuals.length - 1)) * 100;
};


const runMultiStepPrediction = async (model, X, y, minMax, activeFeatures, finalNumFeatures, lookback, futureDays, logsUpdater, predTarget = 'price', prevPrices = null) => {
    logsUpdater('正在計算歷史擬合路徑...');
    if (X.length === 0) {
        logsUpdater('錯誤: 歷史數據不足，無法進行預測。');
        return { historyPreds: [], futurePrices: [], influenceScores: null, actualHistory: [], dirProbs: [] };
    }

    // ── WebGL 記憶體防護：tensor 數量超過閾值時先做 backend reset ──
    const memBefore = tf.memory();
    if (memBefore.numTensors > 800) {
        logsUpdater(`⚠️ 偵測到殘留 tensor 過多 (${memBefore.numTensors})，正在清理...`);
        try { await window.SilverBackend.restore(); } catch(_) {}
    }

    const denormalizeLocal = (val) => denormalizePrice(val, minMax);
    const sig = (z) => 1 / (1 + Math.exp(-z));
    const stride = predTarget === 'dual' ? 2 : 1;  // dual 輸出 2 個單元
    // 對數報酬率：把標準化輸出還原為 r，再由前一日價格重建價格
    const lrStats  = (minMax && minMax.logRet) ? minMax.logRet : { mean: 0, std: 1 };
    const toLogRet = (z) => z * lrStats.std + lrStats.mean;
    const predLogRets = [];   // 供方向準確率直接由報酬率正負號判定
    // 反正規化價格 → [0,1] 尺度，迭代預測時要把重建價格寫回特徵 0
    const pMin = minMax.price.min, pMax = minMax.price.max;
    const normPrice = (v) => (pMax - pMin) === 0 ? 0 : Math.max(0, Math.min(1, (v - pMin) / (pMax - pMin)));

    // ── 1. 分批次歷史擬合（避免一次性大 tensor 造成 OOM）──
    const BATCH = 128; // 每批推理筆數，可視記憶體壓力調小
    const historyPreds = [];   // 價格（price/dual）或機率轉合成價（direction）
    const dirProbs = [];       // 方向機率（direction/dual）
    let batchErr = null;

    // Warmup（捕捉 WebGL context 遺失問題）
    try {
        const w = tf.zeros([1, lookback, finalNumFeatures]);
        const wp = model.predict(w);
        await wp.data();
        tf.dispose([w, wp]);   // dispose both input AND output
    } catch (e) {
        logsUpdater(`⚠️ 模型預熱失敗，嘗試繼續 (${e.message?.slice(0, 80)})`);
    }

    for (let start = 0; start < X.length; start += BATCH) {
        const chunk = X.slice(start, start + BATCH);
        let chunkTensor, predTensor;
        try {
            chunkTensor = tf.tensor3d(chunk, [chunk.length, lookback, finalNumFeatures]);
            predTensor  = model.predict(chunkTensor);
            const data  = await predTensor.data();
            for (let r = 0; r < chunk.length; r++) {
                if (predTarget === 'logreturn') {
                    // 由「前一日實際價格」重建，避免誤差沿著歷史累積
                    const rr = toLogRet(data[r]);
                    predLogRets.push(rr);
                    const idx  = start + r;
                    const base = (prevPrices && prevPrices[idx] != null)
                        ? prevPrices[idx]
                        : denormalizeLocal(y[idx]);
                    historyPreds.push(base * Math.exp(rr));
                } else if (predTarget === 'price') {
                    historyPreds.push(denormalizeLocal(data[r]));
                } else if (predTarget === 'direction') {
                    const p = data[r];           // sigmoid 機率
                    dirProbs.push(p);
                    historyPreds.push(p);        // 直接存機率（圖表以機率呈現）
                } else { // dual
                    historyPreds.push(denormalizeLocal(data[r * 2]));      // 單元0=價格
                    dirProbs.push(sig(data[r * 2 + 1]));                   // 單元1=方向 logit
                }
            }
        } catch (e) {
            batchErr = e;
            logsUpdater(`❌ 歷史擬合批次失敗 (起始=${start}): ${e.message?.slice(0, 120)}`);
            break;
        } finally {
            if (chunkTensor) tf.dispose(chunkTensor);
            if (predTensor)  tf.dispose(predTensor);
        }
        // 每 512 筆讓 UI 喘口氣
        if (start % 512 === 0 && start > 0) await new Promise(r => setTimeout(r, 0));
    }

    if (batchErr) {
        logsUpdater('歷史擬合中斷，已回傳部分結果。請嘗試重新執行或重新整理頁面。');
        // 仍回傳已收集到的部分結果，不讓整個預測流程崩潰
    } else {
        logsUpdater(`歷史擬合完成 (${historyPreds.length} 筆)。`);
    }

    // ── 2. 迭代未來預測 ──
    // 從 X 資料本身推導真實形狀（不依賴外部參數，避免不一致）
    const iterLookback = X[X.length - 1].length;            // 實際 lookback（行數）
    const iterFeatures = X[X.length - 1][0]?.length ?? 1;   // 實際特徵數（列數）
    logsUpdater(`正在執行 T+${futureDays} 日迭代預測（lookback=${iterLookback}, features=${iterFeatures}）...`);

    // deep-copy 最後一個 window，確保不污染原始 X
    let currentInput = X[X.length - 1].map(row => Float32Array.from(row));
    const futurePrices = [];
    const futureDirProbs = [];
    let lastScaled = currentInput[currentInput.length - 1][0]; // 最後一步的 price scaled 值

    for (let i = 0; i < futureDays; i++) {
        let inputTensor, outputTensor;
        try {
            // 用從資料推導的真實形狀建立 tensor
            const flat = [];
            for (const row of currentInput) for (const v of row) flat.push(v);
            inputTensor  = tf.tensor(flat, [1, iterLookback, iterFeatures]);
            outputTensor = model.predict(inputTensor);
            const out = await outputTensor.data();

            if (predTarget === 'logreturn') {
                // 複利式推進：P_{t+1} = P_t · exp(r)，並把新價格寫回視窗特徵 0
                const rr = toLogRet(out[0]);
                predLogRets.push(rr);
                const base = futurePrices.length
                    ? futurePrices[futurePrices.length - 1]
                    : (prevPrices && prevPrices.length ? denormalizeLocal(y[y.length - 1]) : denormalizeLocal(lastScaled));
                const np = base * Math.exp(rr);
                futurePrices.push(np);
                lastScaled = normPrice(np);
            } else if (predTarget === 'direction') {
                // 純方向：輸出為機率，無價格可迭代 → 以機率推一個微幅價格變化供圖表呈現
                const prob = out[0];
                futureDirProbs.push(prob);
                const stepPct = (prob - 0.5) * 0.02; // ±1% 視機率強度
                lastScaled = Math.max(0, Math.min(1, lastScaled * (1 + stepPct)));
                futurePrices.push(denormalizeLocal(lastScaled));
            } else if (predTarget === 'dual') {
                lastScaled = out[0];                    // 單元0=價格
                futureDirProbs.push(sig(out[1]));       // 單元1=方向機率
                futurePrices.push(denormalizeLocal(lastScaled));
            } else {
                lastScaled = out[0];
                futurePrices.push(denormalizeLocal(lastScaled));
            }

            // 滑動視窗：把（合成/預測的）價格填入 feature 0
            const lastRow   = currentInput[currentInput.length - 1];
            const nextRow   = Float32Array.from(lastRow);
            nextRow[0]      = lastScaled;
            currentInput    = [...currentInput.slice(1), nextRow];

        } catch (e) {
            const msg = e?.message ?? String(e);
            logsUpdater(`⚠️ T+${i+1} 迭代失敗: ${msg.slice(0,100)}`);
            const lastRow = currentInput[currentInput.length - 1];
            const nextRow = Float32Array.from(lastRow);
            nextRow[0]    = lastScaled;
            currentInput  = [...currentInput.slice(1), nextRow];
            futurePrices.push(futurePrices[futurePrices.length - 1] ?? denormalizeLocal(lastScaled));
        } finally {
            if (inputTensor)  tf.dispose(inputTensor);
            if (outputTensor) tf.dispose(outputTensor);
        }
    }

    // ── 3. 因子敏感度分析 ──
    logsUpdater('正在進行因子敏感度分析...');
    let influenceScores = null;
    if (finalNumFeatures > 1) {
        try {
            influenceScores = await calculateFactorInfluence(
                model, X[X.length - 1], minMax, activeFeatures, finalNumFeatures
            );
        } catch (e) {
            logsUpdater(`⚠️ 因子敏感度分析失敗: ${e.message?.slice(0,80)}`);
        }
    }

    // ── 4. 實際歷史價格序列 ──
    const actualHistory = Array.isArray(y) && y.length > 0
        ? y.map(denormalizeLocal)
        : (activeFeatures.find(f => f.key === 'price')?.minMax?.scaled?.slice(lookback) ?? []).map(denormalizeLocal);

    const memAfter = tf.memory();
    console.log(`[Predict] tensors: ${memBefore.numTensors} → ${memAfter.numTensors}, bytes: ${(memAfter.numBytes/1024/1024).toFixed(1)} MB`);

    return { historyPreds, futurePrices, influenceScores, actualHistory, dirProbs, futureDirProbs, predLogRets };
};
// ── 匯出 ──
window.SE = {
  tf, denormalizePrice, getDirectionalHuberLoss, getHuberLoss, getDualHeadLoss, resolveTargetSpec, buildTargetTensor,
  calculateMAPE, calculateRSI, calculateROC, calculateMA, calculateEMA,
  ProxyRotator, AgentStore, splitHoldout, shouldPromote, scoreHoldout, resolvePredictions, recentPerformance, shouldRetrain,
  MarketDataCache, fastYield, makeTicker, snapshotWeights, disposeSnapshot, persistSnapshot, makeBatchYielder,
  prepareTrainingBackend, fetchLiveSilverQuote, fetchMarketData, filterDataByYears,
  createModel, createMultivariateDataset, calculateFactorInfluence, calculateDirectionalAccuracy,
  calculateKFoldDirectionalAccuracy, runMultiStepPrediction,
};
})();

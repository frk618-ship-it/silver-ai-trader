/* Silver AI Trader v2 — 選項定義（模型與因子選項沿用 v20.3）*/
(function () {
'use strict';

// 模型：與 v20.3 相同的四種架構，附上給使用者看的說明
const MODELS = [
  { key: 'BiGRU', name: 'Bi-GRU', tag: '推薦', icon: '⇄',
    desc: '雙向門控循環網路，從前後兩個方向讀取價格序列。穩定、訓練快，最適合當作起點。',
    speed: 4, depth: 3 },
  { key: 'LSTM', name: 'LSTM', tag: '經典', icon: '≋',
    desc: '長短期記憶網路，擅長記住較長的趨勢。參數較少，資料不多時比較不容易過度擬合。',
    speed: 5, depth: 3 },
  { key: 'Transformer', name: 'Transformer', tag: '注意力', icon: '◎',
    desc: '用卷積＋前饋網路的輕量 Transformer 編碼器，捕捉局部型態，訓練速度快。',
    speed: 5, depth: 2 },
  { key: 'TimesFM', name: 'TimesFM', tag: '前沿', icon: '▦',
    desc: '仿 Google TimesFM 的 decoder-only 架構：把時間序列切成 patch，用因果注意力預測下一步。適合較長的 lookback。',
    speed: 3, depth: 4 },
];

// 預測目標：與 v20.3 相同
const TARGETS = [
  { key: 'price', name: '價格迴歸', short: '價格', desc: '預測下一個交易日的價格，漲跌方向由價格變化推導。損失函數加上方向懲罰，兼顧價格與方向。' },
  { key: 'direction', name: '方向分類', short: '方向', desc: '直接預測「漲或跌」的機率，會給出模型信心。不輸出價格，只看方向時最直接。' },
  { key: 'dual', name: '雙頭（價格＋方向）', short: '雙頭', desc: '同時輸出價格與方向機率，兩個任務互相輔助；可同時看到預測價與信心。' },
  { key: 'logreturn', name: '對數報酬率', short: '報酬率', desc: '預測 ln(今日/昨日)，再由前一日價格換算。尺度穩定，價格精度通常最好。' },
];

// 因子：與 v20.3 相同的 25 個（price 固定納入）
const FACTORS = [
  { key: 'price', name: '白銀期貨價格', group: 'silver', fixed: true, src: 'SI=F' },
  { key: 'slvPrices', name: 'SLV ETF 收盤價', group: 'silver', src: 'SLV' },
  { key: 'slvVolume', name: 'SLV ETF 成交量', group: 'silver', src: 'SLV' },
  { key: 'slvVolumeRoc', name: 'SLV 成交量變動率', group: 'silver', src: 'SLV' },
  { key: 'silverInventory', name: '白銀庫存代理', group: 'silver', src: 'SLV÷SI', hint: 'SLV 每股含銀量 ×1000，反映 ETF 溢價／折價' },
  { key: 'rsiPrices', name: 'RSI (5)', group: 'tech', src: '計算' },
  { key: 'rsiPrices12', name: 'RSI (12)', group: 'tech', src: '計算' },
  { key: 'priceROC', name: '日報酬率', group: 'tech', src: '計算' },
  { key: 'ma5Silver', name: '5 日均線', group: 'tech', src: '計算' },
  { key: 'ma20Silver', name: '20 日均線', group: 'tech', src: '計算' },
  { key: 'maCross', name: '均線交叉 (MA5−MA20)', group: 'tech', src: '計算' },
  { key: 'macdLine', name: 'MACD 線', group: 'tech', src: '計算' },
  { key: 'goldPrices', name: '黃金期貨', group: 'cross', src: 'GC=F' },
  { key: 'goldSilverRatio', name: '金銀比', group: 'cross', src: 'GC÷SI' },
  { key: 'copperPrices', name: '銅期貨', group: 'cross', src: 'HG=F' },
  { key: 'oilPrices', name: '原油期貨', group: 'cross', src: 'CL=F' },
  { key: 'dxyPrices', name: '美元指數', group: 'macro', src: 'DXY' },
  { key: 'tnxPrices', name: '10 年期殖利率', group: 'macro', src: '^TNX' },
  { key: 'tipsYield', name: 'TIPS 5 年實質利率', group: 'macro', src: '美國財政部' },
  { key: 'vixPrices', name: 'VIX 恐慌指數', group: 'sentiment', src: '^VIX' },
  { key: 'vixRoc', name: 'VIX 變動率', group: 'sentiment', src: '^VIX' },
  { key: 'fearGreed', name: 'CNN 恐懼與貪婪', group: 'sentiment', src: 'CNN' },
  { key: 'socialSentiment', name: 'StockTwits 情緒', group: 'sentiment', src: 'StockTwits' },
  { key: 'gtrend', name: 'Google 搜尋熱度', group: 'sentiment', src: 'Google Trends' },
  { key: 'cotNetPosition', name: 'COT 管理基金淨部位', group: 'flow', src: 'CFTC' },
];
const GROUPS = {
  silver: { name: '白銀本身', color: 'var(--c-silver)' },
  tech: { name: '技術指標', color: 'var(--c-tech)' },
  cross: { name: '跨市場', color: 'var(--c-cross)' },
  macro: { name: '總體經濟', color: 'var(--c-macro)' },
  sentiment: { name: '市場情緒', color: 'var(--c-sent)' },
  flow: { name: '籌碼', color: 'var(--c-flow)' },
};

const all = () => Object.fromEntries(FACTORS.slice(1).map(f => [f.key, true]));
const only = (keys) => Object.fromEntries(FACTORS.slice(1).map(f => [f.key, keys.includes(f.key)]));
const FACTOR_PRESETS = [
  { key: 'all', name: '全部 25 個', make: all },
  { key: 'lean', name: '精簡 10 個', make: () => only(['slvPrices', 'goldPrices', 'goldSilverRatio', 'dxyPrices', 'tipsYield', 'rsiPrices', 'maCross', 'macdLine', 'cotNetPosition', 'vixPrices']) },
  { key: 'tech', name: '純技術面', make: () => only(['rsiPrices', 'rsiPrices12', 'priceROC', 'ma5Silver', 'ma20Silver', 'maCross', 'macdLine']) },
  { key: 'macro', name: '總經＋跨市場', make: () => only(['goldPrices', 'goldSilverRatio', 'copperPrices', 'oilPrices', 'dxyPrices', 'tnxPrices', 'tipsYield']) },
];

// 訓練強度預設（對應 v20.3 的 epochs / lookback / units 等參數）
const TRAIN_PRESETS = [
  { key: 'quick', name: '快速', desc: '約 1 分鐘，先看看方向', p: { epochs: 120, earlyStopPatience: 20 } },
  { key: 'standard', name: '標準', desc: 'v20.3 預設，平衡速度與品質', p: { epochs: 400, earlyStopPatience: 0 } },
  { key: 'thorough', name: '精細', desc: '更多 epochs，搭配早停避免過度擬合', p: { epochs: 800, earlyStopPatience: 60 } },
];

// v20.3 預設參數
const DEFAULT_PARAMS = {
  algorithm: 'BiGRU', predTarget: 'price',
  lookback: 10, layers: 32, epochs: 400, batchSize: 32, lr: 0.001,
  directionalLambda: 0.3, huberDelta: 1.0, futureDays: 3, historyYears: 2,
  validationMode: 'none', kFolds: 5,
  backend: 'auto', fastRnn: true, earlyStopPatience: 0,
};

window.SV_CONFIG = { MODELS, TARGETS, FACTORS, GROUPS, FACTOR_PRESETS, TRAIN_PRESETS, DEFAULT_PARAMS, defaultFactors: all };
})();

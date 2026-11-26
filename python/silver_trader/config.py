"""全域設定：因子清單、預設參數、資料目錄。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ── 資料目錄（模型、紀錄、快取都放這裡；可用環境變數覆寫）──
HOME = Path(os.environ.get("SILVER_TRADER_HOME", Path.home() / ".silver_trader"))


def home() -> Path:
    """回傳資料目錄（每次呼叫都重新讀環境變數，方便測試切換）。"""
    p = Path(os.environ.get("SILVER_TRADER_HOME", HOME))
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── 市場代碼 ──
SYMBOLS = {
    "SI=F": "白銀期貨",
    "SLV": "SLV ETF",
    "GC=F": "黃金期貨",
    "CL=F": "原油期貨",
    "DX-Y.NYB": "美元指數",
    "^VIX": "VIX",
    "^TNX": "10 年期殖利率",
    "HG=F": "銅期貨",
}
STOOQ_MAP = {
    "SI=F": "si.f", "SLV": "slv.us", "GC=F": "gc.f", "CL=F": "cl.f",
    "DX-Y.NYB": "dx.f", "^VIX": "^vix", "^TNX": "10usy.b", "HG=F": "hg.f",
}
# 缺資料時的前向填補起始值（與 HTML 版相同）
FILL_DEFAULTS = {"GC=F": 2000, "CL=F": 75, "DX-Y.NYB": 100, "^VIX": 15, "^TNX": 4.0, "HG=F": 3.5, "SLV": 22}


@dataclass(frozen=True)
class Factor:
    key: str
    name: str
    group: str


# 順序即模型輸入欄位順序；price 永遠是第 0 欄
FACTORS: list[Factor] = [
    Factor("price", "白銀期貨價格 (Lag)", "白銀"),
    Factor("slvPrices", "SLV ETF 收盤價", "白銀"),
    Factor("slvVolume", "SLV ETF 成交量", "白銀"),
    Factor("slvVolumeRoc", "SLV ETF 成交量 ROC", "白銀"),
    Factor("tipsYield", "實質利率 TIPS 5Y (%)", "總經"),
    Factor("rsiPrices", "白銀 RSI (5)", "技術"),
    Factor("rsiPrices12", "白銀 RSI (12)", "技術"),
    Factor("vixRoc", "VIX 變動率 (ROC)", "情緒"),
    Factor("fearGreed", "CNN Fear & Greed", "情緒"),
    Factor("socialSentiment", "StockTwits 白銀情緒", "情緒"),
    Factor("goldPrices", "黃金價格 (GC=F)", "跨市場"),
    Factor("oilPrices", "原油價格 (CL=F)", "跨市場"),
    Factor("dxyPrices", "美元指數 (DXY)", "總經"),
    Factor("vixPrices", "VIX 恐慌指數", "情緒"),
    Factor("tnxPrices", "10 年期利率 (^TNX)", "總經"),
    Factor("copperPrices", "銅期貨價格 (HG=F)", "跨市場"),
    Factor("silverInventory", "白銀庫存代理 (SLV oz/share×1k)", "白銀"),
    Factor("goldSilverRatio", "金銀比", "跨市場"),
    Factor("cotNetPosition", "COT 管理基金淨部位", "籌碼"),
    Factor("priceROC", "白銀日報酬率 (ROC)", "技術"),
    Factor("ma5Silver", "白銀 MA5", "技術"),
    Factor("ma20Silver", "白銀 MA20", "技術"),
    Factor("maCross", "MA Cross (MA5−MA20)", "技術"),
    Factor("macdLine", "MACD 線 (12−26 EMA)", "技術"),
    Factor("gtrend", "Google Trends「silver」", "情緒"),
]
FACTOR_KEYS = [f.key for f in FACTORS]
OPTIONAL_FACTOR_KEYS = FACTOR_KEYS[1:]
FACTOR_NAME = {f.key: f.name for f in FACTORS}

ALGORITHMS = ["BiGRU", "LSTM", "Transformer", "TimesFM"]
PRED_TARGETS = {
    "price": "價格迴歸",
    "direction": "方向分類",
    "dual": "雙頭（價格＋方向）",
    "logreturn": "對數報酬率",
    "multi": "多日報酬率（直接預測 T+1…T+N）",
}


@dataclass
class TrainParams:
    algorithm: str = "BiGRU"
    # 預設用對數報酬率：實測真實資料時，直接預測「價格水準」的誤差（3–7%）和單日漲跌差不多大，
    # 推算出的明日漲跌幅會失真（例如 +16%）；預測報酬率的幅度合理，價格誤差也較小。
    # 預設「多日報酬率」：一次直接預測 T+1…T+N 的累積報酬率，不靠逐日遞迴（遞迴時其他因子停在今天，越往後越失真）
    pred_target: str = "multi"
    lookback: int = 10
    layers: int = 32
    epochs: int = 400
    batch_size: int = 32
    lr: float = 0.001
    directional_lambda: float = 0.3
    huber_delta: float = 1.0
    future_days: int = 5          # 多日報酬率：模型輸出的天數（訓練時固定）；其他目標：遞迴預測的天數
    history_years: float = 2.0
    validation_mode: str = "none"   # none | kfold
    k_folds: int = 5
    early_stop_patience: int = 0    # 0 = 關閉
    seed: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainParams":
        known = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**known)


def default_factors() -> dict[str, bool]:
    return {k: True for k in OPTIONAL_FACTOR_KEYS}


@dataclass
class BacktestConfig:
    test_ratio: float = 0.3
    initial_capital: float = 100_000.0
    leverage: float = 1.0
    position_pct: float = 1.0
    fee_bps: float = 2.0
    slippage_bps: float = 2.0
    threshold: float = 0.0
    allow_short: bool = True


@dataclass
class SearchConfig:
    strategy: str = "random"   # random | grid | bayesian
    metric: str = "da"         # da | loss
    max_trials: int = 15
    search_epochs: int = 60
    max_samples: int = 520
    patience: int = 8
    seed: int | None = None


LIVE_QUOTE_MAX_AGE_HOURS = 96          # 涵蓋週末＋假日休市
LIVE_QUOTE_MAX_DEVIATION = 0.30        # 即時價與最新收盤最大容許偏差
DATA_YEARS = 5

"""組裝市場資料：抓取 → 對齊到白銀期貨交易日 → 計算衍生特徵。

🔴 規則（與 HTML 版相同）：白銀期貨歷史必須是真實資料、且必須取得新鮮的即時報價，
否則丟出 MarketDataError，UI 停在載入畫面，不會用模擬資料頂替。
其餘輔助因子失敗時以中性值補位，並在狀態表中標示。
"""
from __future__ import annotations

import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from . import config, indicators, sources
from .sources import AuxResult, LiveQuote

log = logging.getLogger(__name__)


class MarketDataError(RuntimeError):
    pass


@dataclass
class SourceStatus:
    name: str
    source: str
    real: bool
    detail: str = ""


@dataclass
class MarketData:
    frame: pd.DataFrame                      # index=日期，欄位=因子 key
    live_quote: LiveQuote
    status: list[SourceStatus] = field(default_factory=list)
    extras: dict = field(default_factory=dict)   # F&G 目前分數、StockTwits 多空比…
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def prices(self) -> np.ndarray:
        return self.frame["price"].to_numpy(dtype=float)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.frame.index

    @property
    def last_close(self) -> float:
        return float(self.frame["price"].iloc[-1])

    def recent(self, years: float) -> "MarketData":
        n = int(years * 365.25)
        cut = self.frame.index[-1] - pd.Timedelta(days=n)
        f = self.frame[self.frame.index > cut]
        if len(f) < 10:
            f = self.frame
        return MarketData(f, self.live_quote, self.status, self.extras, self.fetched_at)


# ── 磁碟快取：只存「真實」抓到的資料，作為網路失敗時的備援 ──
def _cache_path(name: str):
    p = config.home() / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.pkl"


def _cache_put(name: str, obj) -> None:
    try:
        with open(_cache_path(name), "wb") as f:
            pickle.dump({"ts": datetime.now(timezone.utc), "data": obj}, f)
    except Exception as e:  # noqa: BLE001
        log.warning("快取寫入失敗 %s: %s", name, e)


def _cache_get(name: str):
    try:
        with open(_cache_path(name), "rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001
        return None


def _sym_key(s: str) -> str:
    return "hist_" + "".join(c if c.isalnum() else "_" for c in s)


def _ffill_on(index: pd.DatetimeIndex, s: pd.Series | None, default: float, seed_first: bool = False) -> np.ndarray:
    """把序列前向填補到指定日期索引；開頭缺值用 default（或序列第一筆）。"""
    if s is None or len(s) == 0:
        return np.full(len(index), float(default))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    out = s.reindex(index.union(s.index)).ffill().reindex(index)
    first = float(s.iloc[0]) if seed_first else float(default)
    return out.fillna(first).to_numpy(dtype=float)


def _social_sentiment(prices: np.ndarray, real_score: float | None) -> np.ndarray:
    """StockTwits 沒有歷史 API：以 10 日動量建構代理序列，最後 30 天平滑收斂到真實分數。"""
    n = len(prices)
    mom = np.zeros(n)
    mom[1:] = np.diff(prices) / prices[:-1]
    c = np.cumsum(np.insert(mom, 0, 0))
    idx = np.arange(1, n + 1)
    start = np.maximum(idx - 10, 0)
    avg = (c[idx] - c[start]) / (idx - start)
    arr = np.clip(50 + avg * 200, 10, 90)
    if real_score is not None:
        k = min(30, n)
        w = (np.arange(k) + 1) / k
        arr[n - k:] = arr[n - k:] * (1 - w) + real_score * w
    return arr


def build_frame(hist: dict[str, pd.DataFrame], cot: AuxResult, tips: AuxResult, fg: AuxResult,
                st: AuxResult, gt: AuxResult) -> pd.DataFrame:
    si = hist["SI=F"]
    idx = pd.DatetimeIndex(si.index).normalize()
    prices = si["close"].to_numpy(dtype=float)

    def col(sym: str, field_: str = "close") -> np.ndarray:
        df = hist.get(sym)
        s = df[field_] if df is not None else None
        return _ffill_on(idx, s, config.FILL_DEFAULTS.get(sym, 0.0))

    f = pd.DataFrame(index=idx)
    f["price"] = prices
    f["slvPrices"] = col("SLV")
    f["slvVolume"] = _ffill_on(idx, hist["SLV"]["volume"] if "SLV" in hist else None, 5_000_000)
    f["slvVolumeRoc"] = indicators.roc(f["slvVolume"].to_numpy())
    f["goldPrices"] = col("GC=F")
    f["oilPrices"] = col("CL=F")
    f["dxyPrices"] = col("DX-Y.NYB")
    f["vixPrices"] = col("^VIX")
    f["tnxPrices"] = col("^TNX")
    f["copperPrices"] = col("HG=F")
    f["goldSilverRatio"] = f["goldPrices"] / f["price"]
    f["rsiPrices"] = indicators.rsi(prices, 5)
    f["rsiPrices12"] = indicators.rsi(prices, 12)
    f["vixRoc"] = indicators.roc(f["vixPrices"].to_numpy())
    f["tipsYield"] = _ffill_on(idx, tips.series, 0.5)
    f["cotNetPosition"] = _ffill_on(idx, cot.series, 0.0, seed_first=True)
    f["fearGreed"] = _ffill_on(idx, fg.series, fg.extra.get("score", 50.0) if fg.real else 50.0)
    f["gtrend"] = _ffill_on(idx, gt.series, 50.0)
    slv = f["slvPrices"].to_numpy()
    f["silverInventory"] = np.where((slv > 0) & (prices > 0), slv / np.where(prices > 0, prices, 1) * 1000, 930.0)
    f["priceROC"] = indicators.roc(prices)
    f["ma5Silver"] = indicators.sma(prices, 5)
    f["ma20Silver"] = indicators.sma(prices, 20)
    f["maCross"] = np.where(f["ma20Silver"] != 0, (f["ma5Silver"] - f["ma20Silver"]) / f["ma20Silver"], 0.0)
    f["macdLine"] = indicators.ema(prices, 12) - indicators.ema(prices, 26)
    f["socialSentiment"] = _social_sentiment(prices, st.extra.get("score") if st.real else None)
    return f[config.FACTOR_KEYS]


def load_market(progress: Callable[[str, str], None] | None = None) -> MarketData:
    """抓取所有資料。progress(step, message) 供 UI 顯示進度。"""
    def say(step: str, msg: str):
        if progress:
            progress(step, msg)
        log.info("[%s] %s", step, msg)

    now = pd.Timestamp.now()
    years = list(range(now.year - config.DATA_YEARS, now.year + 1))
    symbols = list(config.SYMBOLS)
    say("start", "同時向各資料來源發出請求…")
    with ThreadPoolExecutor(max_workers=8) as ex:
        f_quote = ex.submit(sources.fetch_live_quote)
        f_hist = ex.submit(sources.fetch_histories, symbols, config.DATA_YEARS)
        f_cot = ex.submit(sources.fetch_cot)
        f_tips = ex.submit(sources.fetch_tips, years)
        f_fg = ex.submit(sources.fetch_fear_greed)
        f_st = ex.submit(sources.fetch_stocktwits)
        f_gt = ex.submit(sources.fetch_google_trends, now - pd.DateOffset(years=config.DATA_YEARS))

        # ① 即時報價（必要）
        try:
            quote = f_quote.result(timeout=60)
            say("quote", f"白銀期貨即時報價 ${quote.price:.3f}（{quote.source}）")
        except Exception as e:  # noqa: BLE001
            raise MarketDataError(str(e)) from e

        # ② 歷史價格（白銀期貨必要；其餘可補位）
        hist, hsrc = f_hist.result(timeout=120)
        status: list[SourceStatus] = []
        for s in symbols:
            if s in hist:
                _cache_put(_sym_key(s), hist[s])
                status.append(SourceStatus(config.SYMBOLS[s], hsrc[s], True, f"{len(hist[s])} 筆"))
                continue
            cached = _cache_get(_sym_key(s))
            if cached is not None:
                hist[s] = cached["data"]
                status.append(SourceStatus(config.SYMBOLS[s], "本機快取", True,
                                           f"網路失敗，使用 {cached['ts']:%Y-%m-%d %H:%M} UTC 的快取"))
            else:
                status.append(SourceStatus(config.SYMBOLS[s], "無", False, "以固定值補位"))
        if "SI=F" not in hist:
            raise MarketDataError("無法取得白銀期貨（SI=F）真實歷史價格（Yahoo 與 Stooq 均失敗，且無快取）")
        say("history", f"白銀期貨歷史 {len(hist['SI=F'])} 筆")

        aux = {"COT 籌碼": f_cot, "TIPS 實質利率": f_tips, "Fear & Greed": f_fg,
               "StockTwits 情緒": f_st, "Google Trends": f_gt}
        res: dict[str, AuxResult] = {}
        for name, fut in aux.items():
            try:
                res[name] = fut.result(timeout=40)
            except Exception as e:  # noqa: BLE001
                res[name] = AuxResult(None, "無", False, f"逾時或失敗：{e}")
            r = res[name]
            status.append(SourceStatus(name, r.source, r.real, r.detail if r.real else f"以中性值補位（{r.detail[:80]}）"))
        say("aux", "輔助因子完成")

    frame = build_frame(hist, res["COT 籌碼"], res["TIPS 實質利率"], res["Fear & Greed"],
                        res["StockTwits 情緒"], res["Google Trends"])
    last = float(frame["price"].iloc[-1])
    if abs(quote.price / last - 1) > config.LIVE_QUOTE_MAX_DEVIATION:
        raise MarketDataError(f"即時報價 ${quote.price:.3f} 與最新收盤 ${last:.4f} 差距過大，資料可能有誤")

    extras = {
        "fear_greed": res["Fear & Greed"].extra if res["Fear & Greed"].real else None,
        "stocktwits": res["StockTwits 情緒"].extra if res["StockTwits 情緒"].real else None,
    }
    md = MarketData(frame, quote, [SourceStatus("白銀期貨即時報價", quote.source, True,
                                                f"{quote.time:%Y-%m-%d %H:%M} UTC")] + status, extras)
    _cache_put("market_snapshot", md)
    return md


def refresh_quote(md: MarketData) -> MarketData:
    """只更新即時報價（主畫面定時刷新用）。失敗時保留原報價並丟出例外。"""
    q = sources.fetch_live_quote()
    md.live_quote = q
    return md

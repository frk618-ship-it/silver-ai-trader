"""外部資料來源。

所有網路存取都經過 `http_get` 與 `_yf_*` 兩個入口，方便測試時替換。
Python 端沒有瀏覽器的 CORS 限制，因此不再需要 HTML 版的 proxy 輪替。
"""
from __future__ import annotations

import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


# ══════════════════════════════════════════════════════════════
# 網路入口（測試時 monkeypatch 這幾個函式）
# ══════════════════════════════════════════════════════════════
def http_get(url: str, timeout: float = 12.0, headers: dict | None = None) -> requests.Response:
    h = {"User-Agent": UA, "Accept": "*/*"}
    if headers:
        h.update(headers)
    r = requests.get(url, timeout=timeout, headers=h)
    r.raise_for_status()
    return r


def _yf_download(symbols: list[str], years: int) -> dict[str, pd.DataFrame]:
    """yfinance 批次下載日線；回傳 {symbol: DataFrame[close, volume]}。"""
    import yfinance as yf

    raw = yf.download(symbols, period=f"{years}y", interval="1d", group_by="ticker",
                      auto_adjust=False, threads=True, progress=False)
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
            df = df[["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "volume"})
            df = df.dropna(subset=["close"])
            if len(df) >= 30:
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                out[s] = df
        except Exception as e:  # noqa: BLE001
            log.warning("yfinance %s 解析失敗: %s", s, e)
    return out


def _yf_intraday(symbol: str) -> pd.DataFrame:
    """yfinance 1 分鐘線（近 5 日，涵蓋週末）。"""
    import yfinance as yf

    return yf.Ticker(symbol).history(period="5d", interval="1m", auto_adjust=False)


# ══════════════════════════════════════════════════════════════
# 歷史價格
# ══════════════════════════════════════════════════════════════
def _parse_stooq_csv(text: str, years: int) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df or "close" not in df:
        raise ValueError("Stooq CSV 欄位不足")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df[df["close"] > 0]
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(365.25 * years))
    df = df[df["date"] >= cutoff].sort_values("date")
    if "volume" not in df:
        df["volume"] = 0.0
    df = df.set_index("date")[["close", "volume"]]
    if len(df) < 30:
        raise ValueError(f"Stooq 資料過少（{len(df)} 筆）")
    return df


def fetch_stooq_history(symbol: str, years: int = config.DATA_YEARS) -> pd.DataFrame | None:
    s = config.STOOQ_MAP.get(symbol)
    if not s:
        return None
    try:
        r = http_get(f"https://stooq.com/q/d/l/?s={s}&i=d", timeout=15)
        return _parse_stooq_csv(r.text, years)
    except Exception as e:  # noqa: BLE001
        log.warning("Stooq %s 失敗: %s", symbol, e)
        return None


def fetch_histories(symbols: list[str], years: int = config.DATA_YEARS) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """先用 yfinance 批次抓，失敗的再走 Stooq。回傳 (資料, 來源標籤)。"""
    data: dict[str, pd.DataFrame] = {}
    src: dict[str, str] = {}
    try:
        got = _yf_download(symbols, years)
        for s, df in got.items():
            data[s], src[s] = df, "Yahoo Finance"
    except Exception as e:  # noqa: BLE001
        log.warning("yfinance 批次下載失敗: %s", e)
    for s in symbols:
        if s in data:
            continue
        df = fetch_stooq_history(s, years)
        if df is not None:
            data[s], src[s] = df, "Stooq"
    return data, src


# ══════════════════════════════════════════════════════════════
# 🔴 白銀期貨即時報價
# ══════════════════════════════════════════════════════════════
@dataclass
class LiveQuote:
    price: float
    time: datetime          # UTC
    source: str
    symbol: str = "SI=F"
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def age_hours(self) -> float:
        return (datetime.now(timezone.utc) - self.time).total_seconds() / 3600

    def to_dict(self) -> dict:
        return {"price": self.price, "time": self.time.isoformat(), "source": self.source,
                "symbol": self.symbol, "fetched_at": self.fetched_at.isoformat()}


class LiveQuoteError(RuntimeError):
    pass


def _validate_quote(q: LiveQuote) -> LiveQuote:
    if not np.isfinite(q.price) or not (3 <= q.price <= 1000):
        raise LiveQuoteError(f"報價數值異常（{q.price}）")
    if q.age_hours > config.LIVE_QUOTE_MAX_AGE_HOURS:
        raise LiveQuoteError(f"報價過舊（{q.age_hours:.1f} 小時前）")
    if q.age_hours < -1:
        raise LiveQuoteError("報價時間在未來，時間戳記異常")
    return q


def _quote_yfinance() -> LiveQuote:
    df = _yf_intraday("SI=F")
    df = df.dropna(subset=["Close"]) if df is not None and len(df) else df
    if df is None or len(df) == 0:
        raise LiveQuoteError("yfinance 無 1 分鐘資料")
    ts = pd.Timestamp(df.index[-1])
    ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    return _validate_quote(LiveQuote(float(df["Close"].iloc[-1]), ts.to_pydatetime(), "Yahoo Finance"))


def _quote_yahoo_chart() -> LiveQuote:
    r = http_get("https://query1.finance.yahoo.com/v8/finance/chart/SI=F?range=1d&interval=1m", timeout=10)
    res = (r.json().get("chart") or {}).get("result") or []
    if not res:
        raise LiveQuoteError("Yahoo chart 無資料")
    meta = res[0].get("meta") or {}
    price, t = meta.get("regularMarketPrice"), meta.get("regularMarketTime")
    if price is None or t is None:
        ts = res[0].get("timestamp") or []
        closes = ((res[0].get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        for i in range(len(closes) - 1, -1, -1):
            if closes[i] is not None:
                price, t = closes[i], ts[i]
                break
    if price is None or t is None:
        raise LiveQuoteError("Yahoo chart 缺少價格")
    return _validate_quote(LiveQuote(float(price), datetime.fromtimestamp(int(t), timezone.utc), "Yahoo Finance"))


def _quote_stooq() -> LiveQuote:
    r = http_get("https://stooq.com/q/l/?s=si.f&f=sd2t2ohlcv&h&e=csv", timeout=10)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip().lower() for c in df.columns]
    row = df.iloc[0]
    price = pd.to_numeric(row.get("close"), errors="coerce")
    # Stooq 時間為華沙當地時間；新鮮度判斷差 1~2 小時不影響
    ts = pd.Timestamp(f"{row.get('date')} {row.get('time', '00:00:00')}", tz="Europe/Warsaw").tz_convert("UTC")
    return _validate_quote(LiveQuote(float(price), ts.to_pydatetime(), "Stooq", "SI.F"))


QUOTE_SOURCES = [("Yahoo (yfinance)", _quote_yfinance), ("Yahoo chart API", _quote_yahoo_chart), ("Stooq", _quote_stooq)]


def fetch_live_quote() -> LiveQuote:
    """依序嘗試各來源；全部失敗則丟出 LiveQuoteError（附上每個來源的原因）。"""
    errors = []
    for name, fn in QUOTE_SOURCES:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    raise LiveQuoteError("無法取得白銀期貨（SI=F）即時報價 —— " + "；".join(errors))


# ══════════════════════════════════════════════════════════════
# 輔助因子
# ══════════════════════════════════════════════════════════════
@dataclass
class AuxResult:
    series: pd.Series | None          # 以日期為索引；None = 失敗
    source: str                       # 顯示用來源
    real: bool                        # 是否為真實資料
    detail: str = ""
    extra: dict = field(default_factory=dict)


def fetch_cot() -> AuxResult:
    """CFTC COT（Disaggregated 管理基金淨部位 → Legacy 非商業淨部位）。"""
    sources = [
        ("CFTC Disaggregated", "https://publicreporting.cftc.gov/resource/72hh-3qpy.json",
         "as_of_date_form_yyyy_mm_dd", "m_money_positions_long_all", "m_money_positions_short_all"),
        ("CFTC Legacy", "https://publicreporting.cftc.gov/resource/6dca-aqww.json",
         "report_date_as_yyyy_mm_dd", "noncomm_positions_long_all", "noncomm_positions_short_all"),
    ]
    wheres = [
        "upper(market_and_exchange_names) like '%SILVER%COMMODITY%EXCHANGE%INC%'",
        "upper(market_and_exchange_names) like '%SILVER%'",
    ]
    errors = []
    for label, base, dcol, lcol, scol in sources:
        for w in wheres:
            q = urlencode({"$select": f"{dcol},{lcol},{scol}", "$where": w, "$order": f"{dcol} DESC", "$limit": 400})
            try:
                rows = http_get(f"{base}?{q}", timeout=15).json()
                recs = {}
                for r in rows:
                    d, lg, sh = r.get(dcol), r.get(lcol), r.get(scol)
                    if d and lg is not None and sh is not None:
                        recs[pd.Timestamp(str(d)[:10])] = float(lg) - float(sh)
                if recs:
                    s = pd.Series(recs).sort_index()
                    return AuxResult(s, label, True, f"{len(s)} 筆，最新 {s.index[-1].date()}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{label}: {e}")
    return AuxResult(None, "無", False, "；".join(errors[-2:]))


def fetch_tips(years: list[int]) -> AuxResult:
    """美國財政部 5 年期 TIPS 實質殖利率（每年一個 XML）。"""
    base = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml"
    recs: dict[pd.Timestamp, float] = {}
    errors = []
    for y in years:
        try:
            xml = http_get(f"{base}?data=daily_treasury_real_yield_curve&field_tdr_date_value={y}", timeout=15).text
            for blk in re.findall(r"<m:properties>[\s\S]*?</m:properties>", xml):
                dm = re.search(r"<d:NEW_DATE[^>]*>(\d{4}-\d{2}-\d{2})", blk)
                vm = re.search(r"<d:BC_5YEAR[^>]*>([\d.\-]+)</d:BC_5YEAR>", blk)
                if dm and vm:
                    recs[pd.Timestamp(dm.group(1))] = float(vm.group(1))
        except Exception as e:  # noqa: BLE001
            errors.append(f"{y}: {e}")
    if recs:
        s = pd.Series(recs).sort_index()
        return AuxResult(s, "US Treasury", True, f"{len(s)} 筆")
    return AuxResult(None, "無", False, "；".join(errors[-2:]))


def fetch_fear_greed() -> AuxResult:
    try:
        r = http_get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata", timeout=12,
                     headers={"Accept": "application/json", "Referer": "https://edition.cnn.com/"})
        j = r.json()
        hist = (j.get("fear_and_greed_historical") or {}).get("data") or []
        if not hist:
            raise ValueError("無歷史資料")
        s = pd.Series({pd.Timestamp(int(p["x"]), unit="ms").normalize(): float(p["y"]) for p in hist}).sort_index()
        cur = (j.get("fear_and_greed") or {})
        return AuxResult(s, "CNN", True, f"{len(s)} 筆",
                         {"score": float(cur.get("score", s.iloc[-1])), "rating": cur.get("rating", "")})
    except Exception as e:  # noqa: BLE001
        return AuxResult(None, "無", False, str(e))


def fetch_stocktwits() -> AuxResult:
    bull = bear = 0
    errors = []
    for sym in ("SLV", "SI_F"):
        try:
            j = http_get(f"https://api.stocktwits.com/api/2/streams/symbol/{sym}.json", timeout=10).json()
            for m in j.get("messages") or []:
                s = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
                bull += s == "Bullish"
                bear += s == "Bearish"
        except Exception as e:  # noqa: BLE001
            errors.append(f"{sym}: {e}")
    tagged = bull + bear
    if tagged == 0:
        return AuxResult(None, "無", False, "；".join(errors) or "沒有標記情緒的訊息")
    pct = bull / tagged * 100
    return AuxResult(None, "StockTwits", True, f"{tagged} 則標記訊息",
                     {"score": pct, "bullish_pct": pct, "bearish_pct": 100 - pct, "tagged": tagged})


def fetch_google_trends(start: pd.Timestamp) -> AuxResult:
    t = f"{start:%Y-%m} {datetime.now():%Y-%m}"
    try:
        q = urlencode({"hl": "en-US", "tz": "-480", "req": json.dumps(
            {"comparisonItem": [{"keyword": "silver", "geo": "US", "time": t}], "category": 0, "property": ""})})
        txt = http_get(f"https://trends.google.com/trends/api/explore?{q}", timeout=12).text
        widgets = json.loads(txt[txt.index("{"):]).get("widgets", [])
        w = next(w for w in widgets if w.get("id") == "TIMESERIES")
        q2 = urlencode({"hl": "en-US", "tz": "-480", "token": w["token"], "req": json.dumps(w["request"])})
        txt2 = http_get(f"https://trends.google.com/trends/api/widgetdata/multiline?{q2}", timeout=12).text
        tl = json.loads(txt2[txt2.index("{"):])["default"]["timelineData"]
        s = pd.Series({pd.Timestamp(int(p["time"]), unit="s").normalize(): float((p.get("value") or [50])[0]) for p in tl}).sort_index()
        if s.empty:
            raise ValueError("空資料")
        return AuxResult(s, "Google Trends", True, f"{len(s)} 週")
    except Exception as e:  # noqa: BLE001
        return AuxResult(None, "無", False, str(e)[:160])

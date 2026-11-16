"""測試用的假網路：替換 sources 的網路入口，不連外。

只供測試與截圖使用，正式程式不會載入此檔。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from silver_trader import sources

BASE = {"SI=F": 30.0, "SLV": 27.5, "GC=F": 2400, "CL=F": 75, "DX-Y.NYB": 104, "^VIX": 16, "^TNX": 4.3, "HG=F": 4.4}


def make_history(sym: str, days: int = 1300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(sym)) % 2**32 + seed)
    idx = pd.bdate_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=days)
    steps = rng.normal(0.0003, 0.015, days) + 0.004 * np.sin(np.arange(days) / 9)
    close = BASE[sym] * np.exp(np.cumsum(steps))
    return pd.DataFrame({"close": close, "volume": rng.integers(1e6, 6e6, days).astype(float)}, index=idx)


class FakeResp:
    def __init__(self, text: str):
        self.text = text

    def json(self):
        return json.loads(self.text)


def install(monkeypatch=None, *, quote_ok=True, quote_age_h=0.2, history_ok=True, aux_ok=True, quote_price=None):
    hist = {s: make_history(s) for s in BASE}
    last = float(hist["SI=F"]["close"].iloc[-1])
    qp = quote_price if quote_price is not None else last * 1.003

    def yf_download(symbols, years):
        return {s: hist[s] for s in symbols if history_ok or s != "SI=F"}

    def yf_intraday(symbol):
        if not quote_ok:
            raise RuntimeError("network blocked")
        t = pd.Timestamp(datetime.now(timezone.utc) - timedelta(hours=quote_age_h))
        return pd.DataFrame({"Close": [qp]}, index=pd.DatetimeIndex([t]))

    def http_get(url, timeout=12.0, headers=None):
        if not aux_ok or "stooq" in url or "finance.yahoo" in url:
            raise RuntimeError("network blocked")
        if "publicreporting.cftc.gov" in url:
            d = pd.date_range(end=pd.Timestamp.now(), periods=150, freq="W-TUE")
            return FakeResp(json.dumps([{"as_of_date_form_yyyy_mm_dd": x.strftime("%Y-%m-%dT00:00:00"),
                                         "m_money_positions_long_all": str(40000 + i * 50),
                                         "m_money_positions_short_all": "20000"} for i, x in enumerate(d)]))
        if "treasury.gov" in url:
            y = url.split("=")[-1]
            blocks = "".join(f"<m:properties><d:NEW_DATE m:type=\"Edm.DateTime\">{y}-{m:02d}-15T00:00:00</d:NEW_DATE>"
                             f"<d:BC_5YEAR m:type=\"Edm.Double\">{1.5 + m / 20:.2f}</d:BC_5YEAR></m:properties>"
                             for m in range(1, 13))
            return FakeResp(blocks)
        if "dataviz.cnn.io" in url:
            d = pd.date_range(end=pd.Timestamp.now(), periods=400)
            return FakeResp(json.dumps({"fear_and_greed": {"score": 62.0, "rating": "greed"},
                                        "fear_and_greed_historical": {"data": [{"x": int(x.timestamp() * 1000), "y": 50 + 10 * np.sin(i / 20)} for i, x in enumerate(d)]}}))
        if "stocktwits" in url:
            return FakeResp(json.dumps({"messages": [{"entities": {"sentiment": {"basic": "Bullish" if i % 3 else "Bearish"}}} for i in range(30)]}))
        if "trends.google.com" in url:
            if "explore" in url:
                return FakeResp(")]}'\n" + json.dumps({"widgets": [{"id": "TIMESERIES", "token": "t", "request": {}}]}))
            d = pd.date_range(end=pd.Timestamp.now(), periods=260, freq="W")
            return FakeResp(")]}',\n" + json.dumps({"default": {"timelineData": [{"time": str(int(x.timestamp())), "value": [40 + i % 30]} for i, x in enumerate(d)]}}))
        raise RuntimeError(f"unexpected url {url}")

    targets = [(sources, "_yf_download", yf_download), (sources, "_yf_intraday", yf_intraday), (sources, "http_get", http_get)]
    for mod, name, fn in targets:
        if monkeypatch is not None:
            monkeypatch.setattr(mod, name, fn)
        else:
            setattr(mod, name, fn)
    return hist

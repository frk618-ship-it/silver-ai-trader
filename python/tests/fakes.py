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

    def http_get(url, timeout=12.0, headers=None, session=None):
        if not aux_ok or "stooq" in url or "finance.yahoo" in url:
            raise RuntimeError("network blocked")
        if "publicreporting.cftc.gov" in url:
            d = pd.date_range(end=pd.Timestamp.now(), periods=150, freq="W-TUE")
            return FakeResp(json.dumps([{"report_date_as_yyyy_mm_dd": x.strftime("%Y-%m-%dT00:00:00"),
                                         "m_money_positions_long_all": str(40000 + i * 50),
                                         "m_money_positions_short_all": "20000"} for i, x in enumerate(d)]))
        if "treasury.gov" in url:
            y = url.split("=")[-1]
            blocks = "".join(f"<m:properties><d:NEW_DATE m:type=\"Edm.DateTime\">{y}-{m:02d}-15T00:00:00</d:NEW_DATE>"
                             f"<d:TC_5YEAR m:type=\"Edm.Double\">{1.5 + m / 20:.2f}</d:TC_5YEAR></m:properties>"
                             for m in range(1, 13))
            return FakeResp(blocks)
        if "dataviz.cnn.io" in url:
            d = pd.date_range(end=pd.Timestamp.now(), periods=400)
            return FakeResp(json.dumps({"fear_and_greed": {"score": 62.0, "rating": "greed"},
                                        "fear_and_greed_historical": {"data": [{"x": int(x.timestamp() * 1000), "y": 50 + 10 * np.sin(i / 20)} for i, x in enumerate(d)]}}))
        if "stocktwits" in url:
            return FakeResp(json.dumps({"messages": [{"entities": {"sentiment": {"basic": "Bullish" if i % 3 else "Bearish"}}} for i in range(30)]}))
        if "trends.google.com" in url:
            if "/trends/?" in url:
                return FakeResp("")
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


# ══════════════════════════════════════════════════════════════
# 假 GitHub（Contents API）：給 storage 模組測試用
# ══════════════════════════════════════════════════════════════
import hashlib as _hashlib
import base64 as _b64
from urllib.parse import urlparse as _urlparse


class _R:
    def __init__(self, status, body=None, content=b""):
        self.status_code, self._body, self.content = status, body, content

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeGitHub:
    def __init__(self, repo="me/silver-ai-data", token="tok", private=True, push=True):
        self.repo, self.token, self.private, self.push = repo, token, private, push
        self.files: dict[str, tuple[bytes, str]] = {}
        self.commits: list[str] = []
        self.calls = 0

    def install(self, monkeypatch=None):
        from silver_trader import storage
        if monkeypatch is not None:
            monkeypatch.setattr(storage, "_http", self.handle)
        else:
            storage._http = self.handle
        return self

    def handle(self, method, url, headers=None, params=None, json=None, timeout=None):
        self.calls += 1
        if (headers or {}).get("Authorization") != f"Bearer {self.token}":
            return _R(401, {"message": "Bad credentials"})
        path = _urlparse(url).path
        base = f"/repos/{self.repo}"
        if path == base and method == "GET":
            return _R(200, {"private": self.private, "default_branch": "main", "permissions": {"push": self.push}})
        if not path.startswith(base + "/contents/"):
            return _R(404, {"message": "Not Found"})
        fp = path[len(base + "/contents/"):]
        cur = self.files.get(fp)
        if method == "GET":
            if not cur:
                return _R(404, {"message": "Not Found"})
            if "raw" in (headers or {}).get("Accept", ""):
                return _R(200, None, cur[0])
            return _R(200, {"sha": cur[1], "size": len(cur[0])})
        if method == "PUT":
            if not self.push:
                return _R(403, {"message": "Resource not accessible by personal access token"})
            if cur and json.get("sha") is None:
                return _R(422, {"message": '"sha" wasn\'t supplied.'})
            if cur and json.get("sha") != cur[1]:
                return _R(409, {"message": "is at ... but expected ..."})
            data = _b64.b64decode(json["content"])
            sha = _hashlib.sha1(data + str(len(self.commits)).encode()).hexdigest()
            self.files[fp] = (data, sha)
            self.commits.append(json["message"])
            return _R(201, {"content": {"sha": sha}})
        return _R(405, {"message": "method"})

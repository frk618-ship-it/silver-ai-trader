"""代理記憶：模型註冊表、執行紀錄、預測日誌（SQLite），以及冠軍/挑戰者決策。"""
from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, ts TEXT, type TEXT, algorithm TEXT, pred_target TEXT,
    da REAL, mape REAL, holdout_da REAL, promoted INTEGER, note TEXT, seconds REAL);
CREATE TABLE IF NOT EXISTS models (id INTEGER PRIMARY KEY, ts TEXT, algorithm TEXT, pred_target TEXT, lookback INTEGER,
    layers INTEGER, epochs INTEGER, holdout_da REAL, holdout_mape REAL, holdout_n INTEGER, in_sample_da REAL,
    in_sample_mape REAL, status TEXT, reason TEXT, params TEXT, factors TEXT);
CREATE TABLE IF NOT EXISTS predictions (id INTEGER PRIMARY KEY, made_at TEXT, basis_date TEXT, basis_price REAL,
    target_date TEXT, horizon INTEGER, pred_price REAL, pred_dir INTEGER, confidence REAL, algorithm TEXT,
    pred_target TEXT, resolved INTEGER DEFAULT 0, actual_price REAL, correct INTEGER, abs_pct_err REAL, resolved_at TEXT);
"""


def db_path():
    return config.home() / "agent.db"


@contextmanager
def connect():
    con = sqlite3.connect(db_path())
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean(v):
    return None if isinstance(v, float) and not math.isfinite(v) else v


# ── 決策（純函式）──
def should_promote(champion: dict | None, challenger: dict, da_margin: float = 1.0,
                   mape_margin: float = 0.0, min_holdout: int = 20) -> tuple[bool, str]:
    cda = challenger.get("holdout_da")
    if cda is None or not math.isfinite(cda):
        return False, "挑戰者沒有有效的保留集分數"
    if (challenger.get("holdout_n") or 0) < min_holdout:
        return False, f"保留集僅 {challenger.get('holdout_n') or 0} 筆，不足以判定（需 {min_holdout} 筆）"
    if not champion or champion.get("holdout_da") is None or not math.isfinite(champion["holdout_da"]):
        return True, "目前沒有冠軍模型，挑戰者直接上任"
    d = cda - champion["holdout_da"]
    if d > da_margin:
        return True, f"方向準確率 {cda:.2f}% 勝過冠軍 {champion['holdout_da']:.2f}%（+{d:.2f}）"
    if abs(d) <= da_margin:
        cm, gm = champion.get("holdout_mape"), challenger.get("holdout_mape")
        if cm is not None and gm is not None and math.isfinite(cm) and math.isfinite(gm) and cm - gm > mape_margin:
            return True, f"方向準確率相當，但誤差較低（{gm:.2f}% < {cm:.2f}%）"
    return False, f"未達晉升門檻（DA {cda:.2f}% vs 冠軍 {champion['holdout_da']:.2f}%）"


def should_retrain(perf: dict, min_n: int = 10, da_floor: float = 50.0) -> tuple[bool, str]:
    if not perf or perf.get("n", 0) < min_n:
        return False, f"實測樣本僅 {perf.get('n', 0) if perf else 0} 筆，不足以判斷（需 {min_n} 筆）"
    if perf["da"] < da_floor:
        return True, f"近 {perf['n']} 次方向準確率 {perf['da']:.1f}%，低於 {da_floor:.0f}% 門檻，建議重新訓練"
    return False, f"近 {perf['n']} 次方向準確率 {perf['da']:.1f}%，維持現任冠軍"


def next_trading_days(last: date, n: int) -> list[date]:
    out, d = [], last
    while len(out) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


# ── 寫入 ──
def add_run(**kw) -> None:
    with connect() as c:
        c.execute("INSERT INTO runs (ts,type,algorithm,pred_target,da,mape,holdout_da,promoted,note,seconds) "
                  "VALUES (?,?,?,?,?,?,?,?,?,?)",
                  (_now(), kw.get("type"), kw.get("algorithm"), kw.get("pred_target"), _clean(kw.get("da")),
                   _clean(kw.get("mape")), _clean(kw.get("holdout_da")), int(bool(kw.get("promoted"))),
                   kw.get("note"), kw.get("seconds")))


def champion() -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM models WHERE status='champion' ORDER BY id DESC LIMIT 1").fetchone()
        return dict(r) if r else None


def register_model(rec: dict, promote: bool) -> None:
    with connect() as c:
        if promote:
            c.execute("UPDATE models SET status='retired' WHERE status='champion'")
        c.execute("INSERT INTO models (ts,algorithm,pred_target,lookback,layers,epochs,holdout_da,holdout_mape,holdout_n,"
                  "in_sample_da,in_sample_mape,status,reason,params,factors) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (_now(), rec["algorithm"], rec["pred_target"], rec["lookback"], rec["layers"], rec["epochs"],
                   _clean(rec.get("holdout_da")), _clean(rec.get("holdout_mape")), rec.get("holdout_n"),
                   _clean(rec.get("in_sample_da")), _clean(rec.get("in_sample_mape")),
                   "champion" if promote else "challenger", rec.get("reason"),
                   json.dumps(rec.get("params", {})), json.dumps(rec.get("factors", {}))))


def log_predictions(basis_date: date, basis_price: float, prices, probs, algorithm: str, target: str) -> None:
    days = next_trading_days(basis_date, len(prices))
    with connect() as c:
        for i, (d, p) in enumerate(zip(days, prices)):
            conf = abs(float(probs[i]) - 0.5) * 200 if probs is not None and i < len(probs) else None
            c.execute("INSERT INTO predictions (made_at,basis_date,basis_price,target_date,horizon,pred_price,pred_dir,"
                      "confidence,algorithm,pred_target) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (_now(), basis_date.isoformat(), basis_price, d.isoformat(), i + 1, float(p),
                       int(p >= basis_price), conf, algorithm, target))


def resolve_predictions(price_by_date: dict[str, float]) -> int:
    """把目標日已有收盤價的預測拿來對帳。回傳本次結算筆數。"""
    n = 0
    with connect() as c:
        for r in c.execute("SELECT * FROM predictions WHERE resolved=0").fetchall():
            a = price_by_date.get(r["target_date"])
            if a is None:
                continue
            correct = int((a >= r["basis_price"]) == (r["pred_price"] >= r["basis_price"]))
            err = abs(a - r["pred_price"]) / a * 100 if a else None
            c.execute("UPDATE predictions SET resolved=1, actual_price=?, correct=?, abs_pct_err=?, resolved_at=? WHERE id=?",
                      (a, correct, err, _now(), r["id"]))
            n += 1
    return n


def recent_performance(window: int = 20) -> dict:
    with connect() as c:
        rows = c.execute("SELECT correct, abs_pct_err FROM predictions WHERE resolved=1 AND correct IS NOT NULL "
                         "ORDER BY target_date DESC, horizon ASC LIMIT ?", (window,)).fetchall()
    if not rows:
        return {"n": 0, "da": float("nan"), "mape": float("nan")}
    errs = [r["abs_pct_err"] for r in rows if r["abs_pct_err"] is not None]
    return {"n": len(rows), "da": sum(r["correct"] for r in rows) / len(rows) * 100,
            "mape": sum(errs) / len(errs) if errs else float("nan")}


def table(name: str, limit: int = 200) -> pd.DataFrame:
    with connect() as c:
        return pd.read_sql_query(f"SELECT * FROM {name} ORDER BY id DESC LIMIT ?", c, params=(limit,))


def clear_all() -> None:
    with connect() as c:
        c.executescript("DELETE FROM runs; DELETE FROM models; DELETE FROM predictions;")

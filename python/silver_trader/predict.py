"""推論、指標、多步預測與因子敏感度。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dataset import Dataset
from .models import SilverModel


def mape(actual, pred) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    n = min(len(a), len(p))
    a, p = a[:n], p[:n]
    m = (a != 0) & np.isfinite(a) & np.isfinite(p)
    return float(np.mean(np.abs((a[m] - p[m]) / a[m])) * 100) if m.any() else float("nan")


def direction_accuracy(actual, prev, pred_prices, probs, target: str) -> float:
    """方向準確率：direction/dual 用機率判方向，其餘用預測價格相對前一日。"""
    a, pv = np.asarray(actual, float), np.asarray(prev, float)
    if target in ("direction", "dual") and probs is not None and len(probs):
        up_pred = np.asarray(probs) > 0.5
        up_act = a > pv
    else:
        pp = np.asarray(pred_prices, float)
        up_pred = (pp - pv) >= 0
        up_act = (a - pv) >= 0
    n = min(len(up_pred), len(up_act))
    return float(np.mean(up_pred[:n] == up_act[:n]) * 100) if n else float("nan")


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def _multi_stats(model: SilverModel) -> tuple[np.ndarray, np.ndarray]:
    H = model.horizon
    if model.multi_stats is None:
        return np.zeros(H), np.ones(H)
    return np.asarray(model.multi_stats[0], dtype=float), np.asarray(model.multi_stats[1], dtype=float)


def _live_window(ds: Dataset) -> np.ndarray:
    """以最新一天結尾的時間窗（舊資料集沒有 X_live 時退回 X[-1]）。"""
    return ds.X_live if ds.X_live is not None else ds.X[-1]


def _last_price(ds: Dataset) -> float:
    return float(ds.prices[-1]) if ds.prices is not None else float(ds.denorm_price(ds.y[-1]))


def model_prices(model: SilverModel, ds: Dataset) -> tuple[np.ndarray, np.ndarray | None]:
    """對資料集每個樣本推論，回傳 (預測價格, 方向機率或 None)。"""
    out = model.predict(ds.X)
    t = model.pred_target
    if t == "multi":
        mu, sd = _multi_stats(model)
        return ds.prev_prices * np.exp(out[:, 0] * sd[0] + mu[0]), None     # 第 1 欄 = T+1
    if t == "logreturn":
        r = out[:, 0] * ds.logret_std + ds.logret_mean
        return ds.prev_prices * np.exp(r), None
    if t == "direction":
        probs = out[:, 0]
        # 只有方向：用機率合成一個相對前一日的價格，供圖表與方向判定
        return ds.prev_prices * np.where(probs > 0.5, 1.0001, 0.9999), probs
    if t == "dual":
        return ds.denorm_price(out[:, 0]), _sigmoid(out[:, 1])
    return ds.denorm_price(out[:, 0]), None


@dataclass
class Forecast:
    prices: np.ndarray
    dir_probs: np.ndarray | None


def forecast(model: SilverModel, ds: Dataset, days: int) -> Forecast:
    """未來 N 個交易日。

    · 多日報酬率（multi）：一次直接輸出 T+1…T+H 的累積報酬率，不遞迴。
    · 其他目標：迭代式，每一步把預測價格寫回最後一列的價格特徵再往前滑動
      （其他因子停在最新一天的值，所以越往後越不可靠）。
    兩者都從「以最新一天結尾」的時間窗出發。
    """
    win = _live_window(ds).copy()
    t = model.pred_target
    last_price = _last_price(ds)
    if t == "multi":
        mu, sd = _multi_stats(model)
        r = model.predict(win[None])[0] * sd + mu
        k = min(int(days), model.horizon)
        return Forecast(last_price * np.exp(r[:k]), None)
    last_scaled = float(win[-1, 0])
    prices, probs = [], []
    for _ in range(days):
        o = model.predict(win[None])[0]
        if t == "logreturn":
            base = prices[-1] if prices else last_price
            np_ = base * float(np.exp(o[0] * ds.logret_std + ds.logret_mean))
            last_scaled = float(ds.norm_price(np_))
        elif t == "direction":
            prob = float(o[0])
            probs.append(prob)
            last_scaled = float(np.clip(last_scaled * (1 + (prob - 0.5) * 0.02), 0, 1))
            np_ = float(ds.denorm_price(last_scaled))
        elif t == "dual":
            last_scaled = float(o[0])
            probs.append(float(_sigmoid(o[1])))
            np_ = float(ds.denorm_price(last_scaled))
        else:
            last_scaled = float(o[0])
            np_ = float(ds.denorm_price(last_scaled))
        prices.append(np_)
        nxt = win[-1].copy()
        nxt[0] = last_scaled
        win = np.concatenate([win[1:], nxt[None]], axis=0)
    return Forecast(np.array(prices), np.array(probs) if probs else None)


def factor_influence(model: SilverModel, ds: Dataset, delta: float = 0.05) -> dict[str, float]:
    """中央差分敏感度：把最後一天的某個因子 ±delta，看 T+1 預測價格變化（美元）。"""
    from .config import FACTOR_NAME

    base = _live_window(ds)
    batch = []
    for j in range(1, ds.n_features):
        for sgn in (1, -1):
            w = base.copy()
            w[-1, j] = np.clip(w[-1, j] + sgn * delta, 0, 1)
            batch.append(w)
    if not batch:
        return {}
    out = model.predict(np.stack(batch))
    t = model.pred_target
    if t == "multi":      # 對最後一天（T+H）預測價格的影響
        mu, sd = _multi_stats(model)
        vals = _last_price(ds) * np.exp(out[:, -1] * sd[-1] + mu[-1])
    elif t == "logreturn":
        vals = _last_price(ds) * np.exp(out[:, 0] * ds.logret_std + ds.logret_mean)
    elif t == "direction":
        vals = out[:, 0] * 100        # 機率百分點
    else:
        vals = ds.denorm_price(out[:, 0])
    res = {}
    for k, j in enumerate(range(1, ds.n_features)):
        res[FACTOR_NAME.get(ds.feature_keys[j], ds.feature_keys[j])] = float((vals[2 * k] - vals[2 * k + 1]) / 2)
    return res


def score_holdout(model: SilverModel, ds: Dataset, start: int) -> dict:
    """保留集評分（挑戰者與冠軍用同一段資料比較）。"""
    if start >= len(ds.X):
        return {"holdout_da": float("nan"), "holdout_mape": float("nan"), "holdout_n": 0}
    sub = ds.subset(np.arange(start, len(ds.X)))
    preds, probs = model_prices(model, sub)
    actual = ds.denorm_price(sub.y)
    out = {"holdout_da": direction_accuracy(actual, sub.prev_prices, preds, probs, model.pred_target),
           "holdout_mape": mape(actual, preds) if model.pred_target != "direction" else float("nan"),
           "holdout_n": int(len(sub.X))}
    if model.pred_target == "multi" and model.horizon > 1:
        # T+H 的方向命中率（只算 T+H 已經有實際價格的樣本）
        H = model.horizon
        raw, _, _ = sub.multi_targets(H, stats=(np.zeros(H), np.ones(H)))
        ok = ~np.isnan(raw[:, -1])
        if ok.any():
            mu, sd = _multi_stats(model)
            pred_r = model.predict(sub.X[ok])[:, -1] * sd[-1] + mu[-1]
            out["holdout_da_h"] = float(np.mean((pred_r >= 0) == (raw[ok, -1] >= 0)) * 100)
            out["holdout_n_h"] = int(ok.sum())
            out["horizon"] = H
    return out

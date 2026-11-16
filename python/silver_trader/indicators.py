"""技術指標（與 HTML 版逐一對應，結果一致）。"""
from __future__ import annotations

import numpy as np


def rsi(prices: np.ndarray, period: int = 5) -> np.ndarray:
    """Wilder RSI；前 period 筆填 50（與 HTML 版 calculateRSI 相同）。"""
    p = np.asarray(prices, dtype=float)
    n = len(p)
    out = np.full(n, 50.0)
    if n <= period:
        return out
    diff = np.diff(p)
    gains = np.clip(diff, 0, None)
    losses = np.clip(-diff, 0, None)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()
    out[period] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        g, l = gains[i - 1], losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
        out[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return out


def roc(values: np.ndarray, is_difference: bool = False) -> np.ndarray:
    """變動率；第 0 筆為 0；前值為 0 時退回差值。"""
    v = np.asarray(values, dtype=float)
    out = np.zeros(len(v))
    if len(v) < 2:
        return out
    prev, curr = v[:-1], v[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = (curr - prev) / prev
    diff = curr - prev
    use_diff = is_difference | (prev == 0)
    res = np.where(use_diff, diff, pct)
    res[~np.isfinite(res)] = 0.0
    out[1:] = res
    return out


def sma(prices: np.ndarray, period: int) -> np.ndarray:
    """簡單移動平均；資料不足時用現有資料的平均。"""
    p = np.asarray(prices, dtype=float)
    c = np.cumsum(np.insert(p, 0, 0.0))
    idx = np.arange(1, len(p) + 1)
    start = np.maximum(idx - period, 0)
    return (c[idx] - c[start]) / (idx - start)


def ema(prices: np.ndarray, period: int) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    if len(p) == 0:
        return p
    k = 2 / (period + 1)
    out = np.empty(len(p))
    out[0] = p[0]
    for i in range(1, len(p)):
        out[i] = p[i] * k + out[i - 1] * (1 - k)
    return out

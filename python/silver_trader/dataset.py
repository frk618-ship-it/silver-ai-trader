"""建立模型用的時間窗資料集（對應 HTML 版 createMultivariateDataset）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class Scaler:
    min: float
    max: float

    def scale(self, v):
        r = self.max - self.min
        return (np.asarray(v, dtype=float) - self.min) / r if r else np.zeros_like(np.asarray(v, dtype=float))

    def unscale(self, v):
        return np.asarray(v, dtype=float) * (self.max - self.min) + self.min


@dataclass
class Dataset:
    X: np.ndarray            # [N, lookback, F] float32
    y: np.ndarray            # [N] 正規化價格
    y_dir: np.ndarray        # [N] 1=漲 0=跌（相對前一日）
    y_logret: np.ndarray     # [N] 標準化後的 ln(P_t/P_{t-1})
    prev_prices: np.ndarray  # [N] 前一日實際價格
    target_idx: np.ndarray   # [N] 每個樣本的目標在原始序列中的位置
    feature_keys: list[str]
    scalers: dict[str, Scaler]
    logret_mean: float
    logret_std: float
    dates: pd.DatetimeIndex  # 原始序列日期
    X_live: np.ndarray | None = None   # [lookback, F] 以「最新一天」結尾的時間窗：預測未來用
    prices: np.ndarray | None = None   # 原始價格序列（多日目標用）

    @property
    def price_scaler(self) -> Scaler:
        return self.scalers["price"]

    @property
    def n_features(self) -> int:
        return len(self.feature_keys)

    def denorm_price(self, v):
        return self.price_scaler.unscale(v)

    def norm_price(self, v):
        return np.clip(self.price_scaler.scale(v), 0, 1)

    def multi_targets(self, horizon: int, stats: tuple | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """直接多日目標：每個樣本從「前一日收盤」到 T+1…T+H 的累積對數報酬率（各欄標準化）。

        回傳 (Y [N, H]，未來資料不足的列為 NaN；各欄平均；各欄標準差)。
        stats 可傳入訓練時的 (mean, std)，預測時才能用同一個尺度還原。
        """
        H = int(horizon)
        r = np.full((len(self.target_idx), H), np.nan)
        T = len(self.prices)
        for i, t in enumerate(self.target_idx):
            end = min(T, t + H)
            if end > t:
                r[i, :end - t] = np.log(self.prices[t:end] / self.prices[t - 1])
        if stats is None:
            mu = np.nanmean(r, axis=0)
            sd = np.nanstd(r, axis=0)
            sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
        else:
            mu, sd = (np.asarray(v, dtype=float) for v in stats)
        return ((r - mu) / sd).astype(np.float32), mu, sd

    def target_array(self, pred_target: str, horizon: int = 1, stats: tuple | None = None) -> np.ndarray:
        if pred_target == "multi":
            return self.multi_targets(horizon, stats)[0]
        if pred_target == "logreturn":
            return self.y_logret[:, None]
        if pred_target == "direction":
            return self.y_dir[:, None].astype(np.float32)
        if pred_target == "dual":
            return np.stack([self.y, self.y_dir], axis=1).astype(np.float32)
        return self.y[:, None]

    def subset(self, idx) -> "Dataset":
        return Dataset(self.X[idx], self.y[idx], self.y_dir[idx], self.y_logret[idx], self.prev_prices[idx],
                       self.target_idx[idx], self.feature_keys, self.scalers, self.logret_mean,
                       self.logret_std, self.dates, self.X_live, self.prices)


def active_keys(selected: dict[str, bool]) -> list[str]:
    return ["price"] + [k for k in config.OPTIONAL_FACTOR_KEYS if selected.get(k, False)]


def build_dataset(frame: pd.DataFrame, lookback: int, selected: dict[str, bool],
                  max_samples: int | None = None) -> Dataset:
    keys = active_keys(selected)
    scalers = {}
    cols = []
    for k in keys:
        v = frame[k].to_numpy(dtype=float)
        s = Scaler(float(np.min(v)), float(np.max(v)))
        scalers[k] = s
        cols.append(s.scale(v))
    M = np.stack(cols, axis=1).astype(np.float32)          # [T, F]
    prices = frame["price"].to_numpy(dtype=float)
    T = len(prices)
    if T <= lookback:
        raise ValueError(f"資料只有 {T} 筆，不足以建立 lookback={lookback} 的時間窗")
    start = max(lookback, T - max_samples) if max_samples else lookback
    tidx = np.arange(start, T)
    # 滑動視窗（零拷貝 view 再複製成連續陣列）
    win = np.lib.stride_tricks.sliding_window_view(M, (lookback, M.shape[1]))[:, 0]   # [T-lookback+1, L, F]
    X = np.ascontiguousarray(win[tidx - lookback])
    p_now, p_prev = prices[tidx], prices[tidx - 1]
    y = scalers["price"].scale(p_now).astype(np.float32)
    y_dir = (p_now > p_prev).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.where((p_prev > 0) & (p_now > 0), np.log(p_now / p_prev), 0.0)
    mu = float(lr.mean()) if len(lr) else 0.0
    sd = float(lr.std()) or 1.0
    # 預測未來時要用「以最新一天結尾」的時間窗；X[-1] 結尾在倒數第二天（它的目標是最新一天）
    X_live = np.ascontiguousarray(M[T - lookback:T])
    return Dataset(X, y, y_dir, ((lr - mu) / sd).astype(np.float32), p_prev.astype(float), tidx, keys,
                   scalers, mu, sd, frame.index, X_live, prices.astype(float))


def split_holdout(n: int, frac: float = 0.15, min_hold: int = 20) -> tuple[int, int]:
    """時序保留集：永遠取最後一段、不打亂。回傳 (train_end, hold_len)。"""
    if n <= 0:
        return 0, 0
    hold = int(n * frac)
    if hold < min_hold:
        hold = min(min_hold, n // 4)
    if hold < 1 or n - hold < 30:
        return n, 0
    return n - hold, hold

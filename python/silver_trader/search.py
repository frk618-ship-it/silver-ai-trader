"""AI 超參數搜尋（隨機 / 預設網格 / 簡化貝葉斯）。

與 HTML 版的差別：HTML 版用「訓練資料本身」算 DA 來排名，容易挑到過度擬合的組合；
這裡每個試驗都切出最後 20% 當驗證段，排名依驗證段的 DA / loss。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import config
from .config import SearchConfig, TrainParams
from .dataset import build_dataset
from .predict import direction_accuracy, model_prices
from .training import compute_loss, new_model, train_model

SPACE = {
    "algorithm": ["BiGRU", "LSTM", "Transformer", "TimesFM"],
    "lookback": [5, 10, 15, 20, 30],
    "layers": [16, 32, 48, 64, 96],
    "batch_size": [16, 32, 64],
    "directional_lambda": [0.0, 0.1, 0.2, 0.3, 0.5],
    "huber_delta": [0.5, 1.0, 1.5],
    "lr": [0.0005, 0.001, 0.002, 0.003],
}
GRID = [
    dict(algorithm="BiGRU", lookback=10, layers=32, directional_lambda=0.3, batch_size=32, lr=0.001, huber_delta=1.0),
    dict(algorithm="BiGRU", lookback=20, layers=64, directional_lambda=0.3, batch_size=32, lr=0.001, huber_delta=1.0),
    dict(algorithm="BiGRU", lookback=10, layers=32, directional_lambda=0.1, batch_size=32, lr=0.001, huber_delta=1.0),
    dict(algorithm="BiGRU", lookback=30, layers=64, directional_lambda=0.5, batch_size=16, lr=0.001, huber_delta=1.0),
    dict(algorithm="LSTM", lookback=10, layers=32, directional_lambda=0.3, batch_size=32, lr=0.001, huber_delta=1.0),
    dict(algorithm="LSTM", lookback=20, layers=64, directional_lambda=0.3, batch_size=32, lr=0.001, huber_delta=1.0),
    dict(algorithm="LSTM", lookback=10, layers=48, directional_lambda=0.2, batch_size=32, lr=0.002, huber_delta=1.5),
    dict(algorithm="Transformer", lookback=10, layers=64, directional_lambda=0.3, batch_size=32, lr=0.001, huber_delta=1.0),
    dict(algorithm="Transformer", lookback=20, layers=96, directional_lambda=0.2, batch_size=32, lr=0.001, huber_delta=1.0),
    dict(algorithm="TimesFM", lookback=20, layers=64, directional_lambda=0.3, batch_size=32, lr=0.001, huber_delta=1.0),
]


@dataclass
class Trial:
    idx: int
    params: TrainParams
    factors: dict[str, bool]
    loss: float
    da: float
    seconds: float
    epochs: int

    def row(self) -> dict:
        p = self.params
        return {"#": self.idx, "演算法": p.algorithm, "Lookback": p.lookback, "單元": p.layers, "Batch": p.batch_size,
                "α": p.directional_lambda, "LR": p.lr, "因子數": sum(self.factors.values()) + 1,
                "驗證 DA%": round(self.da, 2), "驗證 Loss": round(self.loss, 5), "Epochs": self.epochs,
                "秒": round(self.seconds, 1)}


def _generate(i: int, cfg: SearchConfig, base: TrainParams, base_f: dict, prev: list[Trial], rng: random.Random):
    keys = config.OPTIONAL_FACTOR_KEYS
    if cfg.strategy == "grid":
        return TrainParams.from_dict({**base.to_dict(), **GRID[i % len(GRID)]}), dict(base_f)
    if cfg.strategy == "bayesian" and len(prev) >= 4:
        best = sorted(prev, key=lambda t: -t.da if cfg.metric == "da" else t.loss)[0]
        tp = {k: (getattr(best.params, k) if rng.random() < 0.6 else rng.choice(v)) for k, v in SPACE.items()}
        f = {k: (best.factors.get(k, True) if rng.random() < 0.7 else rng.random() > 0.3) for k in keys}
        return TrainParams.from_dict({**base.to_dict(), **tp}), f
    tp = {k: rng.choice(v) for k, v in SPACE.items()}
    return TrainParams.from_dict({**base.to_dict(), **tp}), {k: rng.random() > 0.3 for k in keys}


def run_trial(frame: pd.DataFrame, p: TrainParams, factors: dict, cfg: SearchConfig, idx: int,
              best_loss_so_far: float = float("inf")) -> Trial | None:
    t0 = time.perf_counter()
    ds = build_dataset(frame, p.lookback, factors, max_samples=cfg.max_samples)
    n = len(ds.X)
    if n < 40:
        return None
    cut = int(n * 0.8)
    m = new_model(p, ds)

    def prune(ep, loss, res):
        if ep >= 10 and best_loss_so_far < float("inf") and loss > best_loss_so_far * 4:
            raise _Pruned()

    try:
        res = train_model(m, ds, p, np.arange(cut), epochs=cfg.search_epochs, patience=cfg.patience, on_epoch=prune)
        epochs = res.epochs_run
    except _Pruned:
        epochs = 10
    val = ds.subset(np.arange(cut, n))
    preds, probs = model_prices(m, val)
    actual = ds.denorm_price(val.y)
    da = direction_accuracy(actual, val.prev_prices, preds, probs, p.pred_target)
    import torch

    with torch.no_grad():
        X = torch.as_tensor(val.X)
        Y = torch.as_tensor(val.target_array(p.pred_target), dtype=torch.float32)
        prev = torch.as_tensor(val.norm_price(val.prev_prices), dtype=torch.float32)[:, None]
        vloss = float(compute_loss(m(X), Y, prev, p.pred_target, p))
    return Trial(idx, p, factors, vloss, da, time.perf_counter() - t0, epochs)


class _Pruned(Exception):
    pass


def run_search(frame: pd.DataFrame, base: TrainParams, base_factors: dict, cfg: SearchConfig,
               on_trial: Callable[[int, Trial | None, Trial | None], None] | None = None,
               should_stop: Callable[[], bool] | None = None) -> list[Trial]:
    rng = random.Random(cfg.seed)
    trials: list[Trial] = []
    best: Trial | None = None
    for i in range(cfg.max_trials):
        if should_stop and should_stop():
            break
        p, f = _generate(i, cfg, base, base_factors, trials, rng)
        p.seed = None if cfg.seed is None else cfg.seed + i
        try:
            t = run_trial(frame, p, f, cfg, i + 1, min((x.loss for x in trials), default=float("inf")))
        except Exception:  # noqa: BLE001
            t = None
        if t:
            trials.append(t)
            if best is None or (t.da > best.da if cfg.metric == "da" else t.loss < best.loss):
                best = t
        if on_trial:
            on_trial(i, t, best)
    return trials


def best_trial(trials: list[Trial], metric: str) -> Trial | None:
    if not trials:
        return None
    return max(trials, key=lambda t: t.da) if metric == "da" else min(trials, key=lambda t: t.loss)

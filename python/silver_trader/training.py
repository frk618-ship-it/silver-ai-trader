"""損失函數與訓練迴圈。"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from .config import TrainParams
from .dataset import Dataset
from .models import SilverModel, build_model


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ══════════════════════════════════════════════════════════════
# 損失函數
# ══════════════════════════════════════════════════════════════
def directional_huber(pred, y, prev, alpha: float, delta: float):
    """(1-α)·Huber + α·方向懲罰。

    與 HTML 版的差別：HTML 版拿「同一個 batch 裡相鄰的樣本」當前一日，
    但訓練時 batch 會被打亂，相鄰樣本其實不是前一天。這裡改用每個樣本
    真正的前一日價格（正規化後），方向懲罰才名副其實。
    """
    hub = F.huber_loss(pred, y, delta=delta)
    if alpha <= 0:
        return hub
    pen = torch.relu(-(y - prev) * (pred - prev)).mean()
    return (1 - alpha) * hub + alpha * pen


def compute_loss(out, y, prev, target: str, p: TrainParams):
    if target == "direction":
        return F.binary_cross_entropy_with_logits(out, y)
    if target == "dual":
        price = F.huber_loss(out[:, :1], y[:, :1], delta=p.huber_delta)
        bce = F.binary_cross_entropy_with_logits(out[:, 1:], y[:, 1:])
        return price + p.directional_lambda * bce
    if target == "logreturn":
        return F.huber_loss(out, y, delta=p.huber_delta)
    return directional_huber(out, y, prev, p.directional_lambda, p.huber_delta)


# ══════════════════════════════════════════════════════════════
# 訓練
# ══════════════════════════════════════════════════════════════
@dataclass
class TrainResult:
    loss_history: list[float] = field(default_factory=list)
    best_loss: float = float("inf")
    best_epoch: int = -1
    epochs_run: int = 0
    stopped_early: bool = False
    stopped_by_user: bool = False
    seconds: float = 0.0

    @property
    def epochs_per_sec(self) -> float:
        return self.epochs_run / self.seconds if self.seconds > 0 else 0.0


EpochCallback = Callable[[int, float, "TrainResult"], None]


def train_model(model: SilverModel, ds: Dataset, p: TrainParams, idx: np.ndarray | None = None,
                epochs: int | None = None, on_epoch: EpochCallback | None = None,
                should_stop: Callable[[], bool] | None = None, patience: int | None = None,
                restore_best: bool = True) -> TrainResult:
    """以 Adam 訓練；記錄每 epoch 的 loss，結束時套回最佳權重。"""
    idx = np.arange(len(ds.X)) if idx is None else np.asarray(idx)
    epochs = p.epochs if epochs is None else epochs
    patience = p.early_stop_patience if patience is None else patience
    target = model.pred_target
    X = torch.as_tensor(ds.X[idx])
    Y = torch.as_tensor(ds.target_array(target)[idx], dtype=torch.float32)
    prev = torch.as_tensor(ds.norm_price(ds.prev_prices[idx]), dtype=torch.float32)[:, None]
    n = len(X)
    bs = max(1, min(p.batch_size, n))
    opt = torch.optim.Adam(model.parameters(), lr=p.lr)
    res = TrainResult()
    best_state = None
    since = 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        if should_stop and should_stop():
            res.stopped_by_user = True
            break
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = compute_loss(model(X[b]), Y[b], prev[b], target, p)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        ep_loss = tot / n
        res.loss_history.append(ep_loss)
        res.epochs_run = ep + 1
        if ep_loss < res.best_loss - 1e-9:
            res.best_loss, res.best_epoch = ep_loss, ep
            best_state = copy.deepcopy(model.state_dict())
            since = 0
        else:
            since += 1
        res.seconds = time.perf_counter() - t0
        if on_epoch:
            on_epoch(ep, ep_loss, res)
        if patience and since >= patience:
            res.stopped_early = True
            break
    res.seconds = time.perf_counter() - t0
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return res


def new_model(p: TrainParams, ds: Dataset) -> SilverModel:
    set_seed(p.seed)
    return build_model(p.algorithm, p.lookback, ds.n_features, p.layers, p.pred_target)


def walk_forward_cv(ds: Dataset, p: TrainParams, on_fold: Callable[[int, int, str], None] | None = None,
                    should_stop: Callable[[], bool] | None = None) -> dict:
    """時序前推交叉驗證：第 i 折用「之前所有資料」訓練、下一段測試。

    與 HTML 版的差別：HTML 版的 K-Fold 會拿未來資料訓練、去測過去，
    在時間序列上會高估表現；這裡改為只往前推。
    """
    from .predict import direction_accuracy, mape, model_prices

    k = max(2, p.k_folds)
    n = len(ds.X)
    edges = np.linspace(0, n, k + 2).astype(int)
    folds = []
    for i in range(k):
        tr = np.arange(0, edges[i + 1])
        te = np.arange(edges[i + 1], edges[i + 2])
        if len(tr) < 30 or len(te) < 5:
            continue
        if should_stop and should_stop():
            break
        if on_fold:
            on_fold(i, k, f"第 {i + 1}/{k} 折：訓練 {len(tr)} 筆、測試 {len(te)} 筆")
        m = new_model(p, ds)
        train_model(m, ds, p, tr, should_stop=should_stop)
        sub = ds.subset(te)
        preds, probs = model_prices(m, sub)
        actual = ds.denorm_price(sub.y)
        folds.append({"fold": i + 1, "da": direction_accuracy(actual, sub.prev_prices, preds, probs, p.pred_target),
                      "mape": mape(actual, preds) if p.pred_target != "direction" else float("nan"),
                      "n_train": len(tr), "n_test": len(te)})
    if not folds:
        return {"folds": [], "da": float("nan"), "mape": float("nan")}
    return {"folds": folds, "da": float(np.nanmean([f["da"] for f in folds])),
            "mape": float(np.nanmean([f["mape"] for f in folds]))}

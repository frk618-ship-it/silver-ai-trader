"""回測：用模型對測試區間逐筆做 1-step 預測並模擬交易（對應 HTML 版 runBacktest）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .dataset import Dataset
from .models import SilverModel
from .predict import model_prices


@dataclass
class BacktestResult:
    equity: pd.Series
    buy_hold: pd.Series
    trades: pd.DataFrame
    total_return: float
    buy_hold_return: float
    max_drawdown: float
    sharpe: float
    trade_count: int
    win_rate: float
    final_equity: float


def max_drawdown(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0, (peak - eq) / peak, 0)
    return float(dd.max()) if len(dd) else 0.0


def sharpe(rets: np.ndarray, periods: int = 252) -> float:
    if len(rets) < 2:
        return 0.0
    sd = rets.std(ddof=1)
    return float(rets.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def run_backtest(model: SilverModel, ds: Dataset, cfg: BacktestConfig) -> BacktestResult:
    ratio = min(max(cfg.test_ratio, 0.05), 0.8)
    split = int(len(ds.X) * (1 - ratio))
    sub = ds.subset(np.arange(split, len(ds.X)))
    if len(sub.X) < 5:
        raise ValueError("測試區間太短，請提高測試比例或縮短 lookback")
    pred, probs = model_prices(model, sub)
    entry = sub.prev_prices
    exit_ = ds.denorm_price(sub.y)
    if probs is not None and model.pred_target == "direction":
        pred_ret = (probs - 0.5) * 0.02     # 方向模型：用機率強度當訊號
    else:
        pred_ret = (pred - entry) / entry
    cost = (cfg.fee_bps + cfg.slippage_bps) / 10_000
    eq = cfg.initial_capital
    curve, rets, trades = [], [], []
    dates = ds.dates[sub.target_idx]
    for i in range(len(entry)):
        d = 1 if pred_ret[i] > cfg.threshold else (-1 if cfg.allow_short and pred_ret[i] < -cfg.threshold else 0)
        qty = (eq * cfg.position_pct * cfg.leverage / entry[i]) * d if d else 0.0
        pnl = qty * (exit_[i] - entry[i]) - abs(qty) * (entry[i] + exit_[i]) * cost
        prev = eq
        eq += pnl
        curve.append(eq)
        rets.append(eq / prev - 1 if prev > 0 else 0)
        if d:
            trades.append({"日期": dates[i].date(), "方向": "做多" if d > 0 else "放空", "進場": entry[i],
                           "出場": exit_[i], "預測價": pred[i], "預測報酬%": pred_ret[i] * 100, "損益": pnl, "權益": eq})
    curve = np.array(curve)
    wins = sum(t["損益"] > 0 for t in trades)
    bh = cfg.initial_capital * exit_ / entry[0]
    return BacktestResult(
        equity=pd.Series(curve, index=dates), buy_hold=pd.Series(bh, index=dates), trades=pd.DataFrame(trades),
        total_return=float(eq / cfg.initial_capital - 1), buy_hold_return=float(exit_[-1] / entry[0] - 1),
        max_drawdown=max_drawdown(curve), sharpe=sharpe(np.array(rets)), trade_count=len(trades),
        win_rate=wins / len(trades) if trades else 0.0, final_equity=float(eq))

"""一次完整的「訓練 → 預測 → 保留集評分 → 冠軍裁決 → 寫入紀錄」流程。"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import agent, config
from .config import TrainParams
from .dataset import Dataset, build_dataset, split_holdout
from .market import MarketData
from .models import SilverModel, load_model, save_model
from .predict import direction_accuracy, factor_influence, forecast, mape, model_prices, score_holdout
from .training import TrainResult, new_model, train_model, walk_forward_cv

MODES = {"train_new": "重新訓練", "train_resume": "接續訓練", "predict": "只預測（用冠軍模型）"}


def models_dir():
    p = config.home() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def best_path():
    return models_dir() / "best.pt"


def champion_path():
    return models_dir() / "champion.pt"


# ── 使用者設定（參數＋因子）存檔 ──
def settings_path():
    return config.home() / "settings.json"


def load_settings() -> tuple[TrainParams, dict[str, bool]]:
    try:
        d = json.loads(settings_path().read_text(encoding="utf-8"))
        f = config.default_factors()
        f.update({k: bool(v) for k, v in d.get("factors", {}).items() if k in f})
        return TrainParams.from_dict(d.get("params", {})), f
    except Exception:  # noqa: BLE001
        return TrainParams(), config.default_factors()


def save_settings(p: TrainParams, factors: dict[str, bool]) -> None:
    settings_path().write_text(json.dumps({"params": p.to_dict(), "factors": factors}, ensure_ascii=False, indent=2),
                               encoding="utf-8")


@dataclass
class RunResult:
    mode: str
    params: TrainParams
    factors: dict[str, bool]
    dates: pd.DatetimeIndex                 # 每個樣本的目標日期
    actual: np.ndarray
    fitted: np.ndarray
    fitted_probs: np.ndarray | None
    future_dates: list
    future_prices: np.ndarray
    future_probs: np.ndarray | None
    last_close: float
    da: float
    mape: float
    influence: dict[str, float]
    holdout_start: int
    challenger: dict = field(default_factory=dict)
    champion: dict | None = None
    promoted: bool = False
    verdict: str = ""
    train: TrainResult | None = None
    cv: dict | None = None
    model_source: str = ""
    seconds: float = 0.0
    messages: list[str] = field(default_factory=list)

    @property
    def t1_up(self) -> bool:
        return bool(self.future_prices[0] >= self.last_close) if len(self.future_prices) else False

    @property
    def t1_pct(self) -> float:
        return float((self.future_prices[0] / self.last_close - 1) * 100) if len(self.future_prices) else 0.0

    @property
    def t1_confidence(self) -> float | None:
        if self.future_probs is not None and len(self.future_probs):
            return float(abs(self.future_probs[0] - 0.5) * 200)
        return None


def _compatible(meta_cfg: dict, p: TrainParams, ds: Dataset) -> bool:
    return (meta_cfg.get("algorithm") == p.algorithm and meta_cfg.get("lookback") == p.lookback
            and meta_cfg.get("n_feat") == ds.n_features and meta_cfg.get("pred_target") == p.pred_target)


def _try_load(path, p: TrainParams | None, ds: Dataset) -> tuple[SilverModel | None, dict]:
    if not path.exists():
        return None, {}
    try:
        m, meta = load_model(path)
    except Exception:  # noqa: BLE001
        return None, {}
    if meta.get("feature_keys") and meta["feature_keys"] != ds.feature_keys:
        return None, meta
    cfg = m.config
    if cfg["lookback"] != ds.X.shape[1] or cfg["n_feat"] != ds.n_features:
        return None, meta
    if p is not None and not _compatible(cfg, p, ds):
        return None, meta
    return m, meta


def run(md: MarketData, p: TrainParams, factors: dict[str, bool], mode: str = "train_new",
        on_epoch: Callable[[int, float, TrainResult], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None) -> RunResult:
    t0 = time.perf_counter()
    msgs: list[str] = []

    def say(m: str):
        msgs.append(m)
        if on_status:
            on_status(m)

    data = md.recent(p.history_years)
    train_res = cv = None
    model_source = ""
    preloaded = None
    if mode == "predict":
        # 只預測：沿用模型當初訓練時的 lookback 與因子，資料集才對得上
        for path, label in ((champion_path(), "冠軍模型"), (best_path(), "最近一次訓練的最佳模型")):
            if not path.exists():
                continue
            try:
                m, meta = load_model(path)
            except Exception:  # noqa: BLE001
                continue
            mp_ = TrainParams.from_dict({**p.to_dict(), **meta.get("params", {})})
            mp_.future_days, mp_.history_years = p.future_days, p.history_years
            preloaded, model_source = (m, meta, mp_), label
            break
        if preloaded is None:
            raise RuntimeError("還沒有已存的模型，請先訓練。")
        model, meta, p = preloaded
        factors = {**config.default_factors(), **meta.get("factors", {})}
        say(f"使用{model_source}（{p.algorithm}，{config.PRED_TARGETS.get(p.pred_target, p.pred_target)}，lookback {p.lookback}）")

    ds = build_dataset(data.frame, p.lookback, factors)
    train_end, hold = split_holdout(len(ds.X))
    say(f"資料集：{len(ds.X)} 個樣本、{ds.n_features} 個特徵；保留集最後 {hold} 筆不參與訓練")

    if mode == "predict":
        if model.config["n_feat"] != ds.n_features:
            raise RuntimeError("已存模型的特徵數與目前資料不符，請重新訓練。")
    else:
        if p.validation_mode == "kfold":
            say(f"時序前推交叉驗證（{p.k_folds} 折）…")
            cv = walk_forward_cv(ds, p, on_fold=lambda i, k, m: say(m), should_stop=should_stop)
            if cv["folds"]:
                say(f"交叉驗證：平均 DA {cv['da']:.2f}%、MAPE {cv['mape']:.2f}%")
        model = None
        if mode == "train_resume":
            model, _ = _try_load(best_path(), p, ds)
            say("接續上次的最佳權重繼續訓練" if model else "沒有相容的已存模型，改為從零開始")
        if model is None:
            model = new_model(p, ds)
        idx = np.arange(train_end)
        train_res = train_model(model, ds, p, idx, on_epoch=on_epoch, should_stop=should_stop)
        model_source = "本次訓練"
        say(f"訓練完成：{train_res.epochs_run} epochs、{train_res.seconds:.1f} 秒"
            f"（{train_res.epochs_per_sec:.1f} epochs/s），最佳 loss {train_res.best_loss:.6f}"
            + ("，早停" if train_res.stopped_early else "") + ("，使用者中止" if train_res.stopped_by_user else ""))
        save_model(model, best_path(), {"params": p.to_dict(), "factors": factors, "feature_keys": ds.feature_keys,
                                        "best_loss": train_res.best_loss})

    # ── 預測與指標 ──
    fitted, probs = model_prices(model, ds)
    actual = ds.denorm_price(ds.y)
    da = direction_accuracy(actual, ds.prev_prices, fitted, probs, p.pred_target)
    mp = mape(actual, fitted) if p.pred_target != "direction" else float("nan")
    fc = forecast(model, ds, p.future_days)
    infl = factor_influence(model, ds) if ds.n_features > 1 else {}
    last_date = data.frame.index[-1].date()
    fut_dates = agent.next_trading_days(last_date, p.future_days)

    # ── 保留集：挑戰者 vs 冠軍（同一段資料）──
    challenger = score_holdout(model, ds, train_end) if hold else {"holdout_da": float("nan"), "holdout_mape": float("nan"), "holdout_n": 0}
    champ_meta = agent.champion()
    champ_score = None
    promoted, verdict = False, ""
    if mode != "predict":
        if champ_meta and hold:
            cm, _ = _try_load(champion_path(), None, ds)
            if cm is not None and cm.pred_target == p.pred_target:
                champ_score = score_holdout(cm, ds, train_end)
            else:
                say("現任冠軍的輸入形狀或預測目標不同，無法在同一保留集比較，視為無冠軍")
        promoted, verdict = agent.should_promote(champ_score, challenger)
        say(("🏆 晉升為新冠軍：" if promoted else "🛡️ 維持現任冠軍：") + verdict)
        if promoted:
            save_model(model, champion_path(), {"params": p.to_dict(), "factors": factors,
                                                "feature_keys": ds.feature_keys})
        agent.register_model({"algorithm": p.algorithm, "pred_target": p.pred_target, "lookback": p.lookback,
                              "layers": p.layers, "epochs": train_res.epochs_run if train_res else 0,
                              **challenger, "in_sample_da": da, "in_sample_mape": mp, "reason": verdict,
                              "params": p.to_dict(), "factors": factors}, promoted)

    seconds = time.perf_counter() - t0
    agent.add_run(type="predict" if mode == "predict" else "train", algorithm=p.algorithm, pred_target=p.pred_target,
                  da=da, mape=mp, holdout_da=challenger.get("holdout_da"), promoted=promoted, note=verdict or model_source,
                  seconds=seconds)
    agent.log_predictions(last_date, float(data.last_close), fc.prices, fc.dir_probs, p.algorithm, p.pred_target)
    agent.resolve_predictions({d.date().isoformat(): float(v) for d, v in md.frame["price"].items()})

    return RunResult(mode=mode, params=p, factors=factors, dates=ds.dates[ds.target_idx], actual=actual,
                     fitted=fitted, fitted_probs=probs, future_dates=fut_dates, future_prices=fc.prices,
                     future_probs=fc.dir_probs, last_close=data.last_close, da=da, mape=mp, influence=infl,
                     holdout_start=train_end, challenger=challenger, champion=champ_score, promoted=promoted,
                     verdict=verdict, train=train_res, cv=cv, model_source=model_source, seconds=seconds,
                     messages=msgs)

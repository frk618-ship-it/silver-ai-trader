import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fakes  # noqa: E402
from silver_trader import agent, config, indicators, market, pipeline  # noqa: E402
from silver_trader.backtest import run_backtest  # noqa: E402
from silver_trader.config import BacktestConfig, SearchConfig, TrainParams  # noqa: E402
from silver_trader.dataset import build_dataset, split_holdout  # noqa: E402
from silver_trader.models import build_model, load_model, save_model  # noqa: E402
from silver_trader.search import run_search  # noqa: E402


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SILVER_TRADER_HOME", str(tmp_path))
    return tmp_path


# ── 即時報價閘門 ──
def test_gate_passes_with_live_quote(monkeypatch):
    fakes.install(monkeypatch)
    md = market.load_market()
    assert md.live_quote.source == "Yahoo Finance"
    assert list(md.frame.columns) == config.FACTOR_KEYS
    assert not md.frame.isna().any().any()
    assert all(s.real for s in md.status)


@pytest.mark.parametrize("kw,msg", [
    (dict(quote_ok=False), "即時報價"),
    (dict(quote_age_h=200), "即時報價"),
    (dict(history_ok=False), "真實歷史價格"),
    (dict(quote_price=300.0), "差距過大"),
])
def test_gate_blocks(monkeypatch, kw, msg):
    fakes.install(monkeypatch, **kw)
    with pytest.raises(market.MarketDataError, match=msg):
        market.load_market()


def test_aux_failure_is_flagged_not_fatal(monkeypatch):
    fakes.install(monkeypatch, aux_ok=False)
    md = market.load_market()
    bad = [s.name for s in md.status if not s.real]
    assert "COT 籌碼" in bad and "Fear & Greed" in bad
    assert md.frame["fearGreed"].eq(50).all()


def test_history_falls_back_to_cache(monkeypatch):
    fakes.install(monkeypatch)
    market.load_market()                       # 先成功一次，留下快取
    fakes.install(monkeypatch, history_ok=False)
    md = market.load_market()
    si = next(s for s in md.status if s.name == "白銀期貨")
    assert si.source == "本機快取"


# ── 指標與資料集 ──
def test_indicators_match_reference():
    p = np.array([10, 11, 10.5, 12, 11, 13, 12.5, 14], float)
    assert indicators.sma(p, 3)[2] == pytest.approx(np.mean(p[:3]))
    assert indicators.sma(p, 3)[0] == p[0]
    assert indicators.roc(p)[1] == pytest.approx(0.1)
    r = indicators.rsi(p, 5)
    assert (r[:5] == 50).all() and 0 <= r[5] <= 100


def test_dataset_windows_align():
    f = pd.DataFrame({k: np.arange(100, dtype=float) + i for i, k in enumerate(config.FACTOR_KEYS)},
                     index=pd.bdate_range("2024-01-01", periods=100))
    ds = build_dataset(f, 10, config.default_factors())
    assert ds.X.shape == (90, 10, len(config.FACTOR_KEYS))
    # 第一個樣本的最後一列 = 第 9 天，目標 = 第 10 天
    assert ds.denorm_price(ds.y[0]) == pytest.approx(10.0)
    assert ds.prev_prices[0] == 9.0
    assert ds.price_scaler.unscale(ds.X[0, -1, 0]) == pytest.approx(9.0)
    assert split_holdout(90) == (70, 20)


@pytest.mark.parametrize("algo", config.ALGORITHMS)
@pytest.mark.parametrize("target", list(config.PRED_TARGETS))
def test_models_shapes_and_roundtrip(tmp_path, algo, target):
    m = build_model(algo, 12, 7, 16, target)
    x = torch.randn(4, 12, 7)
    out = m(x)
    assert out.shape == (4, 2 if target == "dual" else 1)
    save_model(m, tmp_path / "m.pt", {"a": 1})
    m2, meta = load_model(tmp_path / "m.pt")
    m.eval()
    assert torch.allclose(m(x), m2(x)) and meta == {"a": 1}


# ── 端到端 ──
@pytest.fixture
def md(monkeypatch):
    fakes.install(monkeypatch)
    return market.load_market()


@pytest.mark.parametrize("target", list(config.PRED_TARGETS))
def test_pipeline_train_then_predict(md, target):
    p = TrainParams(epochs=8, pred_target=target, seed=1, future_days=3)
    r = pipeline.run(md, p, config.default_factors(), "train_new")
    assert len(r.future_prices) == 3 and np.isfinite(r.future_prices).all()
    assert 0 <= r.da <= 100
    assert r.promoted                          # 第一個模型直接上任冠軍
    assert len(r.fitted) == len(r.actual) == len(r.dates)
    if target in ("direction", "dual"):
        assert r.future_probs is not None and r.t1_confidence is not None
    r2 = pipeline.run(md, TrainParams(future_days=2), config.default_factors(), "predict")
    assert r2.model_source == "冠軍模型" and len(r2.future_prices) == 2
    assert agent.table("predictions").shape[0] == 5


def test_resume_and_early_stop(md):
    p = TrainParams(epochs=5, seed=2)
    pipeline.run(md, p, config.default_factors(), "train_new")
    p2 = TrainParams(epochs=200, seed=2, early_stop_patience=3)
    r = pipeline.run(md, p2, config.default_factors(), "train_resume")
    assert any("接續" in m for m in r.messages)
    assert r.train.stopped_early and r.train.epochs_run < 200


def test_walk_forward_cv(md):
    p = TrainParams(epochs=3, validation_mode="kfold", k_folds=3, seed=3)
    r = pipeline.run(md, p, config.default_factors(), "train_new")
    assert r.cv and len(r.cv["folds"]) == 3 and math.isfinite(r.cv["da"])


def test_backtest(md):
    p = TrainParams(epochs=5, seed=4)
    pipeline.run(md, p, config.default_factors(), "train_new")
    m, meta = load_model(pipeline.best_path())
    ds = build_dataset(md.recent(p.history_years).frame, p.lookback, config.default_factors())
    bt = run_backtest(m, ds, BacktestConfig())
    assert len(bt.equity) > 50 and bt.trade_count > 0 and 0 <= bt.max_drawdown < 1


def test_search(md):
    cfg = SearchConfig(max_trials=3, search_epochs=4, seed=5, strategy="random")
    seen = []
    trials = run_search(md.frame, TrainParams(), config.default_factors(), cfg, on_trial=lambda i, t, b: seen.append(i))
    assert seen == [0, 1, 2] and len(trials) >= 2
    assert all(0 <= t.da <= 100 for t in trials)


@pytest.mark.parametrize("strategy", ["random", "grid", "bayesian"])
def test_search_over_prediction_targets(md, strategy):
    import random as _r
    from silver_trader.search import _generate

    targets = ["multi", "logreturn", "direction"]
    cfg = SearchConfig(max_trials=6, strategy=strategy, seed=1, targets=targets)
    base = TrainParams(pred_target="price", future_days=1)
    got = [_generate(i, cfg, base, config.default_factors(), [], _r.Random(i))[0] for i in range(6)]
    assert {p.pred_target for p in got} == set(targets)          # 每個目標都有試到，且不會用到 base 的 price
    assert all(p.future_days >= 2 for p in got if p.pred_target == "multi")
    assert _generate(0, SearchConfig(), base, config.default_factors(), [], _r.Random(0))[0].pred_target == "price"


def test_search_runs_mixed_targets(md):
    cfg = SearchConfig(max_trials=3, search_epochs=4, seed=5, targets=["multi", "logreturn", "dual"])
    trials = run_search(md.frame, TrainParams(), config.default_factors(), cfg)
    assert {t.params.pred_target for t in trials} == {"multi", "logreturn", "dual"}
    assert all(0 <= t.da <= 100 for t in trials) and all("預測目標" in t.row() for t in trials)


def test_promotion_rules():
    ok, _ = agent.should_promote(None, {"holdout_da": 55, "holdout_n": 30})
    assert ok
    ok, _ = agent.should_promote({"holdout_da": 55, "holdout_mape": 2}, {"holdout_da": 55.5, "holdout_n": 30, "holdout_mape": 3})
    assert not ok
    ok, _ = agent.should_promote({"holdout_da": 55, "holdout_mape": 3}, {"holdout_da": 55.5, "holdout_n": 30, "holdout_mape": 2})
    assert ok
    ok, _ = agent.should_promote(None, {"holdout_da": 60, "holdout_n": 5})
    assert not ok

"""直接多日預測（multi）與「以最新一天結尾的時間窗」。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fakes  # noqa: E402
from silver_trader import config, market, pipeline  # noqa: E402
from silver_trader.config import TrainParams  # noqa: E402
from silver_trader.dataset import build_dataset  # noqa: E402
from silver_trader.models import build_model, load_model, save_model  # noqa: E402
from silver_trader.predict import factor_influence, forecast, score_holdout  # noqa: E402
from silver_trader.training import _multi_rows, train_model  # noqa: E402


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SILVER_TRADER_HOME", str(tmp_path))


def _frame(n=200):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({k: np.linspace(10, 20, n) * (1 + 0.1 * i) + np.sin(np.arange(n) / 5) for i, k in enumerate(config.FACTOR_KEYS)},
                        index=idx)


def test_live_window_ends_on_the_latest_day():
    f = _frame()
    ds = build_dataset(f, 10, config.default_factors())
    assert ds.X_live.shape == ds.X[-1].shape
    # X[-1] 結尾在倒數第二天；X_live 結尾在最新一天（未來預測要用它）
    assert ds.price_scaler.unscale(ds.X[-1][-1, 0]) == pytest.approx(f["price"].iloc[-2])
    assert ds.price_scaler.unscale(ds.X_live[-1, 0]) == pytest.approx(f["price"].iloc[-1])
    np.testing.assert_array_equal(ds.X_live[:-1], ds.X[-1][1:])


def test_recursive_forecast_starts_from_the_latest_window():
    ds = build_dataset(_frame(), 10, config.default_factors())
    m = build_model("LSTM", 10, ds.n_features, 16, "price")
    fc = forecast(m, ds, 1)
    expected = ds.denorm_price(m.predict(ds.X_live[None])[0, 0])
    assert fc.prices[0] == pytest.approx(float(expected))


def test_multi_targets_and_purge():
    f = _frame()
    ds = build_dataset(f, 10, config.default_factors())
    raw, _, _ = ds.multi_targets(5, stats=(np.zeros(5), np.ones(5)))
    t = ds.target_idx[0]
    p = f["price"].to_numpy()
    assert raw[0] == pytest.approx(np.log(p[t:t + 5] / p[t - 1]))
    assert np.isnan(raw[-1, 1:]).all() and not np.isnan(raw[-5]).any()
    m = build_model("LSTM", 10, ds.n_features, 16, "multi", horizon=5)
    idx = np.arange(100)
    keep, Y = _multi_rows(m, ds, idx)
    assert keep.max() == 99 - 4                     # 目標日不可超出這批訓練資料（不偷看保留集）
    assert Y.shape == (len(keep), 5) and np.allclose(Y.mean(axis=0), 0, atol=1e-5)


def test_multi_direct_forecast_and_roundtrip(tmp_path):
    ds = build_dataset(_frame(), 10, config.default_factors())
    p = TrainParams(algorithm="LSTM", pred_target="multi", future_days=5, epochs=3, seed=1, layers=16)
    m = build_model("LSTM", 10, ds.n_features, 16, "multi", horizon=5)
    train_model(m, ds, p, np.arange(150))
    assert m.multi_stats is not None
    fc = forecast(m, ds, 7)                        # 要 7 天，模型只會 5 天
    assert len(fc.prices) == 5 and fc.dir_probs is None
    mu, sd = (np.array(v) for v in m.multi_stats)
    direct = ds.prices[-1] * np.exp(m.predict(ds.X_live[None])[0] * sd + mu)
    np.testing.assert_allclose(fc.prices, direct, rtol=1e-6)      # 一次直接輸出，不是遞迴
    save_model(m, tmp_path / "m.pt")
    m2, _ = load_model(tmp_path / "m.pt")
    assert m2.horizon == 5 and m2.multi_stats == m.multi_stats
    np.testing.assert_allclose(forecast(m2, ds, 5).prices, fc.prices, rtol=1e-6)
    s = score_holdout(m, ds, 150)
    assert s["horizon"] == 5 and 0 <= s["holdout_da_h"] <= 100 and s["holdout_n_h"] == len(ds.X) - 150 - 4
    infl = factor_influence(m, ds)
    assert len(infl) == ds.n_features - 1


def test_pipeline_multi_end_to_end(monkeypatch):
    fakes.install(monkeypatch)
    md = market.load_market()
    r = pipeline.run(md, TrainParams(epochs=5, seed=3), config.default_factors(), "train_new")
    assert r.params.pred_target == "multi" and len(r.future_prices) == 5 == len(r.future_dates)
    assert "holdout_da_h" in r.challenger
    r2 = pipeline.run(md, TrainParams(future_days=7), config.default_factors(), "predict")
    assert len(r2.future_prices) == 5 and len(r2.future_dates) == 5          # 天數由模型決定
    # 接續訓練：N 不同就不能接續
    r3 = pipeline.run(md, TrainParams(epochs=2, seed=3, future_days=3), config.default_factors(), "train_resume")
    assert any("從零開始" in m for m in r3.messages) and len(r3.future_prices) == 3

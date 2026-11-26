"""冠軍模型預測匯出（給貴金屬預警頁使用）。"""
import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fakes  # noqa: E402
from silver_trader import config, export, market, pipeline, storage  # noqa: E402
from silver_trader.config import TrainParams  # noqa: E402


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SILVER_TRADER_HOME", str(tmp_path / "app"))
    return tmp_path


def test_export_uses_backed_up_champion(monkeypatch, tmp_path):
    gh = fakes.FakeGitHub().install(monkeypatch)
    fakes.install(monkeypatch)
    md = market.load_market()
    p = TrainParams(epochs=4, seed=1, history_years=1.5, pred_target="dual")
    pipeline.run(md, p, config.default_factors(), "train_new")
    storage.backup(storage.GitHubStore("tok", gh.repo), "訓練完成")

    monkeypatch.setenv("SILVER_DATA_TOKEN", "tok")
    monkeypatch.setenv("SILVER_DATA_REPO", gh.repo)
    out_file = tmp_path / "out" / "silver_ai.json"
    assert export.main(["-o", str(out_file), "--days", "7"]) == 0
    out = json.loads(out_file.read_text(encoding="utf-8"))
    assert out["schema"] == 1 and out["symbol"] == "SI=F"
    assert out["model"]["source"] == "冠軍模型" and out["model"]["algorithm"] == "BiGRU"
    assert out["model"]["pred_target"] == "dual" and out["model"]["holdout_da"] is not None
    assert len(out["path"]) == 7
    days = [date.fromisoformat(r["date"]) for r in out["path"]]
    assert all(d.weekday() < 5 for d in days) and days == sorted(days)
    assert all(r["p_up"] is not None and 0 <= r["p_up"] <= 100 for r in out["path"])   # 雙頭模型有方向機率
    assert out["base_price"] > 0 and out["live_quote"]["price"] > 0
    assert 0 < len(out["drivers"]) <= 8
    # 匯出不可改動 App 自己的資料目錄
    assert not (tmp_path / "out" / "models").exists()


def test_export_fails_clearly_without_backup(monkeypatch, tmp_path, capsys):
    gh = fakes.FakeGitHub().install(monkeypatch)
    fakes.install(monkeypatch)
    monkeypatch.setenv("SILVER_DATA_TOKEN", "tok")
    monkeypatch.setenv("SILVER_DATA_REPO", gh.repo)
    assert export.main(["-o", str(tmp_path / "x.json")]) == 1
    assert "還沒有備份" in capsys.readouterr().err

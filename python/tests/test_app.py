"""Streamlit 介面測試（AppTest，在同一個行程內執行，可替換網路）。"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fakes  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(ROOT / "app.py")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("SILVER_TRADER_HOME", str(tmp_path))


def texts(at) -> str:
    parts = [m.value for m in at.markdown] + [e.value for e in at.error] + [c.value for c in at.caption]
    parts += [str(m.value) for m in at.metric]
    return "\n".join(map(str, parts))


def test_gate_blocks_without_live_quote(monkeypatch):
    fakes.install(monkeypatch, quote_ok=False)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("無法取得白銀期貨即時價格" in e.value for e in at.error)
    assert len(at.tabs) == 0                      # 沒進入主畫面
    assert any(b.label == "🔄 重新嘗試" for b in at.button)


def test_main_screen_and_training(monkeypatch):
    fakes.install(monkeypatch)
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs][:6] == ["📊 總覽", "🧠 訓練", "🤖 AI 搜尋", "💹 回測", "🧩 因子與資料", "🏆 代理紀錄"]
    assert "期貨即時" in texts(at)
    # 訓練分頁：設 epochs、按開始
    at.number_input[0].set_value(15)
    start = next(b for b in at.button if b.label == "▶ 開始")
    start.click().run()
    assert not at.exception, at.exception
    assert any("完成！" in s.value for s in at.success)
    assert any("判讀" in m.value for m in at.markdown)
    # 總覽出現判讀卡
    assert "判讀" in texts(at)
    # 回測
    bt = next(b for b in at.button if "執行回測" in b.label)
    bt.click().run()
    assert not at.exception, at.exception
    assert any(m.label == "Sharpe" for m in at.metric)

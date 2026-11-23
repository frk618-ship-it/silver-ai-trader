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


def test_app_backs_up_and_restores(monkeypatch, tmp_path):
    """Streamlit 重新啟動（硬碟清空）後，自動從 GitHub 還原冠軍模型。"""
    import streamlit as st
    gh = fakes.FakeGitHub().install(monkeypatch)
    fakes.install(monkeypatch)
    monkeypatch.setenv("SILVER_TRADER_HOME", str(tmp_path / "run1"))
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=300)
    at.secrets["GITHUB_TOKEN"] = "tok"
    at.secrets["DATA_REPO"] = gh.repo
    at.run()
    assert not at.exception, at.exception
    at.number_input[0].set_value(10)
    next(b for b in at.button if b.label == "▶ 開始").click().run()
    assert not at.exception, at.exception
    assert len(gh.commits) == 1, gh.commits
    assert any("已備份到 GitHub" in i.value for i in at.info)

    # 模擬 Streamlit Cloud 休眠後喚醒：新的資料目錄、新的伺服器行程（清掉 cache_resource）
    monkeypatch.setenv("SILVER_TRADER_HOME", str(tmp_path / "run2"))
    st.cache_resource.clear()
    at2 = AppTest.from_file(APP, default_timeout=300)
    at2.secrets["GITHUB_TOKEN"] = "tok"
    at2.secrets["DATA_REPO"] = gh.repo
    at2.run()
    assert not at2.exception, at2.exception
    assert any("已從 GitHub 還原" in i.value for i in at2.info)
    btn = next(b for b in at2.button if b.label == "用冠軍模型預測")
    assert not btn.disabled
    assert (tmp_path / "run2" / "models" / "champion.pt").exists()


def test_app_without_backup_settings(monkeypatch):
    import streamlit as st
    fakes.install(monkeypatch)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("DATA_REPO", raising=False)
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception, at.exception
    assert any("尚未設定自動備份" in i.value for i in at.info)

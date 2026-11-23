"""模型與紀錄備份到 GitHub（假 GitHub，不連外）。"""
import io
import shutil
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fakes  # noqa: E402
from silver_trader import agent, config, market, pipeline, storage  # noqa: E402
from silver_trader.config import TrainParams  # noqa: E402


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("SILVER_TRADER_HOME", str(h))
    return h


@pytest.fixture
def md(monkeypatch):
    fakes.install(monkeypatch)
    return market.load_market()


@pytest.fixture
def gh(monkeypatch):
    return fakes.FakeGitHub().install(monkeypatch)


def train(md, epochs=4):
    return pipeline.run(md, TrainParams(epochs=epochs, seed=1), config.default_factors(), "train_new")


def test_pack_unpack_roundtrip(md, home, tmp_path):
    train(md)
    data = storage.pack()
    names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
    assert {"agent.db", "models/best.pt", "models/champion.pt", "manifest.json"} <= names
    assert "cache/market_snapshot.pkl" not in names          # 行情快取不備份
    shutil.rmtree(home)
    m = storage.unpack(data)
    assert m["summary"]["has_champion"] and storage.summary()["predictions"] == 3


@pytest.mark.parametrize("bad,msg", [
    ({"../evil.py": b"x"}, "不認得"),
    ({"models/../../x.pt": b"x"}, "不認得"),
    ({"agent.db": b"not sqlite"}, "SQLite"),
    ({"manifest.json": b"{}"}, "空的"),
])
def test_unpack_rejects(bad, msg):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k, v in bad.items():
            z.writestr(k, v)
    with pytest.raises(storage.StorageError, match=msg):
        storage.unpack(buf.getvalue())
    with pytest.raises(storage.StorageError, match="zip"):
        storage.unpack(b"hello")


def test_backup_restore_cycle(md, home, gh):
    store = storage.GitHubStore("tok", gh.repo)
    r = train(md)
    up, h = storage.backup(store, "訓練完成")
    assert up and len(gh.commits) == 1 and "冠軍" in gh.commits[0]
    up2, h2 = storage.backup(store, "再一次", h)                 # 沒變化 → 不產生 commit
    assert not up2 and h2 == h and len(gh.commits) == 1
    pipeline.run(md, TrainParams(future_days=2), config.default_factors(), "predict")
    up3, _ = storage.backup(store, "產生預測", h)                 # 預測紀錄增加 → 再備份
    assert up3 and len(gh.commits) == 2

    shutil.rmtree(home)                                           # 模擬 Streamlit Cloud 重新啟動
    assert not storage.has_local_state()
    m = storage.restore(store)
    assert m and storage.summary()["predictions"] == 5
    r2 = pipeline.run(md, TrainParams(future_days=1), config.default_factors(), "predict")
    assert r2.model_source == "冠軍模型"
    assert agent.champion()["algorithm"] == r.params.algorithm


def test_conflict_retry(md, gh):
    store = storage.GitHubStore("tok", gh.repo)
    train(md)
    storage.backup(store, "a")
    real = gh.handle
    state = {"n": 0}

    def racing(method, url, **kw):          # 第一次 PUT 前，別的地方剛好也寫入
        if method == "PUT" and state["n"] == 0:
            state["n"] = 1
            d, _ = gh.files[storage.REMOTE_PATH]
            gh.files[storage.REMOTE_PATH] = (d, "someone-else")
        return real(method, url, **kw)
    storage._http = racing
    try:
        storage.GitHubStore("tok", gh.repo).upload(storage.pack(), "b")
    finally:
        gh.install()
    assert len(gh.commits) == 2


def test_errors_are_explained(gh, monkeypatch):
    with pytest.raises(storage.StorageError, match="token 無效"):
        storage.GitHubStore("wrong", gh.repo).check()
    with pytest.raises(storage.StorageError, match="找不到 repo"):
        storage.GitHubStore("tok", "me/other").check()
    gh.push = False
    with pytest.raises(storage.StorageError, match="寫入權限"):
        storage.GitHubStore("tok", gh.repo).check()


def test_restore_when_remote_empty(gh):
    assert storage.restore(storage.GitHubStore("tok", gh.repo)) is None


def test_store_from_settings(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("DATA_REPO", raising=False)
    assert storage.store_from({}) is None
    s = storage.store_from({"GITHUB_TOKEN": "t", "DATA_REPO": "a/b"})
    assert s.repo == "a/b" and s.branch is None
    monkeypatch.setenv("GITHUB_TOKEN", "t2")
    monkeypatch.setenv("DATA_REPO", "c/d")
    assert storage.store_from(None).repo == "c/d"

"""模型與紀錄的備份／還原。

Streamlit Community Cloud 的硬碟是暫時的（休眠、重新部署後清空），因此把
冠軍模型、最近一次訓練的模型、預測紀錄（SQLite）與設定打包成一個 zip，
存到使用者自己的私人 GitHub repo（透過 Contents API，一次備份 = 一個 commit）。

設定（Streamlit 的 Secrets 或環境變數）：
    GITHUB_TOKEN = "github_pat_..."              # 只能讀寫該 repo 的 fine-grained token
    DATA_REPO    = "owner/silver-ai-data"         # 私人 repo
    DATA_BRANCH  = "main"                         # 選填，預設用 repo 的預設分支
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import __version__, config

# 備份的檔案（相對於資料目錄）；cache/ 是可重新下載的行情快取，不備份
STATE_FILES = ["agent.db", "settings.json", "models/best.pt", "models/champion.pt"]
MANIFEST = "manifest.json"
REMOTE_PATH = "silver-state.zip"
MAX_ZIP_BYTES = 50 * 1024 * 1024
API = "https://api.github.com"


class StorageError(RuntimeError):
    pass


# ══════════════════════════════════════════════════════════════
# 打包／解包
# ══════════════════════════════════════════════════════════════
def _sqlite_snapshot(src: Path) -> bytes:
    """用 SQLite 的 backup API 取得一致的快照（避免複製到寫到一半的檔案）。"""
    with tempfile.TemporaryDirectory() as td:
        dst = Path(td) / "snap.db"
        s = sqlite3.connect(src)
        d = sqlite3.connect(dst)
        try:
            s.backup(d)
        finally:
            d.close()
            s.close()
        return dst.read_bytes()


def summary(home: Path | None = None) -> dict:
    """目前本機狀態的摘要（顯示用，也寫進 manifest）。"""
    home = Path(home or config.home())
    out = {"has_champion": (home / "models/champion.pt").exists(), "has_best": (home / "models/best.pt").exists()}
    db = home / "agent.db"
    if db.exists():
        try:
            con = sqlite3.connect(db)
            out["predictions"] = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            out["models"] = con.execute("SELECT COUNT(*) FROM models").fetchone()[0]
            r = con.execute("SELECT algorithm, holdout_da, ts FROM models WHERE status='champion' ORDER BY id DESC LIMIT 1").fetchone()
            if r:
                out["champion"] = {"algorithm": r[0], "holdout_da": r[1], "ts": r[2]}
            con.close()
        except sqlite3.Error:
            pass
    return out


def has_local_state(home: Path | None = None) -> bool:
    s = summary(home)
    return s["has_champion"] or s["has_best"] or s.get("predictions", 0) > 0


def pack(home: Path | None = None) -> bytes:
    home = Path(home or config.home())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in STATE_FILES:
            p = home / rel
            if not p.exists():
                continue
            z.writestr(rel, _sqlite_snapshot(p) if rel.endswith(".db") else p.read_bytes())
        # 「有沒有變化」由 content_hash() 只看實際檔案內容判斷，manifest 與 zip 時間戳記不影響
        z.writestr(MANIFEST, json.dumps({"app_version": __version__, "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         "summary": summary(home)},
                                        ensure_ascii=False, sort_keys=True))
    return buf.getvalue()


def content_hash(data: bytes) -> str:
    """只看實際檔案內容（忽略 zip 的時間戳記）的雜湊。"""
    h = hashlib.sha256()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for n in sorted(z.namelist()):
            if n == MANIFEST:
                continue
            h.update(n.encode())
            h.update(z.read(n))
    return h.hexdigest()


def unpack(data: bytes, home: Path | None = None) -> dict:
    """還原：只接受白名單內的檔名，整包取代本機狀態。回傳 manifest。"""
    home = Path(home or config.home())
    if len(data) > MAX_ZIP_BYTES:
        raise StorageError("備份檔太大")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise StorageError("不是有效的備份檔（zip）") from e
    names = set(z.namelist())
    bad = names - set(STATE_FILES) - {MANIFEST}
    if bad:
        raise StorageError(f"備份檔含有不認得的檔案：{', '.join(sorted(bad))[:120]}")
    if not names & set(STATE_FILES):
        raise StorageError("備份檔是空的")
    total = sum(i.file_size for i in z.infolist())
    if total > MAX_ZIP_BYTES * 4:
        raise StorageError("備份檔解壓後太大")
    for n in names & set(STATE_FILES):
        if n.endswith(".db"):
            blob = z.read(n)
            if not blob.startswith(b"SQLite format 3\x00"):
                raise StorageError(f"{n} 不是 SQLite 資料庫")
    manifest = {}
    if MANIFEST in names:
        try:
            manifest = json.loads(z.read(MANIFEST))
        except ValueError:
            manifest = {}
    (home / "models").mkdir(parents=True, exist_ok=True)
    for rel in STATE_FILES:
        p = home / rel
        if rel in names:
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_bytes(z.read(rel))
            os.replace(tmp, p)
        elif p.exists():
            p.unlink()
    return manifest


# ══════════════════════════════════════════════════════════════
# GitHub（Contents API）
# ══════════════════════════════════════════════════════════════
def _http(method: str, url: str, **kw) -> requests.Response:
    """網路入口（測試時替換）。"""
    return requests.request(method, url, timeout=kw.pop("timeout", 30), **kw)


@dataclass
class RemoteInfo:
    sha: str
    size: int


class GitHubStore:
    def __init__(self, token: str, repo: str, branch: str | None = None, path: str = REMOTE_PATH):
        if not token or not repo or "/" not in repo:
            raise StorageError("GITHUB_TOKEN 或 DATA_REPO 設定不完整")
        self.token, self.repo, self.branch, self.path = token.strip(), repo.strip(), (branch or "").strip() or None, path

    @property
    def label(self) -> str:
        return f"{self.repo}/{self.path}"

    def _h(self, raw: bool = False) -> dict:
        return {"Authorization": f"Bearer {self.token}", "X-GitHub-Api-Version": "2022-11-28",
                "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
                "User-Agent": "silver-ai-trader"}

    def _url(self) -> str:
        return f"{API}/repos/{self.repo}/contents/{self.path}"

    @staticmethod
    def _explain(r: requests.Response) -> str:
        msg = ""
        try:
            msg = r.json().get("message", "")
        except ValueError:
            pass
        if r.status_code == 401:
            return "GitHub token 無效或已過期，請重新建立並更新 Streamlit Secrets 的 GITHUB_TOKEN"
        if r.status_code == 403:
            return f"token 沒有這個 repo 的寫入權限（Contents 需設為 Read and write）：{msg}"
        if r.status_code == 404:
            return f"找不到 repo {msg and '（' + msg + '）'}：請確認 DATA_REPO 拼字，且 token 有選到這個 repo"
        return f"GitHub 回應 HTTP {r.status_code}：{msg}"

    def check(self) -> dict:
        r = _http("GET", f"{API}/repos/{self.repo}", headers=self._h())
        if r.status_code != 200:
            raise StorageError(self._explain(r))
        j = r.json()
        perms = j.get("permissions") or {}
        if perms and not perms.get("push"):
            raise StorageError("token 對這個 repo 沒有寫入權限（Contents 需設為 Read and write）")
        return {"private": bool(j.get("private")), "default_branch": j.get("default_branch")}

    def info(self) -> RemoteInfo | None:
        params = {"ref": self.branch} if self.branch else None
        r = _http("GET", self._url(), headers=self._h(), params=params)
        if r.status_code == 404:
            # 檔案不存在 vs. repo 不存在：用 check() 區分
            self.check()
            return None
        if r.status_code != 200:
            raise StorageError(self._explain(r))
        j = r.json()
        return RemoteInfo(j["sha"], int(j.get("size", 0)))

    def download(self) -> bytes | None:
        params = {"ref": self.branch} if self.branch else None
        r = _http("GET", self._url(), headers=self._h(raw=True), params=params)
        if r.status_code == 404:
            self.check()
            return None
        if r.status_code != 200:
            raise StorageError(self._explain(r))
        return r.content

    def upload(self, data: bytes, message: str) -> str:
        for attempt in range(2):
            cur = self.info()
            body = {"message": message, "content": base64.b64encode(data).decode()}
            if cur:
                body["sha"] = cur.sha
            if self.branch:
                body["branch"] = self.branch
            r = _http("PUT", self._url(), headers=self._h(), json=body)
            if r.status_code in (200, 201):
                return r.json()["content"]["sha"]
            if r.status_code in (409, 422) and attempt == 0:
                continue          # 其他地方剛好也寫入了 → 取最新 sha 再試一次
            raise StorageError(self._explain(r))
        raise StorageError("備份失敗（多次衝突）")


def store_from(secrets=None) -> GitHubStore | None:
    """從 Streamlit secrets（或環境變數）建立；沒有設定時回傳 None。"""
    def get(k):
        try:
            if secrets is not None and k in secrets:
                return str(secrets[k])
        except Exception:  # noqa: BLE001  （本機沒有 secrets.toml 時 Streamlit 會丟例外）
            pass
        return os.environ.get(k)
    token, repo = get("GITHUB_TOKEN"), get("DATA_REPO")
    if not token or not repo:
        return None
    return GitHubStore(token, repo, get("DATA_BRANCH"))


def backup(store: GitHubStore, reason: str, last_hash: str | None = None) -> tuple[bool, str]:
    """打包並上傳。內容與上次相同時略過。回傳 (是否有上傳, 新的內容雜湊)。"""
    data = pack()
    h = content_hash(data)
    if h == last_hash:
        return False, h
    s = summary()
    champ = s.get("champion")
    extra = f"；冠軍 {champ['algorithm']} 保留集 DA {champ['holdout_da']:.1f}%" if champ and champ.get("holdout_da") is not None else ""
    store.upload(data, f"{reason}（{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC{extra}）")
    return True, h


def restore(store: GitHubStore) -> dict | None:
    data = store.download()
    if data is None:
        return None
    return unpack(data)

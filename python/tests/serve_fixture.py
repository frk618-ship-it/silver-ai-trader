"""以假網路啟動 App（只供截圖與手動檢查）：streamlit run tests/serve_fixture.py"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import fakes  # noqa: E402

fakes.install()
import os  # noqa: E402
if os.environ.get("FAKE_GITHUB"):        # 預覽備份功能：用假的 GitHub
    if not hasattr(fakes, "_GH"):
        fakes._GH = fakes.FakeGitHub(repo=os.environ.get("DATA_REPO", "me/silver-ai-data"))
    fakes._GH.install()
runpy.run_path(str(HERE.parent / "app.py"), run_name="__main__")

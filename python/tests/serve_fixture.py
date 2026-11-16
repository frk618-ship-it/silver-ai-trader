"""以假網路啟動 App（只供截圖與手動檢查）：streamlit run tests/serve_fixture.py"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import fakes  # noqa: E402

fakes.install()
runpy.run_path(str(HERE.parent / "app.py"), run_name="__main__")

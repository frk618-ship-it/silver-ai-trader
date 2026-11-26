"""Silver AI Trader — Python 版（Streamlit + PyTorch）。

白銀期貨（COMEX SI=F）多因子深度學習預測系統，由 V20.4 HTML 版移植。
"""

import time as _time

__version__ = "21.0.0"
_loaded_at = _time.time()      # app.py 用來判斷記憶體裡的模組是否比原始檔舊

# Silver AI Trader — 白銀期貨 AI 預測

白銀期貨（COMEX **SI=F**）多因子深度學習預測系統：Python + Streamlit + PyTorch。

**網址：https://silver-ai.streamlit.app/**
（GitHub Pages 首頁 https://frk618-ship-it.github.io/silver-ai-trader/ 會自動導向這裡）

- 伺服器端抓取行情，不受瀏覽器 CORS 限制：白銀期貨即時報價、5 年歷史價格與 24 項輔助因子。
- 模型：BiGRU、LSTM、Transformer、TimesFM。預測目標：多日報酬率（預設，直接預測 T+1…T+N）、對數報酬率、價格、方向、雙頭。
- 冠軍／挑戰者保留集驗證、AI 參數搜尋、回測、預測紀錄自動對帳。

完整說明、部署與測試方式見 [`python/README.md`](python/README.md)。

| 路徑 | 說明 |
| --- | --- |
| `python/` | 主程式（Streamlit App、模型、資料來源、測試） |
| `index.html` | GitHub Pages 首頁：直接導向 https://silver-ai.streamlit.app/ |
| `.streamlit/` | Streamlit Cloud 的主題設定 |
| `.devcontainer/` | GitHub Codespaces／Dev Container 開發環境 |

⚠️ 模型輸出僅供研究參考，不構成投資建議。

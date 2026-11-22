# Silver AI Trader — V20.4 Fast-Train

白銀期貨多因子深度學習預測系統。

> ✨ **v2 網頁版（重新設計 UI／UX）** → https://frk618-ship-it.github.io/silver-ai-trader/v2/ ，說明見 [`v2/`](v2/README.md)。
>
> 🐍 **新版：Python（Streamlit + PyTorch）** → 見 [`python/`](python/README.md)，介面重新設計，執行 `streamlit run python/app.py`。
> 以下為 HTML 版說明，HTML 版仍保留在 GitHub Pages 上。

| 檔案 | 說明 |
| --- | --- |
| `v2/` | **v2 網頁版**：重新設計的介面，模型與因子選項同 v20.3 |
| `python/` | **V21 Python 版**：Streamlit 介面 + PyTorch 模型 |
| `index.html` | **V20.4 Fast-Train**：訓練速度優化版（直接用瀏覽器開啟即可） |
| `original-v20_3.html` | 原始 V20.3 instant-start 版本（對照用，未修改） |
| `bench/` | 效能/正確性測試腳本（Playwright + 本機 TF.js） |

## V20.4 做了哪些優化

### 1. 融合 RNN 層 `FusedBiGRU` / `FusedLSTM`（最大宗）
TF.js 原生的 GRU/LSTM 每個時間步都要重新做一次輸入投影、切分遞迴權重，
每一步約 20 個 kernel，BiGRU 兩層 × 兩方向 × lookback 步，再加上反向傳播，
一個 batch 就超過 6000 次 kernel 呼叫，**瓶頸是呼叫次數而非計算量**。

新的融合層：
- 所有時間步、兩個方向的 `X·W` 一次 matMul 算完
- 遞迴權重切分移出迴圈；TF.js 預設閘門 `hardSigmoid = clip(0.2a+0.5)` 的係數預先折進權重
- WebGL：雙向的兩個方向堆疊成一個 batched matMul；WASM/CPU：改用 2D matMul + JS 索引取代 `reverse`

**數學與原生層完全相同**（權重形狀、順序一致，輸出與梯度誤差 < 1e-6，已逐一比對）。
可在「⚡ 訓練加速」面板關閉，退回原生層。

### 2. 運算後端自動測速
- 新增「運算後端」選項：自動（預設）/ WebGL / WASM / CPU
- 自動模式會用「實際要訓練的模型」在每個後端跑幾個 batch，挑最快的並快取結果
  （小型 RNN 在很多裝置上 WASM 反而比 WebGL 快）
- 背景分頁原本切到 CPU 後端，現在改切 **WASM（約快 4 倍）**；本來就在 WASM 時不切換
- TF.js 鎖定 `4.22.0`（原本 `@latest` 會無預警變動），正確設定 WASM 檔案路徑，開啟 `enableProdMode()`

### 3. 訓練迴圈去除額外開銷
原本每個 epoch：React setState ×3、Plotly 整張圖重建、讀 localStorage、`tf.nextFrame()`（≥16ms），
而且 loss 每創新低就把整個模型序列化寫進 localStorage。現在：
- 最佳權重以 `clone()` 存在記憶體，訓練結束才寫一次 localStorage（每 60 秒落地一次防當機）
- UI 與 Loss 圖每 ~250ms 批次更新一次；Loss 圖改用 `Plotly.react` 差異更新
- `fit({ yieldEvery: 'never' })` + 依時間觸發的 `MessageChannel` 讓出：
  不再每 batch 等一個畫面，背景分頁也不會因 `requestAnimationFrame` 被凍結而停住
- 訓練結束直接把最佳權重套回模型，不必再從 localStorage 重新載入
- 訓練完成會顯示耗時與 epochs/s

### 4. 其他
- 可選 **早停（Early Stopping）**：連續 N epochs 未改善即停止（預設關閉，行為與原版相同）
- K-Fold 切分由 `O(n²)` 的 `includes` 改為 `Set`
- 超參數搜尋的 `fit` 也改為 `yieldEvery: 'never'`，避免背景分頁卡住

## 實測（headless Chromium，4 核 CPU，無 GPU）

**整個 App 端到端**（模擬資料、BiGRU 預設參數、30 epochs，從按下「重新訓練模型」到預測完成）：

| 版本 | 耗時 |
| --- | --- |
| V20.3 原版（WebGL/SwiftShader） | 176.5 秒 |
| V20.4（自動測速 → 選 WASM + 融合層 + 精簡迴圈） | **29.9 秒（約 5.9×）** |

> 這台測試機沒有實體 GPU，WebGL 是軟體模擬，所以自動測速選了 WASM。
> 有獨顯的電腦上倍數會不同，但「自動測速」會挑出該裝置上最快的後端。

融合層單獨比較，

BiGRU 32 units、lookback 10、20 特徵、480 筆樣本，每 epoch 毫秒數：

| 後端 | batch | 原生層 | 融合層 | 加速 |
| --- | --- | --- | --- | --- |
| WASM | 32 | 922 | 635 | 1.45× |
| WASM | 128 | 522 | 426 | 1.23× |
| WebGL (SwiftShader) | 32 | 4039 | 1889 | 2.14× |
| CPU (JS) | 32 | ~3900 | — | （背景分頁原本用這個，現在改 WASM ≈ 4× 快） |

LSTM：WebGL 1.45×，WASM 約 1.1×。實際 GPU 上 kernel dispatch 成本更高，融合層效益通常更明顯。

## 相容性注意
- 新版訓練出的 BiGRU/LSTM 模型使用自訂層 `FusedBiGRU` / `FusedLSTM`，
  **舊版 HTML 無法載入**新版匯出的模型；新版可以正常載入舊版模型（原生層仍保留）。
- 若需要與舊版互通，請在「⚡ 訓練加速」關閉「融合 RNN 核心」後再訓練。

## 測試
```bash
cd bench && npm i @tensorflow/tfjs@4.22.0 @tensorflow/tfjs-backend-wasm@4.22.0 playwright
node verify.js     # 融合層 vs 原生層：輸出、梯度、存取往返一致性 + 速度
```

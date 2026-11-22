# 資料中繼：Cloudflare Worker（免費，約 5 分鐘）

## 為什麼需要

- 瀏覽器**不允許網頁直接讀取** Yahoo Finance 的資料：Yahoo 的回應沒有 CORS 標頭。
- Stooq 會拒絕部分連線。
- 程式原本依賴的 5 個免費公開 CORS proxy 已全部失效（2026-11 實測）：
  - corsproxy.io 改為需要 API 金鑰
  - thingproxy 停止服務
  - corsproxy.dev 只回傳網頁
  - allorigins、codetabs 回 5xx

這個 Worker 在 Cloudflare 的伺服器上代為抓取資料，帶上瀏覽器 User-Agent、加上 CORS 標頭後回傳。網頁設定好它的網址後，所有資料都會優先走它。

**它不是開放 proxy：**
- 只轉發白名單裡的資料來源：Yahoo、Stooq、CFTC、美國財政部、CNN、StockTwits、Google Trends。
- 只接受你網站的呼叫：`https://frk618-ship-it.github.io`、本機檔案、localhost。
- 只允許 GET。

免費方案每天 10 萬次請求，本系統載入一次約 20–30 次。

## 部署步驟（網頁操作，不用安裝任何東西）

1. 到 https://dash.cloudflare.com/sign-up 註冊（免費，只需 email）。
2. 左側選單 **Workers & Pages**（新版介面在 **Compute (Workers)** 底下）→ **Create** → **Create Worker**（或「Start with Hello World」）。
3. 名稱填 `silver-proxy` → **Deploy**。
4. 部署完成後按 **Edit code**，把編輯器裡的內容**全部刪掉**，貼上本資料夾的 [`silver-proxy.js`](silver-proxy.js) 全文 → 右上角 **Deploy**。
5. 複製 Worker 網址，長得像 `https://silver-proxy.你的帳號名稱.workers.dev`。
6. 打開 https://frk618-ship-it.github.io/silver-ai-trader/v2/ ：
   - 進站畫面：在「資料中繼網址」貼上網址，按 **測試並儲存**。
   - 已經在主畫面：到「因子與資料」頁的「資料中繼」設定。

   出現「✓ 連線成功，白銀期貨 $xx.xxx」就完成了，系統會自動重新載入資料。

> 設定存在瀏覽器裡，同一個網域下的舊版（…/silver-ai-trader/）也會一起使用這個中繼。換電腦或手機時，要在那台裝置上再貼一次。

## 驗證

瀏覽器直接打開 Worker 網址（不加參數），應看到：

```json
{"ok":true,"service":"silver-proxy","hint":"用法：/?url=<encoded url>"}
```

## 進階

- **換網址或用其他網域開啟網頁**：在 Worker 的 **Settings → Variables** 新增 `ALLOWED_ORIGINS`，值是以逗號分隔的來源，例如 `https://frk618-ship-it.github.io,http://localhost`。
- **用指令部署**（已安裝 Node.js）：在本資料夾執行 `npx wrangler login` 再執行 `npx wrangler deploy`，設定見 [`wrangler.toml`](wrangler.toml)。
- **快取**：即時報價快取 30 秒、日線與其他資料 1 小時，減少對資料來源的請求，也降低被限流的機會。

## 疑難排解

| 訊息 | 原因 |
| --- | --- |
| 連不到這個網址 | 網址打錯、Worker 還沒按 Deploy，或網頁的網域不在 `ALLOWED_ORIGINS` |
| 不允許的來源 | 從白名單以外的網站呼叫；把該網域加進 `ALLOWED_ORIGINS` |
| HTTP 429（Yahoo 回應 429） | Yahoo 暫時限流，等幾分鐘再試；快取會讓之後的請求變少 |

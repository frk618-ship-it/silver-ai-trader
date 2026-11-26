"""PyTorch 模型：BiGRU / LSTM / Transformer / TimesFM（結構對應 HTML 版）。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_UNITS = {"price": 1, "direction": 1, "dual": 2, "logreturn": 1}   # multi：輸出 horizon 個單元


def output_units(pred_target: str, horizon: int = 1) -> int:
    return max(1, int(horizon)) if pred_target == "multi" else OUTPUT_UNITS[pred_target]


class BiGRUNet(nn.Module):
    """兩層雙向 GRU，層間 dropout 0.3；取兩方向最後狀態串接。"""

    def __init__(self, n_feat: int, units: int):
        super().__init__()
        self.rnn = nn.GRU(n_feat, units, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.out_dim = units * 2

    def forward(self, x):
        _, h = self.rnn(x)                       # h: [layers*2, B, U]
        return torch.cat([h[-2], h[-1]], dim=1)  # 最後一層的 fwd / bwd


class LSTMNet(nn.Module):
    def __init__(self, n_feat: int, units: int):
        super().__init__()
        self.rnn = nn.LSTM(n_feat, units, num_layers=2, batch_first=True, dropout=0.3)
        self.out_dim = units

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return h[-1]


class TransformerNet(nn.Module):
    """Dense → Conv1D 殘差 → LayerNorm → FFN 殘差 → LayerNorm → 全域平均池化。"""

    def __init__(self, n_feat: int, units: int):
        super().__init__()
        ff = min(units * 2, 128)
        self.inp = nn.Linear(n_feat, units)
        self.conv = nn.Conv1d(units, units, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm(units, eps=1e-6)
        self.ff = nn.Sequential(nn.Linear(units, ff), nn.ReLU(), nn.Dropout(0.1), nn.Linear(ff, units))
        self.ln2 = nn.LayerNorm(units, eps=1e-6)
        self.out_dim = units

    def forward(self, x):
        x = F.relu(self.inp(x))
        a = F.relu(self.conv(x.transpose(1, 2)).transpose(1, 2))
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff(x))
        return x.mean(dim=1)


class _CausalBlock(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d, eps=1e-6)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(0.1), nn.Linear(ff, d))
        self.ln2 = nn.LayerNorm(d, eps=1e-6)
        self.d = d

    def forward(self, x):
        P = x.shape[1]
        s = self.q(x) @ self.k(x).transpose(1, 2) / math.sqrt(self.d)
        mask = torch.triu(torch.ones(P, P, dtype=torch.bool, device=x.device), 1)
        s = s.masked_fill(mask, -1e9)                     # 因果遮罩：看不到未來的 patch
        x = self.ln1(x + torch.softmax(s, -1) @ self.v(x))
        return self.ln2(x + self.ff(x))


class TimesFMNet(nn.Module):
    """TimesFM 風格 decoder-only：patch 化 → 殘差輸入塊 → 2 層因果注意力 → 取最後 patch → 殘差輸出塊。"""

    def __init__(self, n_feat: int, units: int, lookback: int):
        super().__init__()
        d = units
        ff = min(d * 2, 128)
        self.patch = 1
        for c in range(min(8, lookback // 2), 1, -1):
            if lookback % c == 0:
                self.patch = c
                break
        self.n_patch = lookback // self.patch
        pin = self.patch * n_feat
        self.in_hidden, self.in_proj, self.in_skip = nn.Linear(pin, d), nn.Linear(d, d), nn.Linear(pin, d)
        self.blocks = nn.ModuleList([_CausalBlock(d, ff) for _ in range(2)])
        self.out_hidden, self.out_proj, self.out_skip = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out_dim = d

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, self.n_patch, -1)
        x = self.in_proj(F.relu(self.in_hidden(x))) + self.in_skip(x)
        for b in self.blocks:
            x = b(x)
        x = x[:, -1]
        return self.out_proj(F.relu(self.out_hidden(x))) + self.out_skip(x)


class SilverModel(nn.Module):
    def __init__(self, algorithm: str, lookback: int, n_feat: int, units: int, pred_target: str, horizon: int = 1):
        super().__init__()
        self.horizon = max(1, int(horizon)) if pred_target == "multi" else 1
        self.multi_stats = None   # 多日目標的 (各欄平均, 各欄標準差)，訓練時決定、隨模型存檔
        self.config = dict(algorithm=algorithm, lookback=lookback, n_feat=n_feat, units=units, pred_target=pred_target,
                           horizon=self.horizon)
        if algorithm == "BiGRU":
            self.body = BiGRUNet(n_feat, units)
        elif algorithm == "LSTM":
            self.body = LSTMNet(n_feat, units)
        elif algorithm == "Transformer":
            self.body = TransformerNet(n_feat, units)
        elif algorithm == "TimesFM":
            self.body = TimesFMNet(n_feat, units, lookback)
        else:
            raise ValueError(f"未知的演算法：{algorithm}")
        self.head = nn.Linear(self.body.out_dim, output_units(pred_target, self.horizon))
        self.pred_target = pred_target

    def set_multi_stats(self, mu, sd) -> None:
        self.multi_stats = ([float(v) for v in mu], [float(v) for v in sd])
        self.config["multi_stats"] = [list(self.multi_stats[0]), list(self.multi_stats[1])]

    def forward(self, x):
        """回傳原始輸出（direction 為 logit；dual 欄 1 為 logit）。"""
        return self.head(self.body(x))

    @torch.no_grad()
    def predict(self, X, batch: int = 512) -> "torch.Tensor":
        """推論：direction 轉成機率；其餘維持原始尺度。回傳 numpy。"""
        self.eval()
        outs = []
        X = torch.as_tensor(X, dtype=torch.float32)
        for i in range(0, len(X), batch):
            o = self(X[i:i + batch])
            if self.pred_target == "direction":
                o = torch.sigmoid(o)
            outs.append(o)
        return torch.cat(outs).numpy() if outs else torch.empty(0, output_units(self.pred_target, self.horizon)).numpy()


def build_model(algorithm: str, lookback: int, n_feat: int, units: int, pred_target: str,
                horizon: int = 1) -> SilverModel:
    return SilverModel(algorithm, lookback, n_feat, units, pred_target, horizon)


def save_model(model: SilverModel, path, meta: dict | None = None) -> None:
    torch.save({"config": model.config, "state": model.state_dict(), "meta": meta or {}}, path)


def load_model(path) -> tuple[SilverModel, dict]:
    # weights_only=True：只允許張量與基本型別，不會執行檔案裡的程式碼
    # （模型檔可能來自使用者上傳或雲端備份，不能用會執行 pickle 程式碼的載入方式）
    ck = torch.load(path, map_location="cpu", weights_only=True)
    c = ck["config"]
    m = build_model(c["algorithm"], c["lookback"], c["n_feat"], c["units"], c["pred_target"], c.get("horizon", 1))
    m.load_state_dict(ck["state"])
    if c.get("multi_stats"):
        m.set_multi_stats(*c["multi_stats"])
    m.eval()
    return m, ck.get("meta", {})

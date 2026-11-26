"""匯出白銀冠軍模型的未來 N 個交易日預測，給其他系統使用（例如靶材新聞網站的貴金屬預警頁）。

流程：從私人備份 repo 還原冠軍模型 → 抓最新行情 → 只預測（不訓練）→ 寫出 JSON。

    python -m silver_trader.export -o silver_ai.json --days 7

環境變數（擇一）：
    SILVER_DATA_TOKEN / GITHUB_TOKEN   可讀取備份 repo 的 GitHub token
    SILVER_DATA_REPO  / DATA_REPO      備份 repo，例如 frk618-ship-it/silver-ai-data
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

APP_URL = "https://silver-ai.streamlit.app/"
SCHEMA = 1


def _store():
    from . import storage

    token = os.environ.get("SILVER_DATA_TOKEN") or os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("SILVER_DATA_REPO") or os.environ.get("DATA_REPO")
    if not token or not repo:
        raise SystemExit("缺少 SILVER_DATA_TOKEN / SILVER_DATA_REPO（或 GITHUB_TOKEN / DATA_REPO）")
    return storage.GitHubStore(token, repo, os.environ.get("SILVER_DATA_BRANCH") or os.environ.get("DATA_BRANCH"))


def build(days: int = 7, top: int = 8, store=None, home: Path | None = None) -> dict:
    """還原冠軍模型並產生預測；回傳可寫成 JSON 的 dict。"""
    from . import agent, config, market, pipeline, storage
    from .config import TrainParams
    from .models import load_model

    store = store or _store()
    tmp = None
    if home is None:
        tmp = tempfile.TemporaryDirectory()
        home = Path(tmp.name)
    old = os.environ.get("SILVER_TRADER_HOME")
    os.environ["SILVER_TRADER_HOME"] = str(home)
    try:
        if storage.restore(store) is None:
            raise RuntimeError(f"{store.repo} 還沒有備份：請先在 {APP_URL} 訓練一次")
        path = pipeline.champion_path() if pipeline.champion_path().exists() else pipeline.best_path()
        if not path.exists():
            raise RuntimeError("備份裡沒有模型：請先在 App 訓練一次")
        _, meta = load_model(path)
        mp = TrainParams.from_dict(meta.get("params", {}))
        champ = agent.champion() or {}

        md = market.load_market()
        r = pipeline.run(md, TrainParams(future_days=days, history_years=mp.history_years),
                         config.default_factors(), "predict")
        probs = r.future_probs
        infl = sorted(r.influence.items(), key=lambda kv: -abs(kv[1]))[:top]
        q = md.live_quote
        return {
            "schema": SCHEMA,
            "source": "silver-ai-trader",
            "app_url": APP_URL,
            "symbol": "SI=F",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": {
                "source": r.model_source,
                "algorithm": r.params.algorithm,
                "pred_target": r.params.pred_target,
                "pred_target_name": config.PRED_TARGETS.get(r.params.pred_target, r.params.pred_target),
                "lookback": r.params.lookback,
                "layers": r.params.layers,
                "n_factors": sum(r.factors.values()) + 1,
                "holdout_da": champ.get("holdout_da"),
                "holdout_mape": champ.get("holdout_mape"),
                "holdout_n": champ.get("holdout_n"),
                "trained_at": champ.get("ts"),
            },
            "base_date": md.frame.index[-1].date().isoformat(),
            "base_price": round(float(r.last_close), 4),
            "live_quote": {"price": round(float(q.price), 4), "time": q.time.isoformat(), "source": q.source},
            "path": [{"date": d.isoformat(), "price": round(float(p), 4),
                      "p_up": None if probs is None or i >= len(probs) else round(float(probs[i]) * 100, 1)}
                     for i, (d, p) in enumerate(zip(r.future_dates, r.future_prices))],
            "drivers": [{"name": k, "effect": round(float(v), 4)} for k, v in infl],
            "data_sources": {"real": sum(s.real for s in md.status), "total": len(md.status)},
        }
    finally:
        if old is None:
            os.environ.pop("SILVER_TRADER_HOME", None)
        else:
            os.environ["SILVER_TRADER_HOME"] = old
        if tmp is not None:
            tmp.cleanup()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="匯出白銀冠軍模型的未來預測（JSON）")
    ap.add_argument("-o", "--output", default="silver_ai.json", help="輸出檔（- 代表標準輸出）")
    ap.add_argument("--days", type=int, default=7, help="預測幾個交易日（預設 7）")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        out = build(days=args.days)
    except Exception as e:  # noqa: BLE001
        print(f"匯出失敗：{e}", file=sys.stderr)
        return 1
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if args.output == "-":
        print(text)
    else:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        m, last = out["model"], out["path"][-1]
        print(f"已寫入 {args.output}：{m['algorithm']}（{m['source']}），基準 {out['base_date']} ${out['base_price']:.3f} → "
              f"{last['date']} ${last['price']:.3f}（{(last['price'] / out['base_price'] - 1) * 100:+.2f}%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

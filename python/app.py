"""Silver AI Trader — Streamlit 介面。

執行：streamlit run app.py
"""
from __future__ import annotations

import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from silver_trader import __version__, agent, config, market, pipeline, ui  # noqa: E402
from silver_trader.backtest import run_backtest  # noqa: E402
from silver_trader.config import BacktestConfig, SearchConfig, TrainParams  # noqa: E402
from silver_trader.dataset import build_dataset  # noqa: E402
from silver_trader.models import load_model  # noqa: E402
from silver_trader.search import best_trial, run_search  # noqa: E402

st.set_page_config(page_title="白銀期貨 AI 預測", page_icon="🥈", layout="wide", initial_sidebar_state="collapsed")
st.markdown(ui.CSS, unsafe_allow_html=True)
ss = st.session_state


# ══════════════════════════════════════════════════════════════
# 共用
# ══════════════════════════════════════════════════════════════
def header():
    st.markdown(
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">'
        '<span class="ag-badge">Ag</span><div><p class="ag-title">白銀期貨 AI 預測</p>'
        f'<p class="ag-sub">COMEX SI=F · 多因子深度學習 · v{__version__}</p></div></div>',
        unsafe_allow_html=True)


def init_settings():
    if "params" not in ss:
        ss.params, ss.factors = pipeline.load_settings()


def persist():
    pipeline.save_settings(ss.params, ss.factors)


# ══════════════════════════════════════════════════════════════
# 🔴 閘門：沒有白銀期貨即時報價就不進入主畫面
# ══════════════════════════════════════════════════════════════
def gate():
    header()
    left, _ = st.columns([3, 2])
    with left:
        st.markdown("#### 正在取得市場資料")
        st.markdown('<p class="muted">必須先取得 <b>白銀期貨（SI=F）即時報價</b> 與 <b>真實歷史價格</b>，'
                    "才會進入模型畫面；系統不會用模擬資料頂替。其餘 20 多項輔助因子若暫時抓不到，會以中性值補位並標示出來。</p>",
                    unsafe_allow_html=True)
        if ss.get("gate_error") and not ss.get("gate_retry"):
            st.error(f"**無法取得白銀期貨即時價格**\n\n{ss.gate_error}", icon="🚫")
            st.caption("資料來源可能暫時無法連線。重試通常可以解決；若持續失敗，請檢查網路或稍後再試。")
            if st.button("🔄 重新嘗試", type="primary"):
                ss.gate_retry = True
                st.rerun()
            st.stop()
        ss.gate_retry = False
        with st.status("連線到資料來源…", expanded=True) as status:
            def progress(step, msg):
                icon = {"quote": "🔴", "history": "📈", "aux": "🧩"}.get(step, "⏳")
                st.write(f"{icon} {msg}")
            try:
                t0 = time.perf_counter()
                md = market.load_market(progress)
            except Exception as e:  # noqa: BLE001
                status.update(label="取得失敗", state="error")
                ss.gate_error = str(e)
                st.rerun()
            status.update(label=f"完成（{time.perf_counter() - t0:.1f} 秒）", state="complete")
        ss.md = md
        ss.gate_error = None
        st.rerun()


@st.fragment(run_every="60s")
def live_price_bar():
    md: market.MarketData = ss.md
    q = md.live_quote
    if (datetime.now(timezone.utc) - q.fetched_at).total_seconds() > 50:
        try:
            market.refresh_quote(md)
            q = md.live_quote
        except Exception as e:  # noqa: BLE001
            st.toast(f"即時報價更新失敗，保留上一筆：{str(e)[:80]}", icon="⚠️")
    last = md.last_close
    chg = (q.price / last - 1) * 100
    color = ui.UP if chg >= 0 else ui.DOWN
    c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1.4])
    with c1:
        st.markdown(f'<div class="muted"><span class="dot"></span>&nbsp; 期貨即時 · {ui.esc(q.source)}</div>'
                    f'<div class="live"><span class="p">${q.price:,.3f}</span>'
                    f'<span style="color:{color};font-weight:600">{chg:+.2f}%</span>'
                    f'<span class="muted">vs 最新收盤 ${last:,.4f}</span></div>', unsafe_allow_html=True)
    c2.metric("報價時間", q.time.astimezone().strftime("%m/%d %H:%M"), help="報價本身的時間（你的時區）")
    c3.metric("資料天數", f"{len(md.frame):,}", help=f"{md.frame.index[0]:%Y-%m-%d} ～ {md.frame.index[-1]:%Y-%m-%d}")
    real = sum(s.real for s in md.status)
    c4.metric("資料來源", f"{real}/{len(md.status)} 真實", help="其餘以中性值補位，詳見「因子與資料」分頁")


# ══════════════════════════════════════════════════════════════
# 執行訓練／預測（含即時進度）
# ══════════════════════════════════════════════════════════════
def execute(mode: str, p: TrainParams, factors: dict, box=None):
    box = box or st.container()
    with box:
        prog = st.progress(0.0, text="準備中…")
        c1, c2 = st.columns([3, 2])
        chart_ph = c1.empty()
        log_ph = c2.empty()
    logs: list[str] = []
    last = {"t": 0.0}

    def on_status(m):
        logs.append(m)
        log_ph.markdown("\n".join(f"- {x}" for x in logs[-8:]))

    def on_epoch(ep, loss, res):
        now = time.perf_counter()
        if now - last["t"] < 0.25 and ep + 1 < p.epochs:
            return
        last["t"] = now
        eps = res.epochs_per_sec
        eta = (p.epochs - ep - 1) / eps if eps else 0
        prog.progress(min((ep + 1) / p.epochs, 1.0),
                      text=f"Epoch {ep + 1}/{p.epochs} · loss {loss:.5f} · 最佳 {res.best_loss:.5f} · {eps:.1f} epochs/s · 剩餘約 {eta:.0f} 秒")
        chart_ph.plotly_chart(ui.loss_chart(res.loss_history, res.best_epoch), width="stretch",
                              key=f"loss_{ep}")

    r = pipeline.run(ss.md, p, factors, mode, on_epoch=on_epoch, on_status=on_status)
    prog.progress(1.0, text=f"完成 · 共 {r.seconds:.1f} 秒")
    ss.result = r
    return r


# ══════════════════════════════════════════════════════════════
# 分頁：總覽
# ══════════════════════════════════════════════════════════════
def tab_overview():
    r: pipeline.RunResult | None = ss.get("result")
    md: market.MarketData = ss.md
    if r is None:
        champ = agent.champion()
        st.markdown('<div class="card"><h4 style="margin-top:0">還沒有預測結果</h4>'
                    '<p class="muted">選一種方式開始：</p></div>', unsafe_allow_html=True)
        st.write("")
        c1, c2, c3 = st.columns(3)
        with c1.container(border=True):
            st.markdown("**⚡ 立即預測**")
            st.caption("用目前的冠軍模型直接產出 T+N 判讀，幾秒鐘完成。")
            if st.button("用冠軍模型預測", disabled=not pipeline.champion_path().exists() and not pipeline.best_path().exists(),
                         width="stretch", help=None if champ else "還沒有訓練過模型"):
                ss.pending = ("predict", None)
                st.rerun()
        with c2.container(border=True):
            st.markdown("**🚀 一鍵智慧流程**")
            st.caption("AI 搜尋最佳參數 → 完整訓練 → 預測。第一次使用建議選這個。")
            if st.button("開始一鍵流程", type="primary", width="stretch"):
                ss.pending = ("auto", None)
                st.rerun()
        with c3.container(border=True):
            st.markdown("**🧠 自己調參數**")
            st.caption("到上方「🧠 訓練」分頁選演算法、因子與訓練週期，再按「開始」。")
        st.plotly_chart(ui.price_chart(md.frame.index, md.prices, None, [], [], live_price=md.live_quote.price,
                                       show_days=250), width="stretch")
        return

    up, pct = r.t1_up, r.t1_pct
    vol = float(md.frame["price"].pct_change().tail(60).std() * 100)
    if np.isfinite(vol) and vol > 0 and abs(pct) > 3 * vol:
        st.warning(f"這次預測的明日漲跌 **{pct:+.2f}%**，超過近 60 日每日波動（約 ±{vol:.1f}%）的 3 倍，可信度很低。"
                   + ("建議在「訓練」分頁把預測目標改成「對數報酬率」再訓練一次。" if r.params.pred_target != "logreturn" else "建議重新訓練或換一種模型。"),
                   icon="⚠️")
    left, right = st.columns([1, 2.2])
    with left:
        st.markdown(ui.signal_card(up, pct, float(r.future_prices[0]), r.t1_confidence,
                                   f"{r.future_dates[0]:%m/%d}（T+1）"), unsafe_allow_html=True)
    with right:
        items = []
        prev = r.last_close
        for i, (d, pr) in enumerate(zip(r.future_dates, r.future_prices)):
            ch = (pr / prev - 1) * 100
            conf = f" · 信心 {abs(r.future_probs[i] - .5) * 200:.0f}%" if r.future_probs is not None else ""
            items.append((f"T+{i + 1} · {d:%m/%d}{conf}", f"${pr:,.3f}  {ch:+.2f}%", ui.UP if ch >= 0 else ui.DOWN))
            prev = pr
        st.markdown(ui.chips(items), unsafe_allow_html=True)
        st.write("")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("方向準確率（全期）", ui.fmt_pct(r.da), help="模型對歷史每一天漲跌方向的命中率（含訓練資料，會偏樂觀）")
        hd = r.challenger.get("holdout_da")
        m2.metric("保留集方向準確率", ui.fmt_pct(hd),
                  delta=None if not r.champion else f"{hd - r.champion['holdout_da']:+.2f} vs 冠軍",
                  help="最後一段完全沒參與訓練的資料上的命中率，比全期數字更可信")
        m3.metric("平均誤差 MAPE", ui.fmt_pct(r.mape))
        m4.metric("交叉驗證 DA", ui.fmt_pct(r.cv["da"]) if r.cv else "—", help="只有選擇「時序交叉驗證」時才會計算")
        badge = "🏆 已晉升為冠軍" if r.promoted else ("🛡️ 未晉升" if r.mode != "predict" else f"📦 {r.model_source}")
        st.caption(f"{badge}　{r.verdict}　·　{r.params.algorithm} · {config.PRED_TARGETS[r.params.pred_target]} · "
                   f"lookback {r.params.lookback} · {sum(r.factors.values()) + 1} 個因子 · 用時 {r.seconds:.1f} 秒")
    days = st.segmented_control("顯示範圍", ["3 個月", "6 個月", "1 年", "全部"], default="6 個月",
                                label_visibility="collapsed")
    n = {"3 個月": 66, "6 個月": 130, "1 年": 252, "全部": 100_000}.get(days or "6 個月", 130)
    st.plotly_chart(ui.price_chart(r.dates, r.actual, r.fitted, r.future_dates, r.future_prices, r.holdout_start,
                                   r.last_close, md.live_quote.price, n), width="stretch")
    st.caption("⚠️ 模型輸出僅供研究參考，不構成投資建議。")


# ══════════════════════════════════════════════════════════════
# 分頁：訓練
# ══════════════════════════════════════════════════════════════
def settings_form() -> tuple[TrainParams, dict]:
    p: TrainParams = ss.params
    st.markdown("##### 模型")
    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("演算法", config.ALGORITHMS, index=config.ALGORITHMS.index(p.algorithm),
                        help="BiGRU 是預設；Transformer / TimesFM 對長 lookback 較有利")
    targets = list(config.PRED_TARGETS)
    target = c2.selectbox("預測目標", targets, index=targets.index(p.pred_target),
                          format_func=lambda k: config.PRED_TARGETS[k],
                          help="價格迴歸：預測下一日價格；方向分類：預測漲跌機率；雙頭：兩者兼具；對數報酬率：預測報酬率")
    lookback = c3.slider("Lookback（天）", 5, 60, p.lookback, 1, help="每個樣本看過去幾天的資料")
    c1, c2, c3 = st.columns(3)
    layers = c1.select_slider("隱藏單元", [16, 32, 48, 64, 96, 128], value=p.layers)
    future = c2.slider("預測天數 T+N", 1, 10, p.future_days)
    years = c3.select_slider("使用資料年數", [1.0, 1.5, 2.0, 3.0, 4.0, 5.0], value=float(p.history_years))
    st.markdown("##### 訓練")
    c1, c2, c3, c4 = st.columns(4)
    epochs = c1.number_input("Epochs", 10, 3000, p.epochs, 10)
    bs = c2.select_slider("Batch size", [8, 16, 32, 64, 128], value=p.batch_size)
    lr = c3.select_slider("學習率", [0.0003, 0.0005, 0.001, 0.002, 0.003, 0.005], value=p.lr)
    patience = c4.number_input("早停 patience（0=關）", 0, 500, p.early_stop_patience, 5,
                               help="連續 N 個 epoch 沒有改善就停止")
    c1, c2, c3, c4 = st.columns(4)
    lam = c1.slider("方向懲罰 α", 0.0, 1.0, float(p.directional_lambda), 0.05,
                    help="越高越重視漲跌方向；0 = 純 Huber（只看價格誤差）")
    delta = c2.select_slider("Huber δ", [0.5, 1.0, 1.5, 2.0], value=p.huber_delta)
    val = c3.selectbox("驗證方式", ["none", "kfold"], index=0 if p.validation_mode == "none" else 1,
                       format_func=lambda v: {"none": "保留集（預設）", "kfold": "＋時序交叉驗證"}[v])
    k = c4.slider("交叉驗證折數", 2, 8, p.k_folds, disabled=val != "kfold")
    seed_on = st.checkbox("固定亂數種子（結果可重現）", value=p.seed is not None)
    seed = st.number_input("種子", 0, 10**6, p.seed or 42) if seed_on else None

    st.markdown("##### 因子")
    groups: dict[str, list] = {}
    for f in config.FACTORS[1:]:
        groups.setdefault(f.group, []).append(f.key)
    sel: dict[str, bool] = {}
    for g, keys in groups.items():
        chosen = st.pills(g, keys, selection_mode="multi", default=[k_ for k_ in keys if ss.factors.get(k_, True)],
                          format_func=lambda k_: config.FACTOR_NAME[k_])
        for k_ in keys:
            sel[k_] = k_ in (chosen or [])
    st.caption(f"已選 {sum(sel.values()) + 1} 個特徵（白銀期貨價格固定納入）")
    newp = TrainParams(algorithm=algo, pred_target=target, lookback=int(lookback), layers=int(layers),
                       epochs=int(epochs), batch_size=int(bs), lr=float(lr), directional_lambda=float(lam),
                       huber_delta=float(delta), future_days=int(future), history_years=float(years),
                       validation_mode=val, k_folds=int(k), early_stop_patience=int(patience),
                       seed=int(seed) if seed is not None else None)
    return newp, sel


def tab_train():
    with st.container(border=True):
        p, f = settings_form()
    if p != ss.params or f != ss.factors:
        ss.params, ss.factors = p, f
        persist()
    st.write("")
    c1, c2 = st.columns([2, 1])
    mode = c1.radio("執行方式", list(pipeline.MODES), format_func=pipeline.MODES.get, horizontal=True,
                    help="重新訓練：從零開始；接續訓練：從上次最佳權重繼續；只預測：用冠軍模型，不訓練")
    go = c2.button("▶ 開始", type="primary", width="stretch")
    st.caption("訓練中如果要中止，按右上角的 Stop 或重新整理頁面即可（本次結果不會寫入）。")
    if go:
        try:
            r = execute(mode, p, f)
        except Exception as e:  # noqa: BLE001
            st.error(f"執行失敗：{e}", icon="❌")
        else:
            ss.flash = (f"完成！T+1 {'▲ 看漲' if r.t1_up else '▼ 看跌'} {r.t1_pct:+.2f}%（${r.future_prices[0]:,.3f}），"
                        "完整判讀在「📊 總覽」分頁。")
            st.rerun()      # 讓「總覽」分頁也拿到新結果
    elif ss.get("result") and ss.result.train:
        t = ss.result.train
        with st.expander(f"上次訓練：{t.epochs_run} epochs · {t.seconds:.1f} 秒 · 最佳 loss {t.best_loss:.6f}", expanded=True):
            st.plotly_chart(ui.loss_chart(t.loss_history, t.best_epoch), width="stretch")
            st.markdown("\n".join(f"- {m}" for m in ss.result.messages))
            cv = ss.result.cv
            if cv and cv["folds"]:
                st.dataframe(pd.DataFrame(cv["folds"]).rename(columns={"fold": "折", "da": "DA%", "mape": "MAPE%",
                                                                        "n_train": "訓練筆數", "n_test": "測試筆數"}),
                             hide_index=True, width="stretch")


# ══════════════════════════════════════════════════════════════
# 分頁：AI 搜尋
# ══════════════════════════════════════════════════════════════
def run_search_ui(cfg: SearchConfig, then_train: bool):
    md: market.MarketData = ss.md
    frame = md.recent(ss.params.history_years).frame
    prog = st.progress(0.0, text="搜尋中…")
    table_ph = st.empty()
    rows = []

    def on_trial(i, t, best):
        if t:
            rows.append(t.row())
        df = pd.DataFrame(rows)
        if not df.empty:
            sort_col = "驗證 DA%" if cfg.metric == "da" else "驗證 Loss"
            df = df.sort_values(sort_col, ascending=cfg.metric != "da")
            table_ph.dataframe(df, hide_index=True, width="stretch")
        msg = f"試驗 {i + 1}/{cfg.max_trials}"
        if best:
            msg += f" · 目前最佳 #{best.idx}：DA {best.da:.1f}%、loss {best.loss:.4f}"
        prog.progress((i + 1) / cfg.max_trials, text=msg)

    trials = run_search(frame, ss.params, ss.factors, cfg, on_trial=on_trial)
    b = best_trial(trials, cfg.metric)
    ss.search_trials = trials
    if not b:
        st.warning("沒有有效的試驗結果。")
        return
    st.success(f"最佳組合 #{b.idx}：{b.params.algorithm}、lookback {b.params.lookback}、單元 {b.params.layers}、"
               f"驗證 DA {b.da:.1f}%", icon="🏆")
    if then_train:
        newp = TrainParams.from_dict({**b.params.to_dict(), "epochs": ss.params.epochs, "seed": ss.params.seed,
                                      "future_days": ss.params.future_days, "history_years": ss.params.history_years,
                                      "early_stop_patience": ss.params.early_stop_patience or 30})
        ss.params, ss.factors = newp, {**ss.factors, **b.factors}
        persist()
        st.info("已套用最佳參數，開始完整訓練…", icon="🧠")
        r = execute("train_new", newp, ss.factors)
        ss.flash = f"一鍵流程完成！T+1 {'▲ 看漲' if r.t1_up else '▼ 看跌'} {r.t1_pct:+.2f}%（${r.future_prices[0]:,.3f}）。"
        st.rerun()


def tab_search():
    st.markdown('<p class="muted">每個試驗都只用前 80% 資料訓練，用最後 20% 驗證來排名，避免挑到只是背熟訓練資料的組合。</p>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    strategy = c1.selectbox("策略", ["random", "bayesian", "grid"],
                            format_func=lambda s: {"random": "隨機", "bayesian": "簡化貝葉斯", "grid": "預設網格"}[s])
    metric = c2.selectbox("排名依據", ["da", "loss"], format_func=lambda s: {"da": "驗證方向準確率", "loss": "驗證 Loss"}[s])
    trials = c3.slider("試驗次數", 3, 60, 15)
    ep = c4.slider("每次 epochs", 10, 200, 60, 10)
    cfg = SearchConfig(strategy=strategy, metric=metric, max_trials=trials, search_epochs=ep)
    b1, b2 = st.columns(2)
    if b1.button("🔍 只搜尋", width="stretch"):
        run_search_ui(cfg, then_train=False)
    elif b2.button("🚀 搜尋 → 套用 → 訓練 → 預測", type="primary", width="stretch"):
        run_search_ui(cfg, then_train=True)
    elif ss.get("search_trials"):
        trials_ = ss.search_trials
        df = pd.DataFrame([t.row() for t in trials_])
        st.dataframe(df.sort_values("驗證 DA%", ascending=False), hide_index=True, width="stretch")
        b = best_trial(trials_, "da")
        if b and st.button(f"套用 #{b.idx} 的參數到訓練設定"):
            ss.params = TrainParams.from_dict({**b.params.to_dict(), "epochs": ss.params.epochs})
            ss.factors = {**ss.factors, **b.factors}
            persist()
            st.toast("已套用，請到「訓練」分頁開始訓練", icon="✅")


# ══════════════════════════════════════════════════════════════
# 分頁：回測
# ══════════════════════════════════════════════════════════════
def tab_backtest():
    path = pipeline.champion_path() if pipeline.champion_path().exists() else pipeline.best_path()
    if not path.exists():
        st.info("需要先訓練一個模型才能回測。", icon="ℹ️")
        return
    with st.form("bt"):
        c1, c2, c3, c4 = st.columns(4)
        ratio = c1.slider("測試區間（最後 %）", 5, 80, 30) / 100
        cap = c2.number_input("初始資金", 1000.0, 1e9, 100_000.0, 1000.0)
        lev = c3.slider("槓桿", 0.5, 5.0, 1.0, 0.5)
        pos = c4.slider("每次使用資金比例", 0.1, 1.0, 1.0, 0.1)
        c1, c2, c3, c4 = st.columns(4)
        fee = c1.number_input("單邊手續費 (bps)", 0.0, 100.0, 2.0)
        slip = c2.number_input("單邊滑價 (bps)", 0.0, 100.0, 2.0)
        thr = c3.number_input("進場門檻（預測報酬 %）", 0.0, 5.0, 0.0, 0.05) / 100
        short = c4.checkbox("允許放空", True)
        submitted = st.form_submit_button("執行回測", type="primary", width="stretch")
    if not submitted:
        st.caption(f"將使用：{'冠軍模型' if 'champion' in path.name else '最近一次訓練的模型'}。"
                   "回測區間若與訓練資料重疊，結果會偏樂觀。")
        return
    m, meta = load_model(path)
    mp = TrainParams.from_dict(meta.get("params", {}))
    factors = {**config.default_factors(), **meta.get("factors", {})}
    ds = build_dataset(ss.md.recent(mp.history_years).frame, mp.lookback, factors)
    try:
        bt = run_backtest(m, ds, BacktestConfig(ratio, cap, lev, pos, fee, slip, thr, short))
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        return
    k = st.columns(6)
    k[0].metric("總報酬", ui.fmt_pct(bt.total_return * 100), delta=f"{(bt.total_return - bt.buy_hold_return) * 100:+.2f}% vs 買進持有")
    k[1].metric("買進持有", ui.fmt_pct(bt.buy_hold_return * 100))
    k[2].metric("最大回撤", ui.fmt_pct(bt.max_drawdown * 100))
    k[3].metric("Sharpe", f"{bt.sharpe:.2f}")
    k[4].metric("交易次數", f"{bt.trade_count}")
    k[5].metric("勝率", ui.fmt_pct(bt.win_rate * 100, 1))
    st.plotly_chart(ui.equity_chart(bt.equity, bt.buy_hold), width="stretch")
    if not bt.trades.empty:
        st.dataframe(bt.trades.style.format({"進場": "{:.3f}", "出場": "{:.3f}", "預測價": "{:.3f}",
                                             "預測報酬%": "{:+.2f}", "損益": "{:,.0f}", "權益": "{:,.0f}"}),
                     hide_index=True, width="stretch", height=300)
        st.download_button("下載交易明細 CSV", bt.trades.to_csv(index=False).encode("utf-8-sig"),
                           "backtest_trades.csv", "text/csv")


# ══════════════════════════════════════════════════════════════
# 分頁：因子與資料
# ══════════════════════════════════════════════════════════════
def tab_factors():
    md: market.MarketData = ss.md
    st.markdown("##### 資料來源狀態")
    st.dataframe(pd.DataFrame([{"項目": s.name, "狀態": "✅ 真實" if s.real else "⚠️ 補位", "來源": s.source,
                                "說明": s.detail} for s in md.status]),
                 hide_index=True, width="stretch")
    ex = md.extras
    c1, c2 = st.columns(2)
    if ex.get("fear_greed"):
        c1.metric("CNN Fear & Greed", f"{ex['fear_greed']['score']:.0f}", ex["fear_greed"].get("rating", ""),
                  delta_color="off", delta_arrow="off")
    if ex.get("stocktwits"):
        c2.metric("StockTwits 多方比例", ui.fmt_pct(ex["stocktwits"]["bullish_pct"], 1),
                  f"{ex['stocktwits']['tagged']} 則標記訊息", delta_color="off", delta_arrow="off")
    st.markdown("##### 各因子最新值與近 90 天走勢")
    tail = md.frame.tail(90)
    rows = [{"因子": f.name, "分類": f.group, "最新值": float(md.frame[f.key].iloc[-1]),
             "近 90 天": tail[f.key].round(4).tolist()} for f in config.FACTORS]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=560,
                 column_config={"最新值": st.column_config.NumberColumn(format="%.4f"),
                                "近 90 天": st.column_config.LineChartColumn(width="medium")})
    r = ss.get("result")
    if r and r.influence:
        st.markdown("##### 因子敏感度（最近一次預測）")
        unit = "機率百分點" if r.params.pred_target == "direction" else "USD"
        st.plotly_chart(ui.influence_chart(r.influence, unit), width="stretch")


# ══════════════════════════════════════════════════════════════
# 分頁：代理紀錄
# ══════════════════════════════════════════════════════════════
def tab_agent():
    agent.resolve_predictions({d.date().isoformat(): float(v) for d, v in ss.md.frame["price"].items()})
    champ = agent.champion()
    perf = agent.recent_performance()
    retrain, why = agent.should_retrain(perf)
    c1, c2, c3 = st.columns(3)
    if champ:
        c1.metric("現任冠軍", f"{champ['algorithm']} · LB {champ['lookback']}",
                  f"保留集 DA {champ['holdout_da']:.2f}%" if champ["holdout_da"] is not None else None, delta_color="off", delta_arrow="off")
    else:
        c1.metric("現任冠軍", "尚無")
    c2.metric("近期實測方向準確率", ui.fmt_pct(perf["da"], 1) if perf["n"] else "—", f"{perf['n']} 筆已對帳", delta_color="off", delta_arrow="off")
    c3.metric("實測平均誤差", ui.fmt_pct(perf["mape"]) if perf["n"] else "—")
    (st.warning if retrain else st.info)(why, icon="🔁" if retrain else "🧭")
    t1, t2, t3 = st.tabs(["預測日誌", "模型註冊表", "執行紀錄"])
    with t1:
        df = agent.table("predictions")
        if df.empty:
            st.caption("還沒有預測紀錄。每次預測都會記下 T+1…T+N 的判斷，等實際收盤價出來後自動對帳。")
        else:
            df["結果"] = np.where(df["resolved"] == 0, "⏳ 待對帳", np.where(df["correct"] == 1, "✅ 命中", "❌ 未中"))
            show = df[["target_date", "horizon", "basis_price", "pred_price", "actual_price", "結果", "confidence",
                       "algorithm", "made_at"]].rename(columns={
                "target_date": "目標日", "horizon": "T+", "basis_price": "基準價", "pred_price": "預測價",
                "actual_price": "實際價", "confidence": "信心%", "algorithm": "演算法", "made_at": "預測時間"})
            st.dataframe(show, hide_index=True, width="stretch",
                         column_config={k_: st.column_config.NumberColumn(format="%.3f") for k_ in ["基準價", "預測價", "實際價"]})
    with t2:
        df = agent.table("models")
        st.dataframe(df.drop(columns=["params", "factors"], errors="ignore"), hide_index=True, width="stretch")
    with t3:
        st.dataframe(agent.table("runs"), hide_index=True, width="stretch")
    with st.expander("危險區"):
        if st.button("清除所有代理紀錄與模型", type="secondary"):
            agent.clear_all()
            for pth in (pipeline.best_path(), pipeline.champion_path()):
                pth.unlink(missing_ok=True)
            ss.pop("result", None)
            st.toast("已清除", icon="🗑️")
            st.rerun()


# ══════════════════════════════════════════════════════════════
# 主程式
# ══════════════════════════════════════════════════════════════
init_settings()
if "md" not in ss:
    gate()
    st.stop()

top_l, top_r = st.columns([4, 1])
with top_l:
    header()
with top_r:
    st.write("")
    if st.button("🔄 重新讀取資料", width="stretch", help="重新抓取全部歷史與因子（會再次檢查即時報價）"):
        ss.pop("md", None)
        st.rerun()
live_price_bar()
st.divider()
flash_box = st.container()          # 位置固定：有沒有訊息都佔同一格，分頁才不會被重設回第一頁
if ss.get("flash"):
    flash_box.success(ss.pop("flash"), icon="✅")

pending = ss.pop("pending", None)
tabs = st.tabs(["📊 總覽", "🧠 訓練", "🤖 AI 搜尋", "💹 回測", "🧩 因子與資料", "🏆 代理紀錄"])
with tabs[0]:
    if pending:
        kind, _ = pending
        try:
            if kind == "predict":
                execute("predict", ss.params, ss.factors)
            elif kind == "auto":
                run_search_ui(SearchConfig(max_trials=10, search_epochs=40), then_train=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"執行失敗：{e}", icon="❌")
    tab_overview()
with tabs[1]:
    tab_train()
with tabs[2]:
    tab_search()
with tabs[3]:
    tab_backtest()
with tabs[4]:
    tab_factors()
with tabs[5]:
    tab_agent()

"""UI 共用元件：樣式、圖表、卡片。"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

UP, DOWN, METAL, FAINT, ACCENT = "#3DD68C", "#FF6B6B", "#E3EAEF", "#7C8B96", "#9CC3DA"

CSS = """
<style>
:root { --up:#3DD68C; --down:#FF6B6B; --edge:#22303A; --faint:#7C8B96; --card:#141C22; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1400px; }
div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
.ag-badge { display:inline-flex; align-items:center; justify-content:center; width:42px; height:42px; border-radius:10px;
  background:linear-gradient(160deg,#E8F0F5,#8FA5B3); color:#0B1014; font-weight:800; font-family:monospace; font-size:1.05rem; }
.ag-title { font-size:1.35rem; font-weight:700; color:#F2F6F8; line-height:1.15; margin:0; }
.ag-sub { color:var(--faint); font-size:.82rem; margin:0; }
.card { background:var(--card); border:1px solid var(--edge); border-radius:14px; padding:18px 20px; }
.signal { border-radius:14px; padding:22px 24px; border:1px solid var(--edge); height:100%; }
.signal.up { background:linear-gradient(135deg, rgba(61,214,140,.16), rgba(61,214,140,.03)); }
.signal.down { background:linear-gradient(135deg, rgba(255,107,107,.16), rgba(255,107,107,.03)); }
.signal .lbl { color:var(--faint); font-size:.8rem; letter-spacing:.04em; }
.signal .dir { font-size:2.6rem; font-weight:800; line-height:1.05; margin-top:6px; }
.signal .pct { font-size:1.05rem; font-variant-numeric: tabular-nums; margin-top:2px; }
.signal .px { font-size:1.6rem; color:#F2F6F8; font-variant-numeric: tabular-nums; margin-top:14px; }
.bar { height:6px; border-radius:6px; background:#22303A; overflow:hidden; margin-top:6px; }
.bar > div { height:100%; border-radius:6px; }
.chips { display:flex; gap:10px; flex-wrap:wrap; }
.chip { border:1px solid var(--edge); border-radius:12px; padding:10px 14px; min-width:120px; background:var(--card); }
.chip .d { color:var(--faint); font-size:.75rem; }
.chip .v { font-size:1.05rem; font-variant-numeric: tabular-nums; color:#F2F6F8; }
.live { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
.live .p { font-size:1.9rem; font-weight:700; color:#F2F6F8; font-variant-numeric: tabular-nums; }
.dot { width:9px; height:9px; border-radius:50%; background:var(--down); display:inline-block; animation:pulse 1.6s infinite; }
@keyframes pulse { 0%{opacity:1} 50%{opacity:.25} 100%{opacity:1} }
.pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.74rem; border:1px solid var(--edge); color:var(--faint); }
.pill.ok { color:var(--up); border-color:rgba(61,214,140,.4); }
.pill.warn { color:#F5C26B; border-color:rgba(245,194,107,.4); }
.muted { color:var(--faint); font-size:.85rem; }
.step { display:flex; gap:10px; align-items:flex-start; padding:6px 0; }
</style>
"""


def esc(s) -> str:
    return html.escape(str(s))


def fmt_pct(v, digits=2, sign=False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.{digits}f}%" if sign else f"{v:.{digits}f}%"


def base_layout(fig: go.Figure, height: int = 420, legend=True) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", showlegend=legend,
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        font=dict(color="#C9D3DA"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#22303A")
    fig.update_yaxes(gridcolor="#1B252C", zeroline=False)
    return fig


def price_chart(dates, actual, fitted, future_dates, future_prices, holdout_start: int | None = None,
                last_close: float | None = None, live_price: float | None = None, show_days: int = 180) -> go.Figure:
    dates = pd.DatetimeIndex(dates)
    n = min(show_days, len(dates))
    d, a = dates[-n:], np.asarray(actual)[-n:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=a, name="實際收盤", line=dict(color=METAL, width=2)))
    if fitted is not None:
        fig.add_trace(go.Scatter(x=d, y=np.asarray(fitted)[-n:], name="模型擬合",
                                 line=dict(color=ACCENT, width=1.4, dash="dot"), opacity=.85))
    if holdout_start is not None and holdout_start < len(dates):
        hs = dates[holdout_start]
        if hs >= d[0]:
            fig.add_vrect(x0=hs, x1=d[-1], fillcolor="#9CC3DA", opacity=0.06, line_width=0,
                          annotation_text="保留集（未參與訓練）", annotation_position="top left",
                          annotation_font=dict(size=11, color=FAINT))
    if len(future_prices):
        fx = [d[-1]] + list(pd.to_datetime(future_dates))
        fy = [a[-1]] + list(future_prices)
        up = future_prices[0] >= a[-1]
        fig.add_trace(go.Scatter(x=fx, y=fy, name="預測", mode="lines+markers",
                                 line=dict(color=UP if up else DOWN, width=2.5),
                                 marker=dict(size=8, symbol="diamond")))
    if live_price is not None:
        fig.add_hline(y=live_price, line=dict(color="#FF6B6B", width=1, dash="dash"),
                      annotation_text=f"即時 {live_price:.3f}", annotation_position="bottom right",
                      annotation_font=dict(size=11, color="#FF9B9B"))
    fig = base_layout(fig, 430)
    fig.update_xaxes(rangeslider=dict(visible=False))
    return fig


def loss_chart(losses: list[float], best_epoch: int | None = None) -> go.Figure:
    fig = go.Figure(go.Scatter(y=losses, x=list(range(1, len(losses) + 1)), name="訓練 Loss",
                               line=dict(color="#A78BFA", width=2)))
    if best_epoch is not None and best_epoch >= 0 and best_epoch < len(losses):
        fig.add_trace(go.Scatter(x=[best_epoch + 1], y=[losses[best_epoch]], mode="markers", name="最佳",
                                 marker=dict(color=UP, size=10, symbol="diamond")))
    fig = base_layout(fig, 260)
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(type="log", title="Loss")
    return fig


def influence_chart(infl: dict[str, float], unit: str = "USD") -> go.Figure:
    s = pd.Series(infl).sort_values()
    s = pd.concat([s.head(8), s.tail(8)]).drop_duplicates() if len(s) > 16 else s
    fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                           marker_color=[UP if v >= 0 else DOWN for v in s.values],
                           hovertemplate="%{y}: %{x:.4f} " + unit + "<extra></extra>"))
    fig = base_layout(fig, max(260, 26 * len(s) + 60), legend=False)
    fig.update_xaxes(title=f"因子 +5% 對 T+1 預測的影響（{unit}）", zeroline=True, zerolinecolor="#33434F")
    return fig


def equity_chart(equity: pd.Series, buy_hold: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="策略權益", line=dict(color=ACCENT, width=2.2),
                             fill="tozeroy", fillcolor="rgba(156,195,218,.06)"))
    fig.add_trace(go.Scatter(x=buy_hold.index, y=buy_hold.values, name="買進持有", line=dict(color=FAINT, width=1.4, dash="dot")))
    fig = base_layout(fig, 360)
    lo = min(equity.min(), buy_hold.min())
    hi = max(equity.max(), buy_hold.max())
    fig.update_yaxes(range=[lo * 0.97, hi * 1.03])
    return fig


def signal_card(up: bool, pct: float, price: float, conf: float | None, horizon_label: str = "明日 (T+1)") -> str:
    color = UP if up else DOWN
    conf_html = ""
    if conf is not None:
        conf_html = (f'<div style="margin-top:16px"><div style="display:flex;justify-content:space-between" class="lbl">'
                     f'<span>模型信心</span><span>{conf:.0f}%</span></div>'
                     f'<div class="bar"><div style="width:{min(conf, 100):.0f}%;background:{color}"></div></div></div>')
    return (f'<div class="signal {"up" if up else "down"}"><div class="lbl">{esc(horizon_label)} 判讀</div>'
            f'<div class="dir" style="color:{color}">{"▲ 看漲" if up else "▼ 看跌"}</div>'
            f'<div class="pct" style="color:{color}">{pct:+.2f}%</div>'
            f'<div class="px">${price:,.4f}</div>{conf_html}</div>')


def chips(items: list[tuple[str, str, str | None]]) -> str:
    out = []
    for d, v, color in items:
        out.append(f'<div class="chip"><div class="d">{esc(d)}</div><div class="v" style="color:{color or METAL}">{esc(v)}</div></div>')
    return '<div class="chips">' + "".join(out) + "</div>"

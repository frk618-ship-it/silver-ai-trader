/* Silver AI Trader v2 — 圖表（Lightweight Charts + 輕量 SVG/Canvas）*/
(function () {
'use strict';
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const LC = () => window.LightweightCharts;
const toTime = (d) => d;   // 'YYYY-MM-DD' 字串可直接當 BusinessDay 使用

function baseOptions(el) {
  return {
    autoSize: true,
    layout: { background: { type: 'solid', color: 'transparent' }, textColor: css('--text-2'), fontFamily: css('--font'), fontSize: 12 },
    grid: { vertLines: { visible: false }, horzLines: { color: css('--border') } },
    rightPriceScale: { borderVisible: false, scaleMargins: { top: 0.12, bottom: 0.08 } },
    timeScale: { borderVisible: false, rightOffset: 4, fixLeftEdge: true },
    crosshair: { mode: 1, vertLine: { color: css('--text-3'), labelBackgroundColor: css('--surface-3') },
                 horzLine: { color: css('--text-3'), labelBackgroundColor: css('--surface-3') } },
    localization: { locale: 'zh-TW', priceFormatter: (p) => '$' + p.toFixed(3) },
    handleScroll: { vertTouchDrag: false },
  };
}

const charts = new Map();
function destroy(el) {
  const c = charts.get(el);
  if (c) { try { c.remove(); } catch (_) {} charts.delete(el); }
}

/**
 * 價格圖：實際收盤（面積）、模型擬合（虛線）、未來預測（粗線＋菱形）、即時價（水平線）、保留集起點（標記）
 */
function priceChart(el, o) {
  destroy(el);
  if (!LC()) { el.innerHTML = '<div class="empty faint">圖表元件載入失敗</div>'; return null; }
  const chart = LC().createChart(el, baseOptions(el));
  charts.set(el, chart);
  const up = o.futurePrices && o.futurePrices.length && o.futurePrices[0] >= o.actual[o.actual.length - 1];
  const actual = chart.addAreaSeries({
    lineColor: css('--text'), lineWidth: 2, topColor: css('--accent-soft'), bottomColor: 'transparent',
    priceLineVisible: false, lastValueVisible: true, title: '',
  });
  actual.setData(o.dates.map((d, i) => ({ time: toTime(d), value: o.actual[i] })));
  if (o.fitted) {
    const fit = chart.addLineSeries({ color: css('--accent-2'), lineWidth: 1.5, lineStyle: 2, priceLineVisible: false, lastValueVisible: false });
    fit.setData(o.dates.map((d, i) => ({ time: toTime(d), value: o.fitted[i] })).filter(p => Number.isFinite(p.value)));
  }
  if (o.futurePrices && o.futurePrices.length) {
    const col = up ? css('--up') : css('--down');
    const fc = chart.addLineSeries({ color: col, lineWidth: 3, priceLineVisible: false, lastValueVisible: true,
      pointMarkersVisible: true, pointMarkersRadius: 4 });
    const last = o.dates.length - 1;
    fc.setData([{ time: toTime(o.dates[last]), value: o.actual[last] },
      ...o.futureDates.map((d, i) => ({ time: toTime(d), value: o.futurePrices[i] }))]);
  }
  if (Number.isFinite(o.live)) {
    actual.createPriceLine({ price: o.live, color: css('--down'), lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: '即時' });
  }
  if (o.holdStart != null && o.dates[o.holdStart]) {
    actual.setMarkers([{ time: toTime(o.dates[o.holdStart]), position: 'belowBar', color: css('--text-3'), shape: 'arrowUp', text: '保留集起點' }]);
  }
  setRange(el, o.range || 130, o.dates.length + (o.futureDates ? o.futureDates.length : 0));
  return chart;
}

function setRange(el, bars, total) {
  const c = charts.get(el);
  if (!c) return;
  if (!bars || bars >= total) c.timeScale().fitContent();
  else c.timeScale().setVisibleLogicalRange({ from: total - bars, to: total + 2 });
}

function equityChart(el, curve, bh) {
  destroy(el);
  if (!LC()) return null;
  const chart = LC().createChart(el, { ...baseOptions(el), localization: { locale: 'zh-TW', priceFormatter: (p) => Math.round(p).toLocaleString() } });
  charts.set(el, chart);
  const eq = chart.addAreaSeries({ lineColor: css('--accent'), lineWidth: 2, topColor: css('--accent-soft'), bottomColor: 'transparent', priceLineVisible: false });
  eq.setData(curve.map(p => ({ time: p.date, value: p.value })));
  const b = chart.addLineSeries({ color: css('--text-3'), lineWidth: 1.5, lineStyle: 2, priceLineVisible: false });
  b.setData(bh.map(p => ({ time: p.date, value: p.value })));
  chart.timeScale().fitContent();
  return chart;
}

/** 訓練 loss（對數尺度，Canvas，每次更新整張重畫，成本極低）*/
function lossPlot(canvas, losses, bestEpoch) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext('2d');
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const vals = losses.filter(v => Number.isFinite(v) && v > 0);
  if (vals.length < 2) return;
  const lo = Math.log10(Math.min(...vals)), hi = Math.log10(Math.max(...vals));
  const span = Math.max(hi - lo, 1e-6);
  const X = (i) => 4 + (i / (losses.length - 1)) * (w - 8);
  const Y = (v) => 6 + (1 - (Math.log10(v) - lo) / span) * (h - 12);
  g.strokeStyle = css('--border'); g.lineWidth = 1;
  for (let k = 1; k < 4; k++) { const y = (h * k) / 4; g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke(); }
  const grad = g.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, css('--accent-2')); grad.addColorStop(1, css('--accent'));
  g.strokeStyle = grad; g.lineWidth = 2; g.lineJoin = 'round'; g.beginPath();
  losses.forEach((v, i) => { if (v > 0) (i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))); });
  g.stroke();
  if (bestEpoch != null && bestEpoch >= 0 && losses[bestEpoch] > 0) {
    g.fillStyle = css('--up'); g.beginPath(); g.arc(X(bestEpoch), Y(losses[bestEpoch]), 4, 0, Math.PI * 2); g.fill();
  }
}

/** 迷你走勢圖（SVG 字串）*/
function sparkline(values, color = 'var(--accent)') {
  const v = values.filter(Number.isFinite);
  if (v.length < 2) return '<svg class="spark"></svg>';
  const lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1;
  const pts = v.map((x, i) => `${(i / (v.length - 1) * 100).toFixed(2)},${(30 - (x - lo) / span * 26).toFixed(2)}`).join(' ');
  return `<svg class="spark" viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">
    <polyline points="${pts}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>`;
}

/** 因子敏感度橫條（HTML 字串）*/
function influenceBars(infl, unit) {
  const e = Object.entries(infl || {}).filter(([, v]) => Number.isFinite(v));
  if (!e.length) return '<p class="faint">還沒有資料。</p>';
  e.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const top = e.slice(0, 12);
  const mx = Math.max(...top.map(([, v]) => Math.abs(v))) || 1;
  return `<div class="stack" style="gap:8px">${top.map(([k, v]) => {
    const w = Math.abs(v) / mx * 50;
    const up = v >= 0;
    return `<div style="display:grid;grid-template-columns:minmax(120px,190px) 1fr 84px;gap:10px;align-items:center;font-size:13px">
      <span class="muted" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${k}">${k}</span>
      <div style="position:relative;height:10px;background:var(--surface-2);border-radius:99px">
        <i style="position:absolute;top:0;bottom:0;left:${up ? 50 : 50 - w}%;width:${w}%;border-radius:99px;background:var(--${up ? 'up' : 'down'})"></i>
        <i style="position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--border-2)"></i></div>
      <span class="mono r ${up ? 'up' : 'down'}" style="text-align:right">${v >= 0 ? '+' : ''}${v.toFixed(unit === '%' ? 2 : 4)}${unit === '%' ? '' : ''}</span></div>`;
  }).join('')}</div>`;
}

window.SC = { priceChart, setRange, equityChart, lossPlot, sparkline, influenceBars, destroy };
})();

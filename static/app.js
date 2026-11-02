/* IoT Botnet Detection — frontend (ไม่ต้องใช้ไลบรารีภายนอก กราฟวาดด้วย SVG) */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const BASE = ["RF", "XGB", "AE"];
const ENS = ["Mean", "Weighted", "Stacking", "Vote"];
const COLOR = { RF: "var(--s-rf)", XGB: "var(--s-xgb)", AE: "var(--s-ae)", ENS: "var(--s-ens)",
  benign: "var(--s-benign)", attack: "var(--s-attack)" };
const methodColor = (m) => COLOR[m] || COLOR.ENS;
const LEVELS = [
  { k: 0, label: "ปกติ", icon: "✓", color: "var(--good)" },
  { k: 1, label: "เฝ้าระวัง", icon: "!", color: "var(--warning)" },
  { k: 2, label: "แจ้งเตือน", icon: "✕", color: "var(--critical)" },
];
const METHOD_TH = { RF: "Random Forest", XGB: "XGBoost", AE: "Autoencoder", Mean: "เฉลี่ยเท่ากัน",
  Weighted: "ถ่วงน้ำหนัก", Stacking: "Stacking (LR)", Vote: "โหวต 2 ใน 3" };

const state = { status: null, metrics: null, catalog: null, src: "demo", upFile: null };

/* ---------------------------------------------------------------- utils */
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fInt = (v) => (v == null ? "–" : Math.round(v).toLocaleString("en-US"));
const fPct = (v, d = 2) => (v == null ? "–" : (v * 100).toFixed(d) + "%");
const fNum = (v, d = 4) => (v == null ? "–" : Number(v).toFixed(d));
const fCompact = (v) => (Math.abs(v) >= 1e6 ? (v / 1e6).toFixed(1) + "M" : Math.abs(v) >= 1e3 ? (v / 1e3).toFixed(v >= 1e4 ? 0 : 1) + "k" : String(Math.round(v * 100) / 100));
const attackTH = (a) => (a === "benign" ? "ปกติ (benign)" : a);
const devName = (d) => {
  const e = state.catalog?.devices.find((x) => x.device === d);
  return e ? e.name.replace(/_/g, " ") : `อุปกรณ์ ${d}`;
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch { /* ignore */ }
  if (!r.ok) throw new Error(body?.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `HTTP ${r.status}`);
  return body;
}

/* ---------------------------------------------------------------- tooltip */
const tip = $("#tooltip");
function showTip(evt, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
  if (y + h > window.innerHeight - 8) y = evt.clientY - h - pad;
  tip.style.left = x + "px";
  tip.style.top = y + "px";
}
const hideTip = () => { tip.hidden = true; };
const tipRow = (color, name, val) => `<div class="r"><span>${color ? `<i style="background:${color}"></i>` : ""}${esc(name)}</span><b>${val}</b></div>`;

/* ---------------------------------------------------------------- SVG helpers */
const NS = "http://www.w3.org/2000/svg";
function S(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function newSvg(el, minH) {
  el.innerHTML = "";
  const W = Math.max(el.clientWidth, 260), H = Math.max(el.clientHeight || minH || 230, minH || 0);
  const svg = S("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H }, el);
  return { svg, W, H };
}
function niceTicks(min, max, n = 5) {
  if (max === min) max = min + 1;
  const span = max - min, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10;
  const out = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(12));
  return out;
}
const logTicks = (lo, hi) => { const t = []; for (let e = Math.ceil(Math.log10(lo) - 1e-9); e <= Math.floor(Math.log10(hi) + 1e-9); e++) t.push(Math.pow(10, e)); return t; };
const fmtPow = (v) => (v >= 1 && v < 1e4 ? fCompact(v) : v >= 1e4 ? "1e" + Math.round(Math.log10(v)) : "1e" + Math.round(Math.log10(v)));
/** bar path: ปลายมน 4px ด้านข้อมูล ฐานตรงติดเส้นฐาน */
function barPath(x, y0, w, y1, r = 4) {
  const h = y0 - y1;
  if (h <= 0.5) return `M${x},${y0}h${w}`;
  r = Math.min(r, w / 2, h);
  return `M${x},${y0}V${y1 + r}Q${x},${y1} ${x + r},${y1}H${x + w - r}Q${x + w},${y1} ${x + w},${y1 + r}V${y0}Z`;
}
function hbarPath(x0, y, x1, h, r = 4) {
  const w = x1 - x0;
  if (w <= 0.5) return `M${x0},${y}v${h}`;
  r = Math.min(r, h / 2, w);
  return `M${x0},${y}H${x1 - r}Q${x1},${y} ${x1},${y + r}V${y + h - r}Q${x1},${y + h} ${x1 - r},${y + h}H${x0}Z`;
}

/* ---------------------------------------------------------------- vertical bar chart */
function barChart(el, { labels, values, color, fmt = fInt, yMin = 0, yMax, divider, colors }) {
  const { svg, W, H } = newSvg(el);
  const rot = (W - 54) / labels.length < 64;   // แคบ: เอียงป้ายแกน x กันชนกัน
  const m = { t: 20, r: 8, b: rot ? 50 : 30, l: 46 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const vmax = yMax ?? Math.max(...values, 1e-9) * 1.12;
  const ticks = niceTicks(yMin, vmax, 4).filter((t) => t >= yMin - 1e-12 && t <= vmax + 1e-12);
  const y = (v) => m.t + ih - ((Math.max(v, yMin) - yMin) / (vmax - yMin)) * ih;
  const tickFmt = vmax - yMin < 0.01 ? (v) => v.toFixed(4) : vmax <= 1 ? (v) => v.toFixed(2) : fCompact;
  ticks.forEach((t) => {
    S("line", { x1: m.l, x2: W - m.r, y1: y(t), y2: y(t), class: "gridline" }, svg);
    S("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end" }, svg).textContent = tickFmt(t);
  });
  const band = iw / labels.length, bw = Math.min(42, band * 0.62);
  labels.forEach((lab, i) => {
    const cx = m.l + band * i + band / 2, v = values[i];
    const c = colors ? colors[i] : color;
    S("path", { d: barPath(cx - bw / 2, y(yMin), bw, y(v)), style: `fill:${c}` }, svg);
    S("text", { x: cx, y: y(v) - 5, "text-anchor": "middle", class: "val" }, svg).textContent = fmt(v);
    const ly = H - m.b + 16;
    S("text", rot ? { x: cx + 4, y: ly - 4, "text-anchor": "end", transform: `rotate(-35 ${cx + 4} ${ly - 4})` }
      : { x: cx, y: ly, "text-anchor": "middle" }, svg).textContent = lab;
    const hit = S("rect", { x: cx - band / 2, y: m.t, width: band, height: ih, class: "hit" }, svg);
    hit.addEventListener("mousemove", (e) => showTip(e, `<div class="t">${esc(lab)}</div>${tipRow(c, METHOD_TH[lab] || "", fmt(v))}`));
    hit.addEventListener("mouseleave", hideTip);
  });
  S("line", { x1: m.l, x2: W - m.r, y1: y(yMin), y2: y(yMin), class: "baseline" }, svg);
  if (divider) {
    const x = m.l + band * divider;
    S("line", { x1: x, x2: x, y1: m.t - 8, y2: H - m.b, class: "divider" }, svg);
  }
}

/* ---------------------------------------------------------------- horizontal bar chart */
function hbarChart(el, { labels, values, colors, color, fmt = fInt, max, labelW = 130, rowH = 24, sub }) {
  const H = labels.length * rowH + 14;
  el.style.height = H + "px";
  const { svg, W } = newSvg(el, H);
  const m = { l: labelW, r: 64 };
  const vmax = max ?? Math.max(...values, 1e-9);
  const x = (v) => m.l + (Math.max(v, 0) / vmax) * (W - m.l - m.r);
  labels.forEach((lab, i) => {
    const yy = 6 + i * rowH, bh = rowH - 10;
    const c = colors ? colors[i] : color;
    S("text", { x: m.l - 8, y: yy + bh / 2 + 4, "text-anchor": "end" }, svg).textContent = lab;
    S("rect", { x: m.l, y: yy, width: W - m.l - m.r, height: bh, rx: 3, style: "fill:var(--surface-2)" }, svg);
    S("path", { d: hbarPath(m.l, yy, x(values[i]), bh), style: `fill:${c}` }, svg);
    S("text", { x: x(values[i]) + 6, y: yy + bh / 2 + 4, class: "val" }, svg).textContent = fmt(values[i]);
    const hit = S("rect", { x: 0, y: yy - 4, width: W, height: rowH, class: "hit" }, svg);
    hit.addEventListener("mousemove", (e) => showTip(e, `<div class="t">${esc(lab)}</div>${tipRow(c, sub ? sub(i) : "", fmt(values[i]))}`));
    hit.addEventListener("mouseleave", hideTip);
  });
  S("line", { x1: m.l, x2: m.l, y1: 2, y2: H - 4, class: "baseline" }, svg);
}

/* ---------------------------------------------------------------- line chart */
function lineChart(el, { series, xLog, yLog, xDomain, yDomain, xTitle, yTitle, xFmt, yFmt, tipTitle }) {
  const { svg, W, H } = newSvg(el);
  const m = { t: 12, r: 14, b: 40, l: 52 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const allX = series.flatMap((s) => s.x), allY = series.flatMap((s) => s.y).filter((v) => !yLog || v > 0);
  const [x0, x1] = xDomain || [Math.min(...allX), Math.max(...allX)];
  let [y0, y1] = yDomain || [Math.min(...allY), Math.max(...allY)];
  if (yLog && !yDomain) { y0 = Math.pow(10, Math.floor(Math.log10(y0))); y1 = Math.pow(10, Math.ceil(Math.log10(y1))); }
  const tx = xLog ? (v) => Math.log10(Math.max(v, x0)) : (v) => v;
  const ty = yLog ? (v) => Math.log10(Math.max(v, y0)) : (v) => v;
  const X = (v) => m.l + ((tx(v) - tx(x0)) / (tx(x1) - tx(x0))) * iw;
  const Y = (v) => m.t + ih - ((ty(v) - ty(y0)) / (ty(y1) - ty(y0))) * ih;
  const xt = xLog ? logTicks(x0, x1) : niceTicks(x0, x1, 6);
  const yt = yLog ? logTicks(y0, y1) : niceTicks(y0, y1, 5);
  yt.forEach((t) => {
    S("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "gridline" }, svg);
    S("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, svg).textContent = (yFmt || (yLog ? fmtPow : (v) => +v.toFixed(3)))(t);
  });
  xt.forEach((t) => {
    S("text", { x: X(t), y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent = (xFmt || (xLog ? fmtPow : (v) => +v.toFixed(3)))(t);
  });
  S("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, class: "baseline" }, svg);
  if (xTitle) S("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = xTitle;
  if (yTitle) { const t = S("text", { x: 12, y: m.t + ih / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${m.t + ih / 2})` }, svg); t.textContent = yTitle; }
  series.forEach((s) => {
    const d = s.x.map((xv, i) => `${i ? "L" : "M"}${X(Math.min(Math.max(xv, x0), x1)).toFixed(1)},${Y(Math.min(Math.max(s.y[i], y0), y1)).toFixed(1)}`).join("");
    S("path", { d, fill: "none", style: `stroke:${s.color};stroke-width:${s.width || 2};${s.dash ? "stroke-dasharray:6 4;" : ""}stroke-linejoin:round;stroke-linecap:round` }, svg);
  });
  // crosshair
  const vline = S("line", { y1: m.t, y2: m.t + ih, class: "hover-line", visibility: "hidden" }, svg);
  const dots = series.map((s) => S("circle", { r: 4, style: `fill:${s.color};stroke:var(--surface);stroke-width:2`, visibility: "hidden" }, svg));
  const hit = S("rect", { x: m.l, y: m.t, width: iw, height: ih, class: "hit" }, svg);
  const ref = series[0];
  hit.addEventListener("mousemove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    let best = 0, bd = Infinity;
    ref.x.forEach((xv, i) => { if (xv < x0 || xv > x1) return; const d = Math.abs(X(xv) - px); if (d < bd) { bd = d; best = i; } });
    const xv = ref.x[best];
    vline.setAttribute("x1", X(xv)); vline.setAttribute("x2", X(xv)); vline.setAttribute("visibility", "visible");
    series.forEach((s, k) => { dots[k].setAttribute("cx", X(s.x[best])); dots[k].setAttribute("cy", Y(Math.max(s.y[best], y0))); dots[k].setAttribute("visibility", "visible"); });
    showTip(e, `<div class="t">${tipTitle ? tipTitle(xv) : xv}</div>` + series.map((s) => tipRow(s.color, s.name, (s.fmt || ((v) => fNum(v, 4)))(s.y[best]))).join(""));
  });
  hit.addEventListener("mouseleave", () => { hideTip(); vline.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); });
}

/* ---------------------------------------------------------------- histogram (grouped bars per bin) */
function histChart(el, { edges, series, yLog, threshold }) {
  const { svg, W, H } = newSvg(el);
  const m = { t: 18, r: 10, b: 38, l: 48 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const nb = edges.length - 1;
  const maxC = Math.max(1, ...series.flatMap((s) => s.counts));
  const y0 = yLog ? 0.8 : 0, y1 = yLog ? Math.pow(10, Math.ceil(Math.log10(maxC * 1.2))) : maxC * 1.1;
  const ty = yLog ? (v) => Math.log10(Math.max(v, y0)) : (v) => v;
  const Y = (v) => m.t + ih - ((ty(v) - ty(y0)) / (ty(y1) - ty(y0))) * ih;
  const X = (v) => m.l + ((v - edges[0]) / (edges[nb] - edges[0])) * iw;
  (yLog ? logTicks(1, y1) : niceTicks(0, y1, 4)).forEach((t) => {
    S("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "gridline" }, svg);
    S("text", { x: m.l - 6, y: Y(t) + 4, "text-anchor": "end" }, svg).textContent = yLog ? fmtPow(t) : fCompact(t);
  });
  niceTicks(0, 1, 5).forEach((t) => { S("text", { x: X(t), y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent = t.toFixed(1); });
  S("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle" }, svg).textContent = "calibrated P(attack)";
  const bin = iw / nb, gap = 2, bw = Math.max(1, (bin - gap * (series.length + 1)) / series.length);
  for (let i = 0; i < nb; i++) {
    series.forEach((s, k) => {
      const c = s.counts[i];
      if (c > 0) S("path", { d: barPath(X(edges[i]) + gap + k * (bw + gap), Y(y0), bw, Y(c), Math.min(2, bw / 2)), style: `fill:${s.color}` }, svg);
    });
    const hit = S("rect", { x: X(edges[i]), y: m.t, width: bin, height: ih, class: "hit" }, svg);
    hit.addEventListener("mousemove", (e) => showTip(e, `<div class="t">คะแนน ${edges[i].toFixed(3)}–${edges[i + 1].toFixed(3)}</div>` + series.map((s) => tipRow(s.color, s.name, fInt(s.counts[i]))).join("")));
    hit.addEventListener("mouseleave", hideTip);
  }
  S("line", { x1: m.l, x2: W - m.r, y1: Y(y0), y2: Y(y0), class: "baseline" }, svg);
  if (threshold != null) {
    const x = X(Math.min(Math.max(threshold, 0), 1));
    S("line", { x1: x, x2: x, y1: m.t - 6, y2: m.t + ih, class: "thr" }, svg);
    const t = S("text", { x: x + (x > W - 120 ? -6 : 6), y: m.t + 2, class: "thr-label", "text-anchor": x > W - 120 ? "end" : "start" }, svg);
    t.textContent = `threshold ${threshold.toFixed(3)}`;
  }
}

/* sequential heat (blue) 0..7 */
function heatClass(v, lo, hi) {
  if (v == null) return { bg: "var(--heat-0)", dark: false };
  const k = Math.max(0, Math.min(7, Math.round(((v - lo) / (hi - lo)) * 7)));
  return { bg: `var(--heat-${k})`, dark: k >= 4 };
}

/* ---------------------------------------------------------------- overview */
function bestIdx(rows, key, mode) {
  const vals = rows.map((r) => r[key]).filter((v) => v != null);
  const target = mode === "min" ? Math.min(...vals) : Math.max(...vals);
  return new Set(rows.map((r, i) => (r[key] === target ? i : -1)).filter((i) => i >= 0));
}

function renderOverview() {
  const M = state.metrics, st = state.status;
  const emptyEl = $("#overview-empty"), body = $("#overview-body");
  if (!M) {
    emptyEl.hidden = false; body.hidden = true;
    emptyEl.innerHTML = st?.ready
      ? `โหลดโมเดลแล้ว แต่ยังไม่มีผลวัด (<code>metrics.json</code>) — รัน <code>python train.py</code> เพื่อสร้างกราฟ ส่วนแท็บ "อัปโหลด CSV" ใช้งานได้เลย`
      : `ยังไม่มีโมเดล — เปิด Terminal ในโฟลเดอร์โปรเจกต์แล้วรัน<br><br><code>python train.py</code><br><br>จากนั้นกด <b>รีเฟรช</b> หน้านี้`;
    return;
  }
  emptyEl.hidden = true; body.hidden = false;
  const R = M.results, byM = Object.fromEntries(R.map((r) => [r.Model, r]));
  const def = M.fusion.default_method, best = M.fusion.best_single;
  const methods = R.map((r) => r.Model);

  // KPI
  const dFN = byM[def].FN - byM[best].FN, dFP = byM[def].FP - byM[best].FP;
  const delta = (d) => (d === 0 ? "เท่ากับ" : d < 0 ? `<span class="ok-text">น้อยกว่า ${fInt(-d)}</span> จาก` : `มากกว่า ${fInt(d)} จาก`);
  $("#kpis").innerHTML = `
    <div class="kpi"><div class="label">ข้อมูลที่ใช้ฝึก+ทดสอบ</div><div class="value">${fInt(M.dataset.rows)}<span class="unit">แถว</span></div>
      <div class="note">${M.dataset.features} features · ${M.dataset.files} ไฟล์ · โจมตี ${fPct(M.dataset.attack_rate, 1)}</div></div>
    <div class="kpi"><div class="label">วิธีตัดสินที่ใช้ (เลือกจากชุด calibration)</div><div class="value">${esc(def)}</div>
      <div class="note">F1 บนชุด test = ${fNum(byM[def].F1, 6)}</div></div>
    <div class="kpi"><div class="label">การโจมตีที่พลาด (FN) · ${esc(def)}</div><div class="value">${fInt(byM[def].FN)}</div>
      <div class="note">${delta(dFN)} ${esc(best)} (โมเดลเดี่ยวที่ดีที่สุด)</div></div>
    <div class="kpi"><div class="label">แจ้งเตือนผิด (FP) · ${esc(def)}</div><div class="value">${fInt(byM[def].FP)}</div>
      <div class="note">${delta(dFP)} ${esc(best)} · FPR ${fPct(byM[def].FPR, 3)}</div></div>`;

  // results table
  const cols = [["Accuracy", "max", (v) => fNum(v, 6)], ["Precision", "max", (v) => fNum(v, 6)], ["Recall", "max", (v) => fNum(v, 6)],
    ["F1", "max", (v) => fNum(v, 6)], ["ROC-AUC", "max", (v) => fNum(v, 6)], ["PR-AUC", "max", (v) => fNum(v, 6)],
    ["FPR", "min", (v) => fPct(v, 4)], ["FNR", "min", (v) => fPct(v, 4)], ["FP", "min", fInt], ["FN", "min", fInt]];
  const bests = Object.fromEntries(cols.map(([k, mode]) => [k, bestIdx(R, k, mode)]));
  $("#results-table").innerHTML = `<thead><tr><th>Model</th>${cols.map(([k]) => `<th>${k}</th>`).join("")}</tr></thead><tbody>` +
    R.map((r, i) => `<tr class="${i === 3 ? "sep" : ""}"><td><b>${esc(r.Model)}</b> <span class="tag">${METHOD_TH[r.Model] || ""}</span>${r.Model === def ? ' <span class="pill">ใช้งาน</span>' : ""}</td>` +
      cols.map(([k, , f]) => `<td class="${bests[k].has(i) ? "best" : ""}">${f(r[k])}</td>`).join("") + "</tr>").join("") + "</tbody>";

  // bars
  const barColors = methods.map((m) => methodColor(m));
  barChart($("#chart-fn"), { labels: methods, values: R.map((r) => r.FN), colors: barColors, divider: 3 });
  barChart($("#chart-fp"), { labels: methods, values: R.map((r) => r.FP), colors: barColors, divider: 3 });
  const f1 = R.map((r) => r.F1), lo = Math.min(...f1);
  barChart($("#chart-f1"), { labels: methods, values: f1, colors: barColors, divider: 3, fmt: (v) => v.toFixed(4),
    yMin: Math.max(0, lo - (1 - lo) * 1.5 - 1e-4), yMax: 1 });

  // ROC
  const rocEns = ["Mean", "Weighted", "Stacking"].includes(def) ? def
    : ["Mean", "Weighted", "Stacking"].reduce((a, b) => (byM[b].F1 > byM[a].F1 ? b : a));
  const rocSeries = [...BASE, rocEns].map((m) => ({ name: m, x: M.roc[m].fpr.map((v) => Math.max(v, 1e-5)), y: M.roc[m].tpr,
    color: methodColor(m), dash: !BASE.includes(m), width: BASE.includes(m) ? 2 : 2.5 }));
  lineChart($("#chart-roc"), { series: rocSeries, xLog: true, xDomain: [1e-5, 1], yDomain: [0, 1], xTitle: "False Positive Rate (log)",
    yTitle: "True Positive Rate", yFmt: (v) => v.toFixed(1), tipTitle: (x) => `FPR = ${x < 1e-4 ? x.toExponential(1) : fPct(x, 3)}` });
  $("#legend-roc").innerHTML = [...BASE, rocEns].map((m) => `<span class="key"><i class="${BASE.includes(m) ? "" : "dash"}" style="${BASE.includes(m) ? `background:${methodColor(m)}` : `border-color:${methodColor(m)}`}"></i>${m} (AUC=${fNum(byM[m]["ROC-AUC"], 6)})</span>`).join("");

  // hist + confusion selectors
  fillSelect($("#hist-method"), methods, def);
  fillSelect($("#cm-method"), methods, def);
  const drawHist = () => {
    const m = $("#hist-method").value, h = M.hist.by_method[m];
    histChart($("#chart-hist"), { edges: M.hist.edges, yLog: true, threshold: M.fusion.thresholds[m],
      series: [{ name: "ปกติ", counts: h.benign, color: COLOR.benign }, { name: "โจมตี", counts: h.attack, color: COLOR.attack }] });
  };
  const drawCM = () => {
    const m = $("#cm-method").value, cm = M.confusion[m];
    const mx = Math.max(...cm.flat(), 2);
    const cell = (v, lab) => {
      const k = v <= 0 ? 0 : Math.max(1, Math.round((Math.log10(v) / Math.log10(mx)) * 7));
      return `<div class="cell" style="background:var(--heat-${k});color:${k >= 4 ? "#fff" : "var(--ink)"}">${fInt(v)}<small>${lab}</small></div>`;
    };
    $("#cm").innerHTML = `<div></div><div class="lab">ทำนาย: ปกติ</div><div class="lab">ทำนาย: โจมตี</div>
      <div class="lab">จริง: ปกติ</div>${cell(cm[0][0], "TN ถูก")}${cell(cm[0][1], "FP แจ้งเตือนผิด")}
      <div class="lab">จริง: โจมตี</div>${cell(cm[1][0], "FN พลาด")}${cell(cm[1][1], "TP จับได้")}`;
  };
  $("#hist-method").onchange = drawHist;
  $("#cm-method").onchange = drawCM;
  drawHist(); drawCM();

  // fusion info
  const F = M.fusion;
  const wbars = BASE.map((m) => `<div class="wbar"><span>${m}</span><div class="track"><div class="fill" style="width:${F.weights[m] * 100}%;background:${methodColor(m)}"></div></div><span class="num">${F.weights[m].toFixed(1)}</span></div>`).join("");
  const maxC = Math.max(...BASE.map((m) => Math.abs(F.stacking_coef[m])), 1e-9);
  const cbars = BASE.map((m) => `<div class="wbar"><span>${m}</span><div class="track"><div class="fill" style="width:${Math.abs(F.stacking_coef[m]) / maxC * 100}%;background:${methodColor(m)}"></div></div><span class="num">${F.stacking_coef[m].toFixed(2)}</span></div>`).join("");
  $("#fusion-info").innerHTML = `<div class="fusion-grid">
      <div><h3>น้ำหนัก Weighted</h3>${wbars}</div>
      <div><h3>สัมประสิทธิ์ Stacking</h3>${cbars}</div></div>
    <div class="table-wrap" style="margin-top:12px"><table class="data"><thead><tr><th>วิธี</th><th>threshold</th><th>คะแนนบนชุด calibration (${M.config.threshold_mode === "fpr" ? "recall ที่ FPR ≤ " + fPct(M.config.target_fpr, 1) : "F1"})</th></tr></thead><tbody>
    ${Object.keys(F.thresholds).map((m) => `<tr><td>${m}${m === def ? ' <span class="pill">ใช้งาน</span>' : ""}</td><td>${fNum(F.thresholds[m], 4)}</td><td>${fNum(F.cal_objective?.[m], 6)}</td></tr>`).join("")}
    </tbody></table></div>`;

  // gain table
  $("#gain-table").innerHTML = `<thead><tr><th>Ensemble</th><th>ผิดทั้งหมด (${esc(best)})</th><th>ผิดทั้งหมด (ensemble)</th><th>แก้ได้</th><th>ทำพัง</th><th>ΔFN (พลาด)</th><th>ΔFP (แจ้งเตือนผิด)</th></tr></thead><tbody>` +
    M.gain.map((g) => `<tr><td>${g.Ensemble}</td><td>${fInt(g.wrong_single)}</td><td>${fInt(g.wrong_ensemble)}</td><td>${fInt(g.fixed)}</td><td>${fInt(g.broken)}</td>
      <td class="${g.dFN < 0 ? "ok-text" : ""}">${g.dFN > 0 ? "+" : ""}${fInt(g.dFN)}</td><td class="${g.dFP < 0 ? "ok-text" : ""}">${g.dFP > 0 ? "+" : ""}${fInt(g.dFP)}</td></tr>`).join("") + "</tbody>";

  // per-attack heat
  $("#attack-table").innerHTML = `<thead><tr><th>ชนิด</th><th>n</th>${methods.map((m) => `<th>${m}</th>`).join("")}</tr></thead><tbody>` +
    M.per_attack.map((r) => {
      const ben = r.attack === "benign";
      return `<tr><td>${ben ? "<b>benign</b> <span class='tag'>FPR</span>" : esc(r.attack)}</td><td>${fInt(r.n)}</td>` + methods.map((m) => {
        if (ben) return `<td>${fPct(r[m], 2)}</td>`;
        const h = heatClass(r[m], 0.9, 1.0);
        return `<td class="h ${h.dark ? "dark" : ""}" style="background:${h.bg}">${fPct(r[m], 2)}</td>`;
      }).join("") + "</tr>";
    }).join("") + "</tbody>";

  // unseen
  const U = M.unseen;
  $("#unseen-block").hidden = !U;
  if (U) {
    $("#unseen-title").textContent = `Recall ต่อการโจมตี '${U.holdout_family}' ที่ไม่เคยเห็นตอนฝึก (${fInt(U.rows.unseen_attacks)} แถว)`;
    const um = U.results.map((r) => r.Model);
    barChart($("#chart-unseen-recall"), { labels: um, values: U.results.map((r) => r.Recall), colors: um.map(methodColor), divider: 3, fmt: (v) => fPct(v, 1), yMax: 1.08 });
    barChart($("#chart-unseen-fpr"), { labels: um, values: U.results.map((r) => r.FPR), colors: um.map(methodColor), divider: 3, fmt: (v) => fPct(v, 2) });
  }

  // dataset
  const ba = Object.entries(M.dataset.by_attack);
  hbarChart($("#chart-data"), { labels: ba.map(([k]) => k), values: ba.map(([, v]) => v), colors: ba.map(([k]) => (k === "benign" ? COLOR.benign : COLOR.attack)), labelW: 120 });
  $("#data-hint").innerHTML = `สุ่ม ${fInt(M.config.benign_per_file)} แถว/ไฟล์ปกติ และ ${fInt(M.config.attack_per_file)} แถว/ไฟล์โจมตี · Train ${fInt(M.dataset.split.train)} · Calibration ${fInt(M.dataset.split.cal)} · Test ${fInt(M.dataset.split.test)}`;

  // AE loss
  const h = M.ae_history;
  if (h?.loss?.length) {
    const ep = h.loss.map((_, i) => i + 1);
    const ser = [{ name: "train", x: ep, y: h.loss, color: COLOR.AE }];
    if (h.val_loss) ser.push({ name: "validation", x: ep, y: h.val_loss, color: COLOR.AE, dash: true });
    lineChart($("#chart-ae"), { series: ser, yLog: true, xTitle: "epoch", yTitle: "MSE", xFmt: (v) => String(Math.round(v)), tipTitle: (x) => `epoch ${x}` });
  }
}

/* ---------------------------------------------------------------- selects */
function fillSelect(sel, methods, def) {
  sel.innerHTML = methods.map((m) => `<option value="${m}" ${m === def ? "selected" : ""}>${m} · ${METHOD_TH[m] || ""}${m === def ? " (ค่าเริ่มต้น)" : ""}</option>`).join("");
}

function refreshDetectControls() {
  const cat = state.catalog, src = state.src;
  const devSel = $("#dev-select"), attSel = $("#att-select");
  const prevDev = devSel.value, prevAtt = attSel.value;
  const devs = (cat?.devices || []).filter((d) => d.attacks.some((a) => (src === "demo" ? a.demo_rows > 0 : a.file_id)));
  devSel.innerHTML = (src === "demo" ? `<option value="">ทุกอุปกรณ์</option>` : "") +
    devs.map((d) => `<option value="${d.device}">${d.device} · ${esc(d.name.replace(/_/g, " "))}</option>`).join("");
  if ([...devSel.options].some((o) => o.value === prevDev)) devSel.value = prevDev;
  const updateAtt = () => {
    const d = devs.find((x) => String(x.device) === devSel.value);
    let atts;
    if (d) atts = d.attacks.filter((a) => (src === "demo" ? a.demo_rows > 0 : a.file_id));
    else {
      const seen = new Map();
      devs.forEach((dd) => dd.attacks.forEach((a) => { if (a.demo_rows > 0) seen.set(a.attack, (seen.get(a.attack) || 0) + a.demo_rows); }));
      atts = [...seen.entries()].map(([attack, demo_rows]) => ({ attack, demo_rows })).sort((a, b) => (a.attack === "benign" ? -1 : b.attack === "benign" ? 1 : a.attack.localeCompare(b.attack)));
    }
    attSel.innerHTML = (src === "demo" ? `<option value="">ทุกชนิด (ปน)</option>` : "") +
      atts.map((a) => `<option value="${src === "demo" ? a.attack : a.file_id}">${esc(attackTH(a.attack))}${src === "demo" ? ` · ${fInt(a.demo_rows)} แถว` : ""}</option>`).join("");
    if ([...attSel.options].some((o) => o.value === prevAtt)) attSel.value = prevAtt;
    $("#run-btn").disabled = !state.status?.ready || attSel.options.length === 0;
  };
  devSel.onchange = updateAtt;
  updateAtt();
  const hint = $("#src-hint");
  if (src === "demo") {
    hint.textContent = state.status?.has_demo ? "สุ่มจากชุด Test ที่แยกไว้ตอนฝึก — โมเดลไม่เคยเห็นแถวเหล่านี้ ผลจึงเทียบได้ยุติธรรม"
      : "ยังไม่มีชุดตัวอย่าง (demo_pool.joblib) — รัน python train.py";
  } else {
    hint.textContent = cat?.data_dir ? `อ่านไฟล์ CSV จากโฟลเดอร์ ${cat.data_dir} โดยตรง (ไฟล์ใหญ่อาจใช้เวลาอ่านครั้งแรกสักครู่)`
      : "ยังไม่พบชุดข้อมูล — วางโฟลเดอร์ N-BaIoT ไว้ในโฟลเดอร์ data/ ของโปรเจกต์ แล้วรีสตาร์ตเซิร์ฟเวอร์";
  }
}

/* ---------------------------------------------------------------- result view */
function renderResult(el, data) {
  const s = data.summary, rows = data.rows, method = s.method, truth = !!s.eval;
  const thr = state.status.thresholds;
  const lv = (k) => s.levels[String(k)] || 0;
  const src = data.source || {};
  const srcText = src.type === "demo" ? "ชุดทดสอบ (ไม่เคยใช้ฝึก)" : src.type === "file" ? `ไฟล์ ${esc(src.file)} (${fInt(src.rows_in_file)} แถวในไฟล์)` : src.type === "upload" ? `อัปโหลด ${esc(src.file)}` : "API";
  const ev = truth ? s.eval[method] : null;

  const tiles = [
    `<div class="kpi"><div class="label">แถวที่ตรวจ</div><div class="value">${fInt(s.n)}</div><div class="note">${srcText}</div></div>`,
    ...[2, 1, 0].map((k) => {
      const L = LEVELS[k];
      return `<div class="kpi"><div class="label"><span class="badge l${k}"><span class="ic">${L.icon}</span>${L.label}</span></div><div class="value">${fInt(lv(k))}</div><div class="note">${fPct(lv(k) / s.n, 1)} ของทั้งหมด</div></div>`;
    }),
  ];
  const evalTiles = truth ? `<div class="kpis">
      <div class="kpi"><div class="label">ของจริงในชุดนี้</div><div class="value">${fInt(s.truth.attack)}<span class="unit">โจมตี</span></div><div class="note">${fInt(s.truth.benign)} แถวปกติ</div></div>
      <div class="kpi"><div class="label">จับการโจมตีได้ · ${esc(method)}</div><div class="value">${ev.detection_rate == null ? "–" : fPct(ev.detection_rate, 2)}</div><div class="note">${ev.detection_rate == null ? "ไม่มีแถวโจมตี" : `พลาด ${fInt(ev.FN)} แถว`}</div></div>
      <div class="kpi"><div class="label">แจ้งเตือนผิด · ${esc(method)}</div><div class="value">${ev.fpr == null ? "–" : fPct(ev.fpr, 2)}</div><div class="note">${ev.fpr == null ? "ไม่มีแถวปกติ" : `${fInt(ev.FP)} แถวปกติถูกเตือน`}</div></div>
      <div class="kpi"><div class="label">AE เตือนแต่ RF/XGB ไม่เตือน</div><div class="value">${fInt(s.ae_only)}</div><div class="note">อาจเป็นรูปแบบที่ไม่เคยเห็นตอนฝึก</div></div>
    </div>` : `<div class="kpis"><div class="kpi"><div class="label">AE เตือนแต่ RF/XGB ไม่เตือน</div><div class="value">${fInt(s.ae_only)}</div><div class="note">อาจเป็นรูปแบบที่ไม่เคยเห็นตอนฝึก</div></div></div>`;

  const levelBar = `<div class="levelbar">${[0, 1, 2].filter((k) => lv(k) > 0).map((k) => `<div style="flex:${lv(k)};background:${LEVELS[k].color}" title="${LEVELS[k].label} ${fInt(lv(k))}"></div>`).join("")}</div>`;

  const evalTable = truth ? `<div class="card"><div class="card-head"><h3>เทียบทุกวิธีบนข้อมูลชุดนี้</h3><p class="hint">ใช้ threshold ของแต่ละวิธีที่เลือกไว้ตอนฝึก</p></div>
    <div class="table-wrap"><table class="data"><thead><tr><th>วิธี</th><th>จับได้ (recall)</th><th>แจ้งเตือนผิด (FPR)</th><th>Accuracy</th><th>TP</th><th>FN</th><th>FP</th><th>TN</th></tr></thead><tbody>
    ${Object.entries(s.eval).map(([m, e], i) => `<tr class="${i === 3 ? "sep" : ""}"><td><b>${m}</b>${m === method ? ' <span class="pill">ที่เลือก</span>' : ""}</td><td>${e.detection_rate == null ? "–" : fPct(e.detection_rate, 2)}</td><td>${e.fpr == null ? "–" : fPct(e.fpr, 2)}</td><td>${fPct(e.accuracy, 2)}</td><td>${fInt(e.TP)}</td><td>${fInt(e.FN)}</td><td>${fInt(e.FP)}</td><td>${fInt(e.TN)}</td></tr>`).join("")}
    </tbody></table></div></div>` : "";

  el.innerHTML = `<div class="result">
    ${s.notes?.length ? `<ul class="notes">${s.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}
    <div class="kpis">${tiles.join("")}</div>
    <div class="card" style="padding:10px 16px"><div class="hint">สัดส่วนระดับแจ้งเตือน (วิธีตัดสิน: <b>${esc(method)}</b>, threshold ${fNum(s.threshold, 4)})</div>${levelBar}
      <div class="legend">${LEVELS.map((L) => `<span class="key"><i class="sq" style="background:${L.color}"></i>${L.label} ${fInt(lv(L.k))}</span>`).join("")}</div></div>
    ${evalTiles}
    <div class="grid-2">
      <div class="card"><div class="card-head"><h3>แต่ละวิธีเตือนกี่ %</h3><p class="hint">สัดส่วนแถวที่ถูกตัดสินว่าเป็นการโจมตี</p></div><div class="chart auto" id="r-flag"></div></div>
      <div class="card"><div class="card-head"><h3>การกระจายคะแนน · ${esc(method)}</h3><p class="hint">${truth ? "แยกสีตามประเภทจริง" : "ไม่ทราบประเภทจริง"} · เส้นประ = threshold</p></div><div class="chart" id="r-hist"></div>
        <div class="legend">${truth ? `<span class="key"><i class="sq" style="background:${COLOR.benign}"></i>ปกติ</span><span class="key"><i class="sq" style="background:${COLOR.attack}"></i>โจมตี</span>` : `<span class="key"><i class="sq" style="background:var(--s-rf)"></i>ทุกแถว</span>`}</div></div>
    </div>
    ${evalTable}
    <div class="card">
      <div class="card-head row"><div><h3>รายแถว</h3><p class="hint">แสดง ${fInt(data.rows_returned)} จาก ${fInt(s.n)} แถว · คะแนน = P(โจมตี) หลังปรับสเกล · ตัวหนา = เกิน threshold · คลิกแถวเพื่อดูรายละเอียด</p></div>
        <select class="select-sm r-sort" aria-label="เรียงลำดับ"><option value="orig">ลำดับเดิม</option><option value="risk">ความเสี่ยงสูงก่อน</option><option value="safe">ความเสี่ยงต่ำก่อน</option></select></div>
      <div class="rows-scroll"><table class="data r-rows"></table></div>
      <div class="r-detail" style="margin-top:14px"></div>
    </div>
  </div>`;

  const methods = Object.keys(s.flag_rate);
  hbarChart($("#r-flag", el), { labels: methods, values: methods.map((m) => s.flag_rate[m]), colors: methods.map(methodColor), fmt: (v) => fPct(v, 1), max: 1, labelW: 80,
    sub: (i) => `${fInt(s.flagged[methods[i]])} แถว` });
  const hs = truth ? [{ name: "ปกติ", counts: s.hist.benign, color: COLOR.benign }, { name: "โจมตี", counts: s.hist.attack, color: COLOR.attack }]
    : [{ name: "ทุกแถว", counts: s.hist.all, color: "var(--s-rf)" }];
  histChart($("#r-hist", el), { edges: s.hist.edges, series: hs, yLog: false, threshold: s.threshold });

  const cols = [...BASE, method];
  const scoreCell = (r, m) => {
    const v = r.scores[m], on = r.flags[m] === 1;
    return `<td><span class="scorecell ${on ? "on" : ""}">${v.toFixed(3)}<span class="mini"><b style="width:${Math.max(2, v * 100)}%;background:${methodColor(m)}"></b></span></span></td>`;
  };
  const badge = (k) => `<span class="badge l${k}"><span class="ic">${LEVELS[k].icon}</span>${LEVELS[k].label}</span>`;
  const draw = (order) => {
    let rs = rows.slice();
    if (order === "risk") rs.sort((a, b) => b.level - a.level || b.scores[method] - a.scores[method]);
    if (order === "safe") rs.sort((a, b) => a.level - b.level || a.scores[method] - b.scores[method]);
    $(".r-rows", el).innerHTML = `<thead><tr><th>#</th>${rows[0]?.device != null ? '<th class="l">อุปกรณ์</th>' : ""}${truth ? '<th class="l">ประเภทจริง</th>' : ""}${cols.map((m, i) => `<th>${m}${i === 3 ? " (ตัดสิน)" : ""}</th>`).join("")}<th>ระดับ</th></tr></thead><tbody>` +
      rs.map((r) => `<tr class="clickable" data-i="${r.i}"><td>${r.i + 1}</td>${r.device != null ? `<td class="l">${r.device} · ${esc(devName(r.device))}</td>` : ""}${truth ? `<td class="l">${r.attack ? esc(r.attack) : r.truth ? "โจมตี" : "ปกติ"}</td>` : ""}${cols.map((m) => scoreCell(r, m)).join("")}<td style="text-align:left">${badge(r.level)}${r.ae_only ? ' <span class="tag">AE เท่านั้น</span>' : ""}</td></tr>`).join("") + "</tbody>";
    $$(".r-rows tbody tr", el).forEach((tr) => tr.addEventListener("click", () => {
      $$(".r-rows tr.selected", el).forEach((x) => x.classList.remove("selected"));
      tr.classList.add("selected");
      showDetail(rows[+tr.dataset.i]);
    }));
  };
  const showDetail = (r) => {
    const box = $(".r-detail", el);
    const ms = Object.keys(r.scores);
    box.innerHTML = `<div class="detail"><div><h3>แถว #${r.i + 1} ${badge(r.level)}</h3>
      <p class="hint">${r.device != null ? `อุปกรณ์ ${r.device} · ${esc(devName(r.device))} · ` : ""}${r.attack ? `ประเภทจริง ${esc(r.attack)} · ` : ""}AE reconstruction MSE = ${r.ae_mse.toExponential(3)}</p>
      <table class="data"><thead><tr><th>วิธี</th><th>P(โจมตี)</th><th>threshold</th><th>ผล</th></tr></thead><tbody>
      ${ms.map((m) => `<tr><td>${m}</td><td>${r.scores[m].toFixed(4)}</td><td>${fNum(thr[m], 4)}</td><td>${r.flags[m] ? "<b>โจมตี</b>" : "ปกติ"}</td></tr>`).join("")}</tbody></table></div>
      <div><h3>คะแนนเทียบ threshold</h3><div class="chart auto d-chart"></div></div></div>`;
    const c = $(".d-chart", box);
    hbarChart(c, { labels: ms, values: ms.map((m) => r.scores[m]), colors: ms.map(methodColor), fmt: (v) => v.toFixed(3), max: 1, labelW: 80 });
    // วาดขีด threshold ทับ
    const svg = c.querySelector("svg"), W = +svg.getAttribute("width");
    ms.forEach((m, i) => {
      const x = 80 + Math.min(Math.max(thr[m], 0), 1) * (W - 80 - 64), y = 6 + i * 24;
      S("line", { x1: x, x2: x, y1: y - 3, y2: y + 17, class: "thr" }, svg);
    });
  };
  $(".r-sort", el).onchange = (e) => draw(e.target.value);
  draw("orig");
}

/* ---------------------------------------------------------------- actions */
async function runDetect() {
  const btn = $("#run-btn"), out = $("#detect-result");
  const n = Math.max(1, Math.min(20000, +$("#n-input").value || 300));
  const method = $("#method-select").value;
  const dev = $("#dev-select").value, att = $("#att-select").value;
  let url;
  if (state.src === "demo") {
    const q = new URLSearchParams({ n, method });
    if (dev) q.set("device", dev);
    if (att) q.set("attack", att);
    url = `/api/score/demo?${q}`;
  } else {
    url = `/api/score/file?${new URLSearchParams({ file_id: att, n, method })}`;
  }
  btn.disabled = true; btn.textContent = "กำลังตรวจ…";
  out.innerHTML = `<div class="spinner">กำลังให้คะแนนด้วย RF, XGBoost และ Autoencoder…</div>`;
  try {
    renderResult(out, await api(url));
  } catch (e) {
    out.innerHTML = `<div class="banner">${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "ตรวจจับ";
  }
}

async function runUpload() {
  const f = state.upFile, out = $("#upload-result"), btn = $("#up-btn");
  if (!f) return;
  const fd = new FormData();
  fd.append("file", f);
  fd.append("method", $("#up-method-select").value);
  fd.append("truth", $("#truth-select").value);
  btn.disabled = true; btn.textContent = "กำลังตรวจ…";
  out.innerHTML = `<div class="spinner">กำลังอัปโหลดและตรวจ ${esc(f.name)} (${(f.size / 1e6).toFixed(1)} MB)…</div>`;
  try {
    renderResult(out, await api("/api/score/upload", { method: "POST", body: fd }));
  } catch (e) {
    out.innerHTML = `<div class="banner">${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false; btn.textContent = "ตรวจไฟล์";
  }
}

function setUploadFile(f) {
  state.upFile = f || null;
  $("#drop-text").textContent = f ? `${f.name} · ${(f.size / 1e6).toFixed(1)} MB` : "ลากไฟล์มาวาง หรือคลิกเพื่อเลือก";
  $("#up-btn").disabled = !f || !state.status?.ready;
}

/* ---------------------------------------------------------------- init */
function switchTab(name) {
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $$(".tab-panel").forEach((p) => { p.hidden = p.id !== `tab-${name}`; });
  if (name === "overview" && state.metrics) renderOverview(); // วาดใหม่ตามความกว้างจริง
  history.replaceState(null, "", `#${name}`);
}

async function init() {
  $$(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  $$("#src-seg button").forEach((b) => b.addEventListener("click", () => {
    $$("#src-seg button").forEach((x) => x.classList.toggle("active", x === b));
    state.src = b.dataset.src;
    refreshDetectControls();
  }));
  $("#run-btn").addEventListener("click", runDetect);
  $("#up-btn").addEventListener("click", runUpload);
  $("#file-input").addEventListener("change", (e) => setUploadFile(e.target.files[0]));
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setUploadFile(e.dataTransfer.files[0]));

  const stEl = $("#status"), stText = $("#status-text"), banner = $("#banner");
  try {
    state.status = await api("/api/status");
  } catch (e) {
    stEl.className = "status bad"; stText.textContent = "เชื่อมต่อเซิร์ฟเวอร์ไม่ได้";
    banner.hidden = false; banner.textContent = `เรียก /api/status ไม่ได้: ${e.message}\nตรวจว่ารัน uvicorn app:app อยู่`;
    return;
  }
  const st = state.status;
  stEl.className = `status ${st.ready ? "ok" : "bad"}`;
  stText.textContent = st.ready ? `โมเดลพร้อม · ${st.n_features} features${st.trained_at ? " · ฝึกเมื่อ " + st.trained_at.replace("T", " ") : ""}` : "ยังไม่มีโมเดล";
  if (!st.ready) { banner.hidden = false; banner.textContent = st.error; }

  const [metrics, catalog] = await Promise.allSettled([api("/api/metrics"), api("/api/catalog")]);
  state.metrics = metrics.status === "fulfilled" ? metrics.value : null;
  state.catalog = catalog.status === "fulfilled" ? catalog.value : { devices: [] };
  if (st.ready) {
    fillSelect($("#method-select"), st.methods, st.default_method);
    fillSelect($("#up-method-select"), st.methods, st.default_method);
  }
  const fileBtn = $('#src-seg button[data-src="file"]');
  if (!state.catalog.devices.some((d) => d.attacks.some((a) => a.file_id))) { fileBtn.disabled = true; fileBtn.title = "ไม่พบไฟล์ในโฟลเดอร์ข้อมูล"; }
  if (!st.has_demo && !fileBtn.disabled) fileBtn.click();
  refreshDetectControls();
  renderOverview();
  const start = location.hash.slice(1);
  if (["overview", "detect", "upload", "api"].includes(start)) switchTab(start);

  let rT;
  window.addEventListener("resize", () => { clearTimeout(rT); rT = setTimeout(() => { if (!$("#tab-overview").hidden && state.metrics) renderOverview(); }, 200); });
}

init();

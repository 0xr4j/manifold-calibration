// Interactive calibration study. Every number on the page is computed here from data.json,
// with the same formulas as metrics.py.
"use strict";

const BINS = [0, .02, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, .98, 1.0];
const HORIZON_NAMES = { o7: "7 days after opening", mid: "Halfway", d30: "30 days before end", d7: "7 days before end", d1: "1 day before end" };
const $ = (id) => document.getElementById(id);
const pct = (x, d = 0) => (100 * x).toFixed(d) + "%";
const cents = (x) => (x >= 0 ? "+" : "−") + Math.abs(100 * x).toFixed(1) + "¢";

// ---------- statistics ----------
const clip = (p) => Math.min(1 - 0.005, Math.max(0.005, p));
const logit = (p) => { p = clip(p); return Math.log(p / (1 - p)); };
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const mean = (a) => a.reduce((s, v) => s + v, 0) / (a.length || 1);

function brier(ps, ys) { let s = 0; for (let i = 0; i < ps.length; i++) s += (ps[i] - ys[i]) ** 2; return s / (ps.length || 1); }

function wilson(k, n, z = 1.96) {
  if (!n) return [NaN, NaN];
  const ph = k / n, d = 1 + z * z / n;
  const c = (ph + z * z / (2 * n)) / d;
  const h = z * Math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d;
  return [c - h, c + h];
}

function calTable(ps, ys, bins = BINS) {
  const rows = [];
  for (let b = 0; b < bins.length - 1; b++) {
    let n = 0, k = 0, sp = 0;
    for (let i = 0; i < ps.length; i++) {
      const p = ps[i];
      const inBin = p >= bins[b] && (p < bins[b + 1] || (b === bins.length - 2 && p <= 1));
      if (inBin) { n++; k += ys[i]; sp += p; }
    }
    if (!n) continue;
    const [lo, hi] = wilson(k, n);
    rows.push({ n, meanPrice: sp / n, rate: k / n, lo, hi });
  }
  return rows;
}

// Logistic regression of outcome on logit(price), fitted with Newton's method (2 parameters).
function calFit(ps, ys) {
  let a = 0, b = 1;
  for (let it = 0; it < 50; it++) {
    let g0 = 0, g1 = 0, h00 = 1e-6, h01 = 0, h11 = 1e-6;
    for (let i = 0; i < ps.length; i++) {
      const x = logit(ps[i]), mu = sigmoid(a + b * x), w = mu * (1 - mu), r = ys[i] - mu;
      g0 += r; g1 += r * x; h00 += w; h01 += w * x; h11 += w * x * x;
    }
    const det = h00 * h11 - h01 * h01;
    if (!isFinite(det) || Math.abs(det) < 1e-12) break;
    const da = (h11 * g0 - h01 * g1) / det, db = (h00 * g1 - h01 * g0) / det;
    a += da; b += db;
    if (Math.abs(da) < 1e-9 && Math.abs(db) < 1e-9) break;
  }
  return { intercept: a, slope: b };
}

function rng(seed) { // mulberry32
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ---------- SVG charts ----------
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function text(parent, x, y, s, attrs = {}) { const t = el("text", { x, y, ...attrs }, parent); t.textContent = s; return t; }

function frame(svg, { x0, y0, w, h, xDomain, yDomain, xTicks, yTicks, xFmt, yFmt, xLabel, yLabel, grid = true }) {
  const sx = (v) => x0 + (v - xDomain[0]) / (xDomain[1] - xDomain[0]) * w;
  const sy = (v) => y0 + h - (v - yDomain[0]) / (yDomain[1] - yDomain[0]) * h;
  const g = el("g", { class: "axis" }, svg);
  for (const t of yTicks || []) {
    if (grid) el("line", { x1: x0, x2: x0 + w, y1: sy(t), y2: sy(t), class: "grid" }, g);
    text(g, x0 - 8, sy(t) + 4, yFmt ? yFmt(t) : t, { "text-anchor": "end" });
  }
  for (const t of xTicks || []) {
    text(g, sx(t), y0 + h + 18, xFmt ? xFmt(t) : t, { "text-anchor": "middle" });
  }
  el("line", { x1: x0, x2: x0 + w, y1: y0 + h, y2: y0 + h }, g);
  if (xLabel) text(svg, x0 + w / 2, y0 + h + 38, xLabel, { class: "label", "text-anchor": "middle" });
  if (yLabel) text(svg, 16, y0 + h / 2, yLabel, { class: "label", "text-anchor": "middle", transform: `rotate(-90 16 ${y0 + h / 2})` });
  return { sx, sy };
}

function calibrationChart(svg, rows, { curve, hist } = {}) {
  svg.innerHTML = "";
  const W = 640, x0 = 58, w = W - x0 - 16;
  const mainH = hist ? 300 : 330;
  const ticks = [0, .2, .4, .6, .8, 1];
  const { sx, sy } = frame(svg, { x0, y0: 12, w, h: mainH, xDomain: [0, 1], yDomain: [0, 1], xTicks: hist ? [] : ticks, yTicks: ticks,
    yFmt: (t) => pct(t), xFmt: (t) => pct(t), xLabel: hist ? null : "Market price (chance of YES)", yLabel: "Share that resolved YES" });
  el("line", { x1: sx(0), y1: sy(0), x2: sx(1), y2: sy(1), class: "diag" }, svg);
  if (curve) {
    let d = "";
    for (let i = 0; i <= 100; i++) { const p = 0.01 + 0.98 * i / 100; d += (i ? "L" : "M") + sx(p).toFixed(1) + "," + sy(curve(p)).toFixed(1); }
    el("path", { d, fill: "none", stroke: "var(--accent)", "stroke-width": 2 }, svg);
  }
  for (const r of rows) {
    const x = sx(r.meanPrice);
    el("line", { x1: x, x2: x, y1: sy(Math.max(0, r.lo)), y2: sy(Math.min(1, r.hi)), stroke: "var(--dot)", "stroke-width": 1.5 }, svg);
    for (const v of [r.lo, r.hi]) el("line", { x1: x - 4, x2: x + 4, y1: sy(Math.min(1, Math.max(0, v))), y2: sy(Math.min(1, Math.max(0, v))), stroke: "var(--dot)", "stroke-width": 1.5 }, svg);
    const c = el("circle", { cx: x, cy: sy(r.rate), r: 5, fill: "var(--dot)" }, svg);
    const t = el("title", {}, c);
    t.textContent = `Priced ${pct(r.meanPrice, 1)} on average · ${r.n} markets · ${pct(r.rate, 1)} resolved YES (95% CI ${pct(Math.max(0, r.lo))}–${pct(Math.min(1, r.hi))})`;
  }
  if (hist) {
    const counts = new Array(50).fill(0);
    for (const p of hist) counts[Math.min(49, Math.floor(p * 50))]++;
    const max = Math.max(1, ...counts);
    const y0 = 12 + mainH + 14, h = 70;
    const { sx: hx, sy: hy } = frame(svg, { x0, y0, w, h, xDomain: [0, 1], yDomain: [0, max], xTicks: ticks, yTicks: [], xFmt: (t) => pct(t),
      xLabel: "Market price (chance of YES)", grid: false });
    counts.forEach((c, i) => el("rect", { x: hx(i / 50) + 0.5, width: Math.max(0, hx(1 / 50) - hx(0) - 1), y: hy(c), height: y0 + h - hy(c), fill: "var(--muted)", opacity: 0.6 }, svg));
    text(svg, x0 + 4, y0 + 10, "where prices sat", { class: "label" });
  }
}

function barChart(svg, groups, series, { yMax, yFmt, yLabel }) {
  svg.innerHTML = "";
  const x0 = 58, y0 = 14, w = 640 - x0 - 16, h = 270;
  const yTicks = [0, yMax / 4, yMax / 2, 3 * yMax / 4, yMax];
  const { sy } = frame(svg, { x0, y0, w, h, xDomain: [0, 1], yDomain: [0, yMax], yTicks, yFmt, yLabel });
  const gw = w / groups.length, bw = Math.min(46, (gw - 20) / series.length);
  groups.forEach((gname, gi) => {
    const cx = x0 + gw * gi + gw / 2;
    series.forEach((s, si) => {
      const v = s.values[gi];
      if (v == null || isNaN(v)) return;
      const x = cx - (series.length * bw) / 2 + si * bw;
      el("rect", { x: x + 2, width: bw - 4, y: sy(v), height: y0 + h - sy(v), fill: s.color, rx: 3 }, svg);
      text(svg, x + bw / 2, sy(v) - 5, yFmt(v), { "text-anchor": "middle", class: "label" });
    });
    text(svg, cx, y0 + h + 20, gname, { "text-anchor": "middle", class: "label" });
  });
}

function segmented(container, options, value, onChange) {
  container.innerHTML = "";
  for (const [v, label] of options) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = label; b.setAttribute("aria-pressed", String(v === value));
    b.addEventListener("click", () => {
      container.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", "false"));
      b.setAttribute("aria-pressed", "true");
      onChange(v);
    });
    container.appendChild(b);
  }
}

// ---------- theme ----------
(function theme() {
  const root = document.documentElement;
  try { const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; } catch (e) { /* storage unavailable */ }
  $("theme").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* ignore */ }
  });
})();

// ---------- Step 1: event contract ----------
function contract() {
  const p = $("c-price").value / 100, q = $("c-belief").value / 100;
  $("c-price-o").textContent = pct(p); $("c-belief-o").textContent = pct(q);
  $("c-cost").textContent = "$" + p.toFixed(2);
  $("c-win").textContent = "+$" + (1 - p).toFixed(2);
  $("c-lose").textContent = "−$" + p.toFixed(2);
  const ev = q - p;
  $("c-ev").textContent = (ev >= 0 ? "+" : "−") + "$" + Math.abs(ev).toFixed(2);
  $("c-ev").className = ev > 0.005 ? "pos" : ev < -0.005 ? "neg" : "";
  $("c-text").textContent = Math.abs(ev) < 0.005
    ? "Your estimate equals the price, so buying either side has zero expected profit."
    : ev > 0
      ? `You think YES is underpriced by ${pct(ev)}. Buying YES earns ${pct(ev)} of $1 per contract on average, if you are right. Over many such bets, that edge adds up.`
      : `You think YES is overpriced by ${pct(-ev)}. Buying NO (at $${(1 - p).toFixed(2)}) earns ${pct(-ev)} of $1 per contract on average, if you are right.`;
}
["c-price", "c-belief"].forEach((id) => $(id).addEventListener("input", contract));
contract();

// ---------- Step 2: calibration simulator ----------
let simSeed = 7;
function simulate() {
  const n = +$("s-n").value, slope = +$("s-slope").value, bias = +$("s-bias").value;
  $("s-n-o").textContent = n.toLocaleString(); $("s-slope-o").textContent = slope.toFixed(2); $("s-bias-o").textContent = (bias >= 0 ? "+" : "") + bias.toFixed(2);
  const r = rng(simSeed);
  const truth = (p) => sigmoid(bias + slope * logit(p));
  const ps = [], qs = [], ys = [];
  for (let i = 0; i < n; i++) {
    const p = 0.02 + 0.96 * r();
    const q = truth(p);
    ps.push(p); qs.push(q); ys.push(r() < q ? 1 : 0);
  }
  calibrationChart($("s-chart"), calTable(ps, ys), { curve: truth });
  const fit = calFit(ps, ys);
  $("s-brier").textContent = brier(ps, ys).toFixed(3);
  $("s-brier-true").textContent = brier(qs, ys).toFixed(3);
  $("s-fit").textContent = fit.slope.toFixed(2);
  const at90 = truth(0.9), at10 = truth(0.1);
  let msg;
  if (Math.abs(slope - 1) < 0.03 && Math.abs(bias) < 0.03) {
    msg = `A fair market. The dots scatter around the dashed line only because of luck; with ${n.toLocaleString()} markets the 95% intervals show how much scatter to expect. Roll again to see it move.`;
  } else {
    msg = `In this market a 90% price really means ${pct(at90)} and a 10% price really means ${pct(at10)}. ` +
      (slope < 1 ? "Prices are too extreme, so long shots are underpriced and heavy favourites overpriced. " :
       slope > 1 ? "Prices are too timid, so favourites are underpriced and long shots overpriced. " : "") +
      `The prices score a Brier of ${brier(ps, ys).toFixed(3)}; the true chances would score ${brier(qs, ys).toFixed(3)}. That gap is the edge a trader could capture.`;
    if (n < 300) msg += " With this few markets, luck can hide the pattern.";
  }
  $("s-text").textContent = msg;
}
["s-n", "s-slope", "s-bias"].forEach((id) => $(id).addEventListener("input", simulate));
$("s-roll").addEventListener("click", () => { simSeed++; simulate(); });
const preset = (slope, bias) => () => { $("s-slope").value = slope; $("s-bias").value = bias; simulate(); };
$("s-reset").addEventListener("click", preset(1, 0));
$("s-over").addEventListener("click", preset(0.6, 0));
$("s-timid").addEventListener("click", preset(1.5, 0));
simulate();

// ---------- real data ----------
let D, ROWS, CUTS;
const state = { h: "o7", liq: "all", early: false };
// The backtest only trades 7 days after opening: every other horizon is counted from the
// resolution time, which a trader cannot know in advance.
const bt = { mode: "band", h: "o7", side: "YES" };

async function load() {
  try {
    const res = await fetch("data.json");
    D = await res.json();
  } catch (e) {
    document.querySelectorAll("#results .panel, #lookahead .panel, #backtest .panel").forEach((p) => {
      p.innerHTML = '<p class="loading">Could not load data.json. Open this page through a web server (for example GitHub Pages), not as a local file.</p>';
    });
    return;
  }
  // Expand snapshots into rows the charts can filter.
  // day = day the market resolved; tday = day the price was read (when a trade would happen).
  ROWS = D.snapshots.map(([mi, hi, p, tday]) => {
    const m = D.markets[mi];
    return { mi, h: D.horizons[hi], p: p / 1000, tday, y: m[2], traders: m[3], day: m[4], early: m[5] };
  });
  const traders = D.markets.map((m) => m[3]).sort((a, b) => a - b);
  CUTS = [traders[Math.floor(traders.length / 3)], traders[Math.floor(2 * traders.length / 3)]];
  hero(); results(); lookahead(); backtest();
}

function hero() {
  const s = D.summary, head = s.horizons[D.headline];
  document.querySelectorAll('[data-k="markets"]').forEach((e) => (e.textContent = s.markets.toLocaleString()));
  document.querySelector('.stat [data-k="skill"]').textContent = pct(head.skill_vs_base_rate);
  document.querySelector('.stat [data-k="slope"]').textContent = head.slope.toFixed(2);
  const ci = s.slope_ci_headline;
  if (ci) document.querySelector('[data-k="slope-ci"]').textContent = `95% CI ${ci[0].toFixed(2)}–${ci[1].toFixed(2)}`;
  document.querySelector('.stat [data-k="early"]').textContent = pct(s.early_resolution.yes);
  document.querySelector('[data-k="early-no"]').textContent = pct(s.early_resolution.no);
  $("f-range").textContent = `Markets created ${s.created_range[0]} – ${s.created_range[1]}`;
}

function liqOk(r) {
  if (state.liq === "all") return true;
  const t = r.traders;
  return state.liq === "0" ? t < CUTS[0] : state.liq === "1" ? t >= CUTS[0] && t < CUTS[1] : t >= CUTS[1];
}

function results() {
  segmented($("r-h"), Object.entries(HORIZON_NAMES), state.h, (v) => { state.h = v; drawResults(); });
  segmented($("r-liq"), [["all", "All"], ["0", `Fewest (<${CUTS[0]})`], ["1", `Middle`], ["2", `Most (${CUTS[1]}+)`]], state.liq, (v) => { state.liq = v; drawResults(); });
  $("r-early").addEventListener("change", (e) => { state.early = e.target.checked; drawResults(); });
  drawResults();
}

function drawResults() {
  const rows = ROWS.filter((r) => r.h === state.h && liqOk(r) && (!state.early || !r.early));
  const ps = rows.map((r) => r.p), ys = rows.map((r) => r.y);
  calibrationChart($("r-chart"), calTable(ps, ys), { hist: ps });
  const base = mean(ys), b = brier(ps, ys), bb = brier(ps.map(() => base), ys), fit = calFit(ps, ys);
  $("r-n").textContent = rows.length.toLocaleString();
  $("r-brier").textContent = b.toFixed(3);
  $("r-base").textContent = bb.toFixed(3);
  $("r-skill").textContent = pct(1 - b / bb);
  $("r-slope").textContent = fit.slope.toFixed(2);
  $("r-hint").textContent = fit.slope < 0.9
    ? `Slope ${fit.slope.toFixed(2)}: these prices were too extreme. Long shots won more often than priced and favourites less.`
    : fit.slope > 1.1
      ? `Slope ${fit.slope.toFixed(2)}: these prices were too timid. Outcomes were more lopsided than the prices said.`
      : `Slope ${fit.slope.toFixed(2)}: close to a fair market at this horizon.`;
  if (state.h !== "o7") $("r-hint").textContent += " This moment is counted back from when the market resolved, which nobody knows in advance (see the look-ahead trap).";
  upsets(rows);
}

function upsets(rows) {
  const fill = (tbody, list) => {
    tbody.innerHTML = "";
    if (!list.length) { tbody.innerHTML = '<tr><td class="loading">None in this selection.</td></tr>'; return; }
    for (const r of list) {
      const m = D.markets[r.mi];
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      const a = document.createElement("a"); a.href = m[1]; a.textContent = m[0]; a.rel = "noopener"; a.target = "_blank";
      td.appendChild(a);
      const num = document.createElement("td"); num.className = "num"; num.textContent = pct(r.p, 1);
      tr.append(td, num); tbody.appendChild(tr);
    }
  };
  const byTraders = (a, b) => b.traders - a.traders;
  fill($("r-upsets-yes"), rows.filter((r) => r.p <= 0.10 && r.y === 1).sort(byTraders).slice(0, 6));
  fill($("r-upsets-no"), rows.filter((r) => r.p >= 0.90 && r.y === 0).sort(byTraders).slice(0, 6));
}

function lookahead() {
  const hs = D.horizons;
  const score = (rows) => brier(rows.map((r) => r.p), rows.map((r) => r.y));
  const all = hs.map((h) => score(ROWS.filter((r) => r.h === h)));
  const onTime = hs.map((h) => score(ROWS.filter((r) => r.h === h && !r.early)));
  const yMax = Math.ceil(Math.max(...all, ...onTime) * 20) / 20;
  barChart($("l-chart"), hs.map((h) => HORIZON_NAMES[h]), [
    { values: all, color: "var(--accent-2)" }, { values: onTime, color: "var(--accent)" }],
    { yMax, yFmt: (v) => v.toFixed(3), yLabel: "Brier score (lower is better)" });
  const e = D.summary.early_resolution;
  barChart($("l-early"), ["Resolved YES", "Resolved NO"], [{ values: [e.yes, e.no], color: "var(--accent)" }],
    { yMax: Math.ceil(Math.max(e.yes, e.no) * 10) / 10 || 1, yFmt: (v) => pct(v), yLabel: "Share resolved while still open" });
  const d1 = hs.indexOf("d1");
  const early = score(ROWS.filter((r) => r.h === "d1" && r.early));
  $("l-text").textContent = `${pct(e.yes)} of YES markets were resolved while trading was still open, against ${pct(e.no)} of NO markets. ` +
    `A day before resolution, markets that ran to their deadline score a Brier of ${onTime[d1].toFixed(3)}: with a day left, the answer is usually settled. ` +
    `Markets resolved while still open score ${early.toFixed(3)}: the event often came as a surprise. ` +
    `Either way, a trader could not have known which clock they were on, so this study's headline and backtest use the price 7 days after opening.`;
}

function backtest() {
  segmented($("b-mode"), [["band", "Price band"], ["fit", "Fitted correction"]], bt.mode, (v) => { bt.mode = v; drawBacktest(); });
  segmented($("b-side"), [["YES", "Buy YES"], ["NO", "Buy NO"]], bt.side, (v) => { bt.side = v; drawBacktest(); });
  ["b-lo", "b-hi", "b-fee", "b-edge"].forEach((id) => $(id).addEventListener("input", drawBacktest));
  drawBacktest();
}

function drawBacktest() {
  const fee = $("b-fee").value / 100;
  $("b-fee-o").textContent = (+$("b-fee").value).toFixed(1) + "¢";
  $("b-band-controls").hidden = bt.mode !== "band";
  $("b-fit-controls").hidden = bt.mode !== "fit";
  const rows = ROWS.filter((r) => r.h === bt.h).sort((a, b) => a.tday - b.tday || a.day - b.day);
  let trades = [];
  if (bt.mode === "band") {
    let lo = $("b-lo").value / 100, hi = $("b-hi").value / 100;
    $("b-lo-o").textContent = pct(lo); $("b-hi-o").textContent = pct(hi);
    if (lo > hi) [lo, hi] = [hi, lo];
    for (const r of rows) if (r.p >= lo && r.p <= hi) trades.push({ r, side: bt.side });
  } else {
    const edge = $("b-edge").value / 100;
    $("b-edge-o").textContent = pct(edge);
    // Walk-forward: on split day S, learn only from markets already resolved, then trade only
    // prices read on or after S. No trade uses an outcome that was unknown when it was placed.
    const split = rows[Math.floor(rows.length / 2)].tday;
    const train = rows.filter((r) => r.day < split), test = rows.filter((r) => r.tday >= split);
    const fit = calFit(train.map((r) => r.p), train.map((r) => r.y));
    const date = new Date(split * 86400000).toLocaleDateString(undefined, { month: "short", year: "numeric" });
    $("b-fit-text").textContent = `On ${date}, learn from the ${train.length.toLocaleString()} markets already resolved: fair chance = ` +
      `sigmoid(${fit.intercept.toFixed(2)} + ${fit.slope.toFixed(2)} × logit(price)). Then trade the ${test.length.toLocaleString()} markets reaching day 7 after that date, when the gap is at least the minimum edge.`;
    for (const r of test) {
      const fair = sigmoid(fit.intercept + fit.slope * logit(r.p));
      if (fair - r.p >= edge && edge > 0 || (edge === 0 && fair > r.p)) trades.push({ r, side: "YES" });
      else if (r.p - fair >= edge && edge > 0 || (edge === 0 && fair < r.p)) trades.push({ r, side: "NO" });
    }
  }
  const profits = trades.map(({ r, side }) => (side === "YES" ? r.y - r.p : r.p - r.y) - fee);
  const costs = trades.map(({ r, side }) => (side === "YES" ? r.p : 1 - r.p) + fee);
  const n = profits.length, avg = mean(profits);
  const sd = Math.sqrt(profits.reduce((s, v) => s + (v - avg) ** 2, 0) / Math.max(1, n - 1));
  const half = 1.96 * sd / Math.sqrt(Math.max(1, n));
  const wins = trades.filter(({ r, side }) => (side === "YES" ? r.y === 1 : r.y === 0)).length;
  $("b-n").textContent = n.toLocaleString();
  $("b-hit").textContent = n ? pct(wins / n) : "–";
  $("b-avg").textContent = n ? `${cents(avg)} (${cents(avg - half)} to ${cents(avg + half)})` : "–";
  $("b-avg").className = n && avg - half > 0 ? "pos" : n && avg + half < 0 ? "neg" : "";
  const staked = costs.reduce((s, v) => s + v, 0), profit = profits.reduce((s, v) => s + v, 0);
  $("b-roi").textContent = n ? (profit >= 0 ? "+" : "") + pct(profit / staked, 1) : "–";
  // Cumulative profit in resolution order.
  const svg = $("b-chart"); svg.innerHTML = "";
  if (!n) { text(svg, 320, 180, "No trades match this rule.", { "text-anchor": "middle", class: "label" }); return; }
  const cum = []; let s = 0; for (const v of profits) { s += v; cum.push(s); }
  const lo = Math.min(0, ...cum), hi = Math.max(0.01, ...cum);
  const pad = (hi - lo) * 0.08;
  const yT = [lo - pad, 0, hi + pad].sort((a, b) => a - b);
  const { sx, sy } = frame(svg, { x0: 64, y0: 14, w: 640 - 64 - 16, h: 280, xDomain: [0, n], yDomain: [lo - pad, hi + pad],
    xTicks: [0, Math.round(n / 2), n], yTicks: yT, yFmt: (v) => (v >= 0 ? "$" : "−$") + Math.abs(v).toFixed(1),
    xLabel: "Trades, in the order they were placed", yLabel: "Cumulative profit ($)" });
  el("line", { x1: sx(0), x2: sx(n), y1: sy(0), y2: sy(0), stroke: "var(--muted)", "stroke-dasharray": "4 4" }, svg);
  let d = `M${sx(0)},${sy(0)}`;
  cum.forEach((v, i) => (d += `L${sx(i + 1).toFixed(1)},${sy(v).toFixed(1)}`));
  el("path", { d, fill: "none", stroke: profit >= 0 ? "var(--good)" : "var(--bad)", "stroke-width": 2 }, svg);
}

load();

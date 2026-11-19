// Section 1: interactive toy 2D guidance visualization.
import { DEFAULT_COMPONENTS, fieldAt, gaussianPairs, guidanceAt, labelOf, perturb, simulate } from './gmm.js';

const R = 3.4;            // world half-width
const N_STEPS = 80;
const N_PART = 400;
const HEAT = 120;         // heatmap resolution
const ARROWS = 15;        // arrow grid per side

const $ = (id) => document.getElementById(id);
const canvas = $('toy-canvas');
const ctx = canvas.getContext('2d');
const chart = $('toy-chart');
const cctx = chart.getContext('2d');
const heat = document.createElement('canvas');
heat.width = heat.height = HEAT;
const hctx = heat.getContext('2d');

const state = {
  method: 'cg', scale: 1.5, tStart: 1, estimated: true, kappa: 0.5,
  bg: 'density', arBase: true, arGuide: true, arTotal: false, particles: true, x0: false, x0kind: 'guided',
  frame: 50, playing: false,   // start mid-way (t ~ 0.38), where contours and guidance are visible hover: null, pinned: -1, x0Path: false,
};
const model = { comps: DEFAULT_COMPONENTS, clsComps: null, condComps: null, estimated: false, target: [-0.2, 2.3] };
const noise = gaussianPairs(N_PART, 7);
const chartNoise = gaussianPairs(200, 99);
let frames = [], finalLabel = [];

const uiFont = () => getComputedStyle(document.body).fontFamily;  // canvas text follows the page font
function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function rgb(hex) {
  const h = hex.replace('#', '');
  const v = h.length === 3 ? h.split('').map((c) => parseInt(c + c, 16)) : [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
  return v;
}
let C = {};
function readColors() {
  C = { ink: css('--ink'), muted: css('--muted'), line: css('--line'), accent: css('--accent'), photo: css('--photo'),
    pencil: css('--pencil'), base: css('--base'), total: css('--total'), card: css('--card'),
    mcg: css('--m-cg'), mcfg: css('--m-cfg'), mtfg: css('--m-tfg'),
    density: css('--density') };   // density heatmap tint (theme-sky); falls back to ink
}

// ---------------------------------------------------------------- model --
function rebuildModel() {
  model.estimated = state.estimated;
  model.clsComps = perturb(DEFAULT_COMPONENTS, state.kappa, 11);
  model.condComps = perturb(DEFAULT_COMPONENTS, state.kappa, 23);
}
function runCfg() { return { method: state.method, scale: state.scale, tStart: state.tStart, nSteps: N_STEPS }; }
function resimulate() {
  frames = simulate(model, runCfg(), noise);
  const last = frames[frames.length - 1];
  finalLabel = Array.from(last.x, (x, i) => labelOf(model.comps, x, last.y[i]) > 0.5);
  draw();
}

// ------------------------------------------------------------- geometry --
let W = 0, H = 0, dpr = 1;
function resize() {
  dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  canvas.width = Math.round(w * dpr); canvas.height = Math.round(w * dpr);
  W = canvas.width; H = canvas.height;
  const cw = chart.clientWidth;
  chart.width = Math.round(cw * dpr); chart.height = Math.round(cw * 0.625 * dpr);
  draw(); drawChart();
}
const px = (x) => ((x + R) / (2 * R)) * W;
const py = (y) => ((R - y) / (2 * R)) * H;
const wx = (p) => (p / W) * 2 * R - R;
const wy = (p) => R - (p / H) * 2 * R;

// ------------------------------------------------------------ guidance --
function guidanceNow(t, x, y, f) {
  const active = state.method !== 'none' && t <= state.tStart + 1e-9 && state.scale !== 0;
  if (!active) return null;
  const g = guidanceAt(model, state.method, t, x, y, f);
  return [state.scale * g[0], state.scale * g[1]];
}
function cgCfgAt(t, x, y) {
  return [guidanceAt(model, 'cg', t, x, y), guidanceAt(model, 'cfg', t, x, y)];
}

// ------------------------------------------------------------- drawing --
const stateKey = (t) => [t, state.method, state.scale, state.tStart, state.estimated, state.kappa, state.bg,
  model.target[0], model.target[1], C.ink].join('|');
let heatKey = '', badgeKey = '';
function drawHeat(t) {
  const key = stateKey(t);
  if (key === heatKey) { ctx.drawImage(heat, 0, 0, W, H); return; }
  heatKey = key;
  const img = hctx.createImageData(HEAT, HEAT);
  const vals = new Float64Array(HEAT * HEAT);
  let vmax = -Infinity;
  const mode = state.bg;
  for (let j = 0; j < HEAT; j++) {
    for (let i = 0; i < HEAT; i++) {
      const x = -R + ((i + 0.5) / HEAT) * 2 * R, y = R - ((j + 0.5) / HEAT) * 2 * R;
      let v;
      if (mode === 'density') {
        v = fieldAt(model.comps, t, x, y).logp;
      } else if (mode === 'target') {
        const f = fieldAt(model.comps, t, x, y, state.method === 'tfg');
        const s = state.method === 'none' ? 0 : state.scale;
        if (state.method === 'cg') {
          const pc = model.estimated ? fieldAt(model.clsComps, t, x, y).pc : f.pc;
          v = f.logp + s * Math.log(Math.max(pc, 1e-300));
        } else if (state.method === 'cfg') {
          const lc = model.estimated ? fieldAt(model.condComps, t, x, y).logpxc : f.logpxc;
          v = f.logp + s * (lc - f.logp);
        } else if (state.method === 'tfg') {
          const ex = f.x0[0] - model.target[0], ey = f.x0[1] - model.target[1];
          v = f.logp - s * 0.5 * (ex * ex + ey * ey);
        } else v = f.logp;
      } else if (mode === 'classifier') {
        v = (model.estimated && state.method === 'cg' ? fieldAt(model.clsComps, t, x, y) : fieldAt(model.comps, t, x, y)).pc;
      } else {
        const [a, b] = cgCfgAt(t, x, y);
        v = Math.hypot(a[0] - b[0], a[1] - b[1]);
      }
      vals[j * HEAT + i] = v;
      if (v > vmax) vmax = v;
    }
  }
  const ink = rgb(C.density || C.ink), ph = rgb(C.photo), pe = rgb(C.pencil), ac = rgb(C.accent);
  for (let k = 0; k < vals.length; k++) {
    let r, g, b, a;
    if (mode === 'density' || mode === 'target') {
      const p = Math.exp(vals[k] - vmax);
      [r, g, b] = ink; a = 0.62 * Math.pow(p, 0.55);
    } else if (mode === 'classifier') {
      const pc = vals[k];
      const m = pc;
      r = ph[0] * (1 - m) + pe[0] * m; g = ph[1] * (1 - m) + pe[1] * m; b = ph[2] * (1 - m) + pe[2] * m;
      a = 0.12 + 0.5 * Math.pow(Math.abs(2 * pc - 1), 0.8);
    } else {
      const rel = vmax > 1e-12 ? vals[k] / vmax : 0;
      [r, g, b] = ac; a = 0.55 * Math.pow(rel, 0.8);
    }
    img.data.set([r, g, b, Math.round(255 * a)], 4 * k);
  }
  hctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(heat, 0, 0, W, H);
}

function arrow(x0, y0, dx, dy, color, width) {
  const L = Math.hypot(dx, dy);
  if (L < 1.5 * dpr) return;
  const x1 = x0 + dx, y1 = y0 + dy, h = Math.min(6 * dpr, L * 0.4), ang = Math.atan2(dy, dx);
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x1 - h * Math.cos(ang - 0.45), y1 - h * Math.sin(ang - 0.45));
  ctx.lineTo(x1 - h * Math.cos(ang + 0.45), y1 - h * Math.sin(ang + 0.45));
  ctx.closePath(); ctx.fill();
}

function drawArrows(t) {
  if (!state.arBase && !state.arGuide && !state.arTotal) return;
  const cell = W / ARROWS;
  const pts = [];
  const mags = [];
  for (let j = 0; j < ARROWS; j++) for (let i = 0; i < ARROWS; i++) {
    const x = wx((i + 0.5) * cell), y = wy((j + 0.5) * cell);
    const f = fieldAt(model.comps, t, x, y, state.method === 'tfg');
    const g = guidanceNow(t, x, y, f);
    pts.push({ x, y, s: f.score, g });
    mags.push(Math.hypot(f.score[0], f.score[1]));
  }
  mags.sort((a, b) => a - b);
  const ref = Math.max(mags[Math.floor(0.8 * mags.length)], 1e-9);
  const len = (vx, vy) => {
    const m = Math.hypot(vx, vy);
    if (m < 1e-12) return [0, 0];
    const L = cell * 0.85 * Math.tanh(m / ref);
    return [(vx / m) * L, -(vy / m) * L];
  };
  for (const p of pts) {
    const X = px(p.x), Y = py(p.y);
    if (state.arBase) {
      // with guidance on, fade the base score so the guidance arrows stand out
      const [dx, dy] = len(p.s[0], p.s[1]);
      ctx.globalAlpha = state.method !== 'none' && state.arGuide ? 0.3 : 1;
      arrow(X, Y, dx, dy, C.base, 1.4 * dpr);
      ctx.globalAlpha = 1;
    }
    if (state.arGuide && p.g) { const [dx, dy] = len(p.g[0], p.g[1]); arrow(X, Y, dx, dy, C.accent, 1.8 * dpr); }
    if (state.arTotal) {
      const gx = p.g ? p.g[0] : 0, gy = p.g ? p.g[1] : 0;
      const [dx, dy] = len(p.s[0] + gx, p.s[1] + gy); arrow(X, Y, dx, dy, C.total, 1.4 * dpr);
    }
  }
}

function drawDataRings() {
  ctx.setLineDash([4 * dpr, 4 * dpr]);
  ctx.lineWidth = 1 * dpr;
  for (const c of model.comps) {
    ctx.strokeStyle = c.q > 0.6 ? C.pencil : c.q < 0.4 ? C.photo : C.muted;
    ctx.globalAlpha = 0.7;
    ctx.beginPath(); ctx.arc(px(c.mu[0]), py(c.mu[1]), (2 * c.s / (2 * R)) * W, 0, 2 * Math.PI); ctx.stroke();
  }
  ctx.globalAlpha = 1; ctx.setLineDash([]);
}

const X0_EVERY = 3;  // draw clean predictions for every 3rd particle to keep the picture readable
function x0Arrays(fr) {
  return state.x0kind === 'guided' ? [fr.x0gx, fr.x0gy] : [fr.x0bx, fr.x0by];
}
function drawParticles(fr, dim) {
  const [kx, ky] = x0Arrays(fr);
  ctx.save();
  if (dim) ctx.globalAlpha = 0.3;
  if (state.particles) {
    for (let i = 0; i < fr.x.length; i++) {
      ctx.fillStyle = finalLabel[i] ? C.pencil : C.photo;
      ctx.beginPath(); ctx.arc(px(fr.x[i]), py(fr.y[i]), 2.3 * dpr, 0, 2 * Math.PI); ctx.fill();
    }
  }
  if (state.x0) {
    const base = dim ? 0.3 : 1;
    ctx.lineWidth = 1 * dpr; ctx.strokeStyle = C.ink; ctx.globalAlpha = 0.28 * base;
    ctx.beginPath();
    for (let i = 0; i < fr.x.length; i += X0_EVERY) { ctx.moveTo(px(fr.x[i]), py(fr.y[i])); ctx.lineTo(px(kx[i]), py(ky[i])); }
    ctx.stroke();
    ctx.globalAlpha = 0.95 * base;
    for (let i = 0; i < fr.x.length; i += X0_EVERY) {
      const X = px(kx[i]), Y = py(ky[i]);
      ctx.lineWidth = 3.2 * dpr; ctx.strokeStyle = C.card;
      ctx.beginPath(); ctx.arc(X, Y, 4.2 * dpr, 0, 2 * Math.PI); ctx.stroke();
      ctx.lineWidth = 1.8 * dpr; ctx.strokeStyle = finalLabel[i] ? C.pencil : C.photo;
      ctx.beginPath(); ctx.arc(X, Y, 4.2 * dpr, 0, 2 * Math.PI); ctx.stroke();
    }
  }
  ctx.restore();
}

// Hovered particle: full trajectory x_t (solid) and the path of its clean prediction x0_hat (dashed).
function nearestParticle(fr) {
  if (!state.hover) return -1;
  let best = -1, bd = (14 * dpr) ** 2;
  const hx = px(state.hover[0]), hy = py(state.hover[1]);
  for (let i = 0; i < fr.x.length; i++) {
    const d = (px(fr.x[i]) - hx) ** 2 + (py(fr.y[i]) - hy) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}
function star(X, Y, r, fill, stroke) {
  ctx.beginPath();
  for (let k = 0; k < 10; k++) {
    const rr = k % 2 ? r * 0.45 : r, a = -Math.PI / 2 + (k * Math.PI) / 5;
    k ? ctx.lineTo(X + rr * Math.cos(a), Y + rr * Math.sin(a)) : ctx.moveTo(X + rr * Math.cos(a), Y + rr * Math.sin(a));
  }
  ctx.closePath();
  ctx.lineWidth = 2.5 * dpr; ctx.strokeStyle = stroke; ctx.stroke();
  ctx.fillStyle = fill; ctx.fill();
}

// Focused particle (pinned, else hovered):
//   x_t trail up to the current step (fading with age), current x0_hat ring, actual endpoint star,
//   and optionally the x0_hat path up to now (dashed).
function focusIndex(fr) { return state.pinned >= 0 ? state.pinned : nearestParticle(fr); }
function drawFocus(fr, i) {
  if (i < 0) return;
  const cur = Math.min(state.frame, frames.length - 1);
  const col = finalLabel[i] ? C.pencil : C.photo;
  const endF = frames[frames.length - 1];

  // trail: segment-wise alpha so older parts fade out
  ctx.lineCap = 'round';
  for (let k = 1; k <= cur; k++) {
    const a0 = frames[k - 1], a1 = frames[k];
    const age = (cur - k) / Math.max(cur, 1);
    ctx.globalAlpha = 0.2 + 0.8 * (1 - age);
    ctx.lineWidth = 2.4 * dpr; ctx.strokeStyle = col;
    ctx.beginPath(); ctx.moveTo(px(a0.x[i]), py(a0.y[i])); ctx.lineTo(px(a1.x[i]), py(a1.y[i])); ctx.stroke();
  }
  ctx.globalAlpha = 1;

  const [kx, ky] = x0Arrays(fr);
  if (state.x0Path && cur > 0) {
    const gx = state.x0kind === 'guided' ? 'x0gx' : 'x0bx', gy = state.x0kind === 'guided' ? 'x0gy' : 'x0by';
    ctx.globalAlpha = 0.65; ctx.lineWidth = 1.4 * dpr; ctx.strokeStyle = col; ctx.setLineDash([4 * dpr, 4 * dpr]);
    ctx.beginPath();
    for (let k = 0; k <= cur; k++) { const X = px(frames[k][gx][i]), Y = py(frames[k][gy][i]); k ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
    ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
  }

  // actual endpoint (where this trajectory really ends)
  star(px(endF.x[i]), py(endF.y[i]), 8 * dpr, col, C.card);
  // current state -> current clean prediction
  ctx.lineWidth = 1.5 * dpr; ctx.strokeStyle = C.ink; ctx.setLineDash([2 * dpr, 3 * dpr]);
  ctx.beginPath(); ctx.moveTo(px(fr.x[i]), py(fr.y[i])); ctx.lineTo(px(kx[i]), py(ky[i])); ctx.stroke();
  ctx.setLineDash([]);
  ctx.lineWidth = 3 * dpr; ctx.strokeStyle = C.card;
  ctx.beginPath(); ctx.arc(px(fr.x[i]), py(fr.y[i]), 5.5 * dpr, 0, 2 * Math.PI); ctx.stroke();
  ctx.fillStyle = col; ctx.beginPath(); ctx.arc(px(fr.x[i]), py(fr.y[i]), 5 * dpr, 0, 2 * Math.PI); ctx.fill();
  ctx.lineWidth = 5 * dpr; ctx.strokeStyle = C.card;
  ctx.beginPath(); ctx.arc(px(kx[i]), py(ky[i]), 7 * dpr, 0, 2 * Math.PI); ctx.stroke();
  ctx.lineWidth = 2.5 * dpr; ctx.strokeStyle = col;
  ctx.beginPath(); ctx.arc(px(kx[i]), py(ky[i]), 7 * dpr, 0, 2 * Math.PI); ctx.stroke();

  ctx.font = `${12 * dpr}px ${uiFont()}`; ctx.fillStyle = C.ink;
  ctx.fillText('x_t', px(fr.x[i]) + 9 * dpr, py(fr.y[i]) - 9 * dpr);
  ctx.fillText('x̂₀ (estimate)', px(kx[i]) + 10 * dpr, py(ky[i]) + 16 * dpr);
  ctx.fillText('endpoint', px(endF.x[i]) + 10 * dpr, py(endF.y[i]) - 10 * dpr);
  if (state.pinned >= 0) {
    ctx.fillStyle = C.muted;
    ctx.fillText('pinned (click empty space to release)', 10 * dpr, H - 10 * dpr);
  }
}

// Classifier contours p_t(pencil | x) = 0.25 / 0.5 / 0.75 (CG and CFG only): the guidance g_t is the gradient
// of log p_t(pencil | x), so its arrows cross these lines at right angles, pointing uphill toward pencil.
const CON = 90, CON_LEVELS = [0.25, 0.5, 0.75];
let conKey = '', conSegs = [];
function contourSegments(t) {
  const useCls = model.estimated && state.method === 'cg';   // same classifier as the background view
  const key = [t, useCls, state.kappa].join('|');
  if (key === conKey) return conSegs;
  conKey = key;
  const comps = useCls ? model.clsComps : model.comps;
  const n = CON + 1, v = new Float64Array(n * n);
  const gx = (i) => -R + (i / CON) * 2 * R, gy = (j) => R - (j / CON) * 2 * R;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) v[j * n + i] = fieldAt(comps, t, gx(i), gy(j)).pc;
  conSegs = CON_LEVELS.map((lv) => {
    const segs = [];
    const lerp = (a, b, va, vb) => a + ((lv - va) / (vb - va)) * (b - a);
    for (let j = 0; j < CON; j++) for (let i = 0; i < CON; i++) {
      const a = v[j * n + i], b = v[j * n + i + 1], c = v[(j + 1) * n + i + 1], d = v[(j + 1) * n + i];
      const pts = [];   // crossings on edges top (a-b), right (b-c), bottom (d-c), left (a-d)
      if ((a < lv) !== (b < lv)) pts.push([lerp(gx(i), gx(i + 1), a, b), gy(j)]);
      if ((b < lv) !== (c < lv)) pts.push([gx(i + 1), lerp(gy(j), gy(j + 1), b, c)]);
      if ((d < lv) !== (c < lv)) pts.push([lerp(gx(i), gx(i + 1), d, c), gy(j + 1)]);
      if ((a < lv) !== (d < lv)) pts.push([gx(i), lerp(gy(j), gy(j + 1), a, d)]);
      if (pts.length >= 2) segs.push([pts[0], pts[1]]);
      if (pts.length === 4) segs.push([pts[2], pts[3]]);
    }
    return { lv, segs };
  });
  return conSegs;
}
function drawContours(t) {
  if (state.method !== 'cg' && state.method !== 'cfg') return;
  ctx.save();
  ctx.font = `${11 * dpr}px ${uiFont()}`;
  for (const { lv, segs } of contourSegments(t)) {
    if (!segs.length) continue;
    const mid = lv === 0.5;
    // soft, wide bands in the guidance color: they read as background behind the arrows
    // width grows with the level (0.25 -> 0.5 -> 0.75): thicker = more confidently pencil
    ctx.strokeStyle = C.accent; ctx.globalAlpha = 0.14 + 0.16 * lv;
    ctx.lineWidth = (2 + 8 * lv) * dpr;
    ctx.lineCap = 'round';
    ctx.beginPath();
    let lab = null;
    for (const [p, q] of segs) {
      ctx.moveTo(px(p[0]), py(p[1])); ctx.lineTo(px(q[0]), py(q[1]));
      // 0.5 is labelled at its top end, 0.25 / 0.75 where they cross the lower half (y = -R/2),
      // where the three lines are well apart
      const score = mid ? p[1] : -Math.abs(p[1] + R / 2);
      if (!lab || score > lab.score) lab = { x: p[0], y: p[1], score };
    }
    ctx.stroke();
    ctx.setLineDash([]);
    if (lab) {
      const text = mid ? 'p(pencil | x) = 0.5' : `${lv}`;
      const tw = ctx.measureText(text).width;
      const X = Math.min(px(lab.x) + 5 * dpr, W - tw - 6 * dpr);
      const Y = mid ? Math.max(py(lab.y) + 14 * dpr, 30 * dpr) : Math.min(py(lab.y) + 4 * dpr, H - 8 * dpr);
      ctx.globalAlpha = 0.9; ctx.fillStyle = C.accent;
      ctx.fillText(text, X, Y);
    }
  }
  ctx.restore();
}

function drawTarget() {
  if (state.method !== 'tfg') return;
  const X = px(model.target[0]), Y = py(model.target[1]), r = 8 * dpr;
  ctx.strokeStyle = C.accent; ctx.lineWidth = 3 * dpr;
  ctx.beginPath(); ctx.moveTo(X - r, Y - r); ctx.lineTo(X + r, Y + r); ctx.moveTo(X + r, Y - r); ctx.lineTo(X - r, Y + r); ctx.stroke();
}

function draw() {
  if (!frames.length || !W) return;
  const fr = frames[Math.min(state.frame, frames.length - 1)];
  const t = fr.t;
  ctx.fillStyle = C.card; ctx.fillRect(0, 0, W, H);
  drawHeat(t);
  drawDataRings();
  drawContours(t);
  drawArrows(t);
  const fi = focusIndex(fr);
  drawParticles(fr, fi >= 0);
  drawFocus(fr, fi);
  drawTarget();
  // guidance-off indicator
  const on = state.method !== 'none' && t <= state.tStart + 1e-9 && state.scale !== 0;
  ctx.font = `${12 * dpr}px ${uiFont()}`;
  ctx.fillStyle = on ? C.accent : C.muted;
  ctx.fillText(state.method === 'none' ? 'no guidance' : on ? `${state.method.toUpperCase()} guidance ON` : 'guidance OFF (t > start)', 10 * dpr, 18 * dpr);
  $('toy-leg-x0').style.display = state.x0 ? '' : 'none';
  $('toy-tlabel').textContent = `t = ${t.toFixed(2)}`;
  $('toy-time').value = state.frame;
  updateReadout(t);
  updateBadge(t);
}

// ------------------------------------------------------------- readouts --
const fmtv = (v) => `(${v[0].toFixed(2)}, ${v[1].toFixed(2)})`;
function updateReadout(t) {
  const el = $('toy-readout');
  if (!state.hover) { el.innerHTML = '<div class="ro-title">Hover the canvas to read the fields at a point.</div>'; return; }
  const [x, y] = state.hover;
  const f = fieldAt(model.comps, t, x, y);
  const [gcg, gcfg] = cgCfgAt(t, x, y);
  const sg = f.sg;
  const sc = (v) => [sg * v[0], sg * v[1]];
  const diff = sg * Math.hypot(gcg[0] - gcfg[0], gcg[1] - gcfg[1]);
  const pcEst = model.estimated ? fieldAt(model.clsComps, t, x, y).pc : f.pc;
  el.innerHTML = `<div class="ro-title">at x = ${fmtv([x, y])}, t = ${t.toFixed(2)} (σ-scaled vectors)</div>
  <table>
    <tr><td>p<sub>t</sub>(pencil | x)${model.estimated ? ' true / CG model' : ''}</td><td>${f.pc.toFixed(3)}${model.estimated ? ' / ' + pcEst.toFixed(3) : ''}</td></tr>
    <tr><td>σ ∇log p<sub>t</sub>(x)</td><td>${fmtv(sc(f.score))}</td></tr>
    <tr><td>σ g<sub>CG</sub></td><td>${fmtv(sc(gcg))}</td></tr>
    <tr><td>σ g<sub>CFG</sub></td><td>${fmtv(sc(gcfg))}</td></tr>
    <tr><td>σ |g<sub>CG</sub> − g<sub>CFG</sub>|</td><td>${diff.toExponential(1)}</td></tr>
    <tr><td>Tweedie x̂<sub>0</sub></td><td>${fmtv(f.x0)}</td></tr>
  </table>`;
}

function updateBadge(t) {
  const key = stateKey(t);
  if (key === badgeKey) return;
  badgeKey = key;
  let maxD = 0, maxG = 0, cosSum = 0, n = 0;
  const G = 25;
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
    const x = -R + ((i + 0.5) / G) * 2 * R, y = R - ((j + 0.5) / G) * 2 * R;
    const [a, b] = cgCfgAt(t, x, y);
    maxD = Math.max(maxD, Math.hypot(a[0] - b[0], a[1] - b[1]));
    const nb = Math.hypot(b[0], b[1]), na = Math.hypot(a[0], a[1]);
    maxG = Math.max(maxG, nb);
    if (na > 1e-12 && nb > 1e-12) { cosSum += (a[0] * b[0] + a[1] * b[1]) / (na * nb); n++; }
  }
  const rel = maxD / Math.max(maxG, 1e-12);
  $('toy-equal').innerHTML = model.estimated
    ? `Estimated models at t = ${t.toFixed(2)}: CG and CFG fields differ. Max relative gap <b>${rel.toFixed(2)}</b>, mean cosine <b>${(cosSum / Math.max(n, 1)).toFixed(2)}</b>.`
    : `Oracle CG vs CFG at t = ${t.toFixed(2)}: max |g<sub>CG</sub> − g<sub>CFG</sub>| / max |g| = <b>${rel.toExponential(1)}</b>. The two formulas give the same field (up to floating point).`;
}

// ---------------------------------------------------------------- chart --
let chartData = null, chartJob = 0;
// The sweep (21 scales x 1-2 methods x 200 particles x 40 steps, about 0.5 s) runs in small chunks and yields to the
// browser between scales, so it never freezes the page; a newer request cancels an older one.
async function computeChart() {
  const job = ++chartJob;
  const ss = []; for (let s = 0; s <= 5.0001; s += 0.25) ss.push(+s.toFixed(2));
  const methods = state.method === 'tfg' ? ['tfg'] : ['cg', 'cfg'];
  const vals = methods.map(() => []);
  for (const s of ss) {
    methods.forEach((method, j) => {
      const fr = simulate(model, { method, scale: s, tStart: state.tStart, nSteps: 40 }, chartNoise);
      const last = fr[fr.length - 1];
      let acc = 0;
      for (let i = 0; i < last.x.length; i++) {
        acc += method === 'tfg' ? Math.hypot(last.x[i] - model.target[0], last.y[i] - model.target[1])
          : (labelOf(model.comps, last.x[i], last.y[i]) > 0.5 ? 1 : 0);
      }
      vals[j].push(acc / last.x.length);
    });
    await new Promise((r) => setTimeout(r, 0));
    if (job !== chartJob) return;          // superseded by a newer request
  }
  chartData = methods[0] === 'tfg'
    ? { kind: 'tfg', ss, series: [{ name: 'TFG', vals: vals[0], color: C.mtfg, dash: [] }] }
    : { kind: 'frac', ss, series: [
      { name: 'CG', vals: vals[0], color: C.mcg, dash: [] },
      { name: 'CFG', vals: vals[1], color: C.mcfg, dash: [6, 4] },
    ] };
  drawChart();
}
let chartTimer = null;
function scheduleChart() { clearTimeout(chartTimer); chartTimer = setTimeout(computeChart, 120); }

function drawChart() {
  if (!chartData || !chart.width) return;
  const w = chart.width, h = chart.height, m = { l: 44 * dpr, r: 12 * dpr, t: 14 * dpr, b: 44 * dpr };
  cctx.clearRect(0, 0, w, h);
  const ymax = chartData.kind === 'frac' ? 1 : Math.max(...chartData.series[0].vals) * 1.1;
  const X = (s) => m.l + (s / 5) * (w - m.l - m.r);
  const Y = (v) => h - m.b - (v / ymax) * (h - m.t - m.b);
  cctx.font = `${11 * dpr}px ${uiFont()}`;
  cctx.strokeStyle = C.line; cctx.fillStyle = C.muted; cctx.lineWidth = 1 * dpr;
  for (let k = 0; k <= 4; k++) {
    const v = (ymax * k) / 4;
    cctx.beginPath(); cctx.moveTo(m.l, Y(v)); cctx.lineTo(w - m.r, Y(v)); cctx.stroke();
    cctx.fillText(chartData.kind === 'frac' ? `${Math.round(v * 100)}%` : v.toFixed(1), 4 * dpr, Y(v) + 4 * dpr);
  }
  for (let s = 0; s <= 5; s++) cctx.fillText(`${s}`, X(s) - 3 * dpr, h - m.b + 16 * dpr);
  cctx.fillText('guidance scale s', w / 2 - 40 * dpr, h - 6 * dpr);
  // current s
  cctx.strokeStyle = C.muted; cctx.setLineDash([2 * dpr, 3 * dpr]);
  cctx.beginPath(); cctx.moveTo(X(state.scale), m.t); cctx.lineTo(X(state.scale), h - m.b); cctx.stroke();
  cctx.setLineDash([]);
  chartData.series.forEach((sr, idx) => {
    cctx.strokeStyle = sr.color; cctx.lineWidth = 2 * dpr; cctx.setLineDash(sr.dash.map((d) => d * dpr));
    cctx.beginPath();
    sr.vals.forEach((v, i) => { const xx = X(chartData.ss[i]), yy = Y(v); i ? cctx.lineTo(xx, yy) : cctx.moveTo(xx, yy); });
    cctx.stroke(); cctx.setLineDash([]);
    cctx.fillStyle = sr.color;
    cctx.fillText(sr.name + (model.estimated && sr.name !== 'TFG' ? ' (est.)' : ''), w - m.r - 80 * dpr, h - m.b - 28 * dpr + idx * 14 * dpr);
  });
  $('toy-chart-title').textContent = chartData.kind === 'frac'
    ? 'Pencil fraction vs. guidance scale' : 'Distance to TFG target vs. guidance scale';
  const CG = '<b class="m-cg">CG</b>', CFG = '<b class="m-cfg">CFG</b>', TFG = '<b class="m-tfg">TFG</b>';
  $('toy-chart-cap').innerHTML = chartData.kind === 'frac'
    ? (model.estimated
      ? `Fraction ending as pencil vs s. With estimation error, ${CG} and ${CFG} no longer agree.`
      : `Fraction ending as pencil vs s. Oracle ${CG} and ${CFG} coincide (the curves overlap).`)
    : `Mean final distance to the ${TFG} target vs s.`;
}

// ------------------------------------------------------------ controls --
function seg(id, key, after) {
  $(id).addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    for (const x of $(id).children) x.classList.toggle('on', x === b);
    state[key] = b.dataset.v;
    after && after();
  });
}
seg('toy-method', 'method', () => {
  $('toy-tfg-hint').hidden = state.method !== 'tfg';
  resimulate(); scheduleChart();
});
seg('toy-model', 'estimated', () => {
  state.estimated = state.estimated === 'estimated';
  $('toy-kappa').disabled = !state.estimated;
  rebuildModel(); resimulate(); scheduleChart();
});
$('toy-scale').addEventListener('input', (e) => {
  state.scale = +e.target.value; $('toy-scale-v').textContent = `s = ${state.scale.toFixed(2)}`;
  resimulate(); drawChart();
});
$('toy-kappa').addEventListener('input', (e) => {
  state.kappa = +e.target.value; $('toy-kappa-v').textContent = `κ = ${state.kappa.toFixed(2)}`;
  rebuildModel(); resimulate(); scheduleChart();
});
seg('toy-bg', 'bg', () => draw());
[['toy-ar-base', 'arBase'], ['toy-ar-guide', 'arGuide'], ['toy-ar-total', 'arTotal'], ['toy-particles', 'particles'], ['toy-x0', 'x0'],
  ['toy-x0path', 'x0Path']]
  .forEach(([id, key]) => $(id).addEventListener('change', (e) => { state[key] = e.target.checked; draw(); }));
$('toy-time').max = N_STEPS;
$('toy-time').addEventListener('input', (e) => { state.frame = +e.target.value; stop(); draw(); });

let last = 0;
function tick(ts) {
  if (!state.playing) return;
  if (ts - last > 45) {
    last = ts;
    if (state.frame >= N_STEPS) { stop(); return; }
    state.frame++; draw();
  }
  requestAnimationFrame(tick);
}
function stop() { state.playing = false; $('toy-play').textContent = '▶'; }
$('toy-play').addEventListener('click', () => {
  if (state.playing) { stop(); return; }
  if (state.frame >= N_STEPS) state.frame = 0;
  state.playing = true; $('toy-play').textContent = '❚❚'; requestAnimationFrame(tick);
});

function eventWorld(e) {
  const r = canvas.getBoundingClientRect();
  return [wx(((e.clientX - r.left) / r.width) * W), wy(((e.clientY - r.top) / r.height) * H)];
}
let dragging = false;
canvas.addEventListener('pointerdown', (e) => {
  const w = eventWorld(e);
  // TFG: plain click moves the target, Shift+click pins. Other methods: click pins / releases.
  if (state.method === 'tfg' && !e.shiftKey) {
    dragging = true; model.target = w; resimulate(); canvas.setPointerCapture(e.pointerId);
    return;
  }
  state.hover = w;
  state.pinned = nearestParticle(frames[Math.min(state.frame, frames.length - 1)]);
  draw();
});
canvas.addEventListener('pointermove', (e) => {
  state.hover = eventWorld(e);
  if (dragging) { model.target = state.hover; resimulate(); } else { draw(); }
});
canvas.addEventListener('pointerup', () => { if (dragging) { dragging = false; scheduleChart(); } });
canvas.addEventListener('pointerleave', () => { state.hover = null; draw(); });

const onTheme = () => { readColors(); heatKey = ''; draw(); computeChart(); };
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', onTheme);
window.addEventListener('themechange', onTheme);
window.addEventListener('resize', resize);

// Optional URL state, e.g. ?method=cfg&s=2&frame=50&model=estimated&bg=classifier
function applyUrlState() {
  const q = new URLSearchParams(location.search);
  const click = (id, v) => { const b = $(id).querySelector(`button[data-v="${v}"]`); b && b.click(); };
  if (q.has('s')) { $('toy-scale').value = q.get('s'); $('toy-scale').dispatchEvent(new Event('input')); }
  if (q.has('model')) click('toy-model', q.get('model'));
  if (q.has('kappa')) { $('toy-kappa').value = q.get('kappa'); $('toy-kappa').dispatchEvent(new Event('input')); }
  if (q.has('method')) click('toy-method', q.get('method'));
  if (q.has('bg')) click('toy-bg', q.get('bg'));
  if (q.has('frame')) state.frame = Math.min(N_STEPS, +q.get('frame'));
  if (q.has('pin')) state.pinned = +q.get('pin');
  if (q.has('x0path')) { state.x0Path = true; $('toy-x0path').checked = true; }
}

readColors();
rebuildModel();
resimulate();
applyUrlState();
resize();
computeChart();

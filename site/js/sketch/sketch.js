// Section 3: draw an outline, then a 32x32 sketch diffusion model (TF.js, in the browser) draws a cat that follows it.
// Training-free guidance: eps_tilde = eps + s * c * sigma_t * grad_x loss(ink(x0_hat(x))), gradient through the UNet.
// Loss (chosen with pipeline/sketch/guided_test.py): weighted MSE between blurred ink maps; extra ink away from the
// outline costs only 20%, so the model can still add eyes and whiskers.

const TFJS = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
const TFWASM = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/';
const ASSETS = 'assets/sketch/';
const C_GUIDE = 1e4;          // fixed strength constant: s = 1 gives a guidance term of roughly unit rms
const STROKE = 1.6 / 32;      // pen width as a fraction of the canvas (matches the training rasterization)
const SLOW_MS = 400;          // a guided step (batch 2) slower than this switches to the light settings

const $ = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const uiFont = () => getComputedStyle(document.body).fontFamily;

const state = {
  strokes: [], drawing: null,
  s: 0.25, steps: 20, frac: 0.6, seed: 1, batch: 3, slow: false,
  frames: null, frame: 0, running: false, ready: false,
  showOutline: true, showGrad: false,
};
let unetMod = null, W = null, G = null;   // module, weights, blur kernel

// --------------------------------------------------------------- presets --
const PRESETS = {
  head: [ellipse(0.5, 0.58, 0.31, 0.3), [[0.25, 0.4], [0.28, 0.09], [0.44, 0.3]], [[0.56, 0.3], [0.72, 0.09], [0.75, 0.4]]],
  oval: [ellipse(0.5, 0.64, 0.38, 0.26)],
  ears: [[[0.2, 0.55], [0.27, 0.12], [0.45, 0.4]], [[0.55, 0.4], [0.73, 0.12], [0.8, 0.55]]],
};
function ellipse(cx, cy, rx, ry, n = 48) {
  return Array.from({ length: n + 1 }, (_, k) => [cx + rx * Math.cos((2 * Math.PI * k) / n), cy + ry * Math.sin((2 * Math.PI * k) / n)]);
}

// --------------------------------------------------------------- drawing --
const pad = $('sk-pad');
function fit(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return c.getContext('2d');
}
function strokePath(ctx, strokes, size, width, color) {
  ctx.strokeStyle = color; ctx.lineWidth = width; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  for (const s of strokes) {
    ctx.beginPath();
    s.forEach(([x, y], i) => (i ? ctx.lineTo(x * size, y * size) : ctx.moveTo(x * size, y * size)));
    if (s.length === 1) ctx.lineTo(s[0][0] * size + 0.01, s[0][1] * size);
    ctx.stroke();
  }
}
function drawPad() {
  const ctx = fit(pad), w = pad.width;
  ctx.fillStyle = css('--card'); ctx.fillRect(0, 0, w, w);
  // 32x32 grid hint
  ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
  for (let k = 8; k < 32; k += 8) {
    ctx.beginPath(); ctx.moveTo((k / 32) * w, 0); ctx.lineTo((k / 32) * w, w);
    ctx.moveTo(0, (k / 32) * w); ctx.lineTo(w, (k / 32) * w); ctx.stroke();
  }
  const all = state.drawing ? [...state.strokes, state.drawing] : state.strokes;
  strokePath(ctx, all, w, 0.7 * STROKE * w, css('--m-tfg'));   // drawn 30% thinner than the rasterized target
  if (!all.length) {
    ctx.fillStyle = css('--muted'); ctx.font = `${14 * (window.devicePixelRatio || 1)}px ${uiFont()}`;
    ctx.textAlign = 'center'; ctx.fillText('Draw an outline here', w / 2, w / 2);
  }
}
function padPoint(e) {
  const r = pad.getBoundingClientRect();
  return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
}
pad.addEventListener('pointerdown', (e) => { pad.setPointerCapture(e.pointerId); state.drawing = [padPoint(e)]; drawPad(); });
pad.addEventListener('pointermove', (e) => { if (state.drawing) { state.drawing.push(padPoint(e)); drawPad(); } });
const endStroke = () => { if (state.drawing) { state.strokes.push(state.drawing); state.drawing = null; drawPad(); } };
pad.addEventListener('pointerup', endStroke);
pad.addEventListener('pointercancel', endStroke);

// strokes -> 32x32 ink map in [0, 1] (drawn at 4x, then downsampled, like the training data)
function rasterize(strokes) {
  const big = document.createElement('canvas'); big.width = big.height = 128;
  const b = big.getContext('2d');
  b.fillStyle = '#000'; b.fillRect(0, 0, 128, 128);
  strokePath(b, strokes, 128, STROKE * 128, '#fff');
  const small = document.createElement('canvas'); small.width = small.height = 32;
  const s = small.getContext('2d');
  s.imageSmoothingEnabled = true; s.imageSmoothingQuality = 'high';
  s.drawImage(big, 0, 0, 32, 32);
  const px = s.getImageData(0, 0, 32, 32).data;
  return Float32Array.from({ length: 1024 }, (_, i) => px[4 * i] / 255);
}

// ---------------------------------------------------------- TF.js setup --
function loadScript(src) {
  return new Promise((res, rej) => {
    const el = document.createElement('script'); el.src = src; el.onload = res; el.onerror = rej;
    document.head.appendChild(el);
  });
}
function setStatus(html) { $('sk-status').innerHTML = html; }

async function init() {
  setStatus('Loading TensorFlow.js…');
  await loadScript(TFJS);
  await loadScript(`${TFWASM}tf-backend-wasm.min.js`).catch(() => {});
  if (tf.wasm) tf.wasm.setWasmPaths(TFWASM);
  for (const b of ['webgl', 'wasm', 'cpu']) {
    try { if (await tf.setBackend(b)) break; } catch (e) { /* try the next backend */ }
  }
  await tf.ready();
  setStatus(`Loading the sketch model (${tf.getBackend()})…`);
  unetMod = await import('./unet.js');
  ({ W } = await unetMod.loadWeights(ASSETS));
  G = blurKernel();
  // speed probe: one guided step at batch 2, after warm-up
  const probe = async () => tf.tidy(() => guidedGrad(tf.randomNormal([2, 32, 32, 1]), 0.5, tf.zeros([2, 32, 32, 1])).grad).data();
  await probe();
  const t0 = performance.now(); await probe();
  const ms = performance.now() - t0;
  state.slow = ms > SLOW_MS;
  state.batch = state.slow ? 2 : 3;
  $('sk-mode').hidden = !state.slow;
  buildResults();
  state.ready = true;
  $('sk-go').disabled = false;
  setStatus(`Ready · ${tf.getBackend()} · one guided step ≈ ${ms.toFixed(0)} ms${state.slow ? ' (light mode)' : ''}`);
  if (new URLSearchParams(location.search).has('skauto')) generate();   // for screenshots and tests
}

function blurKernel(sigma = 1, r = 2) {
  const v = [];
  let sum = 0;
  for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) { const k = Math.exp(-(x * x + y * y) / (2 * sigma * sigma)); v.push(k); sum += k; }
  return tf.tensor4d(v.map((k) => k / sum), [2 * r + 1, 2 * r + 1, 1, 1]);
}
const blur = (x) => tf.conv2d(x, G, 1, 'same');

// weighted MSE between blurred ink maps, per sample -> (B,)
function outlineLoss(x0, m) {
  const ink = tf.div(tf.add(tf.clipByValue(x0, -1, 1), 1), 2);
  const i = blur(ink);
  const mmax = tf.add(tf.max(m, [1, 2, 3], true), 1e-6);
  // w = 0.2 + 0.8 m/max(m) where there is extra ink (i > m), else 1. The step H(i - m) is built from sign(),
  // whose gradient is zero in TF.js (tf.where/greater have no gradient at all).
  const H = tf.div(tf.add(tf.sign(tf.sub(i, m)), 1), 2);
  const w = tf.sub(1, tf.mul(H, tf.sub(0.8, tf.mul(0.8, tf.div(m, mmax)))));
  return tf.mean(tf.mul(w, tf.square(tf.sub(i, m))), [1, 2, 3]);
}

// eps, x0_hat and grad_x loss at (x, t) in one forward/backward pass
function guidedGrad(x, t, mBlur) {
  const B = x.shape[0];
  const [a, s] = unetMod.alphaSigma(t);
  const tt = tf.fill([B], t);
  let eps = null;
  const f = (xx) => {
    const e = unetMod.unet(W, xx, tt);
    eps = tf.keep(e);
    return tf.sum(outlineLoss(tf.div(tf.sub(xx, tf.mul(e, s)), a), mBlur));
  };
  const { value, grad } = tf.valueAndGrad(f)(x);
  return { eps, grad, value };
}

// ------------------------------------------------------------- sampling --
async function generate() {
  if (!state.ready || state.running) return;
  if (!state.strokes.length) { setStatus('Draw an outline first (or pick a preset).'); return; }
  state.running = true; $('sk-go').disabled = true;
  const n = state.batch, N = state.steps, nGuided = Math.round(state.frac * N);
  const M = rasterize(state.strokes);
  const mBlur = tf.keep(tf.tidy(() => blur(tf.tensor(Array.from({ length: n }, () => Array.from(M)).flat(), [n, 32, 32, 1]))));
  const xInit = tf.randomNormal([n, 32, 32, 1], 0, 1, 'float32', state.seed);
  let xg = tf.keep(xInit.clone()), xu = tf.keep(xInit.clone());
  xInit.dispose();
  state.frames = [];
  state.outline = M;
  const lossG = [], lossU = [];
  const t0 = performance.now();
  for (let k = 0; k < N; k++) {
    const t = 1 - k / N, tp = 1 - (k + 1) / N;
    const [a, s] = unetMod.alphaSigma(t), [ap, sp] = unetMod.alphaSigma(tp);
    const guided = k < nGuided && state.s > 0;
    const last = k === N - 1;
    const res = tf.tidy(() => {
      // guided chain
      let eps, x0g, dl;
      if (guided) {
        const r = guidedGrad(xg, t, mBlur);
        eps = tf.add(r.eps, tf.mul(r.grad, state.s * C_GUIDE * s));
        r.eps.dispose();
      } else {
        eps = unetMod.unet(W, xg, tf.fill([n], t));
      }
      x0g = tf.clipByValue(tf.div(tf.sub(xg, tf.mul(eps, s)), a), -1, 1);
      // d loss / d x0_hat (where guidance wants to add or remove ink), for the overlay
      dl = tf.grad((z) => tf.sum(outlineLoss(z, mBlur)))(x0g);
      const epsG = tf.div(tf.sub(xg, tf.mul(x0g, a)), s);
      const xgNext = last ? x0g : tf.add(tf.mul(x0g, ap), tf.mul(epsG, sp));
      // unguided chain from the same noise
      const eu = unetMod.unet(W, xu, tf.fill([n], t));
      const x0u = tf.clipByValue(tf.div(tf.sub(xu, tf.mul(eu, s)), a), -1, 1);
      const epsU = tf.div(tf.sub(xu, tf.mul(x0u, a)), s);
      const xuNext = last ? x0u : tf.add(tf.mul(x0u, ap), tf.mul(epsU, sp));
      return { xgNext: tf.keep(xgNext), xuNext: tf.keep(xuNext), x0g: tf.keep(x0g), x0u: tf.keep(x0u), dl: tf.keep(dl),
        lg: tf.keep(tf.mean(outlineLoss(x0g, mBlur))), lu: tf.keep(tf.mean(outlineLoss(x0u, mBlur))) };
    });
    xg.dispose(); xu.dispose();
    xg = res.xgNext; xu = res.xuNext;
    const [g, u, d, lg, lu] = await Promise.all([res.x0g.data(), res.x0u.data(), res.dl.data(), res.lg.data(), res.lu.data()]);
    ['x0g', 'x0u', 'dl', 'lg', 'lu'].forEach((key) => res[key].dispose());
    lossG.push(lg[0]); lossU.push(lu[0]);
    state.frames.push({ t, guided, g, u, d, lossG: lossG.slice(), lossU: lossU.slice() });
    state.frame = k;
    $('sk-time').max = N - 1; $('sk-time').value = k;
    drawResults();
    setStatus(`Step ${k + 1} / ${N}${guided ? ' · guided' : ''} · ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    await new Promise(requestAnimationFrame);
  }
  xg.dispose(); xu.dispose(); mBlur.dispose();
  state.running = false; $('sk-go').disabled = false;
  setStatus(`Done in ${((performance.now() - t0) / 1000).toFixed(1)} s · scrub the slider to replay`);
}

// ------------------------------------------------------------- results --
function buildResults() {
  const host = $('sk-results');
  host.innerHTML = '';
  for (let i = 0; i < state.batch; i++) {
    const card = document.createElement('div');
    card.className = 'sk-card';
    card.innerHTML = `<canvas class="sk-big" data-i="${i}" data-kind="g"></canvas>
      <div class="sk-sub"><canvas class="sk-small" data-i="${i}" data-kind="u"></canvas><span>same noise,<br>no guidance</span></div>`;
    host.appendChild(card);
  }
}
const toRGB = (hex) => [1, 3, 5].map((k) => parseInt(hex.slice(k, k + 2), 16));
function paint(canvas, vals, opts = {}) {
  const ctx = fit(canvas), w = canvas.width;
  const off = document.createElement('canvas'); off.width = off.height = 32;
  const o = off.getContext('2d');
  const img = o.createImageData(32, 32);
  const paper = toRGB(css('--card')), ink = toRGB(css('--ink'));
  const add = toRGB(css('--m-cg')), rem = toRGB(css('--pencil')), guide = toRGB(css('--m-tfg'));
  let dmax = 1e-12;
  if (opts.grad) for (const v of opts.grad) dmax = Math.max(dmax, Math.abs(v));
  for (let p = 0; p < 1024; p++) {
    const a = Math.min(1, Math.max(0, (vals[p] + 1) / 2));
    let c = paper.map((pv, k) => pv + (ink[k] - pv) * a);
    if (opts.outline) { const m = opts.outline[p] * 0.28; c = c.map((v, k) => v + (guide[k] - v) * m); }
    if (opts.grad) {
      const g = opts.grad[p] / dmax;                      // < 0: loss falls if ink is added here
      const col = g < 0 ? add : rem, m = Math.min(1, Math.abs(g)) * 0.85;
      c = c.map((v, k) => v + (col[k] - v) * m);
    }
    img.data.set([c[0], c[1], c[2], 255], 4 * p);
  }
  o.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;   // crisp 32x32 pixels: this is exactly what the model produces
  ctx.drawImage(off, 0, 0, w, w);
}
function drawResults() {
  const f = state.frames && state.frames[state.frame];
  for (const c of document.querySelectorAll('#sk-results canvas')) {
    const i = +c.dataset.i;
    if (!f) { const ctx = fit(c); ctx.fillStyle = css('--bg'); ctx.fillRect(0, 0, c.width, c.height); continue; }
    const sl = (arr) => arr.subarray(1024 * i, 1024 * (i + 1));
    if (c.dataset.kind === 'g') {
      paint(c, sl(f.g), { outline: state.showOutline ? state.outline : null, grad: state.showGrad && f.guided ? sl(f.d) : null });
    } else paint(c, sl(f.u));
  }
  if (f) $('sk-tlabel').textContent = `step ${state.frame + 1} / ${state.frames.length} · t = ${f.t.toFixed(2)}${f.guided ? ' · guided' : ''}`;
  drawLoss();
}
function drawLoss() {
  const c = $('sk-loss'), ctx = fit(c), w = c.width, h = c.height, dpr = window.devicePixelRatio || 1;
  ctx.clearRect(0, 0, w, h);
  const f = state.frames && state.frames[state.frames.length - 1];
  const mg = { l: 44 * dpr, r: 10 * dpr, t: 24 * dpr, b: 26 * dpr };
  ctx.font = `${10.5 * dpr}px ${uiFont()}`;
  ctx.fillStyle = css('--muted');
  if (!f) { ctx.textAlign = 'center'; ctx.fillText('Generate to see the outline loss per step', w / 2, h / 2); return; }
  const N = state.steps, all = [...f.lossG, ...f.lossU];
  const ymax = Math.max(...all) * 1.1 || 1;
  const X = (k) => mg.l + (k / Math.max(1, N - 1)) * (w - mg.l - mg.r);
  const Y = (v) => h - mg.b - (v / ymax) * (h - mg.t - mg.b);
  // guided region
  const nG = Math.round(state.frac * N);
  ctx.fillStyle = css('--line'); ctx.globalAlpha = 0.5;
  ctx.fillRect(mg.l, mg.t, X(Math.max(0, nG - 1)) - mg.l, h - mg.t - mg.b); ctx.globalAlpha = 1;
  ctx.strokeStyle = css('--line'); ctx.lineWidth = dpr;
  ctx.fillStyle = css('--muted'); ctx.textAlign = 'right';
  for (let k = 0; k <= 3; k++) {
    const v = (ymax * k) / 3;
    ctx.beginPath(); ctx.moveTo(mg.l, Y(v)); ctx.lineTo(w - mg.r, Y(v)); ctx.stroke();
    ctx.fillText(v.toFixed(3), mg.l - 5 * dpr, Y(v) + 3.5 * dpr);
  }
  ctx.textAlign = 'center'; ctx.fillText('sampling step', (mg.l + w - mg.r) / 2, h - 4 * dpr);
  const line = (vals, color, dash) => {
    ctx.strokeStyle = css(color); ctx.lineWidth = 2 * dpr; ctx.setLineDash(dash.map((d) => d * dpr));
    ctx.beginPath(); vals.forEach((v, k) => (k ? ctx.lineTo(X(k), Y(v)) : ctx.moveTo(X(k), Y(v)))); ctx.stroke(); ctx.setLineDash([]);
  };
  line(f.lossU, '--m-none', [4, 3]);
  line(f.lossG, '--m-tfg', []);
  ctx.textAlign = 'left';
  [['guided', '--m-tfg'], ['no guidance', '--m-none']].forEach(([lab, col], k) => {
    ctx.fillStyle = css(col); ctx.fillRect(mg.l + 6 * dpr + k * 90 * dpr, 7 * dpr, 10 * dpr, 3 * dpr);
    ctx.fillText(lab, mg.l + 19 * dpr + k * 90 * dpr, 12 * dpr);
  });
  ctx.fillStyle = css('--muted'); ctx.textAlign = 'right';
  ctx.fillText('shaded: guidance on', w - mg.r, 12 * dpr);
  ctx.strokeStyle = css('--ink'); ctx.setLineDash([3 * dpr, 3 * dpr]); ctx.lineWidth = dpr;
  ctx.beginPath(); ctx.moveTo(X(state.frame), mg.t); ctx.lineTo(X(state.frame), h - mg.b); ctx.stroke(); ctx.setLineDash([]);
}

// ------------------------------------------------------------- controls --
function bindRange(id, key, fmt, after) {
  const el = $(id), out = $(`${id}-v`);
  const upd = () => { state[key] = +el.value; out.textContent = fmt(state[key]); after && after(); };
  el.addEventListener('input', upd); upd();
}
bindRange('sk-s', 's', (v) => `s = ${v.toFixed(2)}`);
bindRange('sk-steps', 'steps', (v) => `${v}`);
bindRange('sk-frac', 'frac', (v) => `first ${Math.round(v * 100)}%`);
$('sk-clear').addEventListener('click', () => { state.strokes = []; drawPad(); });
$('sk-undo').addEventListener('click', () => { state.strokes.pop(); drawPad(); });
for (const b of document.querySelectorAll('[data-preset]')) {
  b.addEventListener('click', () => { state.strokes = PRESETS[b.dataset.preset].map((s) => s.map((p) => [...p])); drawPad(); });
}
$('sk-seed').addEventListener('click', () => { state.seed = 1 + Math.floor(Math.random() * 1e6); $('sk-seed-v').textContent = `#${state.seed}`; });
$('sk-go').addEventListener('click', generate);
$('sk-time').addEventListener('input', (e) => { if (state.frames && !state.running) { state.frame = +e.target.value; drawResults(); } });
$('sk-show-outline').addEventListener('change', (e) => { state.showOutline = e.target.checked; drawResults(); });
$('sk-show-grad').addEventListener('change', (e) => { state.showGrad = e.target.checked; drawResults(); });
const redraw = () => { drawPad(); drawResults(); };
window.addEventListener('resize', redraw);
window.addEventListener('themechange', redraw);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);

// start with a preset outline; load TF.js and the model only when the section comes into view
state.strokes = PRESETS.head.map((s) => s.map((p) => [...p]));
drawPad();
drawLoss();
new IntersectionObserver((entries, obs) => {
  if (entries.some((e) => e.isIntersecting)) { obs.disconnect(); init().catch((e) => setStatus(`Could not start the model: ${e.message || e}`)); }
}, { rootMargin: '400px' }).observe($('draw'));

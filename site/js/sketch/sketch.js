// Section 3: draw an outline, then a small (32x32 or 64x64) sketch diffusion model (TF.js, in the browser) draws a cat that follows it.
// Training-free guidance: eps_tilde = eps + s * c * sigma_t * grad_x loss(ink(x0_hat(x))), gradient through the UNet.
// Loss (chosen with pipeline/sketch/guided_test.py): weighted MSE between blurred ink maps; extra ink away from the
// outline costs only 20%, so the model can still add eyes and whiskers.

const TFJS = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
const TFWASM = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@4.22.0/dist/';
// model folder (?skmodel=sketch64 to try another export); size and guidance constants come from its manifest
const ASSETS = `assets/${new URLSearchParams(location.search).get('skmodel') || 'sketch64'}/`;
let SIZE = 32;                // model resolution
let C_GUIDE = 1e4;            // strength constant c (manifest guide.c, calibrated with guided_test.py)
let BLUR = 1;                 // loss blur sigma in pixels (manifest guide.blur_sigma)
const LIGHT_SEED = 63471;     // default starting noise in light mode (best of #1 / #933569 / #63471 with the Head preset)
const strokeFrac = () => 1.6 / SIZE;   // pen width as a fraction of the canvas (training: 1.6 px)
const NPIX = () => SIZE * SIZE;

const $ = (id) => document.getElementById(id);
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const uiFont = () => getComputedStyle(document.body).fontFamily;

const state = {
  strokes: [], drawing: null,
  s: 0.25, steps: 20, frac: 0.6, seed: 1, batch: 3, slow: false,
  frames: null, frame: 0, running: false, ready: false,
  showOutline: true, showGrad: true,
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
  // quarter grid hint
  ctx.strokeStyle = css('--line'); ctx.lineWidth = 1;
  for (let k = 1; k < 4; k++) {
    ctx.beginPath(); ctx.moveTo((k / 4) * w, 0); ctx.lineTo((k / 4) * w, w);
    ctx.moveTo(0, (k / 4) * w); ctx.lineTo(w, (k / 4) * w); ctx.stroke();
  }
  const all = state.drawing ? [...state.strokes, state.drawing] : state.strokes;
  strokePath(ctx, all, w, Math.max(0.7 * strokeFrac() * w, 6), css('--m-tfg'));   // drawn 30% thinner than the rasterized target
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

// strokes -> SIZE x SIZE ink map in [0, 1] (drawn at 4x, then downsampled, like the training data)
function rasterize(strokes) {
  const B4 = 4 * SIZE;
  const big = document.createElement('canvas'); big.width = big.height = B4;
  const b = big.getContext('2d');
  b.fillStyle = '#000'; b.fillRect(0, 0, B4, B4);
  strokePath(b, strokes, B4, strokeFrac() * B4, '#fff');
  const small = document.createElement('canvas'); small.width = small.height = SIZE;
  const s = small.getContext('2d');
  s.imageSmoothingEnabled = true; s.imageSmoothingQuality = 'high';
  s.drawImage(big, 0, 0, SIZE, SIZE);
  const px = s.getImageData(0, 0, SIZE, SIZE).data;
  return Float32Array.from({ length: NPIX() }, (_, i) => px[4 * i] / 255);
}

// ---------------------------------------------------------- TF.js setup --
function loadScript(src) {
  return new Promise((res, rej) => {
    const el = document.createElement('script'); el.src = src; el.onload = res; el.onerror = rej;
    document.head.appendChild(el);
  });
}
function setStatus(html) { $('sk-status').innerHTML = html; }
// loading bar: stage label + overall fraction; yields a frame so the bar actually repaints between heavy steps
async function progress(frac, label) {
  $('sk-progress-fill').style.width = `${Math.round(100 * frac)}%`;
  $('sk-progress-label').textContent = `${label} (${Math.round(100 * frac)}%)`;
  await new Promise(requestAnimationFrame);
}
// software WebGL (no real GPU) reports one of these renderers
function softwareWebGL() {
  const gl = document.createElement('canvas').getContext('webgl2') || document.createElement('canvas').getContext('webgl');
  if (!gl) return true;
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const r = String(dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
  return /swiftshader|basic render|llvmpipe|software|warp/i.test(r);
}

async function init() {
  await progress(0.02, 'Loading TensorFlow.js');
  await loadScript(TFJS);
  await loadScript(`${TFWASM}tf-backend-wasm.min.js`).catch(() => {});
  if (tf.wasm) tf.wasm.setWasmPaths(TFWASM);
  for (const b of ['webgl', 'wasm', 'cpu']) {
    try { if (await tf.setBackend(b)) break; } catch (e) { /* try the next backend */ }
  }
  await tf.ready();
  await progress(0.15, `Downloading the sketch model (${tf.getBackend()})`);
  unetMod = await import('./unet.js');
  let lastShown = 0;
  const loaded = await unetMod.loadWeights(ASSETS, (f) => {
    if (f - lastShown > 0.04 || f === 1) { lastShown = f; progress(0.15 + 0.45 * f, 'Downloading the sketch model'); }
  });
  W = loaded.W;
  SIZE = loaded.config.size;
  if (loaded.guide) {
    C_GUIDE = loaded.guide.c; BLUR = loaded.guide.blur_sigma;
    $('sk-s').value = loaded.guide.s; $('sk-s').dispatchEvent(new Event('input'));
  }
  G = blurKernel(BLUR, Math.ceil(2 * BLUR));
  drawPad();
  // Light mode is decided by the hardware, not by one timing: WASM/CPU backends or software WebGL are "slow".
  // (A cold first run can be slow even on a GPU, which used to trigger the warning by mistake.)
  state.slow = tf.getBackend() !== 'webgl' || softwareWebGL();
  // light mode without GPU: one sample; for the 64x64 model also fewer steps
  state.batch = state.slow ? (SIZE > 32 ? 1 : 2) : 3;
  // Warm-up at the batch that will be used: TF.js compiles WebGL shaders per tensor shape, so doing it now keeps the
  // user's first Generate fast. Guided and plain steps are compiled separately so the bar can move in between.
  await progress(0.62, 'Preparing the GPU kernels (guided step)');
  await warmup(state.batch, [true]);
  await progress(0.85, 'Preparing the GPU kernels (plain step)');
  await warmup(state.batch, [false]);
  await progress(0.95, 'Measuring speed');
  const t0 = performance.now(); await warmup(state.batch);
  const ms = performance.now() - t0;
  $('sk-progress').classList.add('done');
  if (state.slow && SIZE > 32) { $('sk-steps').value = 15; $('sk-steps').dispatchEvent(new Event('input')); }
  // starting noise: #1 by default; light mode (1 sample) uses a seed that gives a nice first result.
  // ?skseed=N overrides both (for testing).
  const qSeed = new URLSearchParams(location.search).get('skseed');
  state.seed = qSeed ? +qSeed : state.slow ? LIGHT_SEED : 1;
  $('sk-seed-v').textContent = `#${state.seed}`;
  $('sk-mode').hidden = !state.slow;
  buildResults();
  state.ready = true;
  $('sk-go').disabled = false;
  setStatus(`Ready · ${tf.getBackend()} · guided + plain step ≈ ${ms.toFixed(0)} ms${state.slow ? ' (light mode)' : ''}`);
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

// One DDIM step for the guided chain xg and its unguided twin xu (both (n, S, S, 1)), with a single UNet call on the
// stacked batch [xg; xu]. On guided steps the loss only uses the first n samples, so the twin gets zero gradient.
// Returns the next states and one packed 1-D tensor for display: [x0g, x0u, dl (or zeros), loss_g, loss_u].
function ddimStep(xg, xu, t, tp, last, guided, mBlur) {
  const n = xg.shape[0], P = n * SIZE * SIZE;
  const [a, s] = unetMod.alphaSigma(t, SIZE), [ap, sp] = unetMod.alphaSigma(tp, SIZE);
  return tf.tidy(() => {
    const X = tf.concat([xg, xu], 0);
    const tt = tf.fill([2 * n], t);
    let E;
    if (guided) {
      let e0 = null;
      const f = (xx) => {
        const e = unetMod.unet(W, xx, tt);
        e0 = tf.keep(e);
        const x0 = tf.div(tf.sub(xx, tf.mul(e, s)), a);
        return tf.sum(outlineLoss(tf.slice(x0, [0, 0, 0, 0], [n, -1, -1, -1]), mBlur));
      };
      const grad = tf.grad(f)(X);
      E = tf.add(e0, tf.mul(grad, state.s * C_GUIDE * s));   // twin rows of grad are zero
      e0.dispose();
    } else {
      E = unetMod.unet(W, X, tt);
    }
    const X0 = tf.clipByValue(tf.div(tf.sub(X, tf.mul(E, s)), a), -1, 1);
    const Eclip = tf.div(tf.sub(X, tf.mul(X0, a)), s);
    const Xn = last ? X0 : tf.add(tf.mul(X0, ap), tf.mul(Eclip, sp));
    const [x0g, x0u] = tf.split(X0, 2, 0);
    const [xgN, xuN] = tf.split(Xn, 2, 0);
    // d loss / d x0_hat (where guidance wants to add or remove ink) for the overlay, only on guided steps
    const dl = guided ? tf.grad((z) => tf.sum(outlineLoss(z, mBlur)))(x0g) : tf.zerosLike(x0g);
    const packed = tf.concat([tf.reshape(x0g, [P]), tf.reshape(x0u, [P]), tf.reshape(dl, [P]),
      tf.reshape(tf.mean(outlineLoss(x0g, mBlur)), [1]), tf.reshape(tf.mean(outlineLoss(x0u, mBlur)), [1])]);
    return { xgN: tf.keep(xgN), xuN: tf.keep(xuN), packed: tf.keep(packed) };
  });
}

// run a guided and/or plain step at batch n and read the result back (compiles shaders for these shapes)
async function warmup(n, kinds = [true, false]) {
  const z = tf.randomNormal([n, SIZE, SIZE, 1]);
  const m = tf.zeros([n, SIZE, SIZE, 1]);
  for (const guided of kinds) {
    const r = ddimStep(z, z, 0.5, 0.45, false, guided, m);
    await r.packed.data();
    r.xgN.dispose(); r.xuN.dispose(); r.packed.dispose();
  }
  z.dispose(); m.dispose();
}

// ------------------------------------------------------------- sampling --
async function generate() {
  if (!state.ready || state.running) return;
  if (!state.strokes.length) { setStatus('Draw an outline first (or pick a preset).'); return; }
  state.running = true; $('sk-go').disabled = true;
  const n = state.batch, N = state.steps, nGuided = Math.round(state.frac * N);
  const M = rasterize(state.strokes);
  const mBlur = tf.keep(tf.tidy(() => blur(tf.tensor(Array.from({ length: n }, () => Array.from(M)).flat(), [n, SIZE, SIZE, 1]))));
  const xInit = tf.randomNormal([n, SIZE, SIZE, 1], 0, 1, 'float32', state.seed);
  let xg = tf.keep(xInit.clone()), xu = tf.keep(xInit.clone());
  xInit.dispose();
  state.frames = [];
  state.outline = M;
  const lossG = [], lossU = [];
  const t0 = performance.now();
  const P = n * SIZE * SIZE;
  for (let k = 0; k < N; k++) {
    const t = 1 - k / N, tp = 1 - (k + 1) / N;
    const guided = k < nGuided && state.s > 0;
    const res = ddimStep(xg, xu, t, tp, k === N - 1, guided, mBlur);
    xg.dispose(); xu.dispose();
    xg = res.xgN; xu = res.xuN;
    const buf = await res.packed.data();            // one GPU -> CPU read per step
    res.packed.dispose();
    const g = buf.subarray(0, P), u = buf.subarray(P, 2 * P), d = buf.subarray(2 * P, 3 * P);
    lossG.push(buf[3 * P]); lossU.push(buf[3 * P + 1]);
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
  const off = document.createElement('canvas'); off.width = off.height = SIZE;
  const o = off.getContext('2d');
  const img = o.createImageData(SIZE, SIZE);
  const paper = toRGB(css('--card')), ink = toRGB(css('--ink'));
  const add = toRGB(css('--m-cg')), rem = toRGB(css('--pencil')), guide = toRGB(css('--m-tfg'));
  let dmax = 1e-12;
  if (opts.grad) for (const v of opts.grad) dmax = Math.max(dmax, Math.abs(v));
  for (let p = 0; p < NPIX(); p++) {
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
  ctx.imageSmoothingEnabled = false;   // crisp model pixels: this is exactly what the model produces
  ctx.drawImage(off, 0, 0, w, w);
}
function drawResults() {
  const f = state.frames && state.frames[state.frame];
  for (const c of document.querySelectorAll('#sk-results canvas')) {
    const i = +c.dataset.i;
    if (!f) { const ctx = fit(c); ctx.fillStyle = css('--bg'); ctx.fillRect(0, 0, c.width, c.height); continue; }
    const sl = (arr) => arr.subarray(NPIX() * i, NPIX() * (i + 1));
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
// Start loading when the section is near the viewport, and only once the browser is idle, so it does not compete
// with the rest of the page loading (requestIdleCallback with a 2 s cap; a plain timeout where unsupported).
const whenIdle = (fn) => ('requestIdleCallback' in window ? requestIdleCallback(fn, { timeout: 2000 }) : setTimeout(fn, 300));
new IntersectionObserver((entries, obs) => {
  if (entries.some((e) => e.isIntersecting)) {
    obs.disconnect();
    whenIdle(() => init().catch((e) => {
      setStatus(`Could not start the model: ${e.message || e}`);
      $('sk-progress-label').textContent = 'Loading failed';
    }));
  }
}, { rootMargin: '600px' }).observe($('draw'));

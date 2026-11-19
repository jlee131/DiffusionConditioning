// Section 2: precomputed Stable Diffusion runs (CG vs CFG vs TFG), synchronized over sampling steps.
// Assets come from pipeline/sd/export.py: site/assets/sd/{index.json, seed{k}/...}.

const BASE = 'assets/sd/';
const METHODS = [
  { key: 'none', label: 'No guidance', note: 'base: photo' },
  { key: 'cfg', label: 'CFG', note: 'two prompts' },
  { key: 'cg', label: 'CG', note: 'trained classifier' },
  { key: 'tfg', label: 'TFG', note: 'style loss on x̂₀' },
];
const $ = (id) => document.getElementById(id);
const uiFont = () => getComputedStyle(document.body).fontFamily;  // canvas text follows the page font
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

const state = { seed: 2, scale: '1', start: '0', frame: 30, playing: false };
let index = null;
const metrics = {};   // seed -> {run: curves}
const images = {};    // url -> HTMLImageElement (loaded or loading)
let dpr = 1;

// ------------------------------------------------------------- loading --
function runName(method) {
  if (method === 'none' || state.scale === '0') return 'none';
  return `${method}_s${state.scale}_start${state.start}`;
}
function hasRun(seed, name) { return !!index.runs[`seed${seed}/${name}`]; }
function img(url) {
  if (!images[url]) {
    const im = new Image();
    im.onload = () => draw();
    im.src = url;
    images[url] = im;
  }
  return images[url];
}
function sprite(seed, name, kind) { return img(`${BASE}seed${seed}/${name}_${kind}.webp`); }
async function loadMetrics(seed) {
  if (!metrics[seed]) {
    const r = await fetch(`${BASE}seed${seed}/metrics.json`);
    metrics[seed] = r.ok ? await r.json() : {};
  }
  return metrics[seed];
}

// ----------------------------------------------------------------- DOM --
function buildColumns() {
  const host = $('sd-cols');
  host.innerHTML = '';
  for (const m of METHODS) {
    const col = document.createElement('div');
    col.className = 'sd-col';
    col.style.setProperty('--mc', `var(--m-${m.key})`);
    col.innerHTML = `
      <h3><span class="m-${m.key}">${m.label}</span><small>${m.note}</small></h3>
      <canvas class="sd-big" data-kind="x0g"></canvas>
      <div class="sd-small">
        <figure><canvas data-kind="x0b"></canvas><figcaption>base x̂₀</figcaption></figure>
        <figure><canvas data-kind="zt"></canvas><figcaption>noisy zₜ</figcaption></figure>
      </div>
      <div class="sd-meter"><span class="lbl">pencil score</span> <b class="mono val">–</b>
        <div class="bar"><i></i></div></div>`;
    col.dataset.method = m.key;
    host.appendChild(col);
  }
}
function buildSeeds() {
  const host = $('sd-seeds');
  host.innerHTML = '';
  for (const s of index.seeds) {
    const b = document.createElement('button');
    b.dataset.v = s;
    b.title = `seed ${s}`;
    b.className = s === state.seed ? 'on' : '';
    if (hasRun(s, 'none')) b.innerHTML = `<img src="${BASE}seed${s}/thumb.webp" alt="seed ${s}">`;
    else { b.textContent = `#${s}`; b.disabled = true; }
    host.appendChild(b);
  }
}

// ---------------------------------------------------------------- draw --
function fitCanvas(c) {
  const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  return c.getContext('2d');
}
function drawCell(canvas, im, cell) {
  const ctx = fitCanvas(canvas);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!im || !im.complete || !im.naturalWidth) {
    ctx.fillStyle = css('--muted');
    ctx.font = `${12 * dpr}px ${uiFont()}`;
    ctx.textAlign = 'center';
    ctx.fillText(im && !im.naturalWidth && im.complete ? 'not computed' : 'loading…', canvas.width / 2, canvas.height / 2);
    return;
  }
  const k = state.frame, cols = index.cols;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(im, (k % cols) * cell, Math.floor(k / cols) * cell, cell, cell, 0, 0, canvas.width, canvas.height);
}
function frameLabel() {
  if (state.frame >= index.n_frames - 1) return 'final image';
  return `step ${state.frame + 1} / 30 · t = ${index.t[state.frame]}`;
}
function draw() {
  if (!index) return;
  const m = metrics[state.seed] || {};
  for (const col of document.querySelectorAll('.sd-col')) {
    const name = runName(col.dataset.method);
    const ok = hasRun(state.seed, name);
    for (const c of col.querySelectorAll('canvas')) {
      const kind = c.dataset.kind;
      drawCell(c, ok ? sprite(state.seed, name, kind) : null, index.sizes[kind]);
    }
    const r = m[name];
    let p = null;
    if (r) p = state.frame >= index.n_frames - 1 ? r.clip_final : r.clip_x0_guided[state.frame];
    col.querySelector('.val').textContent = p == null ? '–' : p.toFixed(2);
    col.querySelector('.bar i').style.width = `${(p || 0) * 100}%`;
    col.querySelector('.lbl').textContent = state.frame >= index.n_frames - 1 ? 'pencil score (final)' : 'pencil score of x̂₀';
  }
  $('sd-time').value = state.frame;
  $('sd-tlabel').textContent = frameLabel();
  drawCharts();
}

// -------------------------------------------------------------- charts --
const charts = [];
function lineChart(canvas, cfg) {
  const ctx = fitCanvas(canvas);
  const w = canvas.width, h = canvas.height;
  const mg = { l: 38 * dpr, r: 10 * dpr, t: 24 * dpr, b: 28 * dpr };
  const n = 30;
  const X = (i) => mg.l + (i / (n - 1)) * (w - mg.l - mg.r);
  const Y = (v) => h - mg.b - ((v - cfg.ymin) / (cfg.ymax - cfg.ymin)) * (h - mg.t - mg.b);
  ctx.clearRect(0, 0, w, h);
  ctx.font = `${10.5 * dpr}px ${uiFont()}`;
  ctx.lineWidth = dpr;
  for (let k = 0; k <= 4; k++) {
    const v = cfg.ymin + ((cfg.ymax - cfg.ymin) * k) / 4;
    ctx.strokeStyle = css('--line');
    ctx.beginPath(); ctx.moveTo(mg.l, Y(v)); ctx.lineTo(w - mg.r, Y(v)); ctx.stroke();
    ctx.fillStyle = css('--muted');
    ctx.textAlign = 'right';
    ctx.fillText(cfg.fmt(v), mg.l - 5 * dpr, Y(v) + 3.5 * dpr);
  }
  ctx.textAlign = 'center';
  for (const i of [0, 9, 19, 29]) ctx.fillText(`${i + 1}`, X(i), h - mg.b + 13 * dpr);
  ctx.fillText('sampling step', (mg.l + w - mg.r) / 2, h - 3 * dpr);
  // start marker
  if (cfg.startStep > 0) {
    ctx.fillStyle = css('--line');
    ctx.globalAlpha = 0.5;
    ctx.fillRect(mg.l, mg.t, X(cfg.startStep) - mg.l, h - mg.t - mg.b);
    ctx.globalAlpha = 1;
  }
  // series
  for (const s of cfg.series) {
    if (!s.vals) continue;
    ctx.strokeStyle = css(s.color);
    ctx.lineWidth = 2 * dpr;
    ctx.setLineDash((s.dash || []).map((d) => d * dpr));
    ctx.beginPath();
    let pen = false;
    s.vals.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      const x = X(i), y = Y(Math.max(cfg.ymin, Math.min(cfg.ymax, v)));
      pen ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      pen = true;
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }
  // legend
  let lx = mg.l + 6 * dpr;
  ctx.textAlign = 'left';
  for (const s of cfg.series) {
    if (!s.vals) continue;
    ctx.fillStyle = css(s.color);
    ctx.fillRect(lx, 6 * dpr, 10 * dpr, 3 * dpr);
    ctx.fillText(s.name, lx + 13 * dpr, 11 * dpr);
    lx += ctx.measureText(s.name).width + 26 * dpr;
  }
  // cursor
  const f = Math.min(state.frame, n - 1);
  ctx.strokeStyle = css('--ink');
  ctx.lineWidth = dpr;
  ctx.setLineDash([3 * dpr, 3 * dpr]);
  ctx.beginPath(); ctx.moveTo(X(f), mg.t); ctx.lineTo(X(f), h - mg.b); ctx.stroke();
  ctx.setLineDash([]);
  canvas._geom = { X0: mg.l / dpr, X1: (w - mg.r) / dpr, n };
}
function drawCharts() {
  const m = metrics[state.seed];
  if (!m) return;
  const run = (k) => m[runName(k)];
  const start = state.scale === '0' ? 0 : +state.start;
  const none = m.none || {};
  const mag = ['cfg', 'cg', 'tfg'].map((k) => ({ name: k.toUpperCase(), color: `--m-${k}`, vals: run(k)?.guid_rms }));
  const peak = Math.max(0.1, ...mag.flatMap((s) => (s.vals || []).filter((v) => v != null)));
  const ymaxMag = [0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2, 3, 4].find((v) => v >= peak * 1.05) ?? Math.ceil(peak);
  lineChart($('sd-chart-mag'), { series: mag, ymin: 0, ymax: ymaxMag, fmt: (v) => v.toFixed(2), startStep: start });
  lineChart($('sd-chart-cos'), {
    series: [
      { name: 'CFG vs CG', color: '--m-cg', vals: none.cos_cfg_cg },
      { name: 'CFG vs TFG', color: '--m-tfg', vals: none.cos_cfg_tfg },
    ], ymin: -0.2, ymax: 1, fmt: (v) => v.toFixed(1), startStep: 0,
  });
  lineChart($('sd-chart-clip'), {
    series: METHODS.map((x) => ({ name: x.key === 'none' ? 'none' : x.label, color: `--m-${x.key}`,
      vals: run(x.key)?.clip_x0_guided, dash: x.key === 'none' ? [4, 3] : [] })),
    ymin: 0, ymax: 1, fmt: (v) => v.toFixed(1), startStep: start,
  });
}
function chartScrub(e) {
  const g = e.currentTarget._geom;
  if (!g) return;
  const r = e.currentTarget.getBoundingClientRect();
  const u = (e.clientX - r.left - g.X0) / (g.X1 - g.X0);
  state.frame = Math.max(0, Math.min(g.n - 1, Math.round(u * (g.n - 1))));
  stop();
  draw();
}

// ------------------------------------------------------------ controls --
function seg(id, key) {
  $(id).addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b || b.disabled) return;
    for (const x of $(id).children) x.classList.toggle('on', x === b);
    state[key] = key === 'seed' ? +b.dataset.v : b.dataset.v;
    if (key === 'seed') loadMetrics(state.seed).then(draw);
    draw();
  });
}
let last = 0;
function tick(ts) {
  if (!state.playing) return;
  if (ts - last > 220) {
    last = ts;
    if (state.frame >= index.n_frames - 1) { stop(); return; }
    state.frame++;
    draw();
  }
  requestAnimationFrame(tick);
}
function stop() { state.playing = false; $('sd-play').textContent = '▶'; }

async function init() {
  const r = await fetch(`${BASE}index.json`).catch(() => null);
  if (!r || !r.ok) {
    $('sd-cols').innerHTML = '<p class="sd-missing">Precomputed assets not found (run pipeline/sd/export.py).</p>';
    return;
  }
  index = await r.json();
  dpr = window.devicePixelRatio || 1;
  // Optional URL state, e.g. ?sdseed=3&sds=2&sdstart=10&sdframe=12
  const q = new URLSearchParams(location.search);
  if (q.has('sdseed')) state.seed = +q.get('sdseed');
  if (q.has('sds')) state.scale = q.get('sds');
  if (q.has('sdstart')) state.start = q.get('sdstart');
  if (q.has('sdframe')) state.frame = Math.max(0, Math.min(index.n_frames - 1, +q.get('sdframe')));
  for (const [id, v] of [['sd-scale', state.scale], ['sd-start', state.start]])
    for (const b of $(id).children) b.classList.toggle('on', b.dataset.v === v);
  if (!hasRun(state.seed, 'none')) state.seed = index.seeds.find((s) => hasRun(s, 'none')) ?? index.seeds[0];
  buildColumns();
  buildSeeds();
  $('sd-ref').src = `${BASE}tfg_reference.webp`;
  seg('sd-seeds', 'seed');
  seg('sd-scale', 'scale');
  seg('sd-start', 'start');
  $('sd-time').max = index.n_frames - 1;
  $('sd-time').addEventListener('input', (e) => { state.frame = +e.target.value; stop(); draw(); });
  $('sd-play').addEventListener('click', () => {
    if (state.playing) { stop(); return; }
    if (state.frame >= index.n_frames - 1) state.frame = 0;
    state.playing = true;
    $('sd-play').textContent = '❚❚';
    requestAnimationFrame(tick);
  });
  for (const id of ['sd-chart-mag', 'sd-chart-cos', 'sd-chart-clip']) {
    $(id).addEventListener('click', chartScrub);
    $(id).addEventListener('mousemove', (e) => { if (e.buttons & 1) chartScrub(e); });
  }
  window.addEventListener('resize', () => { dpr = window.devicePixelRatio || 1; draw(); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
  window.addEventListener('themechange', draw);
  await loadMetrics(state.seed);
  draw();
}
init();

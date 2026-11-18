// Toy 2D diffusion with a labeled Gaussian mixture: every quantity is closed form.
//
// Data:   p0(x) = sum_k w_k N(mu_k, s_k^2 I),  label prob q_k = p(pencil | k)
// Noise:  x_t = alpha_t x0 + sigma_t eps  (VP, alpha^2 + sigma^2 = 1)
// Then p_t is again a mixture with means alpha mu_k and variances v_k = alpha^2 s_k^2 + sigma^2.

export const ALPHA_MIN = 0.05;
const PHI_MAX = Math.acos(ALPHA_MIN);

/** VP cosine-type schedule on t in [0, 1]. */
export function schedule(t) {
  const phi = t * PHI_MAX;
  return { a: Math.cos(phi), s: Math.sin(phi) };
}

export const DEFAULT_COMPONENTS = [
  // photo cats (mostly label 0)
  { w: 0.25, mu: [-1.6, 1.25], s: 0.35, q: 0.05 },
  { w: 0.20, mu: [-1.9, -0.85], s: 0.40, q: 0.05 },
  { w: 0.20, mu: [0.25, -1.9], s: 0.35, q: 0.05 },
  // pencil cats (mostly label 1)
  { w: 0.10, mu: [1.75, 1.45], s: 0.35, q: 0.95 },
  { w: 0.15, mu: [1.55, -0.55], s: 0.30, q: 0.95 },
  // ambiguous
  { w: 0.10, mu: [0.0, 0.55], s: 0.30, q: 0.50 },
];

// Deterministic pseudo-random numbers (for particles and perturbations).
export function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function gaussianPairs(n, seed) {
  const rnd = mulberry32(seed);
  const xs = new Float64Array(n), ys = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const u = Math.max(rnd(), 1e-12), v = rnd();
    const r = Math.sqrt(-2 * Math.log(u));
    xs[i] = r * Math.cos(2 * Math.PI * v);
    ys[i] = r * Math.sin(2 * Math.PI * v);
  }
  return [xs, ys];
}

/**
 * Mixture quantities at one point (x, y) and time with (a, s).
 * labelWeight: null -> unconditional weights w_k; 'c' -> w_k q_k (conditional on pencil).
 * Returns score (gradient of log density), log density, responsibilities r_k, per-component scores.
 */
function mixture(comps, a, sg, x, y, labelWeight) {
  const K = comps.length;
  const logs = new Array(K), sx = new Array(K), sy = new Array(K), v = new Array(K);
  let maxL = -Infinity;
  for (let k = 0; k < K; k++) {
    const c = comps[k];
    const vk = a * a * c.s * c.s + sg * sg;
    const dx = x - a * c.mu[0], dy = y - a * c.mu[1];
    const wk = labelWeight === 'c' ? c.w * c.q : c.w;
    const l = Math.log(Math.max(wk, 1e-300)) - Math.log(2 * Math.PI * vk) - (dx * dx + dy * dy) / (2 * vk);
    logs[k] = l; v[k] = vk; sx[k] = -dx / vk; sy[k] = -dy / vk;
    if (l > maxL) maxL = l;
  }
  let Z = 0;
  for (let k = 0; k < K; k++) Z += Math.exp(logs[k] - maxL);
  const r = new Array(K);
  let gx = 0, gy = 0;
  for (let k = 0; k < K; k++) {
    r[k] = Math.exp(logs[k] - maxL) / Z;
    gx += r[k] * sx[k]; gy += r[k] * sy[k];
  }
  return { gx, gy, logp: maxL + Math.log(Z), r, sx, sy, v };
}

/**
 * Everything the visualization needs at a point.
 *   score     : grad log p_t(x)                         (unconditional / base)
 *   cg        : grad log p_t(c|x) via the classifier p_t(c|x) = sum_k r_k q_k   (oracle CG)
 *   cfg       : grad log p_t(x|c) - grad log p_t(x)     (oracle CFG)
 *   pc        : p_t(c | x)
 *   x0        : Tweedie E[x0 | x_t]
 *   J         : d x0 / d x_t = (alpha / sigma^2) Cov[x0 | x_t]   (2x2, symmetric)
 */
export function fieldAt(comps, t, x, y, wantJ = false) {
  const { a, s: sg } = schedule(t);
  const base = mixture(comps, a, sg, x, y, null);
  const cond = mixture(comps, a, sg, x, y, 'c');
  const K = comps.length;

  // Oracle CG, computed from the classifier directly (independent of the conditional mixture):
  //   grad log sum_k r_k q_k = sum_k r_k q_k (s_k - score) / sum_k r_k q_k
  let pc = 0, nx = 0, ny = 0;
  for (let k = 0; k < K; k++) {
    const rq = base.r[k] * comps[k].q;
    pc += rq;
    nx += rq * (base.sx[k] - base.gx);
    ny += rq * (base.sy[k] - base.gy);
  }
  const cg = [nx / pc, ny / pc];
  const cfg = [cond.gx - base.gx, cond.gy - base.gy];

  // Tweedie posterior mean: per-component posterior N(mpost_k, cvar_k I)
  let x0x = 0, x0y = 0;
  const mpx = new Array(K), mpy = new Array(K), cvar = new Array(K);
  for (let k = 0; k < K; k++) {
    const c = comps[k], vk = base.v[k];
    const gain = (a * c.s * c.s) / vk;
    mpx[k] = c.mu[0] + gain * (x - a * c.mu[0]);
    mpy[k] = c.mu[1] + gain * (y - a * c.mu[1]);
    cvar[k] = (c.s * c.s * sg * sg) / vk;
    x0x += base.r[k] * mpx[k]; x0y += base.r[k] * mpy[k];
  }
  // logpxc = log p_t(x, c) (unnormalized conditional density, up to the constant log p(c))
  const out = { a, sg, score: [base.gx, base.gy], cg, cfg, pc, logp: base.logp, logpxc: cond.logp,
    x0: [x0x, x0y], r: base.r };

  if (wantJ) {
    let cxx = 0, cxy = 0, cyy = 0;
    for (let k = 0; k < K; k++) {
      const rk = base.r[k];
      cxx += rk * (cvar[k] + mpx[k] * mpx[k]);
      cyy += rk * (cvar[k] + mpy[k] * mpy[k]);
      cxy += rk * mpx[k] * mpy[k];
    }
    cxx -= x0x * x0x; cyy -= x0y * x0y; cxy -= x0x * x0y;
    const f = a / Math.max(sg * sg, 1e-9);
    out.J = [f * cxx, f * cxy, f * cyy];  // [Jxx, Jxy, Jyy]
  }
  return out;
}

/** A perturbed copy of the components, to emulate an imperfect learned model. */
export function perturb(comps, kappa, seed) {
  const rnd = mulberry32(seed);
  return comps.map((c) => {
    const ang = 2 * Math.PI * rnd(), len = 0.6 * kappa * (0.5 + rnd());
    return {
      ...c,
      mu: [c.mu[0] + len * Math.cos(ang), c.mu[1] + len * Math.sin(ang)],
      s: c.s * (1 + 0.8 * kappa),
      q: Math.min(0.99, Math.max(0.01, c.q + kappa * 0.4 * (rnd() - 0.5))),
    };
  });
}

/**
 * Guidance direction g_t ~ grad log p_t(c | x) for each method.
 *   oracle:    CG and CFG are identical fields.
 *   estimated: CG uses an imperfect classifier (model clsComps), CFG an imperfect conditional model (condComps).
 *   TFG:       g = -grad_x l(x0_hat(x)) = -J^T (x0_hat - target),  l = |x0_hat - target|^2 / 2.
 */
export function guidanceAt(model, method, t, x, y, f = null) {
  if (method === 'none') return [0, 0];
  if (method === 'tfg') {
    f = f || fieldAt(model.comps, t, x, y, true);
    if (!f.J) f = fieldAt(model.comps, t, x, y, true);
    const [Jxx, Jxy, Jyy] = f.J;
    const ex = f.x0[0] - model.target[0], ey = f.x0[1] - model.target[1];
    return [-(Jxx * ex + Jxy * ey), -(Jxy * ex + Jyy * ey)];
  }
  if (!model.estimated) {
    f = f || fieldAt(model.comps, t, x, y);
    return method === 'cg' ? f.cg : f.cfg;
  }
  if (method === 'cg') return fieldAt(model.clsComps, t, x, y).cg;
  // estimated CFG: imperfect conditional score minus the (exact) unconditional score
  const fc = fieldAt(model.condComps, t, x, y);
  const fb = f || fieldAt(model.comps, t, x, y);
  const condScore = [fc.score[0] + fc.cfg[0], fc.score[1] + fc.cfg[1]];
  return [condScore[0] - fb.score[0], condScore[1] - fb.score[1]];
}

/**
 * Deterministic DDIM (probability-flow) sampling of many particles from shared initial noise.
 * cfg: { method, scale, tStart, nSteps }   guidance active while t <= tStart.
 * Returns frames[i] = { t, x, y, x0bx, x0by, x0gx, x0gy } (positions before step i; last frame = final).
 */
export function simulate(model, cfg, noise) {
  const [nx, ny] = noise;
  const n = nx.length, N = cfg.nSteps;
  let x = Float64Array.from(nx), y = Float64Array.from(ny);
  // x_T = alpha_T x0 + sigma_T eps with alpha_T = 0.05: start from sigma_T * eps (shared initial noise)
  const { s: sT } = schedule(1);
  for (let i = 0; i < n; i++) { x[i] *= sT; y[i] *= sT; }
  const frames = [];
  for (let step = 0; step <= N; step++) {
    const t = 1 - step / N;
    const fr = { t, x: Float64Array.from(x), y: Float64Array.from(y),
      x0bx: new Float64Array(n), x0by: new Float64Array(n), x0gx: new Float64Array(n), x0gy: new Float64Array(n) };
    if (step === N) {
      fr.x0bx.set(x); fr.x0by.set(y); fr.x0gx.set(x); fr.x0gy.set(y);
      frames.push(fr);
      break;
    }
    const { a, s } = schedule(t);
    const { a: ap, s: sp } = schedule(1 - (step + 1) / N);
    const active = cfg.method !== 'none' && t <= cfg.tStart + 1e-9 && cfg.scale !== 0;
    for (let i = 0; i < n; i++) {
      const f = fieldAt(model.comps, t, x[i], y[i], active && cfg.method === 'tfg');
      let gx = f.score[0], gy = f.score[1];
      if (active) {
        const g = guidanceAt(model, cfg.method, t, x[i], y[i], f);
        gx += cfg.scale * g[0]; gy += cfg.scale * g[1];
      }
      // eps = -sigma * score ;  x0 = (x + sigma^2 score) / alpha ;  x' = alpha' x0 + sigma' eps
      const x0x = (x[i] + s * s * gx) / a, x0y = (y[i] + s * s * gy) / a;
      fr.x0bx[i] = f.x0[0]; fr.x0by[i] = f.x0[1];
      fr.x0gx[i] = x0x; fr.x0gy[i] = x0y;
      x[i] = ap * x0x - sp * s * gx;
      y[i] = ap * x0y - sp * s * gy;
    }
    frames.push(fr);
  }
  return frames;
}

/** p(pencil) of a clean point: label probability under the t=0 responsibilities. */
export function labelOf(comps, x, y) {
  const f = fieldAt(comps, 0, x, y);
  return f.pc;
}

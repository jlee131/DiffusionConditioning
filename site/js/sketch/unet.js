// TF.js port of pipeline/sketch/model.py (SketchUNet, cosine schedule). Expects the global `tf` (CDN script).
// Weights come from pipeline/sketch/export_tfjs.py, already in TF.js layout (NHWC, HWIO conv filters).

const S_COS = 0.008;
const clamp = (v) => Math.min(1 - 1e-5, Math.max(1e-5, v));

// ---------------------------------------------------------------- schedule --
// Cosine schedule; for size > 32 the log-SNR is shifted by 2 log(32 / size) (same as model.py).
export function alphaBar(t, size = 32) {
  const f = (u) => Math.cos(((u + S_COS) / (1 + S_COS)) * Math.PI / 2) ** 2;
  let ab = clamp(f(t) / f(0));
  if (size !== 32) {
    const logit = Math.log(ab / (1 - ab)) + 2 * Math.log(32 / size);
    ab = clamp(1 / (1 + Math.exp(-logit)));
  }
  return ab;
}
export function alphaSigma(t, size = 32) {
  const ab = alphaBar(t, size);
  return [Math.sqrt(ab), Math.sqrt(1 - ab)];
}

// ------------------------------------------------------------------ weights --
// onProgress(fraction in [0, 1]) is called while weights.bin downloads.
// The true size comes from the manifest: servers such as GitHub Pages gzip the file, and then Content-Length is the
// compressed size while the stream yields decompressed bytes, so it cannot size the buffer.
export async function loadWeights(base, onProgress = null) {
  const manifest = await fetch(`${base}manifest.json`).then((r) => r.json());
  const res = await fetch(`${base}weights.bin`);
  if (!res.ok) throw new Error(`weights.bin: HTTP ${res.status}`);
  const total = 4 * manifest.tensors.reduce((m, t) => Math.max(m, t.offset + t.shape.reduce((a, b) => a * b, 1)), 0);
  let buf;
  if (onProgress && res.body) {
    const chunks = [];
    const reader = res.body.getReader();
    let got = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length;
      onProgress(Math.min(1, got / total));
    }
    const bytes = new Uint8Array(got);
    let o = 0;
    for (const c of chunks) { bytes.set(c, o); o += c.length; }
    buf = bytes.buffer;
  } else {
    buf = await res.arrayBuffer();
  }
  const all = new Float32Array(buf);
  const W = {};
  for (const { name, shape, offset } of manifest.tensors) {
    const n = shape.reduce((a, b) => a * b, 1);
    W[name] = tf.tensor(all.subarray(offset, offset + n), shape, 'float32');
  }
  return { W, config: { size: 32, ...manifest.config }, guide: manifest.guide || null, step: manifest.step };
}

// ---------------------------------------------------------------------- ops --
const silu = (x) => tf.mul(x, tf.sigmoid(x));

function groupNorm(x, gamma, beta, groups = 8, eps = 1e-5) {
  const [B, H, Wd, C] = x.shape;
  const g = tf.reshape(x, [B, H, Wd, groups, C / groups]);
  const { mean, variance } = tf.moments(g, [1, 2, 4], true);
  const y = tf.reshape(tf.div(tf.sub(g, mean), tf.sqrt(tf.add(variance, eps))), [B, H, Wd, C]);
  return tf.add(tf.mul(y, gamma), beta);
}

// PyTorch Conv2d(k=3, padding=1): stride 1 -> 'same'; stride 2 -> explicit symmetric pad + 'valid'
// (TF 'same' pads asymmetrically for stride 2, which would not match PyTorch).
function conv(x, W, name, stride = 1) {
  const w = W[`${name}.weight`], b = W[`${name}.bias`];
  const k = w.shape[0];
  let y;
  if (k === 1) y = tf.conv2d(x, w, 1, 'valid');
  else if (stride === 1) y = tf.conv2d(x, w, 1, 'same');
  else y = tf.conv2d(tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]]), w, stride, 'valid');
  return tf.add(y, b);
}
const dense = (x, W, name) => tf.add(tf.matMul(x, W[`${name}.weight`]), W[`${name}.bias`]);

function resBlock(x, temb, W, name) {
  let h = conv(silu(groupNorm(x, W[`${name}.n1.weight`], W[`${name}.n1.bias`])), W, `${name}.c1`);
  const tb = dense(silu(temb), W, `${name}.t`);                           // (B, C)
  h = tf.add(h, tf.reshape(tb, [tb.shape[0], 1, 1, tb.shape[1]]));
  h = conv(silu(groupNorm(h, W[`${name}.n2.weight`], W[`${name}.n2.bias`])), W, `${name}.c2`);
  const skip = W[`${name}.skip.weight`] ? conv(x, W, `${name}.skip`) : x;
  return tf.add(h, skip);
}

function timestepEmbedding(t, dim = 128) {       // t: tensor (B,) in [0, 1]
  const half = dim / 2;
  const freqs = tf.exp(tf.mul(tf.range(0, half, 1, 'float32'), -Math.log(10000) / half));
  const args = tf.mul(tf.reshape(tf.mul(t, 1000), [-1, 1]), tf.reshape(freqs, [1, -1]));
  return tf.concat([tf.cos(args), tf.sin(args)], 1);
}

// Nearest 2x upsampling with concat + reshape only (rank <= 4, all gradients well supported).
// tf.image.resizeNearestNeighbor has a wrong gradient on the CPU backend for batch > 1 in tfjs 4.22, and tile's
// gradient does not support rank 6 (both found by site/tests/tfjs_grad_ops.html).
function up2x(x) {
  const [B, H, Wd, C] = x.shape;
  const w2 = tf.reshape(tf.concat([x, x], 3), [B, H, 2 * Wd, C]);              // duplicate each column
  const flat = tf.reshape(w2, [B, H, 2 * Wd * C]);
  return tf.reshape(tf.concat([flat, flat], 2), [B, 2 * H, 2 * Wd, C]);          // duplicate each row
}

// ------------------------------------------------------------------- model --
// x: (B, S, S, 1) NHWC in [-1, 1] space; t: (B,) tensor in [0, 1]. Returns predicted eps (B, S, S, 1).
// The number of levels L is read from the weights (d0.., d{L-1}), matching model.py for any L.
export function unet(W, x, t) {
  let L = 0;
  while (W[`d${L}.0.n1.weight`]) L++;
  let temb = dense(timestepEmbedding(t), W, 'temb.0');
  temb = dense(silu(temb), W, 'temb.2');
  let h = conv(x, W, 'stem');
  const skips = [];
  for (let i = 0; i < L; i++) {
    h = resBlock(h, temb, W, `d${i}.0`); h = resBlock(h, temb, W, `d${i}.1`);
    skips.push(h);
    if (i < L - 1) h = conv(h, W, `down${i}`, 2);
  }
  h = resBlock(h, temb, W, 'mid.0'); h = resBlock(h, temb, W, 'mid.1');
  for (let i = L - 1; i >= 0; i--) {
    h = resBlock(tf.concat([h, skips[i]], 3), temb, W, `u${i}`);
    if (i > 0) h = conv(up2x(h), W, `up${i}`);
  }
  return conv(silu(groupNorm(h, W['out_norm.weight'], W['out_norm.bias'])), W, 'out');
}

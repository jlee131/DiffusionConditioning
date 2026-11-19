// TF.js port of pipeline/sketch/model.py (SketchUNet, cosine schedule). Expects the global `tf` (CDN script).
// Weights come from pipeline/sketch/export_tfjs.py, already in TF.js layout (NHWC, HWIO conv filters).

const S_COS = 0.008;

// ---------------------------------------------------------------- schedule --
export function alphaBar(t) {
  const f = (u) => Math.cos(((u + S_COS) / (1 + S_COS)) * Math.PI / 2) ** 2;
  return Math.min(1 - 1e-5, Math.max(1e-5, f(t) / f(0)));
}
export function alphaSigma(t) {
  const ab = alphaBar(t);
  return [Math.sqrt(ab), Math.sqrt(1 - ab)];
}

// ------------------------------------------------------------------ weights --
export async function loadWeights(base) {
  const [manifest, buf] = await Promise.all([
    fetch(`${base}manifest.json`).then((r) => r.json()),
    fetch(`${base}weights.bin`).then((r) => r.arrayBuffer()),
  ]);
  const all = new Float32Array(buf);
  const W = {};
  for (const { name, shape, offset } of manifest.tensors) {
    const n = shape.reduce((a, b) => a * b, 1);
    W[name] = tf.tensor(all.subarray(offset, offset + n), shape, 'float32');
  }
  return { W, config: manifest.config, step: manifest.step };
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
// x: (B, 32, 32, 1) NHWC in [-1, 1] space; t: (B,) tensor in [0, 1]. Returns predicted eps (B, 32, 32, 1).
export function unet(W, x, t) {
  let temb = dense(timestepEmbedding(t), W, 'temb.0');
  temb = dense(silu(temb), W, 'temb.2');
  let h = conv(x, W, 'stem');
  h = resBlock(h, temb, W, 'd0.0'); h = resBlock(h, temb, W, 'd0.1');
  const s0 = h;
  h = conv(h, W, 'down0', 2);
  h = resBlock(h, temb, W, 'd1.0'); h = resBlock(h, temb, W, 'd1.1');
  const s1 = h;
  h = conv(h, W, 'down1', 2);
  h = resBlock(h, temb, W, 'd2.0'); h = resBlock(h, temb, W, 'd2.1');
  const s2 = h;
  h = resBlock(h, temb, W, 'mid.0'); h = resBlock(h, temb, W, 'mid.1');
  h = resBlock(tf.concat([h, s2], 3), temb, W, 'u2');
  h = conv(up2x(h), W, 'up2');
  h = resBlock(tf.concat([h, s1], 3), temb, W, 'u1');
  h = conv(up2x(h), W, 'up1');
  h = resBlock(tf.concat([h, s0], 3), temb, W, 'u0');
  return conv(silu(groupNorm(h, W['out_norm.weight'], W['out_norm.bias'])), W, 'out');
}

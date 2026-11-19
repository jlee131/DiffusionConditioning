"""Export the sketch UNet (EMA weights) for the browser, plus parity test vectors (seconds, CPU).

weights.bin    float32 little-endian, all tensors back to back, already in TF.js layout:
               conv  (out, in, kh, kw) -> (kh, kw, in, out)   [NHWC conv2d filter]
               dense (out, in)         -> (in, out)
               GroupNorm / bias        -> unchanged
manifest.json  {config, tensors: [{name, shape, offset}]}, offsets in float32 elements
parity.json    inputs and PyTorch float32 outputs for site/tests/sketch_parity.html:
               eps = model(x, t); grad = d/dx mean((x0_hat(x) - target)^2), x0_hat = (x - sigma eps) / alpha;
               alpha_bar at a few t (checks the JS schedule)

Output: site/assets/sketch/{weights.bin, manifest.json, parity.json}
Usage:  python -m pipeline.sketch.export_tfjs [--ckpt runs/sketch/ckpt.pt]
"""
import argparse
import json

import numpy as np
import torch

from ..sd.models import ROOT
from .model import SketchUNet, alpha_bar, alpha_sigma

DST = ROOT / "site" / "assets" / "sketch"


def to_tfjs_layout(name, w):
    w = w.detach().float().cpu().numpy()
    if w.ndim == 4:                       # conv
        return w.transpose(2, 3, 1, 0)
    if w.ndim == 2:                       # linear
        return w.transpose(1, 0)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "sketch" / "ckpt.pt"))
    args = ap.parse_args()
    DST.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = SketchUNet(**{k: tuple(v) if k == "ch" else v for k, v in ck["config"].items()})
    model.load_state_dict(ck["ema"])
    model.eval()

    tensors, chunks, offset = [], [], 0
    for name, w in model.state_dict().items():
        a = np.ascontiguousarray(to_tfjs_layout(name, w), dtype="<f4")
        tensors.append({"name": name, "shape": list(a.shape), "offset": offset})
        chunks.append(a.ravel())
        offset += a.size
    np.concatenate(chunks).tofile(DST / "weights.bin")
    json.dump({"config": model.config, "step": ck.get("step"), "tensors": tensors},
              open(DST / "manifest.json", "w"), indent=1)

    # parity vectors (float32, no autocast)
    g = torch.Generator().manual_seed(123)
    x = torch.randn(2, 1, 32, 32, generator=g)
    t = torch.tensor([0.83, 0.27])
    target = (torch.rand(2, 1, 32, 32, generator=g) > 0.8).float() * 2 - 1
    x.requires_grad_(True)
    eps = model(x, t)
    a, s = alpha_sigma(t)
    x0 = (x - s[:, None, None, None] * eps) / a[:, None, None, None]
    loss = ((x0 - target) ** 2).mean()
    (grad,) = torch.autograd.grad(loss, x)
    tt = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    nhwc = lambda v: v.detach().permute(0, 2, 3, 1).reshape(-1).tolist()  # noqa: E731
    json.dump({"x": nhwc(x), "t": t.tolist(), "target": nhwc(target), "eps": nhwc(eps), "loss": loss.item(),
               "grad": nhwc(grad), "ab_t": tt.tolist(), "ab": alpha_bar(tt).tolist()},
              open(DST / "parity.json", "w"))
    mb = (DST / "weights.bin").stat().st_size / 1e6
    print(f"exported step {ck.get('step')}: {len(tensors)} tensors, {offset / 1e6:.2f}M floats, {mb:.1f} MB -> {DST}")


if __name__ == "__main__":
    main()

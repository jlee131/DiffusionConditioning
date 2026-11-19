"""Export the sketch UNet (EMA weights) for the browser, plus parity test vectors (seconds, CPU).

weights.bin    float32 little-endian, all tensors back to back, already in TF.js layout:
               conv  (out, in, kh, kw) -> (kh, kw, in, out)   [NHWC conv2d filter]
               dense (out, in)         -> (in, out)
               GroupNorm / bias        -> unchanged
manifest.json  {config (incl. size), step, guide: {c, s, blur_sigma}, tensors: [{name, shape, offset}]}
               offsets in float32 elements; `guide` holds the outline-guidance constants the page uses
parity.json    inputs and PyTorch float32 outputs for site/tests/sketch_parity.html:
               eps = model(x, t); grad = d/dx mean((x0_hat(x) - target)^2), x0_hat = (x - sigma eps) / alpha;
               alpha_bar at a few t (checks the JS schedule, including the resolution shift)

Output: site/assets/{out}/{weights.bin, manifest.json, parity.json}
Usage:  python -m pipeline.sketch.export_tfjs                                              (v1: runs/sketch -> sketch)
        python -m pipeline.sketch.export_tfjs --ckpt runs/sketch64/ckpt.pt --out sketch64 --guide-c 4e4
"""
import argparse
import json

import numpy as np
import torch

from ..sd.models import ROOT
from .model import alpha_bar, alpha_sigma, load_sketch_model


def to_tfjs_layout(w):
    w = w.detach().float().cpu().numpy()
    if w.ndim == 4:                       # conv
        return w.transpose(2, 3, 1, 0)
    if w.ndim == 2:                       # linear
        return w.transpose(1, 0)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs" / "sketch" / "ckpt.pt"))
    ap.add_argument("--out", default="sketch", help="site/assets/<out>/")
    ap.add_argument("--guide-c", type=float, default=1e4, help="strength constant c (calibrated with guided_test)")
    ap.add_argument("--guide-s", type=float, default=0.25, help="default strength s")
    ap.add_argument("--blur-sigma", type=float, default=None, help="loss blur in pixels (default size / 32)")
    args = ap.parse_args()
    dst = ROOT / "site" / "assets" / args.out
    dst.mkdir(parents=True, exist_ok=True)

    model, ck = load_sketch_model(args.ckpt, device="cpu")
    size = model.config.get("size", 32)

    tensors, chunks, offset = [], [], 0
    for name, w in model.state_dict().items():
        a = np.ascontiguousarray(to_tfjs_layout(w), dtype="<f4")
        tensors.append({"name": name, "shape": list(a.shape), "offset": offset})
        chunks.append(a.ravel())
        offset += a.size
    np.concatenate(chunks).tofile(dst / "weights.bin")
    guide = {"c": args.guide_c, "s": args.guide_s, "blur_sigma": args.blur_sigma or size / 32}
    json.dump({"config": model.config, "step": ck.get("step"), "guide": guide, "tensors": tensors},
              open(dst / "manifest.json", "w"), indent=1)

    # parity vectors (float32, no autocast)
    g = torch.Generator().manual_seed(123)
    x = torch.randn(2, 1, size, size, generator=g)
    t = torch.tensor([0.83, 0.27])
    target = (torch.rand(2, 1, size, size, generator=g) > 0.8).float() * 2 - 1
    x.requires_grad_(True)
    with torch.enable_grad():
        eps = model(x, t)
        a, s = alpha_sigma(t, size)
        x0 = (x - s[:, None, None, None] * eps) / a[:, None, None, None]
        loss = ((x0 - target) ** 2).mean()
        (grad,) = torch.autograd.grad(loss, x)
    tt = torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0])
    nhwc = lambda v: v.detach().permute(0, 2, 3, 1).reshape(-1).tolist()  # noqa: E731
    json.dump({"size": size, "x": nhwc(x), "t": t.tolist(), "target": nhwc(target), "eps": nhwc(eps),
               "loss": loss.item(), "grad": nhwc(grad), "ab_t": tt.tolist(), "ab": alpha_bar(tt, size).tolist()},
              open(dst / "parity.json", "w"))
    mb = (dst / "weights.bin").stat().st_size / 1e6
    print(f"exported step {ck.get('step')} ({size}x{size}): {len(tensors)} tensors, {offset / 1e6:.2f}M floats, "
          f"{mb:.1f} MB, guide {guide} -> {dst}")


if __name__ == "__main__":
    main()

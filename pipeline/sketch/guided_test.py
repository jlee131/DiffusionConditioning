"""Outline-guided (TFG) sampling test in PyTorch, to choose the loss and the strength scale before the browser UI.

Guided DDIM: eps_tilde = eps + s * c * sigma_t * grad_x loss(ink(x0_hat(x))), applied for the first `frac` of steps,
with the gradient taken through x0_hat and the UNet (as in the browser). Losses on blurred ink maps (G = Gaussian):
    mse      : mean((G*ink - G*M)^2)
    coverage : sum(m * relu(m - i)^2) / sum(m),  m = G*M, i = G*ink   (only missing ink on the outline is penalized)
c is calibrated once per loss: c = 1 / median_t rms(sigma_t grad) on an unguided probe, so s = 1 means a guidance
term about as large as the noise prediction (rms 1).

Output: results/sketch/guided_{loss}.png (rows: outline, unguided, s values; 8 seeds), printed calibration constants
Usage:  python -m pipeline.sketch.guided_test [--steps 20] [--frac 0.6]
"""
import argparse
import statistics

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ..sd.models import ROOT
from .model import SketchUNet, alpha_sigma

DEVICE = "cuda"
RESULTS = ROOT / "results" / "sketch"


def outline_cat_head(size=32, ss=4):
    im = Image.new("L", (size * ss, size * ss), 0)
    d = ImageDraw.Draw(im)
    w = round(1.6 * ss)
    s = lambda pts: [(x * ss, y * ss) for x, y in pts]  # noqa: E731
    d.ellipse([6 * ss, 9 * ss, 26 * ss, 28 * ss], outline=255, width=w)
    d.line(s([(8, 13), (9, 3), (14, 9.5)]), fill=255, width=w, joint="curve")
    d.line(s([(18, 9.5), (23, 3), (24, 13)]), fill=255, width=w, joint="curve")
    return np.asarray(im.resize((size, size), Image.LANCZOS), np.float32) / 255


def outline_body(size=32, ss=4):
    im = Image.new("L", (size * ss, size * ss), 0)
    d = ImageDraw.Draw(im)
    w = round(1.6 * ss)
    d.ellipse([4 * ss, 12 * ss, 28 * ss, 29 * ss], outline=255, width=w)
    return np.asarray(im.resize((size, size), Image.LANCZOS), np.float32) / 255


def gaussian_kernel(sigma=1.0, r=2):
    ax = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(ax[:, None] ** 2 + ax[None] ** 2) / (2 * sigma ** 2))
    return (k / k.sum())[None, None].to(DEVICE)


def make_loss(kind, M):
    G = gaussian_kernel()
    m = F.conv2d(M, G, padding=2)

    def loss(x0):
        ink = (x0.clamp(-1, 1) + 1) / 2
        i = F.conv2d(ink, G, padding=2)
        if kind == "mse":
            return ((i - m) ** 2).mean(dim=(1, 2, 3))
        if kind == "mse_w":   # extra ink away from the outline costs only 20% (room for eyes, whiskers)
            w = torch.where(i > m, 0.2 + 0.8 * (m / m.amax(dim=(2, 3), keepdim=True)), torch.ones_like(m))
            return (w * (i - m) ** 2).mean(dim=(1, 2, 3))
        return (m * F.relu(m - i) ** 2).sum(dim=(1, 2, 3)) / m.sum(dim=(1, 2, 3))
    return loss


def guided_sample(model, loss_fn, n, steps, frac, s, c, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(n, 1, 32, 32, device=DEVICE, generator=g)
    ts = torch.linspace(1, 0, steps + 1, device=DEVICE)
    rms_log = []
    for k in range(steps):
        t, tp = ts[k].expand(n), ts[k + 1].expand(n)
        a, sg = (v[:, None, None, None] for v in alpha_sigma(t))
        ap, sp = (v[:, None, None, None] for v in alpha_sigma(tp))
        guided = k < round(frac * steps)
        with torch.enable_grad():
            xg = x.detach().requires_grad_(guided)
            eps = model(xg, t)
            x0 = (xg - sg * eps) / a
            if guided:
                (grad,) = torch.autograd.grad(loss_fn(x0).sum(), xg)
        eps, x0 = eps.detach(), x0.detach()
        if guided:
            d = sg * grad
            rms_log.append(d.pow(2).mean().sqrt().item())
            eps = eps + s * c * d
            x0 = (x - sg * eps) / a
        x0 = x0.clamp(-1, 1)
        eps = (x - a * x0) / sg
        x = ap * x0 + sp * eps if k < steps - 1 else x0
    return x, rms_log


def to_img(x, scale=3):
    ink = ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte().cpu().numpy()[:, 0]
    row = np.concatenate(list(ink), axis=1)
    return Image.fromarray(255 - row).resize((row.shape[1] * scale, 32 * scale), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--kinds", nargs="+", default=["mse", "mse_w", "coverage"])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    args = ap.parse_args()
    ck = torch.load(ROOT / "runs" / "sketch" / "ckpt.pt", map_location=DEVICE, weights_only=True)
    model = SketchUNet(**{k: tuple(v) if k == "ch" else v for k, v in ck["config"].items()}).to(DEVICE)
    model.load_state_dict(ck["ema"])
    model.eval().requires_grad_(False)
    print(f"checkpoint step {ck['step']}")

    for name, outline in [("head", outline_cat_head()), ("body", outline_body())]:
        M = torch.tensor(outline, device=DEVICE)[None, None].expand(args.n, 1, 32, 32)
        for kind in args.kinds:
            loss_fn = make_loss(kind, M)
            _, rms = guided_sample(model, loss_fn, args.n, args.steps, args.frac, 0.0, 1.0, seed=0)
            c = 1.0 / statistics.median(rms)
            rows = [to_img(torch.tensor(outline)[None, None].expand(args.n, 1, 32, 32) * 2 - 1)]
            labels = ["outline", "s=0"]
            rows.append(to_img(guided_sample(model, loss_fn, args.n, args.steps, args.frac, 0.0, c, 0)[0]))
            for s in args.scales:
                x, _ = guided_sample(model, loss_fn, args.n, args.steps, args.frac, s, c, 0)
                final_loss = loss_fn(x).mean().item()
                rows.append(to_img(x))
                labels.append(f"s={s:g} loss={final_loss:.4f}")
            W, H = rows[0].size
            sheet = Image.new("RGB", (W + 150, H * len(rows)), "white")
            dr = ImageDraw.Draw(sheet)
            for r, (im, lab) in enumerate(zip(rows, labels)):
                sheet.paste(im, (150, r * H))
                dr.text((4, r * H + H // 2 - 6), lab, fill="black")
            sheet.save(RESULTS / f"guided_{name}_{kind}.png")
            print(f"{name:5s} {kind:9s} c = {c:.4g}  (median rms sigma*grad = {1 / c:.3g})")


if __name__ == "__main__":
    main()

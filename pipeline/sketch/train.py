"""Train the 32x32 sketch diffusion model on Quick, Draw! cats (long run: see PLAN.md, run by the user).

eps-prediction MSE, t ~ U(0, 1) with the cosine schedule, horizontal flips, AdamW + warmup + cosine decay,
fp16 autocast, EMA weights (used for samples and export). Resumable from runs/sketch/ckpt.pt.

Output: runs/sketch/ckpt.pt (model, EMA, optimizer, step), runs/sketch/log.json
        results/sketch/samples_{step}.png (64 EMA samples, DDIM 50 steps, dark ink on white)
Usage:  python -m pipeline.sketch.train [--iters 60000] [--batch 256] [--resume]
"""
import argparse
import copy
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..sd.models import ROOT
from .model import SketchUNet, alpha_sigma, ddim_sample

DATA = ROOT / "data" / "sketch" / "cat32.npy"
RUNS = ROOT / "runs" / "sketch"
RESULTS = ROOT / "results" / "sketch"
CKPT = RUNS / "ckpt.pt"
DEVICE = "cuda"


def save_grid(x, path, nrow=8, scale=3):
    """x in [-1, 1], (N, 1, 32, 32) -> dark ink on white PNG."""
    ink = ((x.clamp(-1, 1) + 1) / 2 * 255).round().byte().cpu().numpy()[:, 0]
    n = len(ink) // nrow * nrow
    grid = ink[:n].reshape(n // nrow, nrow, 32, 32).transpose(0, 2, 1, 3).reshape(n // nrow * 32, nrow * 32)
    Image.fromarray(255 - grid).resize((grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--sample-every", type=int, default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    RUNS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    data = torch.from_numpy(np.load(DATA)).to(DEVICE)  # uint8 (N, 32, 32), ~100 MB on GPU
    print(f"data {tuple(data.shape)}", flush=True)

    model = SketchUNet().to(DEVICE)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler()
    step, log = 0, []
    if args.resume and CKPT.exists():
        ck = torch.load(CKPT, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        if (RUNS / "log.json").exists():
            log = json.load(open(RUNS / "log.json"))
        print(f"resumed at step {step}", flush=True)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)

    def lr_at(i):
        if i < args.warmup:
            return args.lr * (i + 1) / args.warmup
        u = (i - args.warmup) / max(1, args.iters - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(u, 1.0))))

    def save():
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "config": model.config}, CKPT)
        json.dump(log, open(RUNS / "log.json", "w"))

    t0, run_loss, n_run = time.time(), 0.0, 0
    while step < args.iters:
        idx = torch.randint(0, len(data), (args.batch,), device=DEVICE)
        x0 = data[idx].float()[:, None] / 127.5 - 1
        flip = torch.rand(args.batch, device=DEVICE) < 0.5
        x0 = torch.where(flip[:, None, None, None], x0.flip(-1), x0)
        t = torch.rand(args.batch, device=DEVICE)
        a, s = alpha_sigma(t)
        eps = torch.randn_like(x0)
        xt = a[:, None, None, None] * x0 + s[:, None, None, None] * eps
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.mse_loss(model(xt, t).float(), eps)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        with torch.no_grad():
            d = min(args.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
        step += 1
        run_loss, n_run = run_loss + loss.item(), n_run + 1

        if step % 500 == 0 or step == args.iters:
            el = time.time() - t0
            print(f"step {step:6d} loss {run_loss / n_run:.4f} lr {lr_at(step):.2e} | {el / 60:.1f} min", flush=True)
            log.append({"step": step, "loss": run_loss / n_run})
            run_loss, n_run = 0.0, 0
        if step % args.sample_every == 0 or step == args.iters:
            g = torch.Generator(device=DEVICE).manual_seed(0)
            save_grid(ddim_sample(ema, 64, 50, DEVICE, g), RESULTS / f"samples_{step:06d}.png")
            save()
    save()
    print(f"done: {CKPT}")


if __name__ == "__main__":
    main()

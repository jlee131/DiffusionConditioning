"""Train the noisy style classifier for CG on latents from make_cg_dataset.py.

z_t = alpha_t z0 + sigma_t eps with t ~ U{1..999} (SD 1.5 schedule), BCE on pencil (1) vs photo (0).

Output: runs/cg_classifier/classifier.pt, log.json
        results/cg_classifier/acc_vs_t.png   (validation accuracy / BCE vs t)

Usage:  python -m pipeline.sd.train_cg_classifier [--iters 15000] [--batch 128]
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .cg_classifier import CKPT, NoisyClassifier
from .models import DEVICE, ROOT

DATA = ROOT / "data" / "cg_latents"
RESULTS = ROOT / "results" / "cg_classifier"
VAL_T = list(range(50, 1000, 100))  # 50, 150, ..., 950


def schedule():
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
    return torch.cumprod(1.0 - betas, 0).float().to(DEVICE)


def load_latents():
    xs, ys = [], []
    for label, cls in enumerate(("photo", "pencil")):
        for p in sorted((DATA / cls).glob("shard_*.pt")):
            try:
                z0 = torch.load(p, map_location="cpu", weights_only=True)["z0"]
            except Exception as e:  # shard being written
                print(f"skip {p.name}: {e}")
                continue
            xs.append(z0)
            ys.append(torch.full((len(z0),), float(label)))
    x, y = torch.cat(xs), torch.cat(ys)
    print(f"loaded {len(x)} latents (pencil {int(y.sum())}, photo {int((1 - y).sum())})")
    return x, y


@torch.no_grad()
def evaluate(model, x, y, ac, gen_seed=0):
    g = torch.Generator(device="cpu").manual_seed(gen_seed)
    noise = torch.randn(x.shape, generator=g).to(DEVICE)
    out = {}
    for t in VAL_T:
        a, s = ac[t].sqrt(), (1 - ac[t]).sqrt()
        logit = torch.cat([model(a * x[i:i + 256] + s * noise[i:i + 256], torch.tensor([t], device=DEVICE))
                           for i in range(0, len(x), 256)])
        out[t] = {"acc": ((logit > 0).float() == y).float().mean().item(),
                  "bce": F.binary_cross_entropy_with_logits(logit, y).item()}
    return out


def plot(ev, path):
    W, H, m = 520, 300, 40
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([m, m, W - m, H - m], outline="black")
    xs = [m + (t / 1000) * (W - 2 * m) for t in VAL_T]
    accs = [m + (1 - ev[t]["acc"]) * (H - 2 * m) / 0.5 for t in VAL_T]  # y axis: acc 0.5 .. 1.0
    d.line(list(zip(xs, accs)), fill=(30, 90, 200), width=2)
    for xx, yy, t in zip(xs, accs, VAL_T):
        d.ellipse([xx - 3, yy - 3, xx + 3, yy + 3], fill=(30, 90, 200))
        d.text((xx - 10, H - m + 5), str(t), fill="black")
    d.text((m, 10), "CG classifier: validation accuracy vs t (y: 0.5 .. 1.0)", fill="black")
    d.text((5, m - 5), "1.0", fill="black")
    d.text((5, H - m - 5), "0.5", fill="black")
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    x, y = load_latents()
    perm = torch.randperm(len(x))
    n_val = int(len(x) * args.val_frac)
    xv, yv = x[perm[:n_val]].to(DEVICE).float(), y[perm[:n_val]].to(DEVICE)
    xt, yt = x[perm[n_val:]].to(DEVICE).float(), y[perm[n_val:]].to(DEVICE)
    ac = schedule()

    model = NoisyClassifier().to(DEVICE)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, train {len(xt)}, val {len(xv)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    scaler = torch.amp.GradScaler()

    log, t0, run_loss, n_run = [], time.time(), 0.0, 0
    for it in range(1, args.iters + 1):
        idx = torch.randint(0, len(xt), (args.batch,), device=DEVICE)
        z0, lab = xt[idx], yt[idx]
        flip = torch.rand(args.batch, device=DEVICE) < 0.5
        z0 = torch.where(flip[:, None, None, None], z0.flip(-1), z0)
        t = torch.randint(1, 1000, (args.batch,), device=DEVICE)
        a, s = ac[t].sqrt()[:, None, None, None], (1 - ac[t]).sqrt()[:, None, None, None]
        zt = a * z0 + s * torch.randn_like(z0)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.binary_cross_entropy_with_logits(model(zt, t).float(), lab)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run_loss, n_run = run_loss + loss.item(), n_run + 1

        if it % 500 == 0 or it == args.iters:
            model.eval()
            ev = evaluate(model, xv, yv, ac)
            model.train()
            accs = " ".join(f"{t}:{ev[t]['acc']:.2f}" for t in VAL_T)
            print(f"it {it:6d} train_bce {run_loss / n_run:.4f} | val acc {accs} | {(time.time() - t0) / 60:.1f} min",
                  flush=True)
            log.append({"it": it, "train_bce": run_loss / n_run, "val": ev})
            run_loss, n_run = 0.0, 0
            torch.save({"model": model.state_dict(), "config": {}, "iters": it}, CKPT)

    json.dump(log, open(CKPT.parent / "log.json", "w"), indent=1)
    plot(log[-1]["val"], RESULTS / "acc_vs_t.png")
    print(f"saved {CKPT}")


if __name__ == "__main__":
    main()

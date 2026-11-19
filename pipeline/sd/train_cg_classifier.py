"""Train the noisy style classifier for CG on latents from make_cg_dataset.py.

z_t = alpha_t z0 + sigma_t eps with t ~ U{1..999} (SD 1.5 schedule).
    --mode 3class (default): softmax CE over (photo, photo_bw, pencil); CG uses f_pencil - f_photo
    --mode binary          : BCE pencil (1) vs photo + photo_bw (0), pos_weight for equal priors

Output: runs/cg_classifier/classifier.pt, log.json
        results/cg_classifier/acc_vs_t.png   (validation pencil-vs-photo accuracy vs t)

Usage:  python -m pipeline.sd.train_cg_classifier [--mode 3class] [--iters 12000] [--batch 128]
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from .cg_classifier import CKPT, CLASSES, NoisyClassifier
from .models import DEVICE, ROOT

DATA = ROOT / "data" / "cg_latents"
RESULTS = ROOT / "results" / "cg_classifier"
VAL_T = list(range(50, 1000, 100))  # 50, 150, ..., 950


def schedule():
    betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
    return torch.cumprod(1.0 - betas, 0).float().to(DEVICE)


def load_latents():
    """y = class index into CLASSES (photo 0, photo_bw 1, pencil 2)."""
    xs, ys = [], []
    for label, cls in enumerate(CLASSES):
        for p in sorted((DATA / cls).glob("shard_*.pt")):
            try:
                z0 = torch.load(p, map_location="cpu", weights_only=True)["z0"]
            except Exception as e:  # shard being written
                print(f"skip {p.name}: {e}")
                continue
            xs.append(z0)
            ys.append(torch.full((len(z0),), float(label)))
    x, y = torch.cat(xs), torch.cat(ys).long()
    print("loaded " + ", ".join(f"{c} {int((y == k).sum())}" for k, c in enumerate(CLASSES)))
    return x, y


@torch.no_grad()
def evaluate(model, x, y, ac, gen_seed=0):
    """acc = pencil-vs-photo accuracy of the CG logit (photo_bw counts as photo); acc3 = 3-way accuracy."""
    g = torch.Generator(device="cpu").manual_seed(gen_seed)
    noise = torch.randn(x.shape, generator=g).to(DEVICE)
    is_pencil = (y == CLASSES.index("pencil")).float()
    out = {}
    for t in VAL_T:
        a, s = ac[t].sqrt(), (1 - ac[t]).sqrt()
        tt = torch.tensor([t], device=DEVICE)
        f = torch.cat([model.logits(a * x[i:i + 256] + s * noise[i:i + 256], tt) for i in range(0, len(x), 256)])
        if model.n_classes == 1:
            logit = f.squeeze(-1)
            out[t] = {"acc": ((logit > 0).float() == is_pencil).float().mean().item()}
        else:
            logit = f[:, CLASSES.index("pencil")] - f[:, CLASSES.index("photo")]
            m = y != CLASSES.index("photo_bw")  # the CG logit compares pencil and photo only
            out[t] = {"acc": ((logit[m] > 0).float() == is_pencil[m]).float().mean().item(),
                      "acc3": (f.argmax(1) == y).float().mean().item(),
                      "bw_as_pencil": (f.argmax(1)[y == 1] == 2).float().mean().item()}
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
    ap.add_argument("--mode", choices=["3class", "binary"], default="3class")
    ap.add_argument("--iters", type=int, default=12000)
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
    # equal effective class priors, so the CG logit stays log p(z|pencil) - log p(z|photo)
    counts = torch.bincount(yt, minlength=len(CLASSES)).float().clamp(min=1)
    is_pencil_t = (yt == CLASSES.index("pencil")).float()
    if args.mode == "binary":
        n_classes = 1
        pos_weight = ((1 - is_pencil_t).sum() / is_pencil_t.sum()).reshape(1)
    else:
        n_classes = 3
        class_w = counts.mean() / counts
    config = {"n_classes": n_classes}

    model = NoisyClassifier(**config).to(DEVICE)
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
            f = model.logits(zt, t).float()
            if n_classes == 1:
                target = (lab == CLASSES.index("pencil")).float()
                loss = F.binary_cross_entropy_with_logits(f.squeeze(-1), target, pos_weight=pos_weight)
            else:
                loss = F.cross_entropy(f, lab, weight=class_w)
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
            print(f"it {it:6d} loss {run_loss / n_run:.4f} | pencil-vs-photo acc {accs} | "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            if n_classes == 3:
                print("          3-way acc " + " ".join(f"{t}:{ev[t]['acc3']:.2f}" for t in VAL_T), flush=True)
            log.append({"it": it, "loss": run_loss / n_run, "val": ev})
            run_loss, n_run = 0.0, 0
            torch.save({"model": model.state_dict(), "config": config, "iters": it}, CKPT)

    json.dump(log, open(CKPT.parent / "log.json", "w"), indent=1)
    plot(log[-1]["val"], RESULTS / "acc_vs_t.png")
    print(f"saved {CKPT}")


if __name__ == "__main__":
    main()

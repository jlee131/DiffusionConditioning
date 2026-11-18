"""Generate clean SD latents for training the noisy style classifier used by CG.

Two classes, sampled with exactly the experiment's prompts and base construction (w_cat):
    photo  : eps_null + w * (eps("a photo of a cat")          - eps_null)
    pencil : eps_null + w * (eps("a pencil drawing of a cat") - eps_null)   (= CFG s=1 target)

With equal class priors, the classifier's logit gradient
    grad_z [log p_t(z | pencil) - log p_t(z | photo)]
is exactly the CFG direction, so an oracle classifier reproduces CFG (see PLAN.md).

Output: data/cg_latents/{photo,pencil}/shard_XXXXX.pt  (z0 latents, fp16, plus seeds)
        data/cg_latents/preview_{class}.png            (decoded thumbnails, sanity check)
Resumable: existing shards are skipped.

Usage:  python -m pipeline.sd.make_cg_dataset --n-per-class 1000 --steps 20 --batch 8
"""
import argparse
import time

import torch
from PIL import Image

from .models import DEVICE, DTYPE, ROOT, StableDiffusion
from .sampler import StyleSetup

OUT = ROOT / "data" / "cg_latents"
SEED_OFFSET = {"photo": 1_000_000, "pencil": 2_000_000}  # disjoint from experiment seeds


@torch.no_grad()
def sample_batch(sd, emb_null, emb_c, seeds, w, n_steps):
    z = torch.cat([torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(s)) for s in seeds])
    z = z.to(DEVICE, DTYPE)
    B = len(seeds)
    emb = torch.cat([emb_null.expand(B, -1, -1), emb_c.expand(B, -1, -1)])
    for t, t_prev in sd.timesteps(n_steps):
        a, s = sd.alpha_sigma(t)
        ap, sp = sd.alpha_sigma(t_prev)
        e = sd.eps(torch.cat([z, z]), t, emb)
        eps = e[:B] + w * (e[B:] - e[:B])
        x0 = (z - s * eps) / a
        z = ap * x0 + sp * eps
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    setup = StyleSetup()
    sd = StableDiffusion()
    emb = sd.encode_text([setup.neg_prompt, setup.base_prompt, setup.style_prompt])
    prompts = {"photo": emb[1:2], "pencil": emb[2:3]}

    n_shards = (args.n_per_class + args.batch - 1) // args.batch
    t_start, done = time.time(), 0
    total = 2 * n_shards
    for k in range(n_shards):
        for cls in ("photo", "pencil"):
            d = OUT / cls
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"shard_{k:05d}.pt"
            done += 1
            if path.exists():
                continue
            seeds = [SEED_OFFSET[cls] + k * args.batch + i for i in range(args.batch)]
            z0 = sample_batch(sd, emb[0:1], prompts[cls], seeds, setup.w_cat, args.steps)
            torch.save({"z0": z0.cpu(), "seeds": seeds, "steps": args.steps, "w_cat": setup.w_cat}, path)
            if k == 0:  # sanity-check thumbnails
                imgs = sd.decode(z0, size=128)
                row = Image.new("RGB", (128 * len(seeds), 128))
                for i in range(len(seeds)):
                    x = ((imgs[i] + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
                    row.paste(Image.fromarray(x), (128 * i, 0))
                row.save(OUT / f"preview_{cls}.png")
            el = time.time() - t_start
            print(f"[{done}/{total}] {cls} shard {k}  elapsed {el / 60:.1f} min", flush=True)
    print("done")


if __name__ == "__main__":
    main()

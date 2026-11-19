"""Full SD precompute grid for Section 2 (long run: about 1 h on an RTX 3060).

Grid per seed: none  +  {cfg, cg, tfg} x scales {0.5, 1, 2} x start steps {0, 10, 20}   (28 runs)
Every one of the 30 DDIM steps is recorded:
    x0g_XX.jpg  guided clean prediction  x0_hat(z_t) under the guided eps   (256 px)
    x0b_XX.jpg  base clean prediction    x0_hat(z_t) under the base eps     (256 px)
    zt_XX.png   latent preview of z_t                                        (64 px)
    final.jpg   decoded final image                                          (512 px)
    metrics.json per-step guidance norms / cosines / CLIP scores (written last = run complete)
The "none" run also probes the TFG direction, so all three directions are logged along the base path.

Setup shared with the cold test: SD 1.5, DDIM 30 steps, w_cat = 5, CG classifier v2 (3-class),
TFG = VGG Gram loss vs a reference pencil drawing (CFG s=1 sample of --ref-seed), lambda calibrated once.

Output: runs/sd_grid/seed{k}/{run}/..., runs/sd_grid/index.json, runs/sd_grid/tfg_reference.jpg
Resumable: runs with an existing metrics.json are skipped.

Usage:  python -m pipeline.sd.precompute [--seeds 0 1 2 3 4 5] [--max-runs N]
"""
import argparse
import json
import statistics
import time

import torch

from .cg_classifier import load_classifier
from .cold_test import to_pil
from .losses import VGGStyleLoss
from .models import ROOT, ClipStyleScorer, StableDiffusion
from .sampler import GuidedSampler, RunSpec, StyleSetup

OUT = ROOT / "runs" / "sd_grid"
CLASSIFIER = ROOT / "runs" / "cg_classifier" / "classifier_v2.pt"
METHODS = ("cfg", "cg", "tfg")
SCALES = (0.5, 1.0, 2.0)
STARTS = (0, 10, 20)


def run_name(method, scale, start):
    return "none" if method == "none" else f"{method}_s{scale:g}_start{start}"


def grid():
    yield "none", 0.0, 0
    for start in STARTS:
        for scale in SCALES:
            for method in METHODS:
                yield method, scale, start


def save_run(d, r):
    d.mkdir(parents=True, exist_ok=True)
    for k, (xg, xb, zt) in enumerate(zip(r.x0_guided, r.x0_base, r.zt_preview)):
        to_pil(xg).save(d / f"x0g_{k:02d}.jpg", quality=92)
        to_pil(xb).save(d / f"x0b_{k:02d}.jpg", quality=92)
        to_pil(zt).save(d / f"zt_{k:02d}.png")
    to_pil(r.final).save(d / "final.jpg", quality=94)
    json.dump({"steps": r.steps, **r.metrics}, open(d / "metrics.json", "w"))  # last: marks completion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--ref-seed", type=int, default=7)
    ap.add_argument("--max-runs", type=int, default=None, help="stop after N new runs (smoke test)")
    args = ap.parse_args()
    assert args.ref_seed not in args.seeds
    OUT.mkdir(parents=True, exist_ok=True)

    setup = StyleSetup()
    sd = StableDiffusion()
    scorer = ClipStyleScorer(["a pencil drawing of a cat", "a photo of a cat"])
    sampler = GuidedSampler(sd, setup, scorer, classifier=load_classifier(CLASSIFIER))

    # TFG task loss and its one-off magnitude calibration (same recipe as the cold test)
    t0 = time.time()
    ref = sampler.run(args.ref_seed, RunSpec("cfg", 1.0, record_every=100)).final
    to_pil(ref).save(OUT / "tfg_reference.jpg", quality=94)
    sampler.task_loss = VGGStyleLoss(scorer.decode_tiny, ref)
    probe = sampler.run(args.seeds[0], RunSpec("none", record_every=100), probe_tfg=True)
    ratios = [c / g for c, g in zip(probe.metrics["cfg_rms"], probe.metrics["tfg_rms"]) if g > 0]
    sampler.tfg_lambda = statistics.median(ratios)
    print(f"setup done: tfg_lambda={sampler.tfg_lambda:.4g}  {time.time() - t0:.0f}s", flush=True)

    index = {"setup": vars(setup), "seeds": args.seeds, "ref_seed": args.ref_seed, "methods": METHODS,
             "scales": SCALES, "starts": STARTS, "tfg_lambda": sampler.tfg_lambda,
             "classifier": CLASSIFIER.name, "t": probe.metrics["t"], "runs": {}}
    total = len(args.seeds) * len(list(grid()))
    done, new, t_start = 0, 0, time.time()
    for seed in args.seeds:
        for method, scale, start in grid():
            done += 1
            name = run_name(method, scale, start)
            d = OUT / f"seed{seed}" / name
            if (d / "metrics.json").exists():
                m = json.load(open(d / "metrics.json"))
            else:
                if args.max_runs is not None and new >= args.max_runs:
                    continue
                t0 = time.time()
                r = sampler.run(seed, RunSpec(method, scale, start, record_every=1), probe_tfg=(method == "none"))
                save_run(d, r)
                m, new = r.metrics, new + 1
                el = (time.time() - t_start) / 60
                print(f"[{done}/{total}] seed {seed} {name:20s} clip={m['clip_final']:.2f}  "
                      f"{time.time() - t0:.0f}s  elapsed {el:.1f} min", flush=True)
            index["runs"][f"seed{seed}/{name}"] = {"method": method, "scale": scale, "start": start,
                                                   "clip_final": m["clip_final"]}
            json.dump(index, open(OUT / "index.json", "w"), indent=1)
    print(f"done ({new} new runs, {(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()

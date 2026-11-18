"""Quick TFG loss/scale exploration on one seed (cold-test helper).

Round 1 (CLIP softmax / text_cos / ref_cos) -> results/sd_cold_test/tfg_explore_seed0.png
Round 2 (VGG Gram style, latent statistics, late start) -> tfg_explore2_seed{seed}.png

Usage:  python -m pipeline.sd.explore_tfg [--seed 0]
"""
import argparse
import statistics
import time

from .cold_test import RESULTS, labeled, sheet, to_pil
from .losses import ClipLoss, LatentStatsLoss, VGGStyleLoss
from .models import ClipStyleScorer, StableDiffusion
from .sampler import GuidedSampler, RunSpec, StyleSetup


def calibrate(sampler, seed):
    """lambda so that the TFG direction has the same median RMS as the CFG direction."""
    sampler.tfg_lambda = 1.0
    probe = sampler.run(seed, RunSpec("none", record_every=100), probe_tfg=True)
    r = [c / g for c, g in zip(probe.metrics["cfg_rms"], probe.metrics["tfg_rms"]) if g > 0]
    sampler.tfg_lambda = statistics.median(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref-seed", type=int, default=7)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    sd = StableDiffusion()
    scorer = ClipStyleScorer(["a pencil drawing of a cat", "a photo of a cat"], n_aug=16)
    sampler = GuidedSampler(sd, StyleSetup(), scorer)

    # reference pencil drawing: the CFG (s=1) sample of another seed
    ref = sampler.run(args.ref_seed, RunSpec("cfg", 1.0, record_every=100))
    ref_img, ref_lat = ref.final, sampler.final_latent.clone()
    base = sampler.run(args.seed, RunSpec("none", record_every=100)).final

    variants = [
        ("clip softmax", ClipLoss(scorer), [(2, 8), (4, 8), (8, 8)]),
        ("vgg gram", VGGStyleLoss(scorer.decode_tiny, ref_img), [(0.5, 0), (1, 0), (2, 0), (2, 8)]),
        ("latent stats", LatentStatsLoss(ref_lat), [(0.5, 0), (1, 0), (2, 0), (2, 8)]),
    ]
    rows, labels = [], []
    for name, loss, settings in variants:
        sampler.task_loss = loss
        calibrate(sampler, args.seed)
        row = [labeled(to_pil(base), "none", 180)]
        for s, start in settings:
            t0 = time.time()
            r = sampler.run(args.seed, RunSpec("tfg", s, start, record_every=100))
            p = r.metrics["clip_final"]
            print(f"{name:12s} s={s:4.1f} start={start:2d} lambda={sampler.tfg_lambda:.3g} p={p:.3f} {time.time() - t0:.0f}s")
            row.append(labeled(to_pil(r.final), f"s={s} start={start} p={p:.2f}", 180))
        rows.append(row)
        labels.append(name)
    rows.append([labeled(to_pil(ref_img), "reference (CFG s=1)", 180)])
    labels.append("ref image")
    sheet(rows, labels).save(RESULTS / f"tfg_explore2_seed{args.seed}.png")


if __name__ == "__main__":
    main()

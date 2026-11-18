"""SD cold test: cat baseline + pencil-style guidance (CFG, TFG) on a small grid.

Raw outputs  -> runs/sd_cold_test/        (gitignored)
Contact sheets -> results/sd_cold_test/   (small PNGs referenced by .documents/RESULTS.md)

Usage:  python -m pipeline.sd.cold_test [--seeds 0 1 2] [--steps 30] [--w-cat 5]
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .losses import VGGStyleLoss
from .models import ROOT, ClipStyleScorer, StableDiffusion
from .sampler import GuidedSampler, RunSpec, StyleSetup

RUNS = ROOT / "runs" / "sd_cold_test"
RESULTS = ROOT / "results" / "sd_cold_test"


def to_pil(img):
    x = ((img[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def labeled(img, text, size):
    im = img.resize((size, size), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size + 22), "white")
    canvas.paste(im, (0, 22))
    ImageDraw.Draw(canvas).text((4, 5), text, fill="black")
    return canvas


def sheet(rows, row_labels=None, pad=4):
    """rows: list of lists of PIL images (same size)."""
    w, h = rows[0][0].size
    lab_w = 110 if row_labels else 0
    W = lab_w + max(len(r) for r in rows) * (w + pad)
    H = len(rows) * (h + pad)
    out = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(out)
    for r, row in enumerate(rows):
        y = r * (h + pad)
        if row_labels:
            d.text((4, y + h // 2), row_labels[r], fill="black")
        for c, im in enumerate(row):
            out.paste(im, (lab_w + c * (w + pad), y))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--w-cat", type=float, default=5.0)
    ap.add_argument("--ref-seed", type=int, default=7)
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    setup = StyleSetup(w_cat=args.w_cat, n_steps=args.steps)
    sd = StableDiffusion()
    scorer = ClipStyleScorer(["a pencil drawing of a cat", "a photo of a cat"])
    sampler = GuidedSampler(sd, setup, scorer)

    # --- TFG task loss: VGG Gram style loss vs a reference pencil drawing ------
    # (reference = CFG s=1 sample of a held-out seed; CLIP losses were adversarial, see RESULTS.md)
    ref = sampler.run(args.ref_seed, RunSpec("cfg", 1.0, record_every=100)).final
    to_pil(ref).save(RESULTS / "tfg_reference.png")
    sampler.task_loss = VGGStyleLoss(scorer.decode_tiny, ref)

    # --- calibrate TFG magnitude against CFG on an unguided probe run ---------
    t0 = time.time()
    probe = sampler.run(args.seeds[0], RunSpec("none", record_every=10), probe_tfg=True)
    ratios = [c / g for c, g in zip(probe.metrics["cfg_rms"], probe.metrics["tfg_rms"]) if g > 0]
    sampler.tfg_lambda = statistics.median(ratios)
    print(f"calibration lambda={sampler.tfg_lambda:.3g}  ({time.time() - t0:.1f}s, "
          f"peak mem {torch.cuda.max_memory_allocated() / 1e9:.2f} GB)")

    grid = [("none", "none", 0.0, 0)]
    for s in (0.5, 1.0, 2.0):
        grid.append((f"cfg_s{s}", "cfg", s, 0))
    grid.append(("cfg_s2.0_start10", "cfg", 2.0, 10))
    for s in (1.0, 2.0, 4.0):
        grid.append((f"tfg_s{s}", "tfg", s, 0))
    grid.append(("tfg_s2.0_start10", "tfg", 2.0, 10))

    summary = {"setup": vars(setup), "tfg_lambda": sampler.tfg_lambda, "runs": {}}
    for seed in args.seeds:
        results = {}
        for name, method, scale, start in grid:
            t0 = time.time()
            r = sampler.run(seed, RunSpec(method, scale, start, record_every=3))
            results[name] = r
            d = RUNS / f"seed{seed}" / name
            d.mkdir(parents=True, exist_ok=True)
            to_pil(r.final).save(d / "final.png")
            json.dump(r.metrics, open(d / "metrics.json", "w"))
            summary["runs"][f"seed{seed}/{name}"] = {
                "clip_final": r.metrics["clip_final"], "sec": round(time.time() - t0, 1)}
            print(f"seed {seed} {name:18s} clip_pencil={r.metrics['clip_final']:.3f}  {time.time() - t0:.1f}s")

        # final-image contact sheet: row 1 CFG, row 2 TFG
        def cell(name):
            return labeled(to_pil(results[name].final), f"{name}  p={results[name].metrics['clip_final']:.2f}", 200)
        rows = [[cell(n) for n in ["none", "cfg_s0.5", "cfg_s1.0", "cfg_s2.0", "cfg_s2.0_start10"]],
                [cell(n) for n in ["none", "tfg_s1.0", "tfg_s2.0", "tfg_s4.0", "tfg_s2.0_start10"]]]
        sheet(rows, ["CFG", "TFG"]).save(RESULTS / f"finals_seed{seed}.png")

        # filmstrip over steps: z_t preview and clean predictions
        steps = results["cfg_s2.0"].steps
        def strip(frames):
            return [labeled(to_pil(f), f"step {k}", 128) for f, k in zip(frames, steps)]
        rows = [strip(results["cfg_s2.0"].zt_preview),
                strip(results["none"].x0_guided),
                strip(results["cfg_s2.0"].x0_base),
                strip(results["cfg_s2.0"].x0_guided),
                strip(results["tfg_s2.0"].x0_base),
                strip(results["tfg_s2.0"].x0_guided)]
        labels = ["z_t (CFG s=2)", "x0 none", "x0 base | CFG2", "x0 guided | CFG2", "x0 base | TFG2", "x0 guided | TFG2"]
        sheet(rows, labels).save(RESULTS / f"filmstrip_seed{seed}.png")

        # step-wise guidance metrics
        summary[f"seed{seed}_metrics"] = {
            n: {k: results[n].metrics[k] for k in ["guid_rms", "cfg_rms", "tfg_rms", "cos_cfg_tfg", "clip_x0_guided"]}
            for n in ["cfg_s2.0", "tfg_s2.0"]}

    json.dump(summary, open(RESULTS / "summary.json", "w"), indent=1)
    print("peak mem %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))


if __name__ == "__main__":
    main()

"""Evaluate the trained CG classifier against CFG on SD (a few minutes).

- Final images: CFG vs CG at the same scales s (oracle theory: identical).
- Per-step cosine(d_cfg, CG direction) and RMS ratio along the unguided and CFG trajectories.

Output: results/cg_eval/finals_seed{k}.png, direction_seed{k}.png, summary.json
Usage:  python -m pipeline.sd.eval_cg [--seeds 0 1 2]
"""
import argparse
import json
import math

from PIL import Image, ImageDraw

from .cg_classifier import load_classifier
from .cold_test import labeled, sheet, to_pil
from .models import ROOT, ClipStyleScorer, StableDiffusion
from .sampler import GuidedSampler, RunSpec, StyleSetup

RESULTS = ROOT / "results" / "cg_eval"
SCALES = (0.5, 1.0, 2.0, 4.0)


def line_plot(series, path, title, ylo, yhi):
    """series: {label: (color, [values per step])}"""
    W, H, m = 560, 300, 40
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([m, m, W - m, H - m], outline="black")
    if ylo < 0 < yhi:
        y0 = m + (yhi / (yhi - ylo)) * (H - 2 * m)
        d.line([(m, y0), (W - m, y0)], fill=(200, 200, 200))
    for k, (label, (color, vals)) in enumerate(series.items()):
        n = len(vals)
        pts = [(m + i / max(n - 1, 1) * (W - 2 * m), m + (yhi - v) / (yhi - ylo) * (H - 2 * m))
               for i, v in enumerate(vals) if not math.isnan(v)]
        if len(pts) > 1:
            d.line(pts, fill=color, width=2)
        d.text((W - m - 160, m + 5 + 14 * k), label, fill=color)
    d.text((m, 10), title, fill="black")
    d.text((5, m - 5), f"{yhi:g}", fill="black")
    d.text((5, H - m - 5), f"{ylo:g}", fill="black")
    d.text((m, H - m + 5), "step 0", fill="black")
    d.text((W - m - 40, H - m + 5), "step end", fill="black")
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    sd = StableDiffusion()
    scorer = ClipStyleScorer(["a pencil drawing of a cat", "a photo of a cat"])
    sampler = GuidedSampler(sd, StyleSetup(), scorer, classifier=load_classifier())

    summary = {}
    for seed in args.seeds:
        runs = {"none": sampler.run(seed, RunSpec("none", record_every=100))}
        for s in SCALES:
            runs[f"cfg_s{s}"] = sampler.run(seed, RunSpec("cfg", s, record_every=100))
            runs[f"cg_s{s}"] = sampler.run(seed, RunSpec("cg", s, record_every=100))
        for k, r in runs.items():
            print(f"seed {seed} {k:10s} clip_pencil={r.metrics['clip_final']:.3f}")

        def cell(n):
            return labeled(to_pil(runs[n].final), f"{n} p={runs[n].metrics['clip_final']:.2f}", 180)
        rows = [[cell("none")] + [cell(f"cfg_s{s}") for s in SCALES],
                [cell("none")] + [cell(f"cg_s{s}") for s in SCALES]]
        sheet(rows, ["CFG", "CG"]).save(RESULTS / f"finals_seed{seed}.png")

        mn, mc = runs["none"].metrics, runs["cfg_s1.0"].metrics
        line_plot({"cos | unguided traj": ((30, 90, 200), mn["cos_cfg_cg"]),
                   "cos | CFG s=1 traj": ((220, 120, 30), mc["cos_cfg_cg"])},
                  RESULTS / f"direction_seed{seed}.png", "cos(CFG direction, CG direction) per step", -0.2, 1.0)
        ratio_n = [c / f if f > 0 else float("nan") for c, f in zip(mn["cg_rms"], mn["cfg_rms"])]
        ratio_c = [c / f if f > 0 else float("nan") for c, f in zip(mc["cg_rms"], mc["cfg_rms"])]
        line_plot({"rms(CG)/rms(CFG) | unguided": ((30, 90, 200), ratio_n),
                   "rms(CG)/rms(CFG) | CFG s=1": ((220, 120, 30), ratio_c)},
                  RESULTS / f"magnitude_seed{seed}.png", "guidance magnitude ratio per step", 0, 3)
        summary[seed] = {"clip_final": {k: r.metrics["clip_final"] for k, r in runs.items()},
                         "cos_unguided": mn["cos_cfg_cg"], "cos_cfg1": mc["cos_cfg_cg"],
                         "ratio_unguided": ratio_n, "ratio_cfg1": ratio_c}
    json.dump(summary, open(RESULTS / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()

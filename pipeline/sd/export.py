"""Pack runs/sd_grid into compact web assets for Section 2 (a minute or two, CPU only).

Per run, three WebP sprite sheets (8 x 4 grid, frames 0..29 = DDIM steps, frame 30 = final image):
    {run}_x0g.webp  guided clean prediction  (192 px cells)
    {run}_x0b.webp  base clean prediction    (128 px cells)
    {run}_zt.webp   z_t latent preview       (64 px cells)
Per seed: metrics.json (per-step curves of every run, rounded) and thumb.webp (unguided final).
Global:   index.json (grid description) and tfg_reference.webp.

Incomplete runs (no metrics.json) are skipped, so this also works on a partial grid.
Output: site/assets/sd/   (commit only at final export)
Usage:  python -m pipeline.sd.export [--quality 80]
"""
import argparse
import json
import math

from PIL import Image

from .models import ROOT

SRC = ROOT / "runs" / "sd_grid"
DST = ROOT / "site" / "assets" / "sd"
COLS, ROWS, N_FRAMES = 8, 4, 31
SIZES = {"x0g": 192, "x0b": 128, "zt": 64}
CURVES = ["guid_rms", "cfg_rms", "cg_rms", "tfg_rms", "cos_cfg_cg", "cos_cfg_tfg", "clip_x0_guided", "clip_x0_base"]


def clean(v, nd=4):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(v, nd)


def sprite(frames, cell):
    sheet = Image.new("RGB", (COLS * cell, ROWS * cell), "white")
    for k, im in enumerate(frames):
        sheet.paste(im.convert("RGB").resize((cell, cell), Image.LANCZOS), ((k % COLS) * cell, (k // COLS) * cell))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", type=int, default=80)
    args = ap.parse_args()
    src_index = json.load(open(SRC / "index.json"))
    DST.mkdir(parents=True, exist_ok=True)

    index = {k: src_index[k] for k in ("seeds", "methods", "scales", "starts", "tfg_lambda", "t", "classifier")}
    index.update({"setup": src_index["setup"], "cols": COLS, "rows": ROWS, "n_frames": N_FRAMES,
                  "sizes": SIZES, "runs": {}})
    Image.open(SRC / "tfg_reference.jpg").resize((256, 256), Image.LANCZOS).save(
        DST / "tfg_reference.webp", quality=args.quality)

    n = 0
    for seed in src_index["seeds"]:
        sd_dir = SRC / f"seed{seed}"
        if not sd_dir.exists():
            continue
        out = DST / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        metrics = {}
        for run_dir in sorted(p for p in sd_dir.iterdir() if (p / "metrics.json").exists()):
            name = run_dir.name
            m = json.load(open(run_dir / "metrics.json"))
            final = Image.open(run_dir / "final.jpg")
            nst = len(m["steps"])
            for key, cell in SIZES.items():
                ext = "png" if key == "zt" else "jpg"
                frames = [Image.open(run_dir / f"{key}_{k:02d}.{ext}") for k in range(nst)]
                frames.append(final if key != "zt" else frames[-1])
                q = args.quality if key != "zt" else 50  # z_t is mostly noise; it compresses badly
                sprite(frames, cell).save(out / f"{name}_{key}.webp", quality=q, method=6)
            metrics[name] = {c: [clean(v) for v in m.get(c, [])] for c in CURVES}
            metrics[name]["clip_final"] = clean(m["clip_final"])
            index["runs"][f"seed{seed}/{name}"] = 1
            if name == "none":
                final.resize((112, 112), Image.LANCZOS).save(out / "thumb.webp", quality=args.quality)
            n += 1
        json.dump(metrics, open(out / "metrics.json", "w"), separators=(",", ":"))
    json.dump(index, open(DST / "index.json", "w"), separators=(",", ":"))
    size = sum(p.stat().st_size for p in DST.rglob("*") if p.is_file()) / 1e6
    print(f"exported {n} runs -> {DST}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()

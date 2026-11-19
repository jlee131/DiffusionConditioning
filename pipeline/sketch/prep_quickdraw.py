"""Download Quick, Draw! "cat" (simplified, CC BY 4.0) and rasterize to 32x32 ink maps (a few minutes).

Each drawing (strokes in a 0..255 box) is centered, scaled to fit with a margin, drawn with antialiased lines at
4x resolution and downsampled. Pixel value = ink in 0..255 (0 = paper). Only drawings the game recognized are kept.
The model works on x0 = 2 * ink / 255 - 1 (paper = -1, ink = +1).

Output: data/sketch/cat.ndjson (raw download), data/sketch/cat{size}[_s{min_strokes}].npy (uint8, N x size x size),
        results/sketch/data_preview{size}[_s{min_strokes}].png
Usage:  python -m pipeline.sketch.prep_quickdraw [--size 32] [--min-strokes 0] [--limit N]
        v2 data: --size 64 --min-strokes 9 (about 60k detailed cats; stroke width stays 1.6 px)
"""
import argparse
import json
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from ..sd.models import ROOT

URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/cat.ndjson"
DATA = ROOT / "data" / "sketch"
RESULTS = ROOT / "results" / "sketch"


def rasterize(strokes, size=32, ss=4, margin=None, width=1.6):
    """strokes: [[xs, ys], ...] in 0..255. Returns uint8 (size, size) ink map. margin defaults to size / 16."""
    margin = size / 16 if margin is None else margin
    xs = np.concatenate([np.asarray(s[0], float) for s in strokes])
    ys = np.concatenate([np.asarray(s[1], float) for s in strokes])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    span = max(x1 - x0, y1 - y0, 1.0)
    big = size * ss
    scale = (big - 2 * margin * ss) / span
    ox = (big - (x1 - x0) * scale) / 2 - x0 * scale
    oy = (big - (y1 - y0) * scale) / 2 - y0 * scale
    im = Image.new("L", (big, big), 0)
    d = ImageDraw.Draw(im)
    w = max(1, round(width * ss))
    for sx, sy in strokes:
        pts = [(x * scale + ox, y * scale + oy) for x, y in zip(sx, sy)]
        if len(pts) == 1:
            pts = pts * 2
        d.line(pts, fill=255, width=w, joint="curve")
        r = w / 2  # round caps
        for px, py in (pts[0], pts[-1]):
            d.ellipse([px - r, py - r, px + r, py + r], fill=255)
    return np.asarray(im.resize((size, size), Image.LANCZOS), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-strokes", type=int, default=0, help="keep drawings with at least this many strokes")
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    raw = DATA / "cat.ndjson"
    if not raw.exists():
        print(f"downloading {URL} ...", flush=True)
        urllib.request.urlretrieve(URL, raw)

    imgs, n_seen = [], 0
    with open(raw, encoding="utf-8") as f:
        for line in f:
            n_seen += 1
            rec = json.loads(line)
            if not rec.get("recognized", False) or len(rec["drawing"]) < args.min_strokes:
                continue
            imgs.append(rasterize(rec["drawing"], args.size))
            if args.limit and len(imgs) >= args.limit:
                break
            if len(imgs) % 20000 == 0:
                print(f"{len(imgs)} rasterized", flush=True)
    arr = np.stack(imgs)
    tag = f"cat{args.size}" + (f"_s{args.min_strokes}" if args.min_strokes else "")
    out = DATA / f"{tag}.npy"
    np.save(out, arr)
    print(f"kept {len(arr)} / {n_seen} drawings -> {out} ({arr.nbytes / 1e6:.0f} MB), mean ink {arr.mean() / 255:.3f}")

    # 8 x 16 preview, inverted to dark ink on white, upscaled 3x
    k = min(128, len(arr))
    grid = arr[:k].reshape(8, k // 8, args.size, args.size).transpose(0, 2, 1, 3).reshape(8 * args.size, -1)
    up = max(1, 96 // args.size)
    name = "data_preview.png" if tag == "cat32" else f"data_preview_{tag}.png"
    Image.fromarray(255 - grid).resize((grid.shape[1] * up, grid.shape[0] * up), Image.NEAREST).save(RESULTS / name)


if __name__ == "__main__":
    main()

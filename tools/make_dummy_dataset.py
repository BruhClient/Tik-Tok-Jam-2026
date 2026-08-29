"""Generate a synthetic labeled dataset so the UI can be exercised immediately.

    python tools/make_dummy_dataset.py --out sample_data --n 400

Produces one folder in the layout the app expects:

    sample_data/
    ├── real/   noisy, textured, JPEG-encoded  (label 0)
    └── ai/     smooth gradients, PNG          (label 1)

The split is crude on purpose - it just needs to produce non-degenerate score
distributions so the tables and charts have something real to show.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageFilter


def _authentic(rng: np.random.Generator, size: int) -> Image.Image:
    base = rng.normal(0.5, 0.22, (size, size, 3))
    xs = np.linspace(0, np.pi * rng.uniform(2, 8), size)
    texture = np.sin(xs)[:, None] * np.cos(xs)[None, :] * rng.uniform(0.05, 0.2)
    base += texture[:, :, None]
    base += rng.normal(0, 0.05, base.shape)          # sensor-ish noise
    arr = np.clip(base * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _ai(rng: np.random.Generator, size: int) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size] / size
    img = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = (
            0.5
            + 0.35 * np.sin(xx * rng.uniform(1.5, 4) + rng.uniform(0, 6))
            * np.cos(yy * rng.uniform(1.5, 4) + rng.uniform(0, 6))
        )
    for _ in range(rng.integers(2, 5)):
        cx, cy, r = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), rng.uniform(0.08, 0.25)
        blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2)))
        for c in range(3):
            img[:, :, c] += blob * rng.uniform(-0.3, 0.35)
    arr = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.6))


def build(out_dir: str, n: int, size: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    real_dir = os.path.join(out_dir, "real")
    ai_dir = os.path.join(out_dir, "ai")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ai_dir, exist_ok=True)

    half = n // 2
    for i in range(n):
        is_ai = i >= half
        img = _ai(rng, size) if is_ai else _authentic(rng, size)
        stem = f"{'ai' if is_ai else 'real'}_{i:04d}"
        if is_ai:
            img.save(os.path.join(ai_dir, stem + ".png"))
        else:
            img.save(os.path.join(real_dir, stem + ".jpg"), quality=py_rng.randint(82, 96))

    print(f"wrote {half} authentic images to {real_dir}")
    print(f"wrote {n - half} AI images to {ai_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sample_data", help="output directory")
    ap.add_argument("--n", type=int, default=400, help="number of images (half real, half AI)")
    ap.add_argument("--size", type=int, default=256, help="image side in pixels")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    build(args.out, args.n, args.size, args.seed)


if __name__ == "__main__":
    main()

"""Generate a synthetic labeled dataset so the UI can be exercised immediately.

    python tools/make_dummy_dataset.py --out sample_data --n 400

"Authentic" images are noisy, textured and saved as JPEG (like camera output).
"AI" images are smooth gradients and soft blobs saved as PNG. The split is
crude on purpose - it just needs to produce non-degenerate score distributions.

Also writes a flat variant with labels.csv and a filename-prefix variant so the
three label-detection modes can be tested.
"""

from __future__ import annotations

import argparse
import csv
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
    out = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.6))
    return out


def build(out_dir: str, n: int, size: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    dirs = {
        "sub_real": os.path.join(out_dir, "real"),
        "sub_ai": os.path.join(out_dir, "ai"),
        "flat": os.path.join(out_dir + "_manifest", "images"),
        "prefix": out_dir + "_prefix",
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    manifest_rows = []
    half = n // 2

    for i in range(n):
        is_ai = i >= half
        img = _ai(rng, size) if is_ai else _authentic(rng, size)
        stem = f"{'ai' if is_ai else 'real'}_{i:04d}"

        # 1) subfolder layout
        sub_dir = dirs["sub_ai"] if is_ai else dirs["sub_real"]
        if is_ai:
            img.save(os.path.join(sub_dir, stem + ".png"))
        else:
            img.save(os.path.join(sub_dir, stem + ".jpg"), quality=py_rng.randint(82, 96))

        # 2) flat + manifest layout (every 2nd image, to keep it small)
        if i % 2 == 0:
            fname = f"img_{i:04d}." + ("png" if is_ai else "jpg")
            img.save(os.path.join(dirs["flat"], fname))
            manifest_rows.append({"image_path": f"images/{fname}", "label": int(is_ai)})

        # 3) filename-prefix layout (every 3rd image)
        if i % 3 == 0:
            img.save(os.path.join(dirs["prefix"], stem + (".png" if is_ai else ".jpg")))

    manifest_path = os.path.join(out_dir + "_manifest", "labels.csv")
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"subfolder set : {out_dir}  ({n} images)")
    print(f"manifest set  : {out_dir}_manifest  ({len(manifest_rows)} images + labels.csv)")
    print(f"prefix set    : {out_dir}_prefix")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="sample_data", help="output directory")
    ap.add_argument("--n", type=int, default=400, help="number of images (half real, half AI)")
    ap.add_argument("--size", type=int, default=256, help="image side in pixels")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    build(args.out, args.n, args.size, args.seed)


if __name__ == "__main__":
    main()

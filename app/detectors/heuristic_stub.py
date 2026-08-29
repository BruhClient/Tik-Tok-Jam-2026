"""PLACEHOLDER backend: hand-rolled low-level image statistics.

Not a serious detector - it is a cheap, explainable signal so the UI shows
non-trivial score distributions, ROC curves and robustness degradation before
a trained model exists. Features are computed on a downscaled grayscale copy:

  * high-frequency energy ratio (FFT)      - generators tend to be smoother
  * noise-residual std (Laplacian proxy)   - cameras carry sensor noise
  * saturation mean                        - synthetic images skew saturated
  * JPEG quantisation-table presence       - originals often re-encoded

Replace with a real model via app/detectors/base.py.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .base import Detector, register


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _features(img: Image.Image, has_quant_table: bool) -> dict:
    small = img.convert("RGB")
    if max(small.size) > 512:
        scale = 512 / max(small.size)
        small = small.resize(
            (max(8, int(small.width * scale)), max(8, int(small.height * scale))),
            Image.BILINEAR,
        )

    rgb = np.asarray(small, dtype=np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)

    # high-frequency energy ratio via FFT
    spec = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean())))
    h, w = spec.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    total = float(spec.sum()) + 1e-8
    hf_ratio = float(spec[r > 0.55].sum()) / total

    # noise residual: gray minus a 3x3 box blur
    pad = np.pad(gray, 1, mode="edge")
    box = (
        pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:]
        + pad[1:-1, :-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:]
        + pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:]
    ) / 9.0
    residual_std = float(np.std(gray - box))

    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    saturation = float(np.mean((mx - mn) / (mx + 1e-6)))

    # local contrast variance: synthetic images are often uniformly detailed
    tiles = gray[: (gray.shape[0] // 16) * 16, : (gray.shape[1] // 16) * 16]
    if tiles.size:
        tiles = tiles.reshape(tiles.shape[0] // 16, 16, tiles.shape[1] // 16, 16)
        tile_std = tiles.std(axis=(1, 3))
        detail_uniformity = float(1.0 - min(1.0, tile_std.std() / (tile_std.mean() + 1e-6)))
    else:
        detail_uniformity = 0.5

    return {
        "hf_ratio": hf_ratio,
        "residual_std": residual_std,
        "saturation": saturation,
        "detail_uniformity": detail_uniformity,
        "quant_table": 1.0 if has_quant_table else 0.0,
    }


def _score_from_features(f: dict) -> float:
    """Weights are hand-tuned intuition, not learned. Placeholder only."""
    z = (
        -14.0 * (f["hf_ratio"] - 0.16)
        - 26.0 * (f["residual_std"] - 0.035)
        + 2.4 * (f["saturation"] - 0.30)
        + 2.0 * (f["detail_uniformity"] - 0.55)
        - 0.45 * f["quant_table"]
        + 0.10
    )
    return _sigmoid(z)


@register
class HeuristicDetector(Detector):
    name = "heuristic"
    display_name = "Frequency heuristic (placeholder)"
    description = (
        "FFT high-frequency ratio, noise residual, saturation and JPEG evidence "
        "combined with hand-set weights. A baseline, not a trained model."
    )
    is_placeholder = True
    batch_size = 8

    def predict_batch(self, paths: list) -> list:
        scores = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    has_quant = bool(getattr(im, "quantization", None))
                    im.load()
                    img = im.convert("RGB")
                    if max(img.size) > 1024:
                        s = 1024 / max(img.size)
                        img = img.resize(
                            (max(8, int(img.width * s)), max(8, int(img.height * s))),
                            Image.BILINEAR,
                        )
                scores.append(round(_score_from_features(_features(img, has_quant)), 6))
            except Exception:
                scores.append(float("nan"))
        return scores

    def predict_images(self, images: list) -> list:
        out = []
        for img in images:
            try:
                out.append(round(_score_from_features(_features(img, False)), 6))
            except Exception:
                out.append(float("nan"))
        return out

    def explain(self, path: str) -> dict:
        """Feature breakdown for the inspector panel."""
        with Image.open(path) as im:
            has_quant = bool(getattr(im, "quantization", None))
            im.load()
            img = im.convert("RGB")
        f = _features(img, has_quant)
        f["score"] = _score_from_features(f)
        return f

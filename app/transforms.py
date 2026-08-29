"""Post-processing / redistribution transforms used by the robustness lab.

Each transform has 5 severity levels (1 = mild, 5 = harsh). Everything runs in
memory on PIL images - nothing is written to disk during a sweep.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _recompress(img: Image.Image, fmt: str, **params) -> Image.Image:
    buf = io.BytesIO()
    out = img.convert("RGB")
    out.save(buf, format=fmt, **params)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# --------------------------------------------------------------------------- #
# individual transform ops
# --------------------------------------------------------------------------- #

def jpeg_compress(img, quality: int):
    return _recompress(img, "JPEG", quality=int(quality), subsampling=2)


def webp_compress(img, quality: int):
    return _recompress(img, "WEBP", quality=int(quality))


def gaussian_blur(img, sigma: float):
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def rescale(img, factor: float):
    """Downscale then back up - the classic detail-destroying resize."""
    w, h = img.size
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    small = img.resize((nw, nh), Image.LANCZOS)
    return small.resize((w, h), Image.BICUBIC)


def center_crop(img, keep: float):
    w, h = img.size
    nw, nh = max(1, int(w * keep)), max(1, int(h * keep))
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))


def brightness_contrast(img, delta: float):
    out = ImageEnhance.Brightness(img).enhance(1.0 + delta)
    return ImageEnhance.Contrast(out).enhance(1.0 - delta * 0.6)


def saturation_shift(img, delta: float):
    return ImageEnhance.Color(img).enhance(1.0 + delta)


def gaussian_noise(img, sigma: float, seed: int = 0):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(seed)
    arr += rng.normal(0.0, float(sigma), arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def sharpen(img, amount: float):
    return ImageEnhance.Sharpness(img).enhance(1.0 + amount)


def social_repost(img, level: int):
    """Downscale + sharpen + JPEG, i.e. what a platform does on re-upload."""
    scale = [0.9, 0.75, 0.6, 0.45, 0.35][level - 1]
    quality = [88, 78, 68, 55, 42][level - 1]
    w, h = img.size
    out = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    out = sharpen(out, 0.4)
    return jpeg_compress(out, quality)


def screenshot(img, level: int):
    """Screenshot-ish: resample to an odd size, mild blur, PNG-clean, then JPEG."""
    scale = [0.95, 0.85, 0.7, 0.55, 0.4][level - 1]
    w, h = img.size
    out = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    out = gaussian_blur(out, 0.3 + 0.2 * level)
    return jpeg_compress(out, [92, 85, 75, 65, 55][level - 1])


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

@dataclass
class TransformSpec:
    key: str
    display_name: str
    description: str
    levels: list                       # 5 parameter values, mild -> harsh
    fn: object = None                  # callable(img, param) -> img
    level_labels: list = field(default_factory=list)

    def apply(self, img: Image.Image, severity: int) -> Image.Image:
        severity = max(1, min(5, int(severity)))
        return self.fn(img, self.levels[severity - 1])

    def label_for(self, severity: int) -> str:
        severity = max(1, min(5, int(severity)))
        if self.level_labels:
            return self.level_labels[severity - 1]
        return str(self.levels[severity - 1])


TRANSFORMS: list = [
    TransformSpec(
        "jpeg", "JPEG recompression",
        "Re-encode as JPEG. The single most common thing that happens to a shared image.",
        [90, 75, 60, 45, 30], jpeg_compress,
        ["q90", "q75", "q60", "q45", "q30"],
    ),
    TransformSpec(
        "webp", "WebP recompression",
        "Re-encode as WebP, as many platforms and CDNs do.",
        [90, 80, 70, 55, 40], webp_compress,
        ["q90", "q80", "q70", "q55", "q40"],
    ),
    TransformSpec(
        "blur", "Gaussian blur",
        "Softens high-frequency generator fingerprints.",
        [0.5, 1.0, 1.5, 2.0, 3.0], gaussian_blur,
        ["σ0.5", "σ1.0", "σ1.5", "σ2.0", "σ3.0"],
    ),
    TransformSpec(
        "rescale", "Downscale → upscale",
        "Resize down and back up, destroying pixel-level traces.",
        [0.75, 0.5, 0.35, 0.25, 0.15], rescale,
        ["75%", "50%", "35%", "25%", "15%"],
    ),
    TransformSpec(
        "crop", "Center crop",
        "Crops away borders; tests reliance on global composition.",
        [0.95, 0.85, 0.75, 0.60, 0.50], center_crop,
        ["keep 95%", "keep 85%", "keep 75%", "keep 60%", "keep 50%"],
    ),
    TransformSpec(
        "bright", "Brightness / contrast",
        "Colour-grading style adjustment.",
        [0.05, 0.10, 0.20, 0.30, 0.40], brightness_contrast,
        ["±5%", "±10%", "±20%", "±30%", "±40%"],
    ),
    TransformSpec(
        "saturation", "Saturation shift",
        "Boosts colour intensity, as filters do.",
        [0.15, 0.30, 0.50, 0.75, 1.00], saturation_shift,
        ["+15%", "+30%", "+50%", "+75%", "+100%"],
    ),
    TransformSpec(
        "noise", "Gaussian noise",
        "Additive sensor-like noise; can mask or mimic generator artifacts.",
        [2, 4, 8, 12, 20], gaussian_noise,
        ["σ2", "σ4", "σ8", "σ12", "σ20"],
    ),
    TransformSpec(
        "social", "Social repost combo",
        "Downscale + sharpen + JPEG: the realistic redistribution pipeline.",
        [1, 2, 3, 4, 5], social_repost,
        ["pass 1", "pass 2", "pass 3", "pass 4", "pass 5"],
    ),
    TransformSpec(
        "screenshot", "Screenshot resample",
        "Odd-ratio resample plus mild blur and re-encode.",
        [1, 2, 3, 4, 5], screenshot,
        ["level 1", "level 2", "level 3", "level 4", "level 5"],
    ),
]

#: identity pass used as the clean baseline cell of a sweep (severity 0)
CLEAN = TransformSpec(
    "clean", "Clean baseline", "No transform - the reference point for every delta.",
    [None] * 5, lambda img, _param: img, ["clean"] * 5,
)

TRANSFORMS_BY_KEY = {t.key: t for t in TRANSFORMS}
TRANSFORMS_BY_KEY[CLEAN.key] = CLEAN


def get_transform(key: str) -> TransformSpec:
    return TRANSFORMS_BY_KEY[key]


def apply_transform(img: Image.Image, key: str, severity: int) -> Image.Image:
    return TRANSFORMS_BY_KEY[key].apply(img, severity)

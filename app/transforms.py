"""Post-processing / redistribution transforms used by the robustness lab.

Each transform has 5 severity levels (1 = mild, 5 = harsh). Everything runs in
memory on PIL images - nothing is written to disk during a sweep.

These are not adversarial attacks. Every one of them is something an image
routinely survives on the way to a screen - a platform re-encode, a resize into
a feed, a screenshot of a repost - which is what makes the drop they cause the
honest measure of whether a detector is usable outside a benchmark.

Adding one: write `fn(img, param) -> img`, then add a TransformSpec with five
parameter values and five human labels. It shows up in the CLI and the GUI
picker automatically.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _recompress(img: Image.Image, fmt: str, **params) -> Image.Image:
    """Encode and decode in memory, so the codec's damage lands in the pixels.

    Re-opening the buffer is the whole point: without the decode you have the
    original image and a discarded byte string.
    """
    buf = io.BytesIO()
    out = img.convert("RGB")
    out.save(buf, format=fmt, **params)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# --------------------------------------------------------------------------- #
# individual transform ops
# --------------------------------------------------------------------------- #

def jpeg_compress(img, quality: int):
    """JPEG at `quality`, 4:2:0 - the chroma subsampling the format ships with."""
    return _recompress(img, "JPEG", quality=int(quality), subsampling=2)


def webp_compress(img, quality: int):
    """WebP at `quality`. What most CDNs and image proxies serve today."""
    return _recompress(img, "WEBP", quality=int(quality))


def gaussian_blur(img, sigma: float):
    """Soften high-frequency detail - where generator fingerprints tend to live."""
    return img.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def rescale(img, factor: float):
    """Downscale then back up - the classic detail-destroying resize.

    Down with LANCZOS and back up with BICUBIC on purpose: that asymmetry is
    what a real pipeline does, and it is what makes the lost detail
    unrecoverable rather than merely resampled.
    """
    w, h = img.size
    nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
    small = img.resize((nw, nh), Image.LANCZOS)
    return small.resize((w, h), Image.BICUBIC)


def center_crop(img, keep: float):
    """Keep the middle `keep` fraction. Tests reliance on global composition."""
    w, h = img.size
    nw, nh = max(1, int(w * keep)), max(1, int(h * keep))
    left, top = (w - nw) // 2, (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh))


def brightness_contrast(img, delta: float):
    """Brighten and flatten together, the way a colour grade actually moves."""
    out = ImageEnhance.Brightness(img).enhance(1.0 + delta)
    return ImageEnhance.Contrast(out).enhance(1.0 - delta * 0.6)


def saturation_shift(img, delta: float):
    """Push colour intensity, as a filter preset does."""
    return ImageEnhance.Color(img).enhance(1.0 + delta)


def gaussian_noise(img, sigma: float, seed: int = 0):
    """Additive sensor-like noise. Seeded, so a sweep is reproducible."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(seed)
    arr += rng.normal(0.0, float(sigma), arr.shape).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def sharpen(img, amount: float):
    """Not a transform of its own - a component of the social repost combo."""
    return ImageEnhance.Sharpness(img).enhance(1.0 + amount)


def social_repost(img, level: int):
    """Downscale + sharpen + JPEG, i.e. what a platform does on re-upload.

    The most realistic cell in the grid, and usually the harshest: the three
    steps compound, and a real repost chains all three too.
    """
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
    """One transform and its five severity settings.

    `levels` holds the parameter passed to `fn`; `level_labels` holds what to
    show a human for each ("q75", "sigma 1.0"). They are parallel, and both are
    indexed by severity - 1 (mild).
    """

    key: str
    display_name: str
    description: str
    levels: list                       # 5 parameter values, mild -> harsh
    fn: object = None                  # callable(img, param) -> img
    level_labels: list = field(default_factory=list)

    def apply(self, img: Image.Image, severity: int) -> Image.Image:
        """Run this transform at 1-5. Out-of-range severities clamp, not crash."""
        severity = max(1, min(5, int(severity)))
        return self.fn(img, self.levels[severity - 1])

    def label_for(self, severity: int) -> str:
        """What to call this severity in a chart, a table or the CLI."""
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

#: lookup by key. CLEAN is reachable here but deliberately not in TRANSFORMS -
#: it is the baseline the sweep always runs, never something you select.
TRANSFORMS_BY_KEY = {t.key: t for t in TRANSFORMS}
TRANSFORMS_BY_KEY[CLEAN.key] = CLEAN


def get_transform(key: str) -> TransformSpec:
    """Look up a spec by key. Raises KeyError on an unknown one."""
    return TRANSFORMS_BY_KEY[key]


def apply_transform(img: Image.Image, key: str, severity: int) -> Image.Image:
    """One-shot convenience: look up `key` and apply it at `severity`."""
    return TRANSFORMS_BY_KEY[key].apply(img, severity)

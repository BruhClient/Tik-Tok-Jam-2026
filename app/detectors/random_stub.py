"""PLACEHOLDER backend: deterministic pseudo-random scores.

Exists so the whole UI - progress, cancel, tables, charts, exports, robustness
sweeps - can be exercised before any model is trained. It reads the label hint
in the file path so the charts show a realistic (not degenerate) separation,
and it degrades under transforms the way a weak detector would.

DELETE OR IGNORE THIS ONCE A REAL MODEL EXISTS.
"""

from __future__ import annotations

import hashlib
import os
import time

from .base import Detector, register


def _stable_unit(text: str) -> float:
    """Deterministic float in [0, 1) from a string."""
    digest = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _label_hint(path: str):
    """Guess ground truth from the path so demo scores look plausible."""
    low = path.replace("\\", "/").lower()
    for token in ("/ai/", "/aigc/", "/fake/", "/generated/", "/synthetic/", "ai_", "fake_"):
        if token in low:
            return 1
    for token in ("/real/", "/authentic/", "/natural/", "/human/", "real_"):
        if token in low:
            return 0
    return None


@register
class RandomDetector(Detector):
    name = "random"
    display_name = "Random (placeholder)"
    description = (
        "Deterministic pseudo-random confidences with a mild label-aware bias. "
        "Useful only for exercising the UI end to end."
    )
    is_placeholder = True
    batch_size = 32

    #: simulated per-image cost, so progress and cancel are observable
    delay_per_image = 0.003

    def _score_path(self, path: str) -> float:
        u = _stable_unit(os.path.abspath(path))
        hint = _label_hint(path)
        if hint == 1:
            score = 0.35 + 0.65 * u ** 0.6          # skewed high
        elif hint == 0:
            score = 0.65 * u ** 1.8                 # skewed low
        else:
            score = u
        return min(max(score, 0.0), 1.0)

    def predict_batch(self, paths: list) -> list:
        scores = [round(self._score_path(p), 6) for p in paths]
        if self.delay_per_image:
            time.sleep(self.delay_per_image * len(paths))
        return scores

    def predict_images(self, images: list) -> list:
        """Score decoded images, pulling the clean score toward chance.

        The robustness worker tags each image with `_aigc_source` (original
        path) and `_aigc_severity` (0-5). Blending toward 0.5 as severity rises
        gives the degradation charts a plausible shape while no real model is
        wired up. A real detector ignores these attributes entirely.
        """
        scores = []
        for img in images:
            src = getattr(img, "_aigc_source", "") or f"{img.size}-{img.mode}"
            severity = int(getattr(img, "_aigc_severity", 0))
            base = self._score_path(src) if getattr(img, "_aigc_source", None) else _stable_unit(src)
            jitter = (_stable_unit(f"{src}|{severity}") - 0.5) * 0.08
            pull = min(0.85, 0.14 * severity)          # decay toward 0.5
            score = base + (0.5 - base) * pull + jitter
            scores.append(round(min(max(score, 0.0), 1.0), 6))
        if self.delay_per_image:
            time.sleep(self.delay_per_image * len(images))
        return scores

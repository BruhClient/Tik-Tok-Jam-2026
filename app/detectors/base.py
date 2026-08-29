"""Detector plugin interface.

This is the ONLY seam the real model needs to touch. To plug in a trained
model, create app/detectors/my_model.py:

    from .base import Detector, register

    @register
    class MyModel(Detector):
        name = "my_model"
        display_name = "EfficientNet-B0 + FFT head"
        description = "Trained on ..."
        is_placeholder = False

        def load(self):
            self.model = torch.load(...)

        def predict_batch(self, paths):
            return [float(p) for p in ...]   # 0.0 = authentic, 1.0 = AI

...then import it in app/detectors/__init__.py. It appears in the toolbar
picker automatically.
"""

from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod

from PIL import Image

_REGISTRY: dict = {}


class Detector(ABC):
    """Base class for all detection backends.

    Scores are confidences in [0, 1] that the image is AI-generated.
    """

    name: str = "base"
    display_name: str = "Base detector"
    description: str = ""
    is_placeholder: bool = False
    # Recommended images per predict_batch call; smaller = smoother progress.
    batch_size: int = 16

    def __init__(self):
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        """Load weights / warm up. Called once, lazily, off the GUI thread."""

    def unload(self) -> None:
        """Release resources."""

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
            self._loaded = True

    # -- inference ---------------------------------------------------------
    @abstractmethod
    def predict_batch(self, paths: list) -> list:
        """Return one float in [0, 1] per input path."""
        raise NotImplementedError

    def predict_images(self, images: list) -> list:
        """Score already-decoded PIL images (used by the robustness sweep).

        The default round-trips through temp files so a path-only detector
        still works. Override this for anything real - it avoids the disk I/O.
        """
        tmp_paths = []
        try:
            for img in images:
                fd, p = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                img.convert("RGB").save(p)
                tmp_paths.append(p)
            return self.predict_batch(tmp_paths)
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def open_image(path: str, max_side: int = 1024) -> Image.Image:
        """Decode with a size cap; JPEG draft mode keeps big files cheap."""
        img = Image.open(path)
        try:
            img.draft("RGB", (max_side, max_side))
        except Exception:
            pass
        img = img.convert("RGB")
        if max(img.size) > max_side:
            scale = max_side / max(img.size)
            img = img.resize(
                (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                Image.BILINEAR,
            )
        return img


def register(cls):
    """Class decorator that adds a detector to the registry."""
    _REGISTRY[cls.name] = cls
    return cls


def available_detectors() -> list:
    """Registered detector classes, real backends listed before placeholders."""
    return sorted(_REGISTRY.values(), key=lambda c: (c.is_placeholder, c.display_name))


def get_detector(name: str) -> Detector:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"No detector registered under {name!r}")
    return cls()

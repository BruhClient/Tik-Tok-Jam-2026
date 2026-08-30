"""Detector registry.

clip_head.py is the real model: a frozen CLIP ViT-L/14 tower plus the trained
MLP head, loaded from the bundle at models/bundle.pt. trained.py stays as the
generic slot for any other checkpoint dropped at models/model.pt.
"""

from .base import (  # noqa: F401
    Detector, register, available_detectors, get_detector, weights_detectors
)

from . import clip_head  # noqa: F401
from . import trained  # noqa: F401

__all__ = ["Detector", "register", "available_detectors", "get_detector",
           "weights_detectors"]

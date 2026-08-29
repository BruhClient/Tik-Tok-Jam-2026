"""Detector registry.

Add a real backend by creating a module here that subclasses Detector and
applies @register, then importing it below.
"""

from .base import Detector, register, available_detectors, get_detector  # noqa: F401

from . import random_stub  # noqa: F401
from . import heuristic_stub  # noqa: F401

# from . import real_model  # <- drop the trained model in here

__all__ = ["Detector", "register", "available_detectors", "get_detector"]

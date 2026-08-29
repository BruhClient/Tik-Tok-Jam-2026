"""Shared application state, owned by MainWindow and observed by the tabs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from PyQt6.QtCore import QObject, pyqtSignal

from .dataset import Dataset, LabelMode
from . import metrics as M


@dataclass
class RunResult:
    """Scores for one detector pass over the current dataset."""

    detector_name: str = ""
    detector_display: str = ""
    is_placeholder: bool = True
    scores: list = field(default_factory=list)     # aligned with dataset.items, NaN = failed
    elapsed: float = 0.0
    cancelled: bool = False
    failures: list = field(default_factory=list)   # (path, message)

    def valid_pairs(self, dataset: Dataset):
        """(y_true, score) for items that are both labeled and successfully scored."""
        y, s = [], []
        for item, score in zip(dataset.items, self.scores):
            if item.label is None or score is None or math.isnan(score):
                continue
            y.append(item.label)
            s.append(score)
        return y, s

    @property
    def n_scored(self) -> int:
        return sum(1 for s in self.scores if s is not None and not math.isnan(s))


@dataclass
class RobustnessCell:
    transform_key: str
    severity: int
    metrics: M.Metrics


class AppState(QObject):
    dataset_changed = pyqtSignal()
    labels_changed = pyqtSignal()
    run_changed = pyqtSignal()            # scores replaced (new run finished / cleared)
    scores_updated = pyqtSignal()         # partial batch arrived during a run
    threshold_changed = pyqtSignal(float)
    detector_changed = pyqtSignal(str)
    robustness_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataset = Dataset()
        self.run: RunResult | None = None
        self.threshold: float = 0.5
        self.detector_name: str = ""
        self.label_mode: LabelMode = LabelMode.AUTO
        self.manifest_path: str | None = None
        self.robustness: dict = {}          # (key, severity) -> RobustnessCell
        self.robustness_baseline: M.Metrics | None = None

    # -- dataset -----------------------------------------------------------
    def set_dataset(self, ds: Dataset):
        self.dataset = ds
        self.run = None
        self.robustness = {}
        self.robustness_baseline = None
        self.dataset_changed.emit()
        self.run_changed.emit()
        self.robustness_changed.emit()

    def notify_labels_changed(self):
        self.labels_changed.emit()

    # -- run ---------------------------------------------------------------
    def start_run(self, detector, n: int):
        self.run = RunResult(
            detector_name=detector.name,
            detector_display=detector.display_name,
            is_placeholder=bool(getattr(detector, "is_placeholder", False)),
            scores=[float("nan")] * n,
        )
        self.run_changed.emit()

    def apply_batch(self, indices, scores):
        if self.run is None:
            return
        for i, s in zip(indices, scores):
            if 0 <= i < len(self.run.scores):
                self.run.scores[i] = float(s)
        self.scores_updated.emit()

    def finish_run(self, cancelled: bool, elapsed: float):
        if self.run is None:
            return
        self.run.cancelled = cancelled
        self.run.elapsed = elapsed
        self.run_changed.emit()

    def clear_run(self):
        self.run = None
        self.run_changed.emit()

    @property
    def has_scores(self) -> bool:
        return self.run is not None and self.run.n_scored > 0

    # -- metrics -----------------------------------------------------------
    def current_metrics(self) -> M.Metrics:
        if self.run is None:
            return M.Metrics(threshold=self.threshold)
        y, s = self.run.valid_pairs(self.dataset)
        return M.compute_metrics(y, s, self.threshold)

    def set_threshold(self, t: float):
        t = float(min(max(t, 0.0), 1.0))
        if abs(t - self.threshold) < 1e-9:
            return
        self.threshold = t
        self.threshold_changed.emit(t)

    # -- robustness --------------------------------------------------------
    def set_baseline(self, m: M.Metrics):
        self.robustness_baseline = m
        self.robustness_changed.emit()

    def set_cell(self, key: str, severity: int, m: M.Metrics):
        self.robustness[(key, severity)] = RobustnessCell(key, severity, m)
        self.robustness_changed.emit()

    def clear_robustness(self):
        self.robustness = {}
        self.robustness_baseline = None
        self.robustness_changed.emit()

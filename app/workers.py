"""Qt threads so the GUI stays responsive while the pipeline runs.

The work itself lives in runner.py / sweep.py and logs to the terminal exactly
as the CLI scripts do - these classes only move it off the GUI thread.
"""

from __future__ import annotations

import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from . import dataset as DS
from . import runner, sweep


class ScanWorker(QThread):
    """Scan a directory for images and labels, without scoring anything.

    Cheap enough to run whenever the folder changes, which is what lets the Run
    page say whether it is looking at labeled or unlabeled data before you
    spend a run finding out.
    """

    finished_ok = pyqtSignal(object)              # Dataset
    failed = pyqtSignal(str)

    def __init__(self, directory: str, parent=None):
        super().__init__(parent)
        self.directory = directory

    def run(self):
        try:
            self.finished_ok.emit(DS.scan_directory(self.directory))
        except Exception as exc:
            self.failed.emit(str(exc))


class ScoreWorker(QThread):
    """Scan a directory and score every image."""

    finished_ok = pyqtSignal(object, object)      # dataset, RunResult
    failed = pyqtSignal(str)

    def __init__(self, directory: str, detector_name: str = None,
                 weights: str = None, label_mode=DS.LabelMode.AUTO, parent=None):
        super().__init__(parent)
        self.directory = directory
        self.detector_name = detector_name
        self.weights = weights
        self.label_mode = label_mode

    def run(self):
        try:
            dataset, result = runner.run_directory(
                self.directory, self.detector_name, self.weights,
                label_mode=self.label_mode)
            self.finished_ok.emit(dataset, result)
        except SystemExit as exc:                 # runner raises these for bad input
            self.failed.emit(str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))


class LoadWorker(QThread):
    """Read a finished predictions.json back in."""

    finished_ok = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, json_path: str, parent=None):
        super().__init__(parent)
        self.json_path = json_path

    def run(self):
        try:
            dataset, result = runner.load_predictions(self.json_path)
            self.finished_ok.emit(dataset, result)
        except SystemExit as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))


class SweepWorker(QThread):
    """Run the robustness sweep, reporting each finished cell."""

    cell_done = pyqtSignal(int, int, str)         # index, total, name
    finished_ok = pyqtSignal(object)              # SweepResult
    failed = pyqtSignal(str)

    def __init__(self, dataset, detector_name, cells, sample, max_side,
                 threshold, weights=None, parent=None):
        super().__init__(parent)
        self.dataset = dataset
        self.detector_name = detector_name
        self.weights = weights
        self.cells = cells
        self.sample = sample
        self.max_side = max_side
        self.threshold = threshold
        self._stop = False

    def cancel(self):
        self._stop = True

    def run(self):
        try:
            detector = runner.prepare_detector(
                self.detector_name, self.weights, total_steps=3)
            result = sweep.run_sweep(
                self.dataset, detector, self.cells,
                sample=self.sample, max_side=self.max_side, threshold=self.threshold,
                on_cell=lambda i, n, name, m: self.cell_done.emit(i, n, name),
                should_stop=lambda: self._stop,
            )
            self.finished_ok.emit(result)
        except SystemExit as exc:                 # missing checkpoint, bad input
            self.failed.emit(str(exc))
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(str(exc))

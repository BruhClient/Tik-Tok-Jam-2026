"""Main window: toolbar, tabs, run orchestration, status bar."""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QComboBox, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSizePolicy, QStatusBar, QTabWidget, QToolBar, QWidget
)

from .. import theme as T
from ..detectors import available_detectors, get_detector
from ..state import AppState
from ..workers import InferenceWorker, ThumbnailLoader
from .components import Badge
from .dataset_tab import DatasetTab
from .results_tab import ResultsTab
from .robustness_tab import RobustnessTab

APP_TITLE = "AIGC Image Detector — evaluation console"


class MainWindow(QMainWindow):
    def __init__(self, start_dir: str = None):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1480, 940)
        self.setMinimumSize(1100, 720)

        self.state = AppState(self)
        self.loader = ThumbnailLoader(size=132, parent=self)
        self.worker: InferenceWorker | None = None
        self._detector_cache: dict = {}

        self._build_toolbar()
        self._build_tabs()
        self._build_statusbar()

        self.state.dataset_changed.connect(self._sync_actions)
        self.state.run_changed.connect(self._sync_actions)
        self.state.labels_changed.connect(self._update_status_counts)
        self.state.dataset_changed.connect(self._update_status_counts)

        self._sync_actions()
        self._update_status_counts()

        if start_dir and os.path.isdir(start_dir):
            self.dataset_tab.load_directory(start_dir)

    # -- construction ------------------------------------------------------
    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setFloatable(False)
        self.addToolBar(tb)

        title = QLabel("  AIGC DETECTOR  ")
        title.setStyleSheet(
            f"color: {T.TEXT}; font-size: 13px; font-weight: 800; letter-spacing: 1px;"
            " background: transparent;")
        tb.addWidget(title)
        tb.addSeparator()

        self.load_btn = QPushButton("📂  Load directory")
        self.load_btn.clicked.connect(lambda: self.dataset_tab.choose_directory())
        tb.addWidget(self.load_btn)

        tb.addWidget(QLabel("  Detector "))
        self.detector_combo = QComboBox()
        self.detector_combo.setMinimumWidth(260)
        for cls in available_detectors():
            suffix = "" if not cls.is_placeholder else "  ·  placeholder"
            self.detector_combo.addItem(cls.display_name + suffix, cls.name)
            self.detector_combo.setItemData(self.detector_combo.count() - 1,
                                            cls.description, Qt.ItemDataRole.ToolTipRole)
        self.detector_combo.currentIndexChanged.connect(self._on_detector_changed)
        tb.addWidget(self.detector_combo)

        self.run_btn = QPushButton("▶  Run detection")
        self.run_btn.setProperty("accent", True)
        self.run_btn.clicked.connect(self.run_detection)
        tb.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("■  Cancel")
        self.cancel_btn.clicked.connect(self.cancel_detection)
        self.cancel_btn.setEnabled(False)
        tb.addWidget(self.cancel_btn)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        self.detector_badge = Badge("placeholder backend", T.WARN)
        tb.addWidget(self.detector_badge)

        self.export_btn = QPushButton("⬇  Export predictions.json")
        self.export_btn.clicked.connect(lambda: self.results_tab.export_json())
        tb.addWidget(self.export_btn)

        run_action = QAction("Run", self)
        run_action.setShortcut(QKeySequence("Ctrl+R"))
        run_action.triggered.connect(self.run_detection)
        self.addAction(run_action)

        open_action = QAction("Open", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(lambda: self.dataset_tab.choose_directory())
        self.addAction(open_action)

        self._on_detector_changed()

    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        self.dataset_tab = DatasetTab(self.state, self.loader)
        self.results_tab = ResultsTab(self.state)
        self.robustness_tab = RobustnessTab(self.state, self._current_detector)

        self.tabs.addTab(self.dataset_tab, "  Dataset  ")
        self.tabs.addTab(self.results_tab, "  Results  ")
        self.tabs.addTab(self.robustness_tab, "  Robustness lab  ")

        for tab in (self.dataset_tab, self.results_tab, self.robustness_tab):
            tab.status_message.connect(self.set_status)
        self.robustness_tab.busy_changed.connect(self._on_sweep_busy)
        self.dataset_tab.dataset_loaded.connect(lambda _p: self.tabs.setCurrentIndex(0))

        self.setCentralWidget(self.tabs)

    def _build_statusbar(self):
        bar = QStatusBar()
        self.setStatusBar(bar)

        self.status_label = QLabel("Ready")
        self.counts_label = QLabel("")
        self.counts_label.setStyleSheet(f"color: {T.TEXT_FAINT};")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setVisible(False)

        bar.addWidget(self.status_label, 1)
        bar.addPermanentWidget(self.counts_label)
        bar.addPermanentWidget(self.progress)

    # -- detector ----------------------------------------------------------
    def _current_detector(self):
        name = self.detector_combo.currentData()
        if name not in self._detector_cache:
            self._detector_cache[name] = get_detector(name)
        return self._detector_cache[name]

    def _on_detector_changed(self):
        det = self._current_detector()
        self.state.detector_name = det.name
        if getattr(det, "is_placeholder", False):
            self.detector_badge.setText("placeholder backend")
            self.detector_badge.set_color(T.WARN)
        else:
            self.detector_badge.setText("model backend")
            self.detector_badge.set_color(T.GOOD)
        self.detector_badge.setToolTip(det.description)

    # -- run ---------------------------------------------------------------
    def run_detection(self):
        if self.worker is not None:
            return
        ds = self.state.dataset
        if not ds.items:
            QMessageBox.information(self, "No dataset",
                                    "Load an image directory first (Ctrl+O).")
            return

        try:
            detector = self._current_detector()
        except Exception as exc:
            QMessageBox.critical(self, "Detector error", str(exc))
            return

        paths = [it.path for it in ds.items]
        self.state.start_run(detector, len(paths))

        self.worker = InferenceWorker(detector, paths, parent=self)
        self.worker.batch_ready.connect(self.state.apply_batch)
        self.worker.progress.connect(self._on_progress)
        self.worker.failed_item.connect(self._on_failed_item)
        self.worker.error.connect(lambda msg: self.set_status("Error: " + msg))
        self.worker.done.connect(self._on_run_done)

        self.progress.setVisible(True)
        self.progress.setRange(0, len(paths))
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.load_btn.setEnabled(False)
        self.detector_combo.setEnabled(False)
        self.tabs.setCurrentIndex(1)
        self.set_status(f"Running {detector.display_name} on {len(paths):,} images…")
        self.worker.start()

    def cancel_detection(self):
        if self.worker is not None:
            self.worker.cancel()
            self.set_status("Cancelling…")

    def _on_progress(self, done: int, total: int, eta: float):
        self.progress.setValue(done)
        self.set_status(f"Scored {done:,}/{total:,} images · ~{eta:.0f}s remaining")

    def _on_failed_item(self, path: str, message: str):
        if self.state.run is not None:
            self.state.run.failures.append((path, message))

    def _on_run_done(self, cancelled: bool, elapsed: float):
        run = self.state.run
        self.state.finish_run(cancelled, elapsed)
        self.worker = None
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.load_btn.setEnabled(True)
        self.detector_combo.setEnabled(True)

        n = run.n_scored if run else 0
        msg = ("Run cancelled — " if cancelled else "Run complete — ")
        msg += f"{n:,} images scored in {elapsed:.1f}s"
        if run and run.failures:
            msg += f" · {len(run.failures)} failed"
        self.set_status(msg)
        self._sync_actions()

    def _on_sweep_busy(self, busy: bool):
        self.run_btn.setEnabled(not busy and bool(self.state.dataset.items))
        self.detector_combo.setEnabled(not busy)
        self.load_btn.setEnabled(not busy)

    # -- status ------------------------------------------------------------
    def set_status(self, text: str):
        self.status_label.setText(text)

    def _update_status_counts(self):
        ds = self.state.dataset
        if not ds.items:
            self.counts_label.setText("no dataset")
            return
        self.counts_label.setText(
            f"{len(ds):,} images · {ds.n_real:,} real · {ds.n_ai:,} AI · "
            f"{ds.n_unlabeled:,} unlabeled")

    def _sync_actions(self):
        has_ds = bool(self.state.dataset.items)
        self.run_btn.setEnabled(has_ds and self.worker is None)
        self.export_btn.setEnabled(self.state.has_scores)

    # -- shutdown ----------------------------------------------------------
    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(3000)
        sweep = getattr(self.robustness_tab, "worker", None)
        if sweep is not None:
            sweep.cancel()
            sweep.wait(3000)
        self.loader.shutdown()
        super().closeEvent(event)

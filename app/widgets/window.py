"""The application shell: an upload screen, then a results screen.

There is no navigation rail. You start by saying what you are uploading, and
what comes back is decided by that: a labeled dataset opens on Insights with
Images and Robustness behind header tabs, while plain images open on the
verdict gallery with no tabs at all - there is nothing else to look at when
there is no ground truth.

The pipeline itself lives in runner.py / sweep.py and logs to the terminal.
This owns the widgets and moves the work onto a thread so the window stays
responsive; it never reimplements the pipeline.
"""

from __future__ import annotations

import math
import os

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QButtonGroup, QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSlider, QStackedWidget, QVBoxLayout, QWidget
)

from .. import export as EX
from .. import metrics as M
from .. import runner
from .. import sweep as SW
from .. import theme as T
from ..dataset import LabelMode
from ..workers import LoadWorker, ScanWorker, ScoreWorker, SweepWorker
from . import components as C
from .components import Dot, Tab
from .gallery import GalleryPage
from .pages import ImagesPage, InsightsPage, RobustnessPage
from .upload import UploadPage

SCREEN_UPLOAD, SCREEN_RESULTS = range(2)
TAB_INSIGHTS, TAB_IMAGES, TAB_ROBUSTNESS, VIEW_GALLERY = range(4)

#: the three tabs a labeled run gets, in order. The gallery is not one of them:
#: it is the whole of the unlabeled screen, so it never needs a tab.
TABS = [
    ("Insights", "How the detector performed"),
    ("Images", "Every prediction, filterable"),
    ("Robustness", "Accuracy under post-processing"),
]

#: the threshold means something on any page that renders a verdict
THRESHOLD_VIEWS = (TAB_INSIGHTS, TAB_IMAGES, VIEW_GALLERY)


class AppWindow(QMainWindow):
    def __init__(self, start_dir: str = None, start_json: str = None,
                 threshold: float = 0.5):
        super().__init__()

        # -- state
        self.dataset = None
        self.result = None
        self.robustness = {}
        self.threshold = threshold
        # What Reset goes back to. A run replaces it with the detector's own
        # operating point, so Reset means "the model's answer", not "0.50".
        self.base_threshold = threshold
        self.worker = None
        self.sweep_worker = None
        self.peek_worker = None
        self.view = TAB_INSIGHTS

        self.setWindowTitle("AIGC Detector")
        self.resize(1400, 900)
        self.setMinimumSize(1080, 700)

        self.upload_page = UploadPage(self)

        self.screens = QStackedWidget()
        self.screens.addWidget(self.upload_page)
        self.screens.addWidget(self._build_results())
        self.setCentralWidget(self.screens)

        self._chart_timer = QTimer(self)
        self._chart_timer.setSingleShot(True)
        self._chart_timer.timeout.connect(lambda: self.refresh(charts=True))

        self.go_upload()

        if start_json:
            self.load_predictions_file(start_json)
        elif start_dir:
            # the command line already said what it wanted by naming a folder,
            # so let the labels decide rather than demanding the tile be clicked
            self.upload_page.set_directory(start_dir)
            self.start_run(LabelMode.AUTO)

    # -- results screen ----------------------------------------------------
    def _build_results(self) -> QWidget:
        screen = QWidget()
        outer = QVBoxLayout(screen)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(28, 20, 28, 22)

        self.insights_page = InsightsPage(self)
        self.images_page = ImagesPage(self)
        self.robustness_page = RobustnessPage(self)
        self.gallery_page = GalleryPage(self)

        self.stack = QStackedWidget()
        for w in (self.insights_page, self.images_page, self.robustness_page,
                  self.gallery_page):
            self.stack.addWidget(w)
        lay.addWidget(self.stack, 1)
        outer.addWidget(body, 1)
        return screen

    def _build_header(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("header")
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(28, 12, 28, 0)
        lay.setSpacing(8)

        # -- row 1: what is loaded, and what you can do with it
        top = QHBoxLayout()
        top.setSpacing(12)

        mark = QLabel()                       # the app mark, drawn not typeset
        mark.setFixedSize(18, 18)
        mark.setStyleSheet(f"background-color: {T.ACCENT}; border-radius: 5px;")

        self.source_label = QLabel("")
        self.source_label.setStyleSheet(
            f"color: {T.TEXT}; font-size: 16px; font-weight: 700;"
            " letter-spacing: -0.3px;")
        self.counts_label = QLabel("")
        self.counts_label.setStyleSheet(
            f"color: {T.TEXT_FAINT}; font-size: {C.FS_SMALL}px;")

        self.detector_dot = Dot(T.TEXT_FAINT)
        self.detector_label = QLabel("")
        self.detector_label.setStyleSheet(
            f"color: {T.TEXT_FAINT}; font-size: {C.FS_SMALL}px;")

        self.new_btn = QPushButton("New upload")
        self.new_btn.setToolTip("Back to the upload screen")
        self.new_btn.clicked.connect(self.go_upload)
        self.export_btn = QPushButton("Export JSON")
        self.export_btn.setProperty("accent", True)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_json)

        top.addWidget(mark)
        top.addWidget(self.source_label)
        top.addWidget(self.counts_label)
        top.addSpacing(6)
        top.addWidget(self.detector_dot)
        top.addWidget(self.detector_label)
        top.addStretch(1)
        top.addWidget(self.new_btn)
        top.addWidget(self.export_btn)
        lay.addLayout(top)

        # -- row 2: tabs on the left, the threshold on the right
        bottom = QHBoxLayout()
        bottom.setSpacing(0)

        self.tab_bar = QWidget()
        self.tab_bar.setProperty("bare", True)     # it sits on the header
        tabs = QHBoxLayout(self.tab_bar)
        tabs.setContentsMargins(0, 0, 0, 0)
        tabs.setSpacing(0)
        self.tabs = []
        group = QButtonGroup(self)
        group.setExclusive(True)
        for i, (name, hint) in enumerate(TABS):
            btn = Tab(name)
            btn.setToolTip(hint)
            btn.clicked.connect(lambda _, idx=i: self.go(idx))
            group.addButton(btn)
            tabs.addWidget(btn)
            self.tabs.append(btn)
        tabs.addStretch(1)

        # an explicit stretch, not the tab strip's - the strip is hidden for
        # unlabeled data, and a hidden widget hands its space to whatever is
        # left, which would stretch the threshold controls across the header
        bottom.addWidget(self.tab_bar)
        bottom.addStretch(1)
        bottom.addWidget(self._build_threshold())
        lay.addLayout(bottom)
        return bar

    def _build_threshold(self) -> QWidget:
        self.threshold_box = QWidget()
        self.threshold_box.setProperty("bare", True)
        row = QHBoxLayout(self.threshold_box)
        row.setContentsMargins(0, 0, 0, 6)
        row.setSpacing(10)

        label = QLabel("Threshold")
        label.setStyleSheet(
            f"color: {T.TEXT_FAINT}; font-size: {C.FS_SMALL}px; font-weight: 500;")
        self.thr_label = QLabel(f"{self.threshold:.3f}")
        self.thr_label.setFixedWidth(48)
        self.thr_label.setStyleSheet(
            f"color: {T.TEXT}; font-size: {C.FS_TITLE}px; font-weight: 600;")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setFixedWidth(160)
        self.slider.setRange(0, 1000)
        self.slider.setValue(int(self.threshold * 1000))
        self.slider.valueChanged.connect(self._on_slider)

        self.best_btn = QPushButton("Best F1")
        self.best_btn.setToolTip("Move the threshold to the value that maximises F1")
        self.best_btn.clicked.connect(self._best_threshold)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(
            lambda: self.set_threshold(self.base_threshold))
        self._sync_reset_tip()

        for w in (label, self.thr_label, self.slider, self.best_btn,
                  self.reset_btn):
            row.addWidget(w)
        return self.threshold_box

    # -- navigation --------------------------------------------------------
    def go_upload(self):
        """Back to the start, with the folder and the choice still in place."""
        self.screens.setCurrentIndex(SCREEN_UPLOAD)

    def go(self, index: int):
        self.view = index
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.tabs):
            btn.setChecked(i == index)
        self.threshold_box.setVisible(index in THRESHOLD_VIEWS)
        page = self.stack.currentWidget()
        if hasattr(page, "refresh"):
            page.refresh()

    def show_results(self):
        """Route to the screen the data actually supports."""
        labeled = self.dataset is not None and self.dataset.has_labels
        self.tab_bar.setVisible(labeled)
        self.screens.setCurrentIndex(SCREEN_RESULTS)
        self.go(TAB_INSIGHTS if labeled else VIEW_GALLERY)

    # -- state helpers -----------------------------------------------------
    def score_at(self, di: int):
        if self.result is None or di < 0 or di >= len(self.result.scores):
            return None
        s = self.result.scores[di]
        return None if s is None or math.isnan(s) else s

    def set_threshold(self, t: float):
        self.slider.setValue(int(round(min(max(t, 0.0), 1.0) * 1000)))

    def _sync_reset_tip(self):
        why = (", the detector's own operating point"
               if self.result is not None else "")
        self.reset_btn.setToolTip(f"Back to {self.base_threshold:.3f}{why}")

    def _on_slider(self, value: int):
        self.threshold = value / 1000.0
        self.thr_label.setText(f"{self.threshold:.3f}")
        self.refresh(charts=False)
        self._chart_timer.start(140)

    def _best_threshold(self):
        if self.result is None or self.dataset is None:
            return
        y, s = self.result.valid_pairs(self.dataset)
        if len(set(y)) < 2:
            return
        self.set_threshold(M.best_threshold(y, s, "f1"))

    def refresh(self, charts: bool = True):
        self.insights_page.refresh(charts=charts)
        self.images_page.refresh()
        self.gallery_page.refresh()
        if charts:
            self.robustness_page.refresh()

    def _sync_header(self):
        if self.dataset is None:
            self.source_label.setText("")
            self.counts_label.setText("")
            self.detector_label.setText("")
            self.detector_dot.set_color(T.TEXT_FAINT)
            self.export_btn.setEnabled(False)
            self.best_btn.setEnabled(False)
            return

        source = self.dataset.root or (self.result.source if self.result else "")
        self.source_label.setText(os.path.basename(source.rstrip("\\/")) or source)
        self.source_label.setToolTip(source)

        counts = f"{len(self.dataset):,} images"
        counts += (f"   ·   {self.dataset.n_real:,} real / {self.dataset.n_ai:,} AI"
                   if self.dataset.has_labels else "   ·   unlabeled")
        self.counts_label.setText(counts)

        if self.result is not None:
            self.detector_label.setText(
                self.result.detector_display.replace(" (placeholder)", ""))
            placeholder = self.result.is_placeholder
            self.detector_dot.set_color(T.WARN if placeholder else T.GOOD)
            tip = ("Placeholder backend — these scores are not real detections."
                   if placeholder else "Model backend")
            if self.result.elapsed:
                tip += f"\nScored in {self.result.elapsed:.1f}s"
            self.detector_dot.setToolTip(tip)
            self.detector_label.setToolTip(tip)

        self.export_btn.setEnabled(self.result is not None and self.result.n_scored > 0)
        self.best_btn.setEnabled(self.dataset.has_labels)

    # -- running -----------------------------------------------------------
    def start_run(self, label_mode: LabelMode = None):
        if self.worker is not None:
            return
        directory = self.upload_page.directory
        if not directory or not os.path.isdir(directory):
            QMessageBox.information(self, "No folder",
                                    "Pick an image folder to score.")
            return
        if label_mode is None:
            label_mode = self.upload_page.label_mode

        runner.log("")
        runner.log(f"=== Detect {directory} ===")

        self.upload_page.set_busy(True, "Working — detailed progress in the terminal.")
        self.worker = ScoreWorker(directory, self.upload_page.detector_name,
                                  self.upload_page.weights, label_mode, parent=self)
        self.worker.finished_ok.connect(self._on_run_done)
        self.worker.failed.connect(self._on_run_failed)
        self.worker.start()

    def peek_directory(self, directory: str):
        """Scan a folder in the background so the upload screen can describe it.

        Scanning is cheap next to scoring, and it is what tells you whether the
        folder can support the kind of upload you just declared - before you
        spend a run finding out.
        """
        if self.peek_worker is not None:
            self.peek_worker.wait(0)      # a stale scan just loses the race
        self.peek_worker = ScanWorker(directory, parent=self)
        self.peek_worker.finished_ok.connect(self._on_peek_done)
        self.peek_worker.failed.connect(lambda _msg: setattr(self, "peek_worker", None))
        self.peek_worker.start()

    def _on_peek_done(self, dataset):
        self.peek_worker = None
        self.upload_page.set_peek(dataset)

    def load_predictions_file(self, path: str):
        if self.worker is not None:
            return
        runner.log("")
        runner.log(f"=== Open {path} ===")
        self.upload_page.set_busy(True, "Reading predictions…")
        self.worker = LoadWorker(path, parent=self)
        self.worker.finished_ok.connect(self._on_run_done)
        self.worker.failed.connect(self._on_run_failed)
        self.worker.start()

    def _on_run_done(self, dataset, result):
        self.worker = None
        self.dataset = dataset
        self.result = result
        self.robustness = {}

        # Adopt the detector's operating point. The calibrated model's is not
        # 0.5, and leaving the slider there would call every image AI.
        self.base_threshold = float(result.threshold)
        self._sync_reset_tip()
        self.set_threshold(self.base_threshold)

        runner.summarize(dataset, result, self.threshold, total_steps=4, step_no=4)

        # a sweep report sitting next to the data is picked up for free
        report = os.path.join(dataset.root or "", SW.DEFAULT_REPORT)
        view = SW.load_report_view(report) if dataset.root else {}
        if view and dataset.has_labels:
            self.robustness = view
            runner.log(f"      robustness report loaded from "
                       f"{os.path.basename(report)}", indent=0)

        self.upload_page.set_busy(False, "")
        self._sync_header()
        self.refresh(charts=True)
        self.show_results()

    def _on_run_failed(self, message: str):
        self.worker = None
        self.upload_page.set_busy(False, "Failed.")
        QMessageBox.critical(self, "Run failed", message)

    # -- sweep -------------------------------------------------------------
    def start_sweep(self):
        if self.sweep_worker is not None:
            return
        if self.dataset is None:
            QMessageBox.information(self, "Nothing loaded",
                                    "Upload a folder first.")
            return
        if not self.dataset.has_labels:
            QMessageBox.information(
                self, "Labels required",
                "The sweep measures accuracy, so it needs ground-truth labels.\n\n"
                "Use real/ and ai/ subfolders, a labels.csv, or real_/ai_ prefixes.")
            return
        cells = self.robustness_page.selected_cells()
        if not cells:
            QMessageBox.information(self, "Nothing selected",
                                    "Tick at least one transform.")
            return

        runner.log("")
        runner.log(f"=== Robustness sweep: {len(cells) + 1} cells ===")
        self.robustness_page.set_busy(True, "Sweeping — progress in the terminal.")
        self.sweep_worker = SweepWorker(
            self.dataset, self.upload_page.detector_name, cells,
            self.robustness_page.sample_spin.value(),
            self.robustness_page.side_spin.value(),
            self.threshold, self.upload_page.weights, parent=self)
        self.sweep_worker.cell_done.connect(
            lambda i, n, name: self.robustness_page.set_busy(True, f"{i}/{n}  {name}"))
        self.sweep_worker.finished_ok.connect(self._on_sweep_done)
        self.sweep_worker.failed.connect(self._on_sweep_failed)
        self.sweep_worker.start()

    def cancel_sweep(self):
        if self.sweep_worker is not None:
            self.sweep_worker.cancel()
            self.robustness_page.set_busy(True, "Cancelling…")

    def _on_sweep_done(self, result):
        self.sweep_worker = None
        self.robustness = result.to_view()
        path = os.path.join(self.dataset.root, SW.DEFAULT_REPORT)
        try:
            written = result.write(path)
            runner.log(f"      report -> {written}")
        except Exception as exc:
            runner.warn(f"could not write the report: {exc}")
        self.robustness_page.set_busy(
            False, f"{len(result.cells)} cells in {result.elapsed:.1f}s")
        self.robustness_page.refresh()

    def _on_sweep_failed(self, message: str):
        self.sweep_worker = None
        self.robustness_page.set_busy(False, "Failed.")
        QMessageBox.critical(self, "Sweep failed", message)

    # -- export ------------------------------------------------------------
    def export_json(self):
        if self.result is None:
            return
        default = os.path.join(self.dataset.root or os.path.expanduser("~"),
                               "predictions.json")
        path, _ = QFileDialog.getSaveFileName(self, "Export predictions", default,
                                              "JSON (*.json)")
        if not path:
            return
        try:
            n = EX.export_predictions_json(path, self.dataset, self.result)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        runner.log(f"      wrote {n:,} predictions -> {os.path.abspath(path)}")
        QMessageBox.information(self, "Exported", f"Wrote {n:,} records to:\n{path}")

    # -- shutdown ----------------------------------------------------------
    def closeEvent(self, event):
        for w in (self.worker, self.sweep_worker, self.peek_worker):
            if w is not None:
                if hasattr(w, "cancel"):
                    w.cancel()
                w.wait(3000)
        super().closeEvent(event)

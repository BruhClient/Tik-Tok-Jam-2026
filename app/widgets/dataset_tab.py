"""Dataset tab: bulk load, label summary, thumbnail grid, inspector."""

from __future__ import annotations

import math
import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSplitter, QVBoxLayout, QWidget
)

from .. import theme as T
from ..dataset import LABEL_MODE_TITLES, LabelMode, apply_labels, scan_directory
from .components import Badge, Card, EmptyState, Hint, SectionTitle, StatCard, score_color
from .image_grid import ImageGrid


class DatasetTab(QWidget):
    dataset_loaded = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(self, state, loader, parent=None):
        super().__init__(parent)
        self.state = state
        self.loader = loader
        self._selected_index = -1

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(12)

        root.addLayout(self._build_header())
        root.addLayout(self._build_stats())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_grid_panel())
        splitter.addWidget(self._build_inspector())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([900, 320])
        root.addWidget(splitter, 1)

        self.state.dataset_changed.connect(self.refresh)
        self.state.labels_changed.connect(self.refresh)
        self.state.run_changed.connect(self._on_scores)
        # during a run only the tiles need repainting; reloading the inspector
        # image on every batch would hit the disk far too often
        self.state.scores_updated.connect(self.grid.refresh_scores)

        self.setAcceptDrops(True)
        self.refresh()

    # -- construction ------------------------------------------------------
    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self.load_btn = QPushButton("📂  Load image directory")
        self.load_btn.setProperty("accent", True)
        self.load_btn.clicked.connect(self.choose_directory)

        self.path_label = QLabel("No dataset loaded")
        self.path_label.setStyleSheet(f"color: {T.TEXT_DIM}; background: transparent;")

        self.label_mode_combo = QComboBox()
        for mode in (LabelMode.AUTO, LabelMode.SUBFOLDER, LabelMode.MANIFEST,
                     LabelMode.FILENAME, LabelMode.NONE):
            self.label_mode_combo.addItem(LABEL_MODE_TITLES[mode], mode)
        self.label_mode_combo.setToolTip("How ground-truth labels are derived")
        self.label_mode_combo.currentIndexChanged.connect(self._on_label_mode)

        self.manifest_btn = QPushButton("Load manifest…")
        self.manifest_btn.setToolTip("Pick a CSV/JSON with image_path + label columns")
        self.manifest_btn.clicked.connect(self.choose_manifest)

        row.addWidget(self.load_btn)
        row.addWidget(self.path_label, 1)
        row.addWidget(QLabel("Labels:"))
        row.addWidget(self.label_mode_combo)
        row.addWidget(self.manifest_btn)
        return row

    def _build_stats(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self.card_total = StatCard("Images", "in this directory")
        self.card_real = StatCard("Authentic", "label 0")
        self.card_ai = StatCard("AI-generated", "label 1")
        self.card_unlabeled = StatCard("Unlabeled", "inference only")
        for c in (self.card_total, self.card_real, self.card_ai, self.card_unlabeled):
            row.addWidget(c)
        return row

    def _build_grid_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(8)

        bar = QHBoxLayout()
        self.source_badge = Badge("no labels", T.TEXT_DIM)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter by filename or folder…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.count_label = QLabel("")
        self.count_label.setStyleSheet(f"color: {T.TEXT_FAINT}; background: transparent;")

        bar.addWidget(self.source_badge)
        bar.addWidget(self.filter_edit, 1)
        bar.addWidget(self.count_label)
        lay.addLayout(bar)

        self.grid = ImageGrid(self.state, self.loader)
        self.grid.item_selected.connect(self.show_item)
        self.empty = EmptyState(
            "Drop a folder here, or click “Load image directory”",
            "Subfolders named real/ and ai/ are picked up automatically; "
            "a labels.csv or real_/ai_ filename prefixes also work.",
        )
        lay.addWidget(self.grid, 1)
        lay.addWidget(self.empty, 1)
        return panel

    def _build_inspector(self) -> QWidget:
        card = Card(padding=14)
        card.layout().addWidget(SectionTitle("Inspector"))

        self.preview = QLabel()
        self.preview.setMinimumHeight(230)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet(
            f"background-color: #17171C; border: 1px solid {T.BORDER};"
            " border-radius: 8px; color: #6B6B78;"
        )
        self.preview.setText("Select an image")

        self.meta_label = Hint("")
        self.meta_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.score_label = QLabel("")
        self.score_label.setStyleSheet(
            f"color: {T.TEXT}; font-size: 20px; font-weight: 700; background: transparent;"
        )

        self.open_btn = QPushButton("Open containing folder")
        self.open_btn.clicked.connect(self._open_folder)
        self.open_btn.setEnabled(False)

        card.layout().addWidget(self.preview)
        card.layout().addWidget(self.score_label)
        card.layout().addWidget(self.meta_label)
        card.layout().addStretch(1)
        card.layout().addWidget(self.open_btn)
        return card

    # -- loading -----------------------------------------------------------
    def choose_directory(self):
        start = self.state.dataset.root or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, "Select image directory", start)
        if path:
            self.load_directory(path)

    def load_directory(self, path: str):
        self.status_message.emit(f"Scanning {path} …")
        mode = self.label_mode_combo.currentData() or LabelMode.AUTO
        ds = scan_directory(path, mode=mode, manifest_path=self.state.manifest_path)
        self.state.label_mode = mode
        self.state.set_dataset(ds)
        self.dataset_loaded.emit(path)
        msg = f"Loaded {len(ds)} images from {path}"
        if ds.skipped:
            msg += f" · skipped {ds.skipped} non-image files"
        self.status_message.emit(msg)

    def choose_manifest(self):
        start = self.state.dataset.root or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select label manifest", start, "Manifests (*.csv *.json);;All files (*)")
        if not path:
            return
        self.state.manifest_path = path
        idx = self.label_mode_combo.findData(LabelMode.MANIFEST)
        if idx >= 0:
            self.label_mode_combo.setCurrentIndex(idx)
        else:
            self._on_label_mode()

    def _on_label_mode(self):
        mode = self.label_mode_combo.currentData() or LabelMode.AUTO
        self.state.label_mode = mode
        if not self.state.dataset.items:
            return
        apply_labels(self.state.dataset, mode=mode, manifest_path=self.state.manifest_path)
        self.state.notify_labels_changed()
        self.status_message.emit("Labels: " + self.state.dataset.label_source_detail)

    # -- drag & drop -------------------------------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if os.path.isdir(p):
                self.load_directory(p)
                break
            if os.path.isfile(p):
                self.load_directory(os.path.dirname(p))
                break

    # -- refresh -----------------------------------------------------------
    def refresh(self):
        ds = self.state.dataset
        has = len(ds) > 0
        self.grid.setVisible(has)
        self.empty.setVisible(not has)

        self.path_label.setText(ds.root if ds.root else "No dataset loaded")
        self.card_total.set_value(f"{len(ds):,}")
        self.card_real.set_value(f"{ds.n_real:,}", color=T.REAL_COLOR if ds.n_real else None)
        self.card_ai.set_value(f"{ds.n_ai:,}", color=T.AI_COLOR if ds.n_ai else None)
        self.card_unlabeled.set_value(f"{ds.n_unlabeled:,}")

        if ds.has_labels:
            self.source_badge.setText(ds.label_source_detail)
            self.source_badge.set_color(T.GOOD)
        else:
            self.source_badge.setText("unlabeled — inference only")
            self.source_badge.set_color(T.WARN if has else T.TEXT_DIM)

        self._apply_filter()

    def _apply_filter(self):
        text = self.filter_edit.text().strip().lower()
        items = self.state.dataset.items
        if text:
            rows = [i for i, it in enumerate(items) if text in it.rel_path.lower()]
        else:
            rows = list(range(len(items)))
        self.grid.set_rows(rows)
        self.count_label.setText(f"showing {len(rows):,} of {len(items):,}")

    def _on_scores(self):
        self.grid.refresh_scores()
        if self._selected_index >= 0:
            self.show_item(self._selected_index)

    # -- inspector ---------------------------------------------------------
    def show_item(self, dataset_index: int):
        items = self.state.dataset.items
        if not (0 <= dataset_index < len(items)):
            return
        self._selected_index = dataset_index
        item = items[dataset_index]
        self.open_btn.setEnabled(True)

        pix = QPixmap(item.path)
        if pix.isNull():
            self.preview.setText("Preview unavailable")
        else:
            self.preview.setPixmap(pix.scaled(
                self.preview.width() - 8, self.preview.height() - 8,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

        run = self.state.run
        score = None
        if run is not None and dataset_index < len(run.scores):
            s = run.scores[dataset_index]
            score = None if s is None or math.isnan(s) else s

        if score is None:
            self.score_label.setText("—")
            self.score_label.setStyleSheet(
                f"color: {T.TEXT_FAINT}; font-size: 20px; font-weight: 700;"
                " background: transparent;")
        else:
            verdict = "AI-generated" if score >= self.state.threshold else "authentic"
            self.score_label.setText(f"{score:.4f}  ·  {verdict}")
            self.score_label.setStyleSheet(
                f"color: {score_color(score)}; font-size: 20px; font-weight: 700;"
                " background: transparent;")

        dims = f"{pix.width()}×{pix.height()}" if not pix.isNull() else "?"
        label_txt = "—" if item.label is None else ("AI-generated" if item.label else "authentic")
        self.meta_label.setText(
            f"<b>{item.name}</b><br>"
            f"{item.rel_path}<br>"
            f"{dims} · {item.size_bytes / 1024:.0f} KB · "
            f"{os.path.splitext(item.name)[1].lstrip('.').upper()}<br>"
            f"ground truth: {label_txt}"
        )

    def _open_folder(self):
        items = self.state.dataset.items
        if 0 <= self._selected_index < len(items):
            folder = os.path.dirname(items[self._selected_index].path)
            os.startfile(folder) if hasattr(os, "startfile") else None

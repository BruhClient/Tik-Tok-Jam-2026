"""Robustness lab: re-run the test set under post-processing and chart the fall-off."""

from __future__ import annotations

import math
import os
import random

from PIL import Image, ImageQt
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
)

from .. import export as EX
from .. import metrics as M
from .. import theme as T
from ..transforms import TRANSFORMS, TRANSFORMS_BY_KEY
from ..workers import RobustnessWorker
from .charts import MplCanvas
from .components import Card, Hint, PlaceholderBanner, SectionTitle, StatCard

TABLE_COLUMNS = ["Transform", "Severity", "N", "Accuracy", "Δ Acc", "AUC", "Δ AUC", "FPR"]


class TransformRow(QWidget):
    """One transform with a master checkbox and five severity toggles."""

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self.spec = spec
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(3)

        levels_txt = " · ".join(spec.label_for(i) for i in range(1, 6))
        self.check = QCheckBox(spec.display_name)
        self.check.setToolTip(f"{spec.description}\n\nSeverities: {levels_txt}")
        self.check.toggled.connect(self._on_master)

        sev_row = QHBoxLayout()
        sev_row.setSpacing(2)
        sev_row.setContentsMargins(24, 0, 0, 0)
        self.levels = []
        for i in range(1, 6):
            cb = QCheckBox(str(i))
            cb.setChecked(True)
            cb.setToolTip(f"severity {i}: {spec.label_for(i)}")
            cb.setStyleSheet(f"color: {T.TEXT_FAINT}; font-size: 10px;")
            cb.toggled.connect(self._on_level)
            sev_row.addWidget(cb)
            self.levels.append(cb)
        short = levels_txt if len(levels_txt) <= 30 else levels_txt[:29] + "…"
        self.levels_label = QLabel(short)
        self.levels_label.setToolTip(levels_txt)
        self.levels_label.setStyleSheet(
            f"color: {T.TEXT_FAINT}; font-size: 9px; background: transparent;")
        self.levels_label.setMinimumWidth(0)
        sev_row.addSpacing(6)
        sev_row.addWidget(self.levels_label, 1)

        lay.addWidget(self.check)
        lay.addLayout(sev_row)
        self._set_levels_enabled(False)

    def _set_levels_enabled(self, on: bool):
        for cb in self.levels:
            cb.setEnabled(on)

    def _on_master(self, checked: bool):
        self._set_levels_enabled(checked)
        if checked and not any(cb.isChecked() for cb in self.levels):
            for cb in self.levels:
                cb.setChecked(True)

    def _on_level(self):
        if self.check.isChecked() and not any(cb.isChecked() for cb in self.levels):
            self.check.setChecked(False)

    def selected_cells(self) -> list:
        if not self.check.isChecked():
            return []
        return [(self.spec.key, i + 1) for i, cb in enumerate(self.levels) if cb.isChecked()]


class RobustnessTab(QWidget):
    status_message = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)

    def __init__(self, state, get_detector_fn, parent=None):
        super().__init__(parent)
        self.state = state
        self._get_detector = get_detector_fn
        self.worker: RobustnessWorker | None = None
        self._sample_paths: list = []
        self._sample_labels: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        self.banner = PlaceholderBanner()
        self.banner.set_text(
            "Placeholder detector — the degradation curve below is synthetic. "
            "The transforms and metrics are real; only the model is not.")
        root.addWidget(self.banner)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([340, 900])
        root.addWidget(splitter, 1)

        self.state.run_changed.connect(self._sync_enabled)
        self.state.dataset_changed.connect(self._sync_enabled)
        self.state.labels_changed.connect(self._sync_enabled)
        self.state.robustness_changed.connect(self.refresh_results)
        self._sync_enabled()

    # -- construction ------------------------------------------------------
    def _build_controls(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(330)
        holder = QWidget()
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(0, 0, 10, 0)
        lay.setSpacing(10)

        intro = Card()
        intro.layout().addWidget(SectionTitle("Robustness sweep"))
        intro.layout().addWidget(Hint(
            "Each selected transform is applied in memory at five severities and the "
            "whole set is re-scored. Everything is compared against a clean baseline "
            "measured with the same pipeline."))
        lay.addWidget(intro)

        group = QGroupBox("Transforms")
        gl = QVBoxLayout(group)
        gl.setSpacing(2)
        self.rows = []
        for spec in TRANSFORMS:
            row = TransformRow(spec)
            self.rows.append(row)
            gl.addWidget(row)
        for key in ("jpeg", "blur", "rescale"):          # sensible defaults
            for row in self.rows:
                if row.spec.key == key:
                    row.check.setChecked(True)
        lay.addWidget(group)

        opts = Card()
        opts.layout().addWidget(SectionTitle("Run options"))

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("Sample size"))
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(10, 100000)
        self.sample_spin.setValue(200)
        self.sample_spin.setSingleStep(50)
        self.sample_spin.setToolTip(
            "Images drawn from the labeled set for each cell. Smaller = faster sweeps.")
        sample_row.addWidget(self.sample_spin, 1)
        opts.layout().addLayout(sample_row)

        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("Max image side"))
        self.side_spin = QSpinBox()
        self.side_spin.setRange(128, 4096)
        self.side_spin.setValue(768)
        self.side_spin.setSingleStep(128)
        self.side_spin.setToolTip("Images are decoded and capped to this size before transforming.")
        side_row.addWidget(self.side_spin, 1)
        opts.layout().addLayout(side_row)

        self.stratify_check = QCheckBox("Balance real / AI in the sample")
        self.stratify_check.setChecked(True)
        opts.layout().addWidget(self.stratify_check)

        self.cell_label = Hint("")
        opts.layout().addWidget(self.cell_label)

        self.run_btn = QPushButton("▶  Run robustness sweep")
        self.run_btn.setProperty("accent", True)
        self.run_btn.clicked.connect(self.run_sweep)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_sweep)
        self.cancel_btn.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.run_btn, 2)
        btn_row.addWidget(self.cancel_btn, 1)
        opts.layout().addLayout(btn_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        opts.layout().addWidget(self.progress)
        self.progress_label = Hint("")
        opts.layout().addWidget(self.progress_label)

        lay.addWidget(opts)
        lay.addStretch(1)
        scroll.setWidget(holder)
        return scroll

    def _build_results(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        cards = QHBoxLayout()
        cards.setSpacing(10)
        self.card_baseline = StatCard("Clean baseline", "accuracy on untouched images")
        self.card_worst = StatCard("Worst case", "hardest transform")
        self.card_avg_drop = StatCard("Mean accuracy drop", "across every cell run")
        self.card_cells = StatCard("Cells evaluated", "transform × severity")
        for c in (self.card_baseline, self.card_worst, self.card_avg_drop, self.card_cells):
            cards.addWidget(c)
        lay.addLayout(cards)

        chart_card = Card(padding=6)
        head = QHBoxLayout()
        head.addWidget(SectionTitle("Degradation curve"))
        head.addStretch(1)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["Accuracy", "ROC AUC", "F1", "False positive rate"])
        self.metric_combo.currentIndexChanged.connect(self.refresh_results)
        head.addWidget(self.metric_combo)
        chart_card.layout().addLayout(head)
        self.chart = MplCanvas(height=3.2)
        chart_card.layout().addWidget(self.chart)
        lay.addWidget(chart_card, 3)

        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(TABLE_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table, 2)

        lay.addWidget(self._build_preview_strip())

        export_row = QHBoxLayout()
        export_row.addStretch(1)
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_export_csv.clicked.connect(lambda: self.export_report("csv"))
        self.btn_export_json = QPushButton("⬇  Export robustness report")
        self.btn_export_json.setProperty("accent", True)
        self.btn_export_json.clicked.connect(lambda: self.export_report("json"))
        export_row.addWidget(self.btn_export_csv)
        export_row.addWidget(self.btn_export_json)
        lay.addLayout(export_row)
        return panel

    def _build_preview_strip(self) -> QWidget:
        card = Card(padding=10)
        head = QHBoxLayout()
        head.addWidget(SectionTitle("Transform preview"))
        head.addStretch(1)
        self.preview_sev = QComboBox()
        self.preview_sev.addItems([f"severity {i}" for i in range(1, 6)])
        self.preview_sev.setCurrentIndex(2)
        self.preview_sev.currentIndexChanged.connect(self.refresh_preview)
        self.preview_btn = QPushButton("Pick image…")
        self.preview_btn.clicked.connect(self.pick_preview_image)
        head.addWidget(self.preview_sev)
        head.addWidget(self.preview_btn)
        card.layout().addLayout(head)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setFixedHeight(150)
        self.preview_holder = QWidget()
        self.preview_layout = QHBoxLayout(self.preview_holder)
        self.preview_layout.setContentsMargins(2, 2, 2, 2)
        self.preview_layout.setSpacing(8)
        self.preview_scroll.setWidget(self.preview_holder)
        card.layout().addWidget(self.preview_scroll)
        self._preview_path = None
        return card

    # -- sampling / running ------------------------------------------------
    def _selected_cells(self) -> list:
        cells = []
        for row in self.rows:
            cells.extend(row.selected_cells())
        return cells

    def _build_sample(self):
        ds = self.state.dataset
        labeled = [i for i, it in enumerate(ds.items) if it.label is not None]
        if not labeled:
            return [], []
        n = min(self.sample_spin.value(), len(labeled))
        rng = random.Random(20260829)

        if self.stratify_check.isChecked():
            reals = [i for i in labeled if ds.items[i].label == 0]
            ais = [i for i in labeled if ds.items[i].label == 1]
            half = max(1, n // 2)
            rng.shuffle(reals)
            rng.shuffle(ais)
            picked = reals[:half] + ais[:half]
            if len(picked) < n:
                rest = [i for i in labeled if i not in set(picked)]
                rng.shuffle(rest)
                picked += rest[: n - len(picked)]
        else:
            picked = labeled[:]
            rng.shuffle(picked)
            picked = picked[:n]

        picked.sort()
        return ([ds.items[i].path for i in picked],
                [ds.items[i].label for i in picked])

    def run_sweep(self):
        if self.worker is not None:
            return
        ds = self.state.dataset
        if not ds.items:
            QMessageBox.information(self, "No dataset", "Load an image directory first.")
            return
        if not ds.has_labels:
            QMessageBox.information(
                self, "Labels required",
                "The robustness sweep measures accuracy, so it needs ground-truth labels.\n\n"
                "Use real/ and ai/ subfolders, a labels.csv, or real_/ai_ filename prefixes.")
            return

        cells = self._selected_cells()
        if not cells:
            QMessageBox.information(self, "Nothing selected",
                                    "Tick at least one transform to sweep.")
            return

        try:
            detector = self._get_detector()
        except Exception as exc:
            QMessageBox.critical(self, "Detector error", str(exc))
            return

        paths, labels = self._build_sample()
        if not paths:
            QMessageBox.information(self, "No labeled images", "Nothing to evaluate.")
            return
        self._sample_paths, self._sample_labels = paths, labels

        self.state.clear_robustness()
        cells = [("clean", 0)] + cells

        self.worker = RobustnessWorker(detector, paths, labels, cells,
                                       max_side=self.side_spin.value(), parent=self)
        self.worker.progress.connect(self._on_progress)
        self.worker.cell_done.connect(self._on_cell_done)
        self.worker.error.connect(lambda msg: self.status_message.emit("Sweep error: " + msg))
        self.worker.done.connect(self._on_done)

        self.progress.setVisible(True)
        self.progress.setRange(0, len(cells))
        self.progress.setValue(0)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.busy_changed.emit(True)
        self.status_message.emit(
            f"Sweeping {len(cells)} cells × {len(paths)} images "
            f"({len(cells) * len(paths):,} scored images)…")
        self.worker.start()

    def cancel_sweep(self):
        if self.worker is not None:
            self.worker.cancel()
            self.status_message.emit("Cancelling sweep…")

    def _on_progress(self, done: int, total: int, text: str):
        self.progress.setValue(done)
        self.progress_label.setText(f"{done}/{total} · {text}")

    def _on_cell_done(self, key: str, severity: int, labels: list, scores: list):
        m = M.compute_metrics(labels, scores, self.state.threshold)
        if key == "clean" and severity == 0:
            self.state.set_baseline(m)
        else:
            self.state.set_cell(key, severity, m)

    def _on_done(self, cancelled: bool, elapsed: float):
        self.worker = None
        self.progress.setVisible(False)
        self.progress_label.setText("")
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.busy_changed.emit(False)
        self.status_message.emit(
            ("Sweep cancelled after " if cancelled else "Sweep finished in ")
            + f"{elapsed:.1f}s")
        self.refresh_results()
        self.refresh_preview()

    # -- results -----------------------------------------------------------
    def _metric_getter(self):
        idx = self.metric_combo.currentIndex()
        return [
            ("Accuracy", lambda m: m.accuracy),
            ("ROC AUC", lambda m: m.auc),
            ("F1", lambda m: m.f1),
            ("False positive rate", lambda m: m.fpr),
        ][idx]

    def refresh_results(self):
        state = self.state
        name, getter = self._metric_getter()
        baseline = state.robustness_baseline

        series = {}
        for (key, severity), cell in state.robustness.items():
            spec = TRANSFORMS_BY_KEY.get(key)
            label = spec.display_name if spec else key
            value = getter(cell.metrics)
            if value == value:
                series.setdefault(label, {})[severity] = value

        base_value = getter(baseline) if baseline else float("nan")
        self.chart.plot_degradation(series, base_value, name)
        self._fill_table()
        self._update_cards()

    def _fill_table(self):
        state = self.state
        baseline = state.robustness_baseline
        cells = sorted(state.robustness.items(), key=lambda kv: (kv[0][0], kv[0][1]))

        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(cells))
        for r, ((key, severity), cell) in enumerate(cells):
            spec = TRANSFORMS_BY_KEY.get(key)
            m = cell.metrics
            d_acc = (m.accuracy - baseline.accuracy) if (baseline and baseline.valid) else float("nan")
            d_auc = (m.auc - baseline.auc) if (baseline and baseline.valid) else float("nan")
            values = [
                spec.display_name if spec else key,
                spec.label_for(severity) if spec else str(severity),
                str(m.n),
                M.fmt(m.accuracy),
                self._delta_text(d_acc),
                M.fmt(m.auc, pct=False),
                self._delta_text(d_auc, pct=False),
                M.fmt(m.fpr),
            ]
            for c, text in enumerate(values):
                cell_item = QTableWidgetItem(text)
                cell_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                if c in (4, 6):
                    delta = d_acc if c == 4 else d_auc
                    cell_item.setForeground(self._delta_brush(delta))
                if c >= 2:
                    cell_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r, c, cell_item)
        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()

    @staticmethod
    def _delta_text(value: float, pct: bool = True) -> str:
        if value is None or value != value:
            return "—"
        sign = "+" if value >= 0 else "−"
        v = abs(value)
        return f"{sign}{v * 100:.1f}pp" if pct else f"{sign}{v:.3f}"

    @staticmethod
    def _delta_brush(value: float):
        from PyQt6.QtGui import QBrush, QColor
        if value is None or value != value:
            return QBrush(QColor(T.TEXT_FAINT))
        if value >= -0.02:
            return QBrush(QColor(T.GOOD))
        if value >= -0.10:
            return QBrush(QColor(T.WARN))
        return QBrush(QColor(T.BAD))

    def _update_cards(self):
        state = self.state
        baseline = state.robustness_baseline
        cells = list(state.robustness.values())

        if baseline and baseline.valid:
            self.card_baseline.set_value(M.fmt(baseline.accuracy),
                                         f"AUC {M.fmt(baseline.auc, pct=False)} · n={baseline.n}")
        else:
            self.card_baseline.set_value("—", "run a sweep")

        self.card_cells.set_value(str(len(cells)), "transform × severity")

        if not cells or not (baseline and baseline.valid):
            self.card_worst.set_value("—", "run a sweep")
            self.card_avg_drop.set_value("—", "run a sweep")
            return

        drops = [(c.metrics.accuracy - baseline.accuracy, c) for c in cells
                 if c.metrics.accuracy == c.metrics.accuracy]
        if not drops:
            return
        drops.sort(key=lambda t: t[0])
        worst_delta, worst = drops[0]
        spec = TRANSFORMS_BY_KEY.get(worst.transform_key)
        self.card_worst.set_value(
            M.fmt(worst.metrics.accuracy),
            f"{spec.display_name if spec else worst.transform_key} · "
            f"{spec.label_for(worst.severity) if spec else worst.severity} "
            f"({self._delta_text(worst_delta)})",
            color=T.BAD if worst_delta < -0.10 else T.WARN if worst_delta < -0.02 else T.GOOD)

        mean_drop = sum(d for d, _ in drops) / len(drops)
        self.card_avg_drop.set_value(
            self._delta_text(mean_drop), f"over {len(drops)} cells",
            color=T.BAD if mean_drop < -0.10 else T.WARN if mean_drop < -0.02 else T.GOOD)

    # -- preview -----------------------------------------------------------
    def pick_preview_image(self):
        start = self.state.dataset.root or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Preview image", start,
            "Images (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff)")
        if path:
            self._preview_path = path
            self.refresh_preview()

    def refresh_preview(self):
        while self.preview_layout.count():
            child = self.preview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        path = self._preview_path
        if not path and self.state.dataset.items:
            path = self.state.dataset.items[0].path
        if not path or not os.path.isfile(path):
            self.preview_layout.addWidget(Hint("Load a dataset or pick an image to preview."))
            return

        severity = self.preview_sev.currentIndex() + 1
        keys = ["clean"] + [k for row in self.rows for (k, _s) in row.selected_cells()[:1]]
        try:
            base = Image.open(path)
            base.draft("RGB", (512, 512))
            base = base.convert("RGB")
            base.thumbnail((320, 320), Image.BILINEAR)
        except Exception as exc:
            self.preview_layout.addWidget(Hint(f"Preview failed: {exc}"))
            return

        for key in keys:
            spec = TRANSFORMS_BY_KEY.get(key)
            if spec is None:
                continue
            try:
                img = base if key == "clean" else spec.apply(base, severity)
                self.preview_layout.addWidget(self._preview_tile(img, spec, severity, key))
            except Exception as exc:
                print(f"[preview] {key}: {exc}")
        self.preview_layout.addStretch(1)

    def _preview_tile(self, img: Image.Image, spec, severity: int, key: str) -> QWidget:
        box = QFrame()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)

        thumb = img.copy()
        thumb.thumbnail((108, 108), Image.BILINEAR)
        pix = QPixmap.fromImage(ImageQt.ImageQt(thumb)).copy()

        pic = QLabel()
        pic.setPixmap(pix)
        pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pic.setStyleSheet(f"border: 1px solid {T.BORDER}; border-radius: 6px; padding: 2px;")

        caption = QLabel(spec.display_name if key != "clean" else "Clean")
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 9px; background: transparent;")

        sub = QLabel("original" if key == "clean" else spec.label_for(severity))
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(f"color: {T.TEXT_FAINT}; font-size: 9px; background: transparent;")

        lay.addWidget(pic)
        lay.addWidget(caption)
        lay.addWidget(sub)
        return box

    # -- misc --------------------------------------------------------------
    def _sync_enabled(self):
        ds = self.state.dataset
        run = self.state.run
        self.banner.setVisible(run is not None and run.is_placeholder)
        ready = bool(ds.items) and ds.has_labels
        self.run_btn.setEnabled(ready and self.worker is None)
        if not ds.items:
            self.cell_label.setText("Load a dataset to enable the sweep.")
        elif not ds.has_labels:
            self.cell_label.setText("This dataset has no labels — accuracy cannot be measured.")
        else:
            n_labeled = len(ds.labeled_indices())
            self.cell_label.setText(f"{n_labeled:,} labeled images available.")

    def export_report(self, fmt: str):
        if not self.state.robustness:
            QMessageBox.information(self, "Nothing to export", "Run a sweep first.")
            return
        default = os.path.join(self.state.dataset.root or os.path.expanduser("~"),
                               f"robustness_report.{fmt}")
        path, _ = QFileDialog.getSaveFileName(self, "Export robustness report", default,
                                              f"{fmt.upper()} (*.{fmt})")
        if not path:
            return
        try:
            if fmt == "json":
                EX.export_robustness_json(path, self.state.dataset, self.state.run, self.state)
            else:
                EX.export_robustness_csv(path, self.state.dataset, self.state.run, self.state)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.status_message.emit(f"Wrote robustness report to {path}")

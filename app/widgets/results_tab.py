"""Results tab: per-image scores, live threshold, metrics, charts, exports."""

from __future__ import annotations

import math
import os

from PyQt6.QtCore import (
    QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QTimer, Qt, pyqtSignal
)
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox, QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QMessageBox, QPushButton, QScrollArea, QSlider, QSplitter, QStyledItemDelegate,
    QTableView, QVBoxLayout, QWidget
)

from .. import export as EX
from .. import metrics as M
from .. import theme as T
from .charts import MplCanvas
from .components import (
    Card, Chip, EmptyState, Hint, PlaceholderBanner, SectionTitle, StatCard, score_color
)

FILTER_ALL, FILTER_CORRECT, FILTER_FP, FILTER_FN, FILTER_UNLABELED = range(5)

COLUMNS = ["#", "Image", "Score", "Ground truth", "Predicted", "Result"]


class ResultsModel(QAbstractTableModel):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.rows: list = []

    def set_rows(self, rows: list):
        self.beginResetModel()
        self.rows = list(rows)
        self.endResetModel()

    def refresh(self):
        if self.rows:
            self.dataChanged.emit(self.index(0, 0),
                                  self.index(len(self.rows) - 1, len(COLUMNS) - 1))

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section]
        return None

    def _score(self, di: int):
        run = self.state.run
        if run is None or di >= len(run.scores):
            return None
        s = run.scores[di]
        return None if s is None or math.isnan(s) else s

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        di = self.rows[index.row()]
        item = self.state.dataset.items[di]
        score = self._score(di)
        col = index.column()
        thr = self.state.threshold
        pred = None if score is None else int(score >= thr)

        if role == Qt.ItemDataRole.UserRole:            # sort key
            return [di, item.rel_path, -1.0 if score is None else score,
                    -1 if item.label is None else item.label,
                    -1 if pred is None else pred,
                    self._result_text(item.label, pred)][col]

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return di + 1
            if col == 1:
                return item.rel_path
            if col == 2:
                return "—" if score is None else f"{score:.4f}"
            if col == 3:
                return "—" if item.label is None else ("AI" if item.label else "Real")
            if col == 4:
                return "—" if pred is None else ("AI" if pred else "Real")
            if col == 5:
                return self._result_text(item.label, pred)

        if role == Qt.ItemDataRole.ToolTipRole:
            return item.path

        if role == Qt.ItemDataRole.ForegroundRole:
            if col == 2 and score is not None:
                return QBrush(QColor(score_color(score)))
            if col == 3 and item.label is not None:
                return QBrush(QColor(T.AI_COLOR if item.label else T.REAL_COLOR))
            if col == 4 and pred is not None:
                return QBrush(QColor(T.AI_COLOR if pred else T.REAL_COLOR))
            if col == 5:
                txt = self._result_text(item.label, pred)
                if txt.startswith("✓"):
                    return QBrush(QColor(T.GOOD))
                if txt.startswith("✗"):
                    return QBrush(QColor(T.BAD))
                return QBrush(QColor(T.TEXT_FAINT))

        if role == Qt.ItemDataRole.TextAlignmentRole and col in (0, 2, 3, 4, 5):
            return int(Qt.AlignmentFlag.AlignCenter)

        if role == Qt.ItemDataRole.UserRole + 1:
            return score
        return None

    @staticmethod
    def _result_text(label, pred) -> str:
        if label is None or pred is None:
            return "—"
        if label == pred:
            return "✓ correct"
        return "✗ false positive" if pred == 1 else "✗ false negative"


class ScoreBarDelegate(QStyledItemDelegate):
    """Draws the confidence as a bar behind the number."""

    def paint(self, painter: QPainter, option, index):
        score = index.data(Qt.ItemDataRole.UserRole + 1)
        if score is None:
            return super().paint(painter, option, index)
        painter.save()
        r = option.rect.adjusted(6, 7, -6, -7)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#25252C")))
        painter.drawRoundedRect(r, 4, 4)
        c = QColor(score_color(score))
        c.setAlpha(150)
        painter.setBrush(QBrush(c))
        fill = r.adjusted(0, 0, -int(r.width() * (1.0 - float(score))), 0)
        painter.drawRoundedRect(fill, 4, 4)
        painter.setPen(QPen(QColor(T.TEXT)))
        f = QFont(painter.font())
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, f"{score:.4f}")
        painter.restore()


class ResultsTab(QWidget):
    status_message = pyqtSignal(str)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._filter = FILTER_ALL

        self._chart_timer = QTimer(self)
        self._chart_timer.setSingleShot(True)
        self._chart_timer.timeout.connect(
            lambda: self.refresh(charts=("hist", "cm", "sweep")))

        self._live_timer = QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.timeout.connect(self._live_tick)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        self.banner = PlaceholderBanner()
        root.addWidget(self.banner)

        root.addLayout(self._build_cards())
        root.addWidget(self._build_threshold_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_table_panel())
        splitter.addWidget(self._build_charts_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setSizes([620, 700])
        root.addWidget(splitter, 1)

        self.state.run_changed.connect(self.refresh)
        self.state.scores_updated.connect(self._light_refresh)
        self.state.dataset_changed.connect(self.refresh)
        self.state.labels_changed.connect(self.refresh)
        self.state.threshold_changed.connect(self._on_threshold_changed)

        self.refresh()

    # -- construction ------------------------------------------------------
    def _build_cards(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self.card_acc = StatCard("Accuracy", "at current threshold")
        self.card_auc = StatCard("ROC AUC", "threshold-independent")
        self.card_f1 = StatCard("F1 (AI class)", "precision · recall")
        self.card_fpr = StatCard("False positive rate", "authentic flagged as AI")
        self.card_scored = StatCard("Scored", "images with a prediction")
        for c in (self.card_acc, self.card_auc, self.card_f1, self.card_fpr, self.card_scored):
            row.addWidget(c)
        return row

    def _build_threshold_bar(self) -> QWidget:
        card = Card(padding=12)
        lay = QHBoxLayout()
        lay.setSpacing(10)

        title = SectionTitle("Decision threshold")
        self.thr_value = QLabel("0.50")
        self.thr_value.setStyleSheet(
            f"color: {T.ACCENT}; font-size: 16px; font-weight: 700; background: transparent;")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(500)
        self.slider.valueChanged.connect(
            lambda v: self.state.set_threshold(v / 1000.0))

        self.btn_f1 = QPushButton("Best F1")
        self.btn_f1.clicked.connect(lambda: self._auto_threshold("f1"))
        self.btn_youden = QPushButton("Youden J")
        self.btn_youden.clicked.connect(lambda: self._auto_threshold("youden"))
        self.btn_half = QPushButton("Reset 0.50")
        self.btn_half.clicked.connect(lambda: self.state.set_threshold(0.5))

        lay.addWidget(title)
        lay.addWidget(self.thr_value)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.btn_f1)
        lay.addWidget(self.btn_youden)
        lay.addWidget(self.btn_half)
        card.layout().addLayout(lay)
        return card

    def _build_table_panel(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(8)

        chip_row = QHBoxLayout()
        chip_row.setSpacing(6)
        self.chips = []
        for i, text in enumerate(["All", "Correct", "False positives",
                                  "False negatives", "Unlabeled"]):
            chip = Chip(text)
            chip.clicked.connect(lambda _, idx=i: self._set_filter(idx))
            chip_row.addWidget(chip)
            self.chips.append(chip)
        self.chips[0].setChecked(True)
        chip_row.addStretch(1)
        lay.addLayout(chip_row)

        self.model = ResultsModel(self.state)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.ItemDataRole.UserRole)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setItemDelegateForColumn(2, ScoreBarDelegate(self.table))
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for col in (0, 2, 3, 4, 5):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        lay.addWidget(self.table, 1)

        self.table_empty = EmptyState(
            "No predictions yet",
            "Load a directory on the Dataset tab, pick a detector in the toolbar, "
            "then hit Run detection.")
        lay.addWidget(self.table_empty, 1)

        lay.addLayout(self._build_export_row())
        return panel

    def _build_export_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        self.relative_check = QCheckBox("relative paths")
        self.relative_check.setToolTip("Write paths relative to the dataset root")

        self.btn_json = QPushButton("⬇  Export predictions.json")
        self.btn_json.setProperty("accent", True)
        self.btn_json.clicked.connect(self.export_json)
        self.btn_csv = QPushButton("Export CSV")
        self.btn_csv.clicked.connect(self.export_csv)
        self.btn_report = QPushButton("Export run report")
        self.btn_report.clicked.connect(self.export_report)

        row.addWidget(self.relative_check)
        row.addStretch(1)
        row.addWidget(self.btn_report)
        row.addWidget(self.btn_csv)
        row.addWidget(self.btn_json)
        return row

    def _build_charts_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        grid = QGridLayout(holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)

        self.chart_hist = MplCanvas()
        self.chart_roc = MplCanvas()
        self.chart_pr = MplCanvas()
        self.chart_cm = MplCanvas()
        self.chart_sweep = MplCanvas()

        for canvas, (r, c, rs, cs) in zip(
            [self.chart_hist, self.chart_roc, self.chart_pr, self.chart_cm, self.chart_sweep],
            [(0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1), (2, 0, 1, 2)],
        ):
            wrapper = Card(padding=6)
            wrapper.layout().addWidget(canvas)
            grid.addWidget(wrapper, r, c, rs, cs)

        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 1)
        scroll.setWidget(holder)
        return scroll

    # -- interaction -------------------------------------------------------
    def _set_filter(self, idx: int):
        self._filter = idx
        for i, chip in enumerate(self.chips):
            chip.setChecked(i == idx)
        self._rebuild_rows()

    def _auto_threshold(self, criterion: str):
        run = self.state.run
        if run is None:
            return
        y, s = run.valid_pairs(self.state.dataset)
        if len(set(y)) < 2:
            self.status_message.emit("Need both classes labeled to optimise the threshold.")
            return
        t = M.best_threshold(y, s, criterion)
        self.state.set_threshold(t)
        self.status_message.emit(f"Threshold set to {t:.3f} ({criterion}).")

    def _on_threshold_changed(self, t: float):
        self.thr_value.setText(f"{t:.2f}")
        target = int(round(t * 1000))
        if self.slider.value() != target:
            self.slider.blockSignals(True)
            self.slider.setValue(target)
            self.slider.blockSignals(False)
        # table + numbers track the slider immediately; charts are debounced so
        # dragging stays smooth on large sets
        self._rebuild_rows()
        self._update_cards()
        self._chart_timer.start(140)

    # -- refresh -----------------------------------------------------------
    def _light_refresh(self):
        """Called on every batch during a run; coalesced so big sets stay smooth."""
        if not self._live_timer.isActive():
            self._live_timer.start(250)

    def _live_tick(self):
        self._rebuild_rows()
        self._update_cards()

    def _rebuild_rows(self):
        state = self.state
        run = state.run
        thr = state.threshold
        rows = []
        for di, item in enumerate(state.dataset.items):
            score = None
            if run is not None and di < len(run.scores):
                s = run.scores[di]
                score = None if s is None or math.isnan(s) else s
            pred = None if score is None else int(score >= thr)

            if self._filter == FILTER_ALL:
                keep = True
            elif self._filter == FILTER_UNLABELED:
                keep = item.label is None
            elif item.label is None or pred is None:
                keep = False
            elif self._filter == FILTER_CORRECT:
                keep = pred == item.label
            elif self._filter == FILTER_FP:
                keep = pred == 1 and item.label == 0
            else:
                keep = pred == 0 and item.label == 1
            if keep:
                rows.append(di)
        self.model.set_rows(rows)

        has_rows = bool(rows) or bool(state.dataset.items)
        self.table.setVisible(has_rows)
        self.table_empty.setVisible(not has_rows)
        if not rows and state.dataset.items:
            self.table_empty.set_text("Nothing matches this filter",
                                      "Try a different chip or move the threshold.")
            self.table.setVisible(False)
            self.table_empty.setVisible(True)

    def _update_cards(self):
        state = self.state
        run = state.run
        m = state.current_metrics()

        if run is None:
            for card in (self.card_acc, self.card_auc, self.card_f1,
                         self.card_fpr, self.card_scored):
                card.set_value("—")
            self.card_scored.set_value("0", f"of {len(state.dataset)} images")
            return

        labeled = state.dataset.has_labels
        if labeled and m.valid:
            self.card_acc.set_value(M.fmt(m.accuracy),
                                    f"{m.tp + m.tn} of {m.n} correct")
            self.card_auc.set_value(M.fmt(m.auc, pct=False), "1.0 = perfect ranking")
            self.card_f1.set_value(M.fmt(m.f1, pct=False),
                                   f"P {M.fmt(m.precision)} · R {M.fmt(m.recall)}")
            self.card_fpr.set_value(M.fmt(m.fpr),
                                    f"{m.fp} of {m.tn + m.fp} authentic images",
                                    color=T.BAD if (m.fpr == m.fpr and m.fpr > 0.1) else T.TEXT)
        else:
            for card, sub in ((self.card_acc, "no labels"), (self.card_auc, "no labels"),
                              (self.card_f1, "no labels"), (self.card_fpr, "no labels")):
                card.set_value("—", sub)

        n_ai = sum(1 for i, s in enumerate(run.scores)
                   if not math.isnan(s) and s >= state.threshold)
        self.card_scored.set_value(f"{run.n_scored:,}",
                                   f"{n_ai:,} flagged AI at {state.threshold:.2f}")

    def refresh(self, charts=("hist", "roc", "pr", "cm", "sweep")):
        state = self.state
        run = state.run

        placeholder = run.is_placeholder if run else True
        self.banner.setVisible(bool(run) and placeholder)
        if run and placeholder:
            self.banner.set_text(
                f"“{run.detector_display}” is a placeholder backend — these numbers "
                "exercise the pipeline, they are not real detection results.")

        self._rebuild_rows()
        self._update_cards()

        for btn in (self.btn_json, self.btn_csv, self.btn_report):
            btn.setEnabled(run is not None and run.n_scored > 0)

        if run is None or run.n_scored == 0:
            for canvas in (self.chart_hist, self.chart_roc, self.chart_pr,
                           self.chart_cm, self.chart_sweep):
                canvas.clear_to_message("Run a detector to populate this chart")
            return

        y, s = run.valid_pairs(state.dataset)
        unlabeled = [sc for it, sc in zip(state.dataset.items, run.scores)
                     if it.label is None and not math.isnan(sc)]
        m = state.current_metrics()

        if "hist" in charts:
            self.chart_hist.plot_score_histogram(y, s, state.threshold, unlabeled)
        if "roc" in charts:
            self.chart_roc.plot_roc(y, s)
        if "pr" in charts:
            self.chart_pr.plot_pr(y, s)
        if "cm" in charts:
            self.chart_cm.plot_confusion(m)
        if "sweep" in charts:
            self.chart_sweep.plot_threshold_sweep(y, s, state.threshold)

    # -- exports -----------------------------------------------------------
    def _default_dir(self) -> str:
        return self.state.dataset.root or os.path.expanduser("~")

    def export_json(self):
        run = self.state.run
        if run is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export predictions", os.path.join(self._default_dir(), "predictions.json"),
            "JSON (*.json)")
        if not path:
            return
        try:
            n = EX.export_predictions_json(path, self.state.dataset, run,
                                           relative=self.relative_check.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.status_message.emit(f"Wrote {n} predictions to {path}")
        QMessageBox.information(
            self, "Export complete",
            f"Wrote {n} records to:\n{path}\n\n"
            'Format: [{"image_path": ..., "pred": 0.0-1.0}, ...]')

    def export_csv(self):
        run = self.state.run
        if run is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", os.path.join(self._default_dir(), "predictions.csv"),
            "CSV (*.csv)")
        if not path:
            return
        try:
            n = EX.export_predictions_csv(path, self.state.dataset, run,
                                          self.state.threshold,
                                          relative=self.relative_check.isChecked())
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.status_message.emit(f"Wrote {n} rows to {path}")

    def export_report(self):
        run = self.state.run
        if run is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export run report", os.path.join(self._default_dir(), "run_report.json"),
            "JSON (*.json)")
        if not path:
            return
        try:
            EX.export_run_report(path, self.state.dataset, run, self.state.threshold)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self.status_message.emit(f"Wrote run report to {path}")

"""The viewer: one window, one finished result.

No background work happens here. Scanning, model loading and scoring are done
by app/runner.py before this window is constructed, and they log to the
terminal. This just draws what came out.
"""

from __future__ import annotations

import math
import os

from PyQt6.QtCore import (
    QAbstractTableModel, QModelIndex, QSortFilterProxyModel, QTimer, Qt
)
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QMainWindow,
    QMessageBox, QPushButton, QSlider, QSplitter, QStyledItemDelegate,
    QTableView, QVBoxLayout, QWidget
)

from .. import export as EX
from .. import metrics as M
from .. import theme as T
from .charts import MplCanvas
from .components import Badge, Card, Chip, SectionTitle, StatCard, score_color

FILTER_ALL, FILTER_FP, FILTER_FN, FILTER_AI, FILTER_REAL = range(5)
COLUMNS = ["Image", "Score", "Truth", "Pred", "Result"]


class ResultsModel(QAbstractTableModel):
    def __init__(self, view, parent=None):
        super().__init__(parent)
        self.view = view          # ResultsWindow, for dataset/scores/threshold
        self.rows: list = []

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = list(rows)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return COLUMNS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        di = self.rows[index.row()]
        item = self.view.dataset.items[di]
        score = self.view.score_at(di)
        pred = None if score is None else int(score >= self.view.threshold)
        col = index.column()

        if role == Qt.ItemDataRole.UserRole:                    # sort key
            return [item.rel_path, -1.0 if score is None else score,
                    -1 if item.label is None else item.label,
                    -1 if pred is None else pred,
                    _result_text(item.label, pred)][col]

        if role == Qt.ItemDataRole.DisplayRole:
            return [item.rel_path,
                    "—" if score is None else f"{score:.4f}",
                    _label_text(item.label),
                    _label_text(pred),
                    _result_text(item.label, pred)][col]

        if role == Qt.ItemDataRole.ToolTipRole:
            return item.path

        if role == Qt.ItemDataRole.ForegroundRole:
            if col == 2 and item.label is not None:
                return QBrush(QColor(T.AI_COLOR if item.label else T.REAL_COLOR))
            if col == 3 and pred is not None:
                return QBrush(QColor(T.AI_COLOR if pred else T.REAL_COLOR))
            if col == 4:
                txt = _result_text(item.label, pred)
                if txt.startswith("✓"):
                    return QBrush(QColor(T.GOOD))
                if txt.startswith("✗"):
                    return QBrush(QColor(T.BAD))
                return QBrush(QColor(T.TEXT_FAINT))

        if role == Qt.ItemDataRole.TextAlignmentRole and col > 0:
            return int(Qt.AlignmentFlag.AlignCenter)

        if role == Qt.ItemDataRole.UserRole + 1:
            return score
        if role == Qt.ItemDataRole.UserRole + 2:
            return di
        return None


def _label_text(label) -> str:
    return "—" if label is None else ("AI" if label == 1 else "Real")


def _result_text(label, pred) -> str:
    if label is None or pred is None:
        return "—"
    if label == pred:
        return "✓"
    return "✗ FP" if pred == 1 else "✗ FN"


class ScoreBarDelegate(QStyledItemDelegate):
    """Confidence drawn as a bar behind the number."""

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
        painter.drawRoundedRect(
            r.adjusted(0, 0, -int(r.width() * (1.0 - float(score))), 0), 4, 4)
        painter.setPen(QPen(QColor(T.TEXT)))
        f = QFont(painter.font())
        f.setBold(True)
        painter.setFont(f)
        painter.drawText(option.rect, Qt.AlignmentFlag.AlignCenter, f"{score:.4f}")
        painter.restore()


class ResultsWindow(QMainWindow):
    """Everything the app shows, in one window."""

    def __init__(self, dataset, result, robustness: dict = None, threshold: float = 0.5):
        super().__init__()
        self.dataset = dataset
        self.result = result
        self.robustness = robustness or {}
        self.threshold = threshold
        self._filter = FILTER_ALL
        self._selected = -1

        self.setWindowTitle(f"AIGC Detector — {os.path.basename(dataset.root) or 'results'}")
        self.resize(1440, 900)
        self.setMinimumSize(1000, 660)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(12)

        root.addWidget(self._build_header())
        root.addLayout(self._build_cards())
        root.addWidget(self._build_threshold())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left, right = self._build_left(), self._build_right()
        # the charts need real estate or matplotlib collapses their axes
        left.setMinimumWidth(420)
        right.setMinimumWidth(560)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([560, 800])
        root.addWidget(splitter, 1)

        self.setCentralWidget(central)

        self._chart_timer = QTimer(self)
        self._chart_timer.setSingleShot(True)
        self._chart_timer.timeout.connect(self._draw_charts)

        self.refresh(charts=True)

    # -- construction ------------------------------------------------------
    def _build_header(self) -> QWidget:
        card = Card(padding=12)
        row = QHBoxLayout()
        row.setSpacing(10)

        title = QLabel(self.dataset.root or self.result.source)
        title.setStyleSheet(
            f"color: {T.TEXT}; font-size: 14px; font-weight: 700; background: transparent;")
        title.setToolTip(self.result.source)

        sub = QLabel(f"{len(self.dataset):,} images  ·  {self.result.detector_display}"
                     + (f"  ·  scored in {self.result.elapsed:.1f}s"
                        if self.result.elapsed else ""))
        sub.setStyleSheet(f"color: {T.TEXT_DIM}; font-size: 11px; background: transparent;")

        left = QVBoxLayout()
        left.setSpacing(2)
        left.addWidget(title)
        left.addWidget(sub)

        row.addLayout(left, 1)

        if self.result.is_placeholder:
            row.addWidget(Badge("placeholder backend", T.WARN))
        if self.dataset.has_labels:
            row.addWidget(Badge(f"{self.dataset.n_real} real / {self.dataset.n_ai} AI",
                                T.GOOD))
        else:
            row.addWidget(Badge("unlabeled", T.TEXT_DIM))

        self.export_btn = QPushButton("⬇  Export predictions.json")
        self.export_btn.setProperty("accent", True)
        self.export_btn.clicked.connect(self.export_json)
        row.addWidget(self.export_btn)

        card.layout().addLayout(row)
        return card

    def _build_cards(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        self.card_acc = StatCard("Accuracy", "at current threshold")
        self.card_auc = StatCard("ROC AUC", "threshold-independent")
        self.card_f1 = StatCard("F1", "AI class")
        self.card_fpr = StatCard("False positives", "authentic flagged as AI")
        for c in (self.card_acc, self.card_auc, self.card_f1, self.card_fpr):
            row.addWidget(c)
        return row

    def _build_threshold(self) -> QWidget:
        card = Card(padding=10)
        row = QHBoxLayout()
        row.setSpacing(10)

        row.addWidget(SectionTitle("Threshold"))
        self.thr_label = QLabel(f"{self.threshold:.2f}")
        self.thr_label.setStyleSheet(
            f"color: {T.ACCENT}; font-size: 15px; font-weight: 700; background: transparent;")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(int(self.threshold * 1000))
        self.slider.valueChanged.connect(self._on_slider)

        best = QPushButton("Best F1")
        best.clicked.connect(lambda: self._set_threshold(self._best("f1")))
        half = QPushButton("0.50")
        half.clicked.connect(lambda: self._set_threshold(0.5))
        enabled = self.dataset.has_labels
        best.setEnabled(enabled)

        row.addWidget(self.thr_label)
        row.addWidget(self.slider, 1)
        row.addWidget(best)
        row.addWidget(half)
        card.layout().addLayout(row)
        return card

    def _build_left(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(8)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        self.chips = []
        for i, text in enumerate(["All", "False positives", "False negatives",
                                  "Flagged AI", "Flagged real"]):
            chip = Chip(text)
            chip.clicked.connect(lambda _, idx=i: self._set_filter(idx))
            chips.addWidget(chip)
            self.chips.append(chip)
        self.chips[0].setChecked(True)
        chips.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setStyleSheet(f"color: {T.TEXT_FAINT}; background: transparent;")
        chips.addWidget(self.count_label)
        lay.addLayout(chips)

        self.model = ResultsModel(self)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.ItemDataRole.UserRole)

        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setItemDelegateForColumn(1, ScoreBarDelegate(self.table))
        self.table.setWordWrap(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for c in range(1, len(COLUMNS)):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.ResizeToContents)
        self.table.clicked.connect(self._on_row_clicked)
        lay.addWidget(self.table, 1)

        lay.addWidget(self._build_preview())
        return panel

    def _build_preview(self) -> QWidget:
        card = Card(padding=10)
        card.setMaximumHeight(148)
        row = QHBoxLayout()
        row.setSpacing(12)

        self.preview = QLabel("Click a row to preview")
        self.preview.setFixedSize(126, 126)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet(
            f"background-color: #17171C; border: 1px solid {T.BORDER};"
            f" border-radius: 8px; color: {T.TEXT_FAINT}; font-size: 11px;")

        self.preview_score = QLabel("")
        self.preview_score.setStyleSheet(
            f"color: {T.TEXT_FAINT}; font-size: 19px; font-weight: 700;"
            " background: transparent;")
        self.preview_meta = QLabel("")
        self.preview_meta.setWordWrap(True)
        self.preview_meta.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_meta.setStyleSheet(
            f"color: {T.TEXT_DIM}; font-size: 11px; background: transparent;")

        info = QVBoxLayout()
        info.setSpacing(4)
        info.addWidget(self.preview_score)
        info.addWidget(self.preview_meta)
        info.addStretch(1)

        row.addWidget(self.preview)
        row.addLayout(info, 1)
        card.layout().addLayout(row)
        return card

    def _build_right(self) -> QWidget:
        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)

        self.chart_hist = MplCanvas()
        self.chart_roc = MplCanvas()
        self.chart_cm = MplCanvas()
        self.chart_robust = MplCanvas()

        for canvas, pos in ((self.chart_hist, (0, 0)), (self.chart_roc, (0, 1)),
                            (self.chart_cm, (1, 0)), (self.chart_robust, (1, 1))):
            wrapper = Card(padding=6)
            wrapper.layout().addWidget(canvas)
            grid.addWidget(wrapper, *pos)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        return panel

    # -- interaction -------------------------------------------------------
    def score_at(self, di: int):
        if di < 0 or di >= len(self.result.scores):
            return None
        s = self.result.scores[di]
        return None if s is None or math.isnan(s) else s

    def _on_slider(self, value: int):
        self.threshold = value / 1000.0
        self.thr_label.setText(f"{self.threshold:.2f}")
        self.refresh(charts=False)
        self._chart_timer.start(140)

    def _set_threshold(self, t: float):
        self.slider.setValue(int(round(t * 1000)))

    def _best(self, criterion: str) -> float:
        y, s = self.result.valid_pairs(self.dataset)
        if len(set(y)) < 2:
            return self.threshold
        return M.best_threshold(y, s, criterion)

    def _set_filter(self, idx: int):
        self._filter = idx
        for i, chip in enumerate(self.chips):
            chip.setChecked(i == idx)
        self.refresh(charts=False)

    def _on_row_clicked(self, index):
        src = self.proxy.mapToSource(index)
        di = self.model.data(self.model.index(src.row(), 0), Qt.ItemDataRole.UserRole + 2)
        if di is not None:
            self._show_preview(int(di))

    # -- refresh -----------------------------------------------------------
    def refresh(self, charts: bool = True):
        self._rebuild_rows()
        self._update_cards()
        if charts:
            self._draw_charts()
        if self._selected >= 0:
            self._show_preview(self._selected)

    def _rebuild_rows(self):
        rows = []
        for di, item in enumerate(self.dataset.items):
            score = self.score_at(di)
            pred = None if score is None else int(score >= self.threshold)
            f = self._filter
            if f == FILTER_ALL:
                keep = True
            elif f == FILTER_AI:
                keep = pred == 1
            elif f == FILTER_REAL:
                keep = pred == 0
            elif item.label is None or pred is None:
                keep = False
            elif f == FILTER_FP:
                keep = pred == 1 and item.label == 0
            else:
                keep = pred == 0 and item.label == 1
            if keep:
                rows.append(di)
        self.model.set_rows(rows)
        self.count_label.setText(f"{len(rows):,} of {len(self.dataset):,}")

    def _update_cards(self):
        if not self.dataset.has_labels:
            for card in (self.card_acc, self.card_auc, self.card_f1, self.card_fpr):
                card.set_value("—", "no ground-truth labels")
            flagged = sum(1 for di in range(len(self.dataset))
                          if (self.score_at(di) or 0) >= self.threshold)
            self.card_acc.set_value(f"{flagged:,}", "images flagged as AI")
            self.card_acc.title_label.setText("FLAGGED AI")
            return

        m = M.compute_metrics(*self.result.valid_pairs(self.dataset), self.threshold)
        self.card_acc.set_value(M.fmt(m.accuracy), f"{m.tp + m.tn} of {m.n} correct")
        self.card_auc.set_value(M.fmt(m.auc, pct=False), "1.0 = perfect ranking")
        self.card_f1.set_value(M.fmt(m.f1, pct=False),
                               f"P {M.fmt(m.precision)} · R {M.fmt(m.recall)}")
        self.card_fpr.set_value(
            M.fmt(m.fpr), f"{m.fp} of {m.tn + m.fp} authentic",
            color=T.BAD if (m.fpr == m.fpr and m.fpr > 0.10) else T.TEXT)

    def _draw_charts(self):
        y, s = self.result.valid_pairs(self.dataset)
        unlabeled = [sc for it, sc in zip(self.dataset.items, self.result.scores)
                     if it.label is None and not math.isnan(sc)]
        self.chart_hist.plot_score_histogram(y, s, self.threshold, unlabeled)
        self.chart_roc.plot_roc(y, s)
        self.chart_cm.plot_confusion(
            M.compute_metrics(y, s, self.threshold) if y else M.Metrics())
        self._draw_robustness()

    def _draw_robustness(self):
        if not self.robustness:
            self.chart_robust.clear_to_message(
                "No robustness report loaded\n\nrun:  python robustness.py <dir>")
            return
        series = self.robustness.get("series", {})
        baseline = self.robustness.get("baseline", float("nan"))
        metric = self.robustness.get("metric", "Accuracy")
        self.chart_robust.plot_degradation(series, baseline, metric)

    def _show_preview(self, di: int):
        self._selected = di
        item = self.dataset.items[di]
        score = self.score_at(di)

        pix = QPixmap(item.path)
        if pix.isNull():
            self.preview.setText("no preview")
            self.preview.setPixmap(QPixmap())
        else:
            self.preview.setPixmap(pix.scaled(
                self.preview.width() - 6, self.preview.height() - 6,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

        if score is None:
            self.preview_score.setText("—")
            self.preview_score.setStyleSheet(
                f"color: {T.TEXT_FAINT}; font-size: 19px; font-weight: 700;"
                " background: transparent;")
        else:
            verdict = "AI-generated" if score >= self.threshold else "authentic"
            self.preview_score.setText(f"{score:.4f}  ·  {verdict}")
            self.preview_score.setStyleSheet(
                f"color: {score_color(score)}; font-size: 19px; font-weight: 700;"
                " background: transparent;")

        dims = f"{pix.width()}×{pix.height()}" if not pix.isNull() else "?"
        self.preview_meta.setText(
            f"<b>{item.name}</b><br>{item.rel_path}<br>"
            f"{dims} · {item.size_bytes / 1024:.0f} KB<br>"
            f"ground truth: {_label_text(item.label)}")

    # -- export ------------------------------------------------------------
    def export_json(self):
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
        print(f"  wrote {n:,} predictions to {path}", flush=True)
        QMessageBox.information(self, "Exported",
                                f"Wrote {n:,} records to:\n{path}")

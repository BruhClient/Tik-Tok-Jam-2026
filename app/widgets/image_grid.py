"""Lazy-loading thumbnail grid that stays responsive on large directories."""

from __future__ import annotations

import math

from PyQt6.QtCore import QAbstractListModel, QModelIndex, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QFontMetrics, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QListView, QStyle, QStyledItemDelegate

from .. import theme as T
from .components import score_color

THUMB = 132
TILE_W = THUMB + 22
TILE_H = THUMB + 46


class ImageGridModel(QAbstractListModel):
    """Rows are indices into dataset.items (filtered subset)."""

    def __init__(self, state, loader, parent=None):
        super().__init__(parent)
        self.state = state
        self.loader = loader
        self.rows: list = []
        loader.ready.connect(self._on_thumb)

    # -- data --------------------------------------------------------------
    def set_rows(self, rows: list):
        self.beginResetModel()
        self.rows = list(rows)
        self.endResetModel()

    def item_for(self, row: int):
        if 0 <= row < len(self.rows):
            return self.state.dataset.items[self.rows[row]]
        return None

    def dataset_index(self, row: int) -> int:
        return self.rows[row] if 0 <= row < len(self.rows) else -1

    def score_for(self, row: int):
        run = self.state.run
        if run is None:
            return None
        di = self.dataset_index(row)
        if di < 0 or di >= len(run.scores):
            return None
        s = run.scores[di]
        return None if s is None or math.isnan(s) else s

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        item = self.item_for(index.row())
        if item is None:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return item.name
        if role == Qt.ItemDataRole.ToolTipRole:
            score = self.score_for(index.row())
            bits = [item.rel_path, f"{item.size_bytes / 1024:.0f} KB"]
            if item.label is not None:
                bits.append("label: " + ("AI-generated" if item.label else "authentic"))
            if score is not None:
                bits.append(f"score: {score:.4f}")
            return "\n".join(bits)
        if role == Qt.ItemDataRole.DecorationRole:
            return self.loader.get(index.row(), item.path)
        if role == Qt.ItemDataRole.UserRole:
            return item.label
        if role == Qt.ItemDataRole.UserRole + 1:
            return self.score_for(index.row())
        return None

    def refresh_scores(self):
        if self.rows:
            self.dataChanged.emit(self.index(0), self.index(len(self.rows) - 1),
                                  [Qt.ItemDataRole.UserRole + 1])

    def _on_thumb(self, row: int, path: str, pix: QPixmap):
        if 0 <= row < len(self.rows):
            idx = self.index(row)
            self.dataChanged.emit(idx, idx, [Qt.ItemDataRole.DecorationRole])


class ImageTileDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index) -> QSize:
        return QSize(TILE_W, TILE_H)

    def paint(self, painter: QPainter, option, index):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        rect = option.rect.adjusted(4, 4, -4, -4)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        # tile background
        bg = QColor(T.CARD_HOVER if (selected or hovered) else T.CARD)
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(QColor(T.ACCENT if selected else T.BORDER), 1.4 if selected else 1))
        painter.drawRoundedRect(QRectF(rect), 8, 8)

        # thumbnail
        pix = index.data(Qt.ItemDataRole.DecorationRole)
        img_rect = QRectF(rect.x() + 7, rect.y() + 7, rect.width() - 14, THUMB - 6)
        if isinstance(pix, QPixmap) and not pix.isNull():
            scaled = pix.scaled(
                int(img_rect.width()), int(img_rect.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = img_rect.x() + (img_rect.width() - scaled.width()) / 2
            y = img_rect.y() + (img_rect.height() - scaled.height()) / 2
            painter.drawPixmap(int(x), int(y), scaled)
        else:
            painter.setBrush(QBrush(QColor("#1F1F26")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(img_rect, 6, 6)
            painter.setPen(QPen(QColor(T.TEXT_FAINT)))
            painter.drawText(img_rect, Qt.AlignmentFlag.AlignCenter, "…")

        # label badge (top-left)
        label = index.data(Qt.ItemDataRole.UserRole)
        if label is not None:
            text = "AI" if label == 1 else "REAL"
            color = QColor(T.AI_COLOR if label == 1 else T.REAL_COLOR)
            self._badge(painter, rect.x() + 10, rect.y() + 10, text, color)

        # score badge (top-right)
        score = index.data(Qt.ItemDataRole.UserRole + 1)
        if score is not None:
            text = f"{score:.2f}"
            color = QColor(score_color(score))
            f = QFont(painter.font())
            f.setPointSizeF(7.5)
            f.setBold(True)
            w = QFontMetrics(f).horizontalAdvance(text) + 12
            self._badge(painter, rect.right() - 10 - w, rect.y() + 10, text, color, width=w)

        # filename
        painter.setPen(QPen(QColor(T.TEXT_DIM)))
        f = QFont(painter.font())
        f.setPointSizeF(8.0)
        painter.setFont(f)
        name_rect = rect.adjusted(8, THUMB + 4, -8, -4)
        fm = QFontMetrics(f)
        name = fm.elidedText(str(index.data(Qt.ItemDataRole.DisplayRole) or ""),
                             Qt.TextElideMode.ElideMiddle, name_rect.width())
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, name)

        painter.restore()

    def _badge(self, painter: QPainter, x: int, y: int, text: str, color: QColor,
               width: int = None):
        f = QFont(painter.font())
        f.setPointSizeF(7.5)
        f.setBold(True)
        painter.setFont(f)
        w = width if width is not None else QFontMetrics(f).horizontalAdvance(text) + 12
        r = QRectF(x, y, w, 16)
        bg = QColor(color)
        bg.setAlpha(48)
        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(color, 1))
        painter.drawRoundedRect(r, 8, 8)
        painter.setPen(QPen(color))
        painter.drawText(r, Qt.AlignmentFlag.AlignCenter, text)


class ImageGrid(QListView):
    item_selected = pyqtSignal(int)     # dataset index

    def __init__(self, state, loader, parent=None):
        super().__init__(parent)
        self.model_ = ImageGridModel(state, loader, self)
        self.setModel(self.model_)
        self.setItemDelegate(ImageTileDelegate(self))
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setMovement(QListView.Movement.Static)
        self.setUniformItemSizes(True)
        self.setSpacing(2)
        self.setMouseTracking(True)
        self.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.setVerticalScrollMode(QListView.ScrollMode.ScrollPerPixel)
        self.verticalScrollBar().setSingleStep(24)
        self.clicked.connect(self._on_clicked)

    def _on_clicked(self, index: QModelIndex):
        self.item_selected.emit(self.model_.dataset_index(index.row()))

    def set_rows(self, rows: list):
        self.model_.set_rows(rows)

    def refresh_scores(self):
        self.model_.refresh_scores()

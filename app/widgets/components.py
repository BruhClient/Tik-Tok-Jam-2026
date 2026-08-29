"""Small reusable UI pieces."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget
)

from .. import theme as T


class Card(QFrame):
    """Rounded surface used as a container everywhere."""

    def __init__(self, parent=None, padding: int = 14):
        super().__init__(parent)
        self.setObjectName("card")
        self.setStyleSheet(
            f"#card {{ background-color: {T.CARD}; border: 1px solid {T.BORDER};"
            f" border-radius: 10px; }}"
        )
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(padding, padding, padding, padding)
        self._layout.setSpacing(8)

    def layout(self) -> QVBoxLayout:  # type: ignore[override]
        return self._layout


class StatCard(Card):
    """Big-number metric tile."""

    def __init__(self, title: str, hint: str = "", parent=None):
        super().__init__(parent, padding=14)
        self.setMinimumWidth(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.title_label = QLabel(title.upper())
        self.title_label.setStyleSheet(
            f"color: {T.TEXT_FAINT}; font-size: 10px; font-weight: 700;"
            " letter-spacing: 1px; background: transparent;"
        )

        self.value_label = QLabel("—")
        self.value_label.setStyleSheet(
            f"color: {T.TEXT}; font-size: 26px; font-weight: 700; background: transparent;"
        )

        self.sub_label = QLabel(hint)
        self.sub_label.setStyleSheet(
            f"color: {T.TEXT_DIM}; font-size: 11px; background: transparent;"
        )
        self.sub_label.setWordWrap(True)

        self._layout.addWidget(self.title_label)
        self._layout.addWidget(self.value_label)
        self._layout.addWidget(self.sub_label)
        self._layout.setSpacing(2)

    def set_value(self, text: str, sub: str = None, color: str = None):
        self.value_label.setText(text)
        self.value_label.setStyleSheet(
            f"color: {color or T.TEXT}; font-size: 26px; font-weight: 700;"
            " background: transparent;"
        )
        if sub is not None:
            self.sub_label.setText(sub)


class Badge(QLabel):
    def __init__(self, text: str = "", color: str = T.TEXT_DIM, parent=None):
        super().__init__(text, parent)
        self.set_color(color)

    def set_color(self, color: str):
        self.setStyleSheet(
            f"color: {color}; background-color: rgba(255,255,255,0.05);"
            f" border: 1px solid {T.BORDER}; border-radius: 10px;"
            " padding: 3px 10px; font-size: 11px; font-weight: 600;"
        )


class Chip(QPushButton):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setProperty("chip", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class SectionTitle(QLabel):
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(
            f"color: {T.TEXT}; font-size: 14px; font-weight: 700; background: transparent;"
        )


class Hint(QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setStyleSheet(
            f"color: {T.TEXT_DIM}; font-size: 11px; background: transparent;"
        )


class PlaceholderBanner(QFrame):
    """Amber warning shown while a stub detector is selected."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("banner")
        self.setStyleSheet(
            f"#banner {{ background-color: rgba(245,166,35,0.10);"
            f" border: 1px solid rgba(245,166,35,0.45); border-radius: 8px; }}"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        icon = QLabel("⚠")
        icon.setStyleSheet(f"color: {T.WARN}; font-size: 15px; background: transparent;")
        self.label = QLabel("Placeholder detector — numbers below are not real results.")
        self.label.setStyleSheet(
            f"color: {T.WARN}; font-size: 12px; font-weight: 600; background: transparent;"
        )
        lay.addWidget(icon)
        lay.addWidget(self.label, 1)

    def set_text(self, text: str):
        self.label.setText(text)


class EmptyState(QWidget):
    """Centred message for panels with nothing to show yet."""

    def __init__(self, title: str, hint: str = "", parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(6)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(
            f"color: {T.TEXT_DIM}; font-size: 15px; font-weight: 600; background: transparent;"
        )
        self.hint_label = QLabel(hint)
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(
            f"color: {T.TEXT_FAINT}; font-size: 12px; background: transparent;"
        )
        lay.addWidget(self.title_label)
        lay.addWidget(self.hint_label)

    def set_text(self, title: str, hint: str = ""):
        self.title_label.setText(title)
        self.hint_label.setText(hint)


def score_color(score: float) -> str:
    """Teal (authentic) -> amber -> red (AI)."""
    if score is None or score != score:
        return T.TEXT_FAINT
    if score < 0.35:
        return T.REAL_COLOR
    if score < 0.65:
        return T.WARN
    return T.AI_COLOR

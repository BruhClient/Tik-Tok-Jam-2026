"""Dark theme: palette constants, Qt stylesheet, matplotlib rcParams."""

from __future__ import annotations

BG = "#0E0E10"
SURFACE = "#141418"
CARD = "#1C1C21"
CARD_HOVER = "#24242B"
BORDER = "#2A2A32"
TEXT = "#EDEDF0"
TEXT_DIM = "#9A9AA6"
TEXT_FAINT = "#6B6B78"

ACCENT = "#FE2C55"
ACCENT_DIM = "#C4213F"
SECONDARY = "#25F4EE"
WARN = "#F5A623"
GOOD = "#3DD68C"
BAD = "#FF5C5C"

REAL_COLOR = "#25F4EE"
AI_COLOR = "#FE2C55"

FONT_STACK = '"Segoe UI", "Inter", "SF Pro Text", system-ui, sans-serif'
MONO_STACK = '"Cascadia Mono", "Consolas", "SF Mono", monospace'


STYLESHEET = f"""
* {{
    font-family: {FONT_STACK};
    font-size: 13px;
    color: {TEXT};
}}

QWidget {{
    background-color: {BG};
}}

QMainWindow, QDialog {{
    background-color: {BG};
}}

/* ---------- Toolbar ---------- */
QToolBar {{
    background-color: {SURFACE};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 8px 12px;
    spacing: 8px;
}}
QToolBar QLabel {{
    color: {TEXT_DIM};
    padding: 0 4px;
}}

/* ---------- Buttons ---------- */
QPushButton {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 14px;
    color: {TEXT};
}}
QPushButton:hover {{
    background-color: {CARD_HOVER};
    border-color: #3A3A45;
}}
QPushButton:pressed {{
    background-color: #121216;
}}
QPushButton:disabled {{
    color: {TEXT_FAINT};
    background-color: #16161A;
    border-color: #222228;
}}
QPushButton[accent="true"] {{
    background-color: {ACCENT};
    border: 1px solid {ACCENT};
    color: #FFFFFF;
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{
    background-color: #FF4569;
    border-color: #FF4569;
}}
QPushButton[accent="true"]:disabled {{
    background-color: #4A1A26;
    border-color: #4A1A26;
    color: #8A6672;
}}
QPushButton[chip="true"] {{
    border-radius: 13px;
    padding: 5px 14px;
    background-color: {CARD};
    color: {TEXT_DIM};
}}
QPushButton[chip="true"]:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: #FFFFFF;
    font-weight: 600;
}}

/* ---------- Inputs ---------- */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 10px;
    selection-background-color: {ACCENT};
}}
QComboBox:hover, QLineEdit:hover {{
    border-color: #3A3A45;
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_DIM};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {CARD};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    outline: none;
    padding: 4px;
}}

QCheckBox {{
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {BORDER};
    background-color: {CARD};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox:disabled {{
    color: {TEXT_FAINT};
}}
QCheckBox::indicator:disabled {{
    background-color: #1A1A20;
    border-color: #26262E;
}}

/* ---------- Slider ---------- */
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: #FFFFFF;
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}

/* ---------- Tabs ---------- */
QTabWidget::pane {{
    border: none;
    background-color: {BG};
}}
QTabBar {{
    background-color: {SURFACE};
}}
QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_DIM};
    padding: 10px 20px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{
    color: {TEXT};
}}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
    font-weight: 600;
}}

/* ---------- Views ---------- */
QListView, QTableView, QTreeView {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    outline: none;
    gridline-color: {BORDER};
    selection-background-color: {ACCENT_DIM};
}}
QTableView::item {{
    padding: 4px 6px;
    border: none;
}}
QTableView::item:selected {{
    background-color: {ACCENT_DIM};
    color: #FFFFFF;
}}
QListView::item:selected {{
    background-color: transparent;
}}
QHeaderView::section {{
    background-color: {CARD};
    color: {TEXT_DIM};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 7px 8px;
    font-weight: 600;
}}
QHeaderView::section:hover {{
    color: {TEXT};
}}
QTableCornerButton::section {{
    background-color: {CARD};
    border: none;
}}

/* ---------- Scrollbars ---------- */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #35353F;
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: #45454F;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #35353F;
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0; width: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}

/* ---------- Misc ---------- */
QProgressBar {{
    background-color: {CARD};
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 4px;
}}
QStatusBar {{
    background-color: {SURFACE};
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
}}
QStatusBar::item {{
    border: none;
}}
QSplitter::handle {{
    background-color: {BORDER};
}}
QSplitter::handle:horizontal {{
    width: 1px;
}}
QToolTip {{
    background-color: {CARD};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 6px 8px;
    border-radius: 4px;
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 12px;
    background-color: {SURFACE};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: {TEXT_DIM};
    font-weight: 600;
}}
"""


def apply_matplotlib_style() -> None:
    """Make matplotlib figures blend into the dark UI."""
    import matplotlib

    matplotlib.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BORDER,
            "axes.labelcolor": TEXT_DIM,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 10,
            "axes.titleweight": "600",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": BORDER,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "text.color": TEXT,
            "xtick.color": TEXT_FAINT,
            "ytick.color": TEXT_FAINT,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": TEXT_DIM,
            "font.size": 9,
            "figure.autolayout": False,
        }
    )

"""The first screen: say what you are uploading, then pick the folder.

The old Run page inferred the kind of data from the folder and quietly went
whichever way the scan fell. Here the choice is explicit and enforced - you
declare a labeled dataset or plain images, and the app either honours it or
tells you the folder cannot support it. That is what decides which results
screen you land on.
"""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget
)

from .. import theme as T
from ..dataset import LabelMode
from ..detectors import available_detectors
from . import components as C
from .components import Card, Hint, SectionTitle

MODE_LABELED, MODE_IMAGES = range(2)

#: what the scanner tries, in order - quoted back when a labeled folder has none
LABEL_SOURCES = ("real/ and ai/ subfolders, a labels.csv manifest, "
                 "or real_ / ai_ filename prefixes")


class ModeTile(QFrame):
    """One of the two upload choices: a title, a line of consequence, a state.

    A QFrame has no :checked pseudo-state, so the selected look is painted from
    here rather than from the stylesheet.
    """

    clicked = pyqtSignal()

    def __init__(self, title: str, detail: str, parent=None):
        super().__init__(parent)
        self.setProperty("tile", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._checked = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(3)

        self.title_label = QLabel(title)
        self.detail_label = QLabel(detail)
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet(
            f"color: {T.TEXT_FAINT}; font-size: {C.FS_SMALL}px;"
            " background: transparent;")
        lay.addWidget(self.title_label)
        lay.addWidget(self.detail_label)
        self._paint()

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        self._checked = bool(checked)
        self._paint()

    def _paint(self):
        if self._checked:
            self.setStyleSheet(
                'QFrame[tile="true"] { background-color: %s;'
                ' border: 1px solid %s; border-radius: %dpx; }'
                % (T.ACCENT_SOFT, T.ACCENT, T.R_CARD))
            color, weight = T.ACCENT_TEXT, 700
        else:
            self.setStyleSheet("")          # back to the stylesheet's resting look
            color, weight = T.TEXT, 600
        self.title_label.setStyleSheet(
            f"color: {color}; font-size: {C.FS_BODY}px; font-weight: {weight};"
            " background: transparent;")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class UploadPage(QWidget):
    """Declare the data, point at it, run it."""

    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app
        self.peek = None                   # Dataset from the last folder scan
        self.mode = None                   # nothing is assumed on your behalf

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 0, 28, 28)
        outer.setSpacing(0)
        outer.addStretch(1)

        column = QWidget()
        column.setMaximumWidth(560)
        col = QVBoxLayout(column)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(14)

        col.addLayout(self._build_brand())
        col.addWidget(self._build_card())
        col.addWidget(Hint(
            "The same run from a terminal:  python detect.py <folder>"))

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(column)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(2)                # sits a little above centre

        self._sync()

    # -- construction ------------------------------------------------------
    def _build_brand(self) -> QHBoxLayout:
        brand = QHBoxLayout()
        brand.setSpacing(11)
        mark = QLabel()                    # the app mark, drawn not typeset
        mark.setFixedSize(26, 26)
        mark.setStyleSheet(f"background-color: {T.ACCENT}; border-radius: 8px;")

        words = QVBoxLayout()
        words.setSpacing(1)
        title = QLabel("AIGC Detector")
        title.setStyleSheet(
            f"color: {T.TEXT}; font-size: 17px; font-weight: 700;"
            " letter-spacing: -0.3px;")
        words.addWidget(title)
        words.addWidget(Hint("Find AI-generated images, and see how well that "
                             "detection holds up."))

        brand.addWidget(mark, 0, Qt.AlignmentFlag.AlignTop)
        brand.addLayout(words, 1)
        return brand

    def _build_card(self) -> QWidget:
        card = Card(padding=20)
        lay = card.layout()
        lay.setSpacing(12)

        lay.addWidget(SectionTitle("What are you uploading"))
        self.tiles = [
            ModeTile("Labeled dataset",
                     "A folder holding real/ and ai/. You get accuracy, AUC, "
                     "ROC and the robustness sweep."),
            ModeTile("Just images",
                     "Any folder of images. You get a verdict per image - "
                     "AI-generated or authentic."),
        ]
        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        for i, tile in enumerate(self.tiles):
            tile.clicked.connect(lambda idx=i: self.set_mode(idx))
            tiles.addWidget(tile, 1)       # equal weight, not equal wordiness
        lay.addLayout(tiles)

        lay.addSpacing(4)
        self.folder_title = SectionTitle("Folder")
        lay.addWidget(self.folder_title)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Choose an image folder...")
        self.path_edit.textChanged.connect(self._on_path_changed)
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_folder)
        src = QHBoxLayout()
        src.setSpacing(8)
        src.addWidget(self.path_edit, 1)
        src.addWidget(self.browse_btn)
        lay.addLayout(src)

        self.peek_label = Hint("")
        lay.addWidget(self.peek_label)

        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {T.BORDER};")
        lay.addSpacing(4)
        lay.addWidget(divider)
        lay.addSpacing(4)

        lay.addWidget(SectionTitle("Model"))
        self.detector_combo = QComboBox()
        for cls in available_detectors():
            bits = []
            if cls.is_placeholder:
                bits.append("placeholder")
            if cls.requires_weights and not cls.is_ready():
                bits.append("no checkpoint")
            suffix = "   ·   " + " · ".join(bits) if bits else ""
            self.detector_combo.addItem(cls.display_name + suffix, cls.name)
            self.detector_combo.setItemData(self.detector_combo.count() - 1,
                                            cls.description,
                                            Qt.ItemDataRole.ToolTipRole)
        self.detector_combo.currentIndexChanged.connect(self._sync)
        lay.addWidget(self.detector_combo)

        self.weights_row = QWidget()
        self.weights_row.setProperty("bare", True)   # it sits on the card
        wr = QHBoxLayout(self.weights_row)
        wr.setContentsMargins(0, 0, 0, 0)
        wr.setSpacing(8)
        self.weights_edit = QLineEdit()
        self.weights_edit.setPlaceholderText(
            "models/model.pt — leave empty for the default")
        weights_browse = QPushButton("Browse...")
        weights_browse.clicked.connect(self.browse_weights)
        wr.addWidget(self.weights_edit, 1)
        wr.addWidget(weights_browse)
        lay.addWidget(self.weights_row)

        lay.addSpacing(6)
        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.run_btn = QPushButton("Run")
        self.run_btn.setProperty("accent", True)
        self.run_btn.setMinimumWidth(112)
        self.run_btn.clicked.connect(lambda: self.app.start_run())
        self.open_btn = QPushButton("Open predictions.json")
        self.open_btn.setProperty("link", True)
        self.open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_btn.setToolTip("Visualise a result file produced earlier")
        self.open_btn.clicked.connect(self.browse_json)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.open_btn)
        actions.addStretch(1)
        lay.addLayout(actions)

        self.status = Hint("")
        lay.addWidget(self.status)
        return card

    # -- state -------------------------------------------------------------
    @property
    def directory(self) -> str:
        return self.path_edit.text().strip().strip('"')

    @property
    def detector_name(self) -> str:
        return self.detector_combo.currentData()

    @property
    def weights(self) -> str:
        return self.weights_edit.text().strip().strip('"') or None

    @property
    def label_mode(self) -> LabelMode:
        """NONE for plain images, so a labeled folder is still scored blind."""
        return LabelMode.NONE if self.mode == MODE_IMAGES else LabelMode.AUTO

    def set_mode(self, mode: int):
        self.mode = mode
        for i, tile in enumerate(self.tiles):
            tile.setChecked(i == mode)
        self._sync()

    def set_directory(self, path: str):
        self.path_edit.setText(path)

    # -- browsing ----------------------------------------------------------
    def browse_folder(self):
        start = self.directory or os.path.expanduser("~")
        path = QFileDialog.getExistingDirectory(self, "Select image folder", start)
        if path:
            self.path_edit.setText(path)

    def browse_json(self):
        start = self.directory or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(self, "Open predictions", start,
                                              "JSON (*.json)")
        if path:
            self.app.load_predictions_file(path)

    def browse_weights(self):
        start = self.weights_edit.text().strip() or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", start,
            "Checkpoints (*.pt *.pth);;All files (*)")
        if path:
            self.weights_edit.setText(path)

    def set_busy(self, busy: bool, message: str = ""):
        for w in (self.detector_combo, self.weights_row, self.path_edit,
                  self.browse_btn, self.open_btn):
            w.setEnabled(not busy)
        for tile in self.tiles:
            tile.setEnabled(not busy)
        self.run_btn.setText("Running…" if busy else "Run")
        self.status.setText(message)
        if busy:
            self.run_btn.setEnabled(False)
        else:
            self._sync()

    # -- folder peek -------------------------------------------------------
    def _on_path_changed(self):
        self.peek = None
        self._sync()
        if os.path.isdir(self.directory):
            self.app.peek_directory(self.directory)

    def set_peek(self, dataset):
        """Report what a background scan found, before anything is scored."""
        if dataset is None or dataset.root != os.path.abspath(self.directory or ""):
            return                          # a later folder won the race
        self.peek = dataset
        self._sync()

    def _peek_text(self):
        """(message, colour, ok) for the current folder under the current mode.

        The scan always runs with AUTO, so one scan answers both modes and
        switching the tile re-reads it rather than rescanning.
        """
        if self.mode is None:
            return "Pick one of the two above to begin.", T.TEXT_FAINT, False
        if not self.directory:
            return "", T.TEXT_DIM, False
        if self.peek is None:
            return ("Reading the folder…" if os.path.isdir(self.directory)
                    else "That folder does not exist."), T.TEXT_FAINT, False
        if not len(self.peek):
            return "No images in this folder.", T.WARN, False

        count = f"{len(self.peek):,} images"
        if self.mode == MODE_IMAGES:
            note = ("labels present but ignored — you asked for plain images"
                    if self.peek.has_labels else "no labels — verdicts only")
            return f"{count}   ·   {note}", T.TEXT_DIM, True
        if not self.peek.has_labels:
            return (f"{count}, but no labels. Looked for {LABEL_SOURCES}.\n"
                    'Fix the folder, or choose "Just images" to score it anyway.',
                    T.BAD, False)
        source = self.peek.label_source_detail.split(" (")[0]
        return (f"{count}   ·   {self.peek.n_real:,} real / {self.peek.n_ai:,} AI"
                f"   ·   {source}", T.TEXT_DIM, True)

    def _sync(self):
        cls = next((c for c in available_detectors()
                    if c.name == self.detector_name), None)
        self.weights_row.setVisible(bool(cls and cls.requires_weights))

        chosen = self.mode is not None
        for w in (self.folder_title, self.path_edit, self.browse_btn):
            w.setEnabled(chosen)

        message, color, ok = self._peek_text()
        self.peek_label.setText(message)
        self.peek_label.setStyleSheet(
            f"color: {color}; font-size: {C.FS_SMALL}px; background: transparent;")
        self.run_btn.setEnabled(ok)
        self.status.setText(
            "Choose a folder." if chosen and not self.directory else "")

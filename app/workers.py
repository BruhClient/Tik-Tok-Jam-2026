"""Background workers. Nothing here touches widgets directly - only signals."""

from __future__ import annotations

import os
import time
import traceback
from collections import OrderedDict

from PIL import Image, ImageQt
from PyQt6.QtCore import QObject, QRunnable, QSize, QThread, QThreadPool, pyqtSignal
from PyQt6.QtGui import QPixmap

from .transforms import get_transform


# --------------------------------------------------------------------------- #
# thumbnails
# --------------------------------------------------------------------------- #

class _ThumbSignals(QObject):
    ready = pyqtSignal(int, str, QPixmap)   # row, path, pixmap
    failed = pyqtSignal(int, str)


class _ThumbTask(QRunnable):
    def __init__(self, row: int, path: str, size: int, signals: _ThumbSignals):
        super().__init__()
        self.row, self.path, self.size, self.signals = row, path, size, signals
        self.setAutoDelete(True)

    def run(self):
        try:
            with Image.open(self.path) as im:
                try:
                    im.draft("RGB", (self.size * 2, self.size * 2))
                except Exception:
                    pass
                im = im.convert("RGB")
                im.thumbnail((self.size, self.size), Image.BILINEAR)
                qim = ImageQt.ImageQt(im)
                pix = QPixmap.fromImage(qim).copy()
            self.signals.ready.emit(self.row, self.path, pix)
        except Exception as exc:
            self.signals.failed.emit(self.row, str(exc))


class ThumbnailLoader(QObject):
    """Lazy thumbnail decoding with an LRU pixmap cache."""

    ready = pyqtSignal(int, str, QPixmap)

    def __init__(self, size: int = 132, cache_limit: int = 2500, parent=None):
        super().__init__(parent)
        self.size = size
        self.cache_limit = cache_limit
        self._cache: OrderedDict = OrderedDict()
        self._pending: set = set()
        self._signals = _ThumbSignals()
        self._signals.ready.connect(self._on_ready)
        self._signals.failed.connect(self._on_failed)
        self._pool = QThreadPool()
        self._pool.setMaxThreadCount(max(2, min(6, (os.cpu_count() or 4) - 1)))

    def get(self, row: int, path: str):
        """Return a cached pixmap or None, queueing a decode when missing."""
        key = self._key(path)
        pix = self._cache.get(key)
        if pix is not None:
            self._cache.move_to_end(key)
            return pix
        if key not in self._pending:
            self._pending.add(key)
            self._pool.start(_ThumbTask(row, path, self.size, self._signals))
        return None

    def clear(self):
        self._cache.clear()
        self._pending.clear()
        self._pool.clear()

    def shutdown(self):
        self._pool.clear()
        self._pool.waitForDone(2000)

    def _key(self, path: str) -> str:
        try:
            return f"{path}|{os.path.getmtime(path)}|{self.size}"
        except OSError:
            return f"{path}|0|{self.size}"

    def _on_ready(self, row: int, path: str, pix: QPixmap):
        key = self._key(path)
        self._pending.discard(key)
        self._cache[key] = pix
        while len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)
        self.ready.emit(row, path, pix)

    def _on_failed(self, row: int, path_or_msg: str):
        # keep the slot symmetric; a broken image just stays blank
        self._pending = {k for k in self._pending}


def placeholder_pixmap(size: int, color: str = "#24242B") -> QPixmap:
    pix = QPixmap(QSize(size, size))
    pix.fill(_qcolor(color))
    return pix


def _qcolor(hex_str: str):
    from PyQt6.QtGui import QColor
    return QColor(hex_str)


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #

class InferenceWorker(QThread):
    """Runs a detector over a list of paths in batches."""

    progress = pyqtSignal(int, int, float)          # done, total, eta_seconds
    batch_ready = pyqtSignal(list, list)            # indices, scores
    failed_item = pyqtSignal(str, str)              # path, message
    error = pyqtSignal(str)
    done = pyqtSignal(bool, float)                  # cancelled, elapsed_seconds

    def __init__(self, detector, paths: list, indices: list = None, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.paths = list(paths)
        self.indices = list(indices) if indices is not None else list(range(len(paths)))
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        started = time.perf_counter()
        total = len(self.paths)
        done = 0
        try:
            self.detector.ensure_loaded()
        except Exception as exc:
            self.error.emit(f"Failed to load detector: {exc}")
            self.done.emit(True, 0.0)
            return

        bs = max(1, int(getattr(self.detector, "batch_size", 16)))
        try:
            for start in range(0, total, bs):
                if self._cancel:
                    break
                chunk_paths = self.paths[start:start + bs]
                chunk_idx = self.indices[start:start + bs]
                try:
                    scores = self.detector.predict_batch(chunk_paths)
                except Exception as exc:
                    traceback.print_exc()
                    for p in chunk_paths:
                        self.failed_item.emit(p, str(exc))
                    scores = [float("nan")] * len(chunk_paths)

                scores = [float(s) if s is not None else float("nan") for s in scores]
                if len(scores) != len(chunk_paths):
                    scores = (scores + [float("nan")] * len(chunk_paths))[:len(chunk_paths)]

                self.batch_ready.emit(chunk_idx, scores)
                done += len(chunk_paths)
                elapsed = time.perf_counter() - started
                eta = (elapsed / done) * (total - done) if done else 0.0
                self.progress.emit(done, total, eta)
        except Exception as exc:
            traceback.print_exc()
            self.error.emit(str(exc))

        self.done.emit(self._cancel, time.perf_counter() - started)


# --------------------------------------------------------------------------- #
# robustness sweep
# --------------------------------------------------------------------------- #

class RobustnessWorker(QThread):
    """Applies (transform, severity) cells in memory and re-scores each set."""

    cell_started = pyqtSignal(str, int)                  # transform key, severity
    cell_done = pyqtSignal(str, int, list, list)         # key, severity, labels, scores
    progress = pyqtSignal(int, int, str)                 # done_cells, total_cells, text
    error = pyqtSignal(str)
    done = pyqtSignal(bool, float)

    def __init__(self, detector, paths: list, labels: list, cells: list,
                 max_side: int = 768, parent=None):
        """cells: list of (transform_key, severity) tuples."""
        super().__init__(parent)
        self.detector = detector
        self.paths = list(paths)
        self.labels = list(labels)
        self.cells = list(cells)
        self.max_side = max_side
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _load(self, path: str):
        img = Image.open(path)
        try:
            img.draft("RGB", (self.max_side, self.max_side))
        except Exception:
            pass
        img = img.convert("RGB")
        if max(img.size) > self.max_side:
            s = self.max_side / max(img.size)
            img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                             Image.BILINEAR)
        return img

    def run(self):
        started = time.perf_counter()
        try:
            self.detector.ensure_loaded()
        except Exception as exc:
            self.error.emit(f"Failed to load detector: {exc}")
            self.done.emit(True, 0.0)
            return

        bs = max(1, int(getattr(self.detector, "batch_size", 16)))
        total_cells = len(self.cells)

        for ci, (key, severity) in enumerate(self.cells):
            if self._cancel:
                break
            try:
                spec = get_transform(key)
            except KeyError:
                continue
            self.cell_started.emit(key, severity)
            label_txt = f"{spec.display_name} · {spec.label_for(severity)}" if severity else "Clean baseline"
            self.progress.emit(ci, total_cells, label_txt)

            scores, labels = [], []
            try:
                for start in range(0, len(self.paths), bs):
                    if self._cancel:
                        break
                    chunk = self.paths[start:start + bs]
                    imgs = []
                    keep_labels = []
                    for j, p in enumerate(chunk):
                        try:
                            img = self._load(p)
                            img = spec.apply(img, severity) if severity else img
                            # hints consumed only by the placeholder detector
                            img._aigc_source = p
                            img._aigc_severity = severity
                            imgs.append(img)
                            keep_labels.append(self.labels[start + j])
                        except Exception as exc:
                            print(f"[robustness] skipped {p}: {exc}")
                    if not imgs:
                        continue
                    chunk_scores = self.detector.predict_images(imgs)
                    scores.extend(float(s) for s in chunk_scores)
                    labels.extend(keep_labels)
            except Exception as exc:
                traceback.print_exc()
                self.error.emit(f"{key}@{severity}: {exc}")
                continue

            if not self._cancel:
                self.cell_done.emit(key, severity, labels, scores)
                self.progress.emit(ci + 1, total_cells, label_txt)

        self.done.emit(self._cancel, time.perf_counter() - started)

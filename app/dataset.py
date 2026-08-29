"""Dataset scanning and ground-truth label inference.

Three label sources are supported and auto-detected in this order:
  1. subfolder  - real/ vs ai/ (and synonyms) anywhere in the relative path
  2. manifest   - labels.csv / labels.json with path + label columns
  3. filename   - regex prefix on the file name, default ^(real|ai|fake)[_-]

Anything that yields no label stays None -> the app runs in inference-only mode.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

REAL_TOKENS = {"real", "authentic", "natural", "human", "nature", "camera", "genuine", "0"}
AI_TOKENS = {"ai", "aigc", "fake", "generated", "synthetic", "gen", "sd", "midjourney", "1"}

MANIFEST_NAMES = ("labels.csv", "labels.json", "manifest.csv", "manifest.json", "ground_truth.csv")
PATH_KEYS = ("image_path", "path", "file", "filename", "filepath", "image", "img")
LABEL_KEYS = ("label", "is_ai", "y", "target", "class", "gt", "ground_truth")

DEFAULT_FILENAME_REGEX = r"^(real|authentic|ai|aigc|fake|generated)[_\-.]"


class LabelMode(str, Enum):
    AUTO = "auto"
    SUBFOLDER = "subfolder"
    MANIFEST = "manifest"
    FILENAME = "filename"
    NONE = "none"


LABEL_MODE_TITLES = {
    LabelMode.AUTO: "Auto-detect",
    LabelMode.SUBFOLDER: "Subfolder names (real/ vs ai/)",
    LabelMode.MANIFEST: "Manifest file (csv/json)",
    LabelMode.FILENAME: "Filename prefix",
    LabelMode.NONE: "Unlabeled (inference only)",
}


@dataclass
class ImageItem:
    path: str                   # absolute path
    rel_path: str               # path relative to dataset root, forward slashes
    label: int | None = None    # 0 = real/authentic, 1 = AI-generated
    size_bytes: int = 0

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass
class Dataset:
    root: str = ""
    items: list = field(default_factory=list)
    label_mode: LabelMode = LabelMode.NONE
    label_source_detail: str = ""
    skipped: int = 0

    def __len__(self) -> int:
        return len(self.items)

    @property
    def n_real(self) -> int:
        return sum(1 for i in self.items if i.label == 0)

    @property
    def n_ai(self) -> int:
        return sum(1 for i in self.items if i.label == 1)

    @property
    def n_unlabeled(self) -> int:
        return sum(1 for i in self.items if i.label is None)

    @property
    def has_labels(self) -> bool:
        return self.n_real + self.n_ai > 0

    def labeled_indices(self) -> list:
        return [i for i, it in enumerate(self.items) if it.label is not None]


# --------------------------------------------------------------------------- #
# scanning
# --------------------------------------------------------------------------- #

def list_images(root: str):
    """Recursively collect image files. Returns (paths, skipped_non_images)."""
    paths = []
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                paths.append(os.path.join(dirpath, fn))
            elif not fn.startswith("."):
                skipped += 1
    paths.sort()
    return paths, skipped


def scan_directory(root, mode=LabelMode.AUTO, manifest_path=None,
                   filename_regex=DEFAULT_FILENAME_REGEX) -> Dataset:
    root = os.path.abspath(root)
    paths, skipped = list_images(root)

    items = []
    for p in paths:
        try:
            size = os.path.getsize(p)
        except OSError:
            size = 0
        rel = os.path.relpath(p, root).replace("\\", "/")
        items.append(ImageItem(path=p, rel_path=rel, size_bytes=size))

    ds = Dataset(root=root, items=items, skipped=skipped)
    apply_labels(ds, mode=mode, manifest_path=manifest_path, filename_regex=filename_regex)
    return ds


def apply_labels(ds: Dataset, mode=LabelMode.AUTO, manifest_path=None,
                 filename_regex=DEFAULT_FILENAME_REGEX) -> Dataset:
    """(Re)apply labels to an existing dataset, in place."""
    for it in ds.items:
        it.label = None
    ds.label_mode = LabelMode.NONE
    ds.label_source_detail = "no labels found - inference only"

    if not ds.items:
        return ds

    if mode == LabelMode.AUTO:
        order = [LabelMode.SUBFOLDER, LabelMode.MANIFEST, LabelMode.FILENAME]
    elif mode == LabelMode.NONE:
        return ds
    else:
        order = [mode]

    for m in order:
        if m == LabelMode.SUBFOLDER:
            n = _label_by_subfolder(ds)
            detail = "subfolder names"
        elif m == LabelMode.MANIFEST:
            mp = manifest_path or _find_manifest(ds.root)
            if not mp:
                continue
            n = _label_by_manifest(ds, mp)
            detail = "manifest: " + os.path.basename(mp)
        else:
            n = _label_by_filename(ds, filename_regex)
            detail = "filename regex: " + filename_regex

        if n > 0:
            ds.label_mode = m
            ds.label_source_detail = "%s (%d/%d labeled)" % (detail, n, len(ds.items))
            return ds
        for it in ds.items:
            it.label = None

    return ds


def _token_label(token: str):
    t = token.strip().lower()
    if t in REAL_TOKENS:
        return 0
    if t in AI_TOKENS:
        return 1
    return None


def _label_by_subfolder(ds: Dataset) -> int:
    n = 0
    for it in ds.items:
        parts = it.rel_path.split("/")[:-1]
        for part in reversed(parts):
            lab = _token_label(part)
            if lab is not None:
                it.label = lab
                n += 1
                break
    return n


def _label_by_filename(ds: Dataset, regex: str) -> int:
    try:
        pat = re.compile(regex, re.IGNORECASE)
    except re.error:
        return 0
    n = 0
    for it in ds.items:
        m = pat.match(it.name)
        if not m:
            continue
        token = m.group(1) if m.groups() else m.group(0)
        lab = _token_label(re.sub(r"[_\-.]$", "", token))
        if lab is not None:
            it.label = lab
            n += 1
    return n


def _find_manifest(root: str):
    for name in MANIFEST_NAMES:
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p
    return None


def _coerce_label(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if float(value) >= 0.5 else 0
    s = str(value).strip().lower()
    if not s:
        return None
    if s in ("1", "true", "yes"):
        return 1
    if s in ("0", "false", "no"):
        return 0
    return _token_label(s)


def _pick_key(keys, candidates):
    lower = {str(k).strip().lower(): k for k in keys}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _load_manifest_rows(manifest_path: str):
    ext = os.path.splitext(manifest_path)[1].lower()
    if ext == ".json":
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for key in ("records", "images", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return list(data[key])
            return [{"image_path": k, "label": v} for k, v in data.items()]
        return [r for r in data if isinstance(r, dict)]
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _label_by_manifest(ds: Dataset, manifest_path: str) -> int:
    try:
        rows = _load_manifest_rows(manifest_path)
    except Exception:
        return 0
    if not rows:
        return 0

    path_key = _pick_key(rows[0].keys(), PATH_KEYS)
    label_key = _pick_key(rows[0].keys(), LABEL_KEYS)
    if path_key is None or label_key is None:
        return 0

    base = os.path.dirname(os.path.abspath(manifest_path))
    lookup = {}
    for row in rows:
        raw = row.get(path_key)
        lab = _coerce_label(row.get(label_key))
        if raw is None or lab is None:
            continue
        raw = str(raw).strip().replace("\\", "/")
        # index under several spellings so relative and absolute paths both resolve
        abs_p = os.path.normcase(os.path.abspath(os.path.join(base, raw)))
        lookup[abs_p] = lab
        lookup[os.path.normcase(raw)] = lab
        lookup.setdefault(os.path.normcase(os.path.basename(raw)), lab)

    n = 0
    for it in ds.items:
        for key in (os.path.normcase(os.path.abspath(it.path)),
                    os.path.normcase(it.rel_path),
                    os.path.normcase(it.name)):
            if key in lookup:
                it.label = lookup[key]
                n += 1
                break
    return n

"""Result writers.

Three outputs, one of which is the contract:

  * predictions.json  the deliverable - never anything but raw scores
  * predictions.csv   the same plus the derived verdict, for a spreadsheet
  * run report        metrics, timing and dataset composition, for the record

predictions.json is the deliverable format required by the problem statement:

    [
      {"image_path": "...", "pred": 0.8731},
      ...
    ]

One record per input image, in scan order, whatever happened to it. The
threshold never appears here and never changes a value: it is a reading of the
scores, not a property of them.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime

from . import metrics as M


def _path_for(item, root: str, relative: bool) -> str:
    """Absolute by default; relative when the caller wants a portable file."""
    if relative:
        return item.rel_path
    return os.path.abspath(item.path)


def export_predictions_json(path: str, dataset, run, relative: bool = False,
                            nan_value: float = 0.5) -> int:
    """Write the required [{image_path, pred}] file. Returns row count.

    An image that failed to decode has a NaN score, which is not valid JSON and
    is not a prediction either. It is written as `nan_value` (0.5 - maximally
    uncommitted) rather than dropped, so the record count always matches the
    file count. The failures are named in the terminal summary.
    """
    records = []
    for item, score in zip(dataset.items, run.scores):
        pred = float(score)
        if math.isnan(pred):
            pred = float(nan_value)
        records.append({
            "image_path": _path_for(item, dataset.root, relative),
            # 6 dp: past float noise, short enough that the file stays readable
            "pred": round(pred, 6),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return len(records)


def export_predictions_csv(path: str, dataset, run, threshold: float,
                           relative: bool = False) -> int:
    """Spreadsheet view: score plus the verdict and correctness at `threshold`.

    Unlike the JSON, this one is threshold-dependent by design. Cells are left
    empty rather than filled with a placeholder wherever the answer is unknown -
    no score, no label, or neither.
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "pred", "predicted_label", "true_label", "correct"])
        for item, score in zip(dataset.items, run.scores):
            has_score = not math.isnan(score)
            pred_label = "" if not has_score else int(score >= threshold)
            true_label = "" if item.label is None else item.label
            correct = ""
            if has_score and item.label is not None:
                correct = int(pred_label == item.label)
            w.writerow([
                _path_for(item, dataset.root, relative),
                "" if not has_score else round(float(score), 6),
                pred_label, true_label, correct,
            ])
    return len(dataset.items)


def build_run_report(dataset, run, threshold: float) -> dict:
    """Everything worth recording about one run, as a plain dict.

    Written by `detect.py --report`. Includes the label source and the threshold
    so a result can be read months later without guessing how it was produced.
    """
    y, s = run.valid_pairs(dataset)
    m = M.compute_metrics(y, s, threshold)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "root": dataset.root,
            "n_images": len(dataset.items),
            "n_real": dataset.n_real,
            "n_ai": dataset.n_ai,
            "n_unlabeled": dataset.n_unlabeled,
            "label_source": dataset.label_source_detail,
        },
        "detector": {
            "name": run.detector_name,
            "display_name": run.detector_display,
        },
        "run": {
            "elapsed_seconds": round(run.elapsed, 3),
            "images_scored": run.n_scored,
            "images_failed": len(run.failures),
            # getattr: only the GUI's worker sets this, and a CLI RunResult
            # has no such field
            "cancelled": getattr(run, "cancelled", False),
            "threshold": threshold,
        },
        "metrics": m.as_dict(),
    }


def export_run_report(path: str, dataset, run, threshold: float) -> dict:
    """Write the run report to `path` and return it."""
    report = build_run_report(dataset, run, threshold)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report

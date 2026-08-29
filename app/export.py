"""Result writers.

predictions.json is the deliverable format required by the problem statement:

    [
      {"image_path": "...", "pred": 0.8731},
      ...
    ]
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime

from . import metrics as M


def _path_for(item, root: str, relative: bool) -> str:
    if relative:
        return item.rel_path
    return os.path.abspath(item.path)


def export_predictions_json(path: str, dataset, run, relative: bool = False,
                            nan_value: float = 0.5) -> int:
    """Write the required [{image_path, pred}] file. Returns row count."""
    records = []
    for item, score in zip(dataset.items, run.scores):
        pred = float(score)
        if math.isnan(pred):
            pred = float(nan_value)
        records.append({
            "image_path": _path_for(item, dataset.root, relative),
            "pred": round(pred, 6),
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    return len(records)


def export_predictions_csv(path: str, dataset, run, threshold: float,
                           relative: bool = False) -> int:
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
            "is_placeholder": run.is_placeholder,
        },
        "run": {
            "elapsed_seconds": round(run.elapsed, 3),
            "images_scored": run.n_scored,
            "images_failed": len(run.failures),
            "cancelled": run.cancelled,
            "threshold": threshold,
        },
        "metrics": m.as_dict(),
    }


def export_run_report(path: str, dataset, run, threshold: float) -> dict:
    report = build_run_report(dataset, run, threshold)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def build_robustness_report(dataset, run, state) -> dict:
    from .transforms import TRANSFORMS_BY_KEY

    baseline = state.robustness_baseline
    cells = []
    for (key, severity), cell in sorted(state.robustness.items()):
        spec = TRANSFORMS_BY_KEY.get(key)
        entry = {
            "transform": key,
            "transform_name": spec.display_name if spec else key,
            "severity": severity,
            "severity_label": spec.label_for(severity) if spec and severity else "clean",
            "metrics": cell.metrics.as_dict(),
        }
        if baseline and baseline.valid:
            entry["delta_accuracy"] = _delta(cell.metrics.accuracy, baseline.accuracy)
            entry["delta_auc"] = _delta(cell.metrics.auc, baseline.auc)
        cells.append(entry)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": dataset.root,
        "detector": run.detector_display if run else "",
        "is_placeholder": run.is_placeholder if run else True,
        "threshold": state.threshold,
        "baseline": baseline.as_dict() if baseline else None,
        "cells": cells,
    }


def _delta(a: float, b: float):
    if a != a or b != b:
        return None
    return round(a - b, 6)


def export_robustness_json(path: str, dataset, run, state) -> dict:
    report = build_robustness_report(dataset, run, state)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def export_robustness_csv(path: str, dataset, run, state) -> int:
    report = build_robustness_report(dataset, run, state)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["transform", "severity", "severity_label", "n", "accuracy",
                    "auc", "f1", "fpr", "recall", "delta_accuracy", "delta_auc"])
        for c in report["cells"]:
            m = c["metrics"]
            w.writerow([
                c["transform"], c["severity"], c["severity_label"], m["n"],
                _r(m["accuracy"]), _r(m["auc"]), _r(m["f1"]), _r(m["fpr"]), _r(m["recall"]),
                c.get("delta_accuracy", ""), c.get("delta_auc", ""),
            ])
    return len(report["cells"])


def _r(v):
    if v is None or v != v:
        return ""
    return round(float(v), 6)

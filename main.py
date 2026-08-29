"""DEV viewer: score a directory, then show the result.

    python main.py                      score the dev dataset, open the viewer
    python main.py <image_directory>    score any other directory
    python main.py predictions.json     just visualise a finished result file

All the work - scanning, loading the detector, scoring - runs first and logs to
this terminal. The window opens on the finished result; nothing loads inside it.

Dev dataset: `sample_data/`, resolved relative to this file. Both splits load
as one pooled set:

    sample_data/
    |-- train/
    |   |-- real/     authentic images -> label 0
    |   `-- ai/       AI-generated     -> label 1
    `-- test/
        |-- real/
        `-- ai/

Production batch runs go through predict.py, which takes any directory, needs
no structure, and never opens a window.
"""

from __future__ import annotations

import json
import os
import sys

#: dataset opened when no argument is given
DEV_DATA_DIR = "sample_data"

#: default decision threshold for the summary and the viewer
DEFAULT_THRESHOLD = 0.5

#: picked up automatically if it sits next to the dataset (see robustness.py)
ROBUSTNESS_REPORT = "robustness_report.json"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from app import runner                                    # noqa: E402
from app.transforms import TRANSFORMS_BY_KEY              # noqa: E402


def dev_dataset_dir() -> str:
    """Absolute dev dataset path, resolved from this file, not the cwd."""
    return os.path.join(PROJECT_ROOT, DEV_DATA_DIR)


def load_robustness(report_path: str) -> dict:
    """Reshape a robustness report into what the viewer's chart wants."""
    if not os.path.isfile(report_path):
        return {}
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception as exc:
        runner.warn(f"could not read {report_path}: {exc}")
        return {}

    series = {}
    for cell in report.get("cells", []):
        spec = TRANSFORMS_BY_KEY.get(cell.get("transform"))
        name = spec.display_name if spec else cell.get("transform", "?")
        acc = (cell.get("metrics") or {}).get("accuracy")
        if acc is not None and acc == acc:
            series.setdefault(name, {})[cell.get("severity", 0)] = acc

    baseline = ((report.get("baseline") or {}).get("accuracy", float("nan")))
    runner.log(f"      robustness report: {len(series)} transform(s) "
               f"from {os.path.basename(report_path)}")
    return {"series": series, "baseline": baseline, "metric": "Accuracy"}


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # ---- work first, with terminal logs ---------------------------------
    if arg and arg.lower().endswith(".json"):
        if not os.path.isfile(arg):
            print(f"error: no such file: {arg}", file=sys.stderr)
            return 2
        dataset, result = runner.load_predictions(arg, total_steps=2)
        runner.summarize(dataset, result, DEFAULT_THRESHOLD, total_steps=2, step_no=2)
        report_dir = os.path.dirname(os.path.abspath(arg))
    else:
        directory = arg or dev_dataset_dir()
        if not os.path.isdir(directory):
            print(f"error: not a directory: {directory}", file=sys.stderr)
            if arg is None:
                print(f"the dev dataset is missing. Expected:\n"
                      f"  {DEV_DATA_DIR}/train/real, {DEV_DATA_DIR}/train/ai\n"
                      f"  {DEV_DATA_DIR}/test/real,  {DEV_DATA_DIR}/test/ai\n"
                      f"or run:  python main.py <directory>", file=sys.stderr)
            return 2
        dataset, result = runner.run_directory(directory)
        runner.summarize(dataset, result, DEFAULT_THRESHOLD)
        report_dir = dataset.root

    robustness = load_robustness(os.path.join(report_dir, ROBUSTNESS_REPORT))

    # ---- then draw it ----------------------------------------------------
    runner.log("")
    runner.log("opening viewer...")

    from PyQt6.QtWidgets import QApplication
    from app import theme
    from app.widgets.window import ResultsWindow

    QApplication.setApplicationName("AIGC Detector")
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(theme.STYLESHEET)
    theme.apply_matplotlib_style()

    window = ResultsWindow(dataset, result, robustness, DEFAULT_THRESHOLD)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

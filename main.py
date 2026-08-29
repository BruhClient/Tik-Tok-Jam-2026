"""AIGC Image Detector - desktop evaluation console (DEV entry point).

    python main.py                    open the dev dataset below
    python main.py <image_directory>  open some other directory instead

Dev dataset: `sample_data/test/`, resolved relative to this file:

    sample_data/
    |-- test/            <- what this console opens
    |   |-- real/        authentic images -> label 0
    |   `-- ai/          AI-generated     -> label 1
    `-- train/           ignored here; training data is not evaluation data

Filenames do not matter and both label folders may nest further. Change
DEV_DATA_DIR to target a different split. Production batch runs go through
predict.py, which takes any directory and needs no structure at all.
"""

from __future__ import annotations

import os
import sys

from PyQt6.QtWidgets import QApplication

from app import theme
from app.widgets.main_window import MainWindow

#: default dataset opened when no directory argument is given (the eval split -
#: never point this at train/, or the accuracy numbers are measured on data the
#: model was fitted to)
DEV_DATA_DIR = os.path.join("sample_data", "test")

#: label subfolders the dev dataset is expected to contain
DEV_LABEL_DIRS = ("real", "ai")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def dev_dataset_dir() -> str:
    """Absolute path of the dev dataset, resolved from this file, not the cwd."""
    return os.path.join(PROJECT_ROOT, DEV_DATA_DIR)


def check_dev_dataset(path: str) -> str | None:
    """Return the path if usable, else print what is wrong and return None."""
    if not os.path.isdir(path):
        print(f"dev dataset not found: {path}", file=sys.stderr)
        _print_expected_layout()
        return None

    missing = [d for d in DEV_LABEL_DIRS if not os.path.isdir(os.path.join(path, d))]
    if missing:
        print(f"dev dataset at {path} is missing: "
              + ", ".join(d + "/" for d in missing), file=sys.stderr)
        _print_expected_layout()
        # still open it - the app will just report the images as unlabeled
    return path


def _print_expected_layout() -> None:
    # ASCII only: this goes to a Windows console that may not be UTF-8
    print(f"expected layout:\n"
          f"  {DEV_DATA_DIR}/\n"
          f"    real/   authentic images\n"
          f"    ai/     AI-generated images\n"
          f"open any other folder with:  python main.py <directory>",
          file=sys.stderr)


def main() -> int:
    QApplication.setApplicationName("AIGC Image Detector")
    QApplication.setOrganizationName("Hackathon")

    app = QApplication(sys.argv)
    app.setStyleSheet(theme.STYLESHEET)
    theme.apply_matplotlib_style()

    if len(sys.argv) > 1:
        start_dir = sys.argv[1]
        if not os.path.isdir(start_dir):
            print(f"not a directory: {start_dir}", file=sys.stderr)
            start_dir = None
    else:
        start_dir = check_dev_dataset(dev_dataset_dir())

    window = MainWindow(start_dir=start_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

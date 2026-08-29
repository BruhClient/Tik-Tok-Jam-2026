"""AIGC Image Detector - desktop evaluation console.

Usage:
    python main.py [image_directory]
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from app import theme
from app.widgets.main_window import MainWindow


def main() -> int:
    QApplication.setApplicationName("AIGC Image Detector")
    QApplication.setOrganizationName("Hackathon")

    app = QApplication(sys.argv)
    app.setStyleSheet(theme.STYLESHEET)
    theme.apply_matplotlib_style()

    start_dir = sys.argv[1] if len(sys.argv) > 1 else None
    window = MainWindow(start_dir=start_dir)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

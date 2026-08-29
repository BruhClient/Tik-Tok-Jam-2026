"""PRODUCTION script: score any image directory, write predictions.json.

    python predict.py <image_dir>
    python predict.py <image_dir> --out results.json --detector heuristic

Output is the required deliverable format:

    [
      {"image_path": "C:/data/test/img_0001.jpg", "pred": 0.8731},
      ...
    ]

`pred` is P(AI-generated) in [0, 1]. No labels, no folder structure and no GUI
are required - point it at any directory and it scores every image it finds,
recursively. If the directory happens to be labeled, an accuracy summary is
printed too; otherwise that section is skipped.

Visualise the output afterwards with:  python main.py <out.json>
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import runner                                          # noqa: E402
from app.detectors import available_detectors                   # noqa: E402
from app.export import export_predictions_json, export_run_report  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="predict.py",
        description="Score an image directory for AI-generated content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("directory", nargs="?", help="directory of images, searched recursively")
    ap.add_argument("--out", "-o", default="predictions.json",
                    help="output JSON path (default: predictions.json)")
    ap.add_argument("--detector", "-d", default=None,
                    help="registered detector name (default: first real backend)")
    ap.add_argument("--threshold", "-t", type=float, default=0.5,
                    help="threshold for the printed summary only (default: 0.5)")
    ap.add_argument("--relative", action="store_true",
                    help="write paths relative to the input directory")
    ap.add_argument("--report", default=None,
                    help="also write a metrics/timing report to this JSON path")
    ap.add_argument("--quiet", "-q", action="store_true", help="suppress progress output")
    ap.add_argument("--list-detectors", action="store_true",
                    help="print the registered backends and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_detectors:
        for cls in available_detectors():
            tag = "  [PLACEHOLDER]" if cls.is_placeholder else ""
            print(f"{cls.name:12s} {cls.display_name}{tag}")
            print(f"{'':12s} {cls.description}")
        return 0

    if not args.directory:
        print("error: an image directory is required "
              "(use --list-detectors to inspect backends)", file=sys.stderr)
        return 2

    runner.QUIET = args.quiet
    dataset, result = runner.run_directory(args.directory, args.detector, total_steps=5)

    runner.step(4, 5, "writing")
    n = export_predictions_json(args.out, dataset, result, relative=args.relative)
    runner.log(f"{n:,} predictions -> {os.path.abspath(args.out)}", indent=6)
    if args.report:
        export_run_report(args.report, dataset, result, args.threshold)
        runner.log(f"run report    -> {os.path.abspath(args.report)}", indent=6)

    runner.summarize(dataset, result, args.threshold, total_steps=5, step_no=5)
    runner.log("")
    runner.log(f"visualise it:  python main.py {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

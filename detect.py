"""THE deliverable: image directory in, predictions.json out.

    python detect.py <image_dir>
    python detect.py <image_dir> --out results.json --weights models/best.pt

The folder decides what you get back. Both cases always write the JSON:

  labeled    a folder holding real/ and ai/ (or a labels.csv, or real_/ai_
             filename prefixes). Both classes are pooled into one evaluation
             set and scored together, then accuracy, AUC, F1, FPR and the
             confusion counts are printed. --best-threshold also reports where
             F1 peaks, which is usually not 0.50.

  unlabeled  any other folder. Scores only - there is no truth to measure
             against, so no metrics. This is not an error; pass
             --require-labels if you expected labels and want it to fail loudly
             when the subfolders turn out to be misnamed.

Output is the required format:

    [
      {"image_path": "C:/data/img_0001.jpg", "pred": 0.8731},
      ...
    ]

`pred` is P(AI-generated) in [0, 1]. Scores are always raw floats - the
threshold only affects what gets printed, never what gets written.

Visualise any of it with:  python gui.py <dir>   or   python gui.py <out.json>
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import metrics as M                                      # noqa: E402
from app import runner                                            # noqa: E402
from app.detectors import available_detectors                     # noqa: E402
from app.export import export_predictions_json, export_run_report  # noqa: E402

TOTAL_STEPS = 5


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="detect.py",
        description="Score an image directory for AI-generated content. "
                    "Reports accuracy too when the folder is labeled.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?",
                    help="directory of images, searched recursively")
    ap.add_argument("--out", "-o", default="predictions.json",
                    help="output JSON path (default: predictions.json)")
    ap.add_argument("--detector", "-d", default=None,
                    help="registered detector name (default: best available)")
    ap.add_argument("--weights", "-w", default=None,
                    help="checkpoint to load (default: models/model.pt)")
    ap.add_argument("--threshold", "-t", type=float, default=None,
                    help="decision threshold for the printed summary "
                         "(default: the detector's own operating point, or 0.5)")
    ap.add_argument("--best-threshold", action="store_true",
                    help="also report the threshold that maximises F1 (labeled only)")
    ap.add_argument("--require-labels", action="store_true",
                    help="fail instead of falling back to scores-only")
    ap.add_argument("--relative", action="store_true",
                    help="write paths relative to the input directory")
    ap.add_argument("--report", default=None,
                    help="also write a metrics/timing report to this JSON path")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="suppress progress output")
    ap.add_argument("--list-detectors", action="store_true",
                    help="print the registered backends and exit")
    return ap.parse_args(argv)


def _report(exc: SystemExit) -> int:
    """Print a SystemExit raised for bad input and turn it into exit code 2."""
    code = exc.code
    if isinstance(code, str):
        print(code, file=sys.stderr)
        return 2
    return int(code or 0)


def list_detectors() -> int:
    for cls in available_detectors():
        tag = (f"  [no checkpoint at {cls.default_weights}]"
               if cls.requires_weights and not cls.is_ready() else "")
        print(f"{cls.name:12s} {cls.display_name}{tag}")
        print(f"{'':12s} {cls.description}")
    print(f"\ndefault: {runner.default_detector_name()}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_detectors:
        return list_detectors()

    if not args.directory:
        print("error: an image directory is required "
              "(use --list-detectors to inspect backends)", file=sys.stderr)
        return 2

    runner.QUIET = args.quiet
    try:
        dataset, result = runner.run_directory(
            args.directory, args.detector, args.weights, total_steps=TOTAL_STEPS)
    except SystemExit as exc:
        # the runner raises SystemExit(message) for bad input; report it as a
        # usage error (2) rather than the 1 a bare raise would give
        return _report(exc)

    if args.require_labels and not dataset.has_labels:
        print("\nerror: --require-labels was set but no ground-truth labels were "
              "found.\n"
              "       use real/ and ai/ subfolders, a labels.csv, or real_/ai_ "
              "prefixes.", file=sys.stderr)
        return 2

    # A calibrated model ships the threshold it was tuned for, and for this one
    # it is nowhere near 0.5. Honour it unless the caller asked for a specific
    # value. Only the printed summary moves - the JSON is always raw scores.
    threshold = args.threshold if args.threshold is not None else result.threshold

    runner.step(4, TOTAL_STEPS, "writing")
    n = export_predictions_json(args.out, dataset, result, relative=args.relative)
    runner.log(f"{n:,} predictions -> {os.path.abspath(args.out)}", indent=6)
    if args.report:
        export_run_report(args.report, dataset, result, threshold)
        runner.log(f"run report    -> {os.path.abspath(args.report)}", indent=6)

    runner.summarize(dataset, result, threshold,
                     total_steps=TOTAL_STEPS, step_no=5)

    if args.best_threshold and dataset.has_labels:
        y, s = result.valid_pairs(dataset)
        best = M.best_threshold(y, s, "f1")
        bm = M.compute_metrics(y, s, best)
        runner.log("")
        runner.log(f"best F1 at threshold {best:.3f}", indent=6)
        runner.log(f"accuracy  {M.fmt(bm.accuracy)}      F1 {M.fmt(bm.f1, pct=False)}"
                   f"      FPR {M.fmt(bm.fpr)}", indent=6)

    runner.log("")
    runner.log(f"see it:  python gui.py {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

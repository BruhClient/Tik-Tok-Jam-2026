"""PRODUCTION entry point: score any image directory, write predictions.json.

    python predict.py <image_dir>
    python predict.py <image_dir> --out results.json --detector heuristic

Output is the required deliverable format:

    [
      {"image_path": "C:/data/test/img_0001.jpg", "pred": 0.8731},
      ...
    ]

`pred` is P(AI-generated) in [0, 1]. No labels, no folder structure and no GUI
are required - point it at any directory and it scores every image it finds,
recursively. If the directory happens to be labeled (real/ + ai/ subfolders, a
labels.csv, or real_/ai_ filename prefixes) an accuracy summary is printed as
well; otherwise that section is simply skipped.

main.py is the interactive console; this is the batch path. Both share the same
detector registry, so a model registered once works in both.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.dataset import scan_directory                     # noqa: E402
from app.detectors import available_detectors, get_detector  # noqa: E402
from app.export import export_predictions_json, export_run_report  # noqa: E402
from app import metrics as M                               # noqa: E402
from app.state import RunResult                            # noqa: E402


def default_detector_name() -> str:
    """Prefer a real backend; fall back to the best placeholder available."""
    detectors = available_detectors()          # real backends sort first
    if not detectors:
        raise RuntimeError("no detectors registered")
    return detectors[0].name


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="predict.py",
        description="Score an image directory for AI-generated content.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("directory", nargs="?",
                    help="directory of images, searched recursively")
    ap.add_argument("--out", "-o", default="predictions.json",
                    help="output JSON path (default: predictions.json)")
    ap.add_argument("--detector", "-d", default=None,
                    help="registered detector name (default: first real backend)")
    ap.add_argument("--threshold", "-t", type=float, default=0.5,
                    help="decision threshold used for the printed summary (default: 0.5)")
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
    if not os.path.isdir(args.directory):
        print(f"error: not a directory: {args.directory}", file=sys.stderr)
        return 2

    ds = scan_directory(args.directory)
    if not ds.items:
        print(f"error: no images found under {args.directory}", file=sys.stderr)
        return 2

    name = args.detector or default_detector_name()
    try:
        detector = get_detector(name)
    except KeyError:
        print(f"error: unknown detector {name!r}. Available: "
              + ", ".join(c.name for c in available_detectors()), file=sys.stderr)
        return 2

    log = (lambda *a, **k: None) if args.quiet else (
        lambda *a, **k: print(*a, file=sys.stderr, **k))

    log(f"directory : {os.path.abspath(args.directory)}")
    log(f"images    : {len(ds):,}"
        + (f"  ({ds.skipped} non-image files skipped)" if ds.skipped else ""))
    log(f"detector  : {detector.display_name}")
    if detector.is_placeholder:
        log("WARNING   : this is a PLACEHOLDER backend - the scores are not "
            "real detections.")

    detector.ensure_loaded()

    paths = [it.path for it in ds.items]
    scores, failures = [], []
    bs = max(1, detector.batch_size)
    started = time.perf_counter()

    for start in range(0, len(paths), bs):
        chunk = paths[start:start + bs]
        try:
            scores.extend(float(s) for s in detector.predict_batch(chunk))
        except Exception as exc:                      # keep going on a bad batch
            failures.extend((p, str(exc)) for p in chunk)
            scores.extend([float("nan")] * len(chunk))
        if not args.quiet:
            done = min(start + bs, len(paths))
            pct = 100.0 * done / len(paths)
            print(f"\r  scoring {done:,}/{len(paths):,} ({pct:.0f}%)",
                  end="", file=sys.stderr, flush=True)
    elapsed = time.perf_counter() - started
    log("")

    run = RunResult(detector_name=detector.name, detector_display=detector.display_name,
                    is_placeholder=detector.is_placeholder, scores=scores,
                    elapsed=elapsed, failures=failures)

    n = export_predictions_json(args.out, ds, run, relative=args.relative)
    log(f"wrote {n:,} predictions to {os.path.abspath(args.out)} in {elapsed:.1f}s"
        + (f"  ({len(failures)} failed to decode)" if failures else ""))

    # optional accuracy summary - only when the directory carries labels
    if ds.has_labels:
        y, s = run.valid_pairs(ds)
        m = M.compute_metrics(y, s, args.threshold)
        log("")
        log(f"labels    : {ds.label_source_detail}")
        log(f"threshold : {args.threshold:.2f}")
        log(f"accuracy  : {M.fmt(m.accuracy)}   AUC: {M.fmt(m.auc, pct=False)}   "
            f"F1: {M.fmt(m.f1, pct=False)}")
        log(f"FPR       : {M.fmt(m.fpr)}   (authentic images flagged as AI)")
        log(f"confusion : TP {m.tp}  FP {m.fp}  TN {m.tn}  FN {m.fn}")

    if args.report:
        export_run_report(args.report, ds, run, args.threshold)
        log(f"wrote report to {os.path.abspath(args.report)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Headless CLI mirroring the deliverable: image directory -> predictions JSON.

    python tools/predict_dir.py sample_data --out predictions.json --detector heuristic

Uses the exact same detector registry and writer as the desktop app, so once a
real backend is registered this script produces the submission file unchanged.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.dataset import scan_directory                    # noqa: E402
from app.detectors import available_detectors, get_detector  # noqa: E402
from app.export import export_predictions_json            # noqa: E402
from app.state import RunResult                           # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="directory of images (searched recursively)")
    ap.add_argument("--out", default="predictions.json", help="output JSON path")
    ap.add_argument("--detector", default="heuristic",
                    help="registered detector name (default: heuristic)")
    ap.add_argument("--relative", action="store_true",
                    help="write paths relative to the input directory")
    ap.add_argument("--list-detectors", action="store_true")
    args = ap.parse_args()

    if args.list_detectors:
        for cls in available_detectors():
            tag = " [placeholder]" if cls.is_placeholder else ""
            print(f"{cls.name:12s} {cls.display_name}{tag}")
        return 0

    if not os.path.isdir(args.directory):
        print(f"error: {args.directory} is not a directory", file=sys.stderr)
        return 2

    ds = scan_directory(args.directory)
    if not ds.items:
        print("error: no images found", file=sys.stderr)
        return 2

    detector = get_detector(args.detector)
    if detector.is_placeholder:
        print(f"warning: '{detector.name}' is a PLACEHOLDER backend — "
              "the scores are not real detections.", file=sys.stderr)
    detector.ensure_loaded()

    paths = [it.path for it in ds.items]
    scores = []
    bs = max(1, detector.batch_size)
    started = time.perf_counter()
    for start in range(0, len(paths), bs):
        scores.extend(float(s) for s in detector.predict_batch(paths[start:start + bs]))
        done = min(start + bs, len(paths))
        print(f"\r  scored {done}/{len(paths)}", end="", file=sys.stderr, flush=True)
    elapsed = time.perf_counter() - started
    print(file=sys.stderr)

    run = RunResult(detector_name=detector.name, detector_display=detector.display_name,
                    is_placeholder=detector.is_placeholder, scores=scores, elapsed=elapsed)
    n = export_predictions_json(args.out, ds, run, relative=args.relative)
    print(f"wrote {n} predictions to {args.out} in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

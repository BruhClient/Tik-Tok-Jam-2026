"""ROBUSTNESS sweep: how far does accuracy fall under post-processing?

    python robustness.py <image_dir>
    python robustness.py <image_dir> --transforms jpeg,blur,rescale --sample 200

Each selected transform is applied in memory at five severities and the whole
sample is re-scored, then compared against a clean baseline measured through
the same pipeline. Writes robustness_report.json next to the dataset, which
`python main.py` picks up and draws as the degradation curve.

Ground-truth labels are required - this measures accuracy, not just scores.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PIL import Image                                    # noqa: E402

from app import metrics as M                             # noqa: E402
from app import runner                                   # noqa: E402
from app.export import export_robustness_json            # noqa: E402
from app.transforms import TRANSFORMS, TRANSFORMS_BY_KEY  # noqa: E402

DEFAULT_TRANSFORMS = "jpeg,blur,rescale,crop,social"
DEFAULT_REPORT = "robustness_report.json"


class _Bag:
    """Minimal stand-in for the state object export.py expects."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.robustness = {}
        self.robustness_baseline = None


class _Cell:
    def __init__(self, key, severity, metrics):
        self.transform_key = key
        self.severity = severity
        self.metrics = metrics


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="robustness.py",
        description="Measure detector robustness under post-processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", nargs="?", help="labeled image directory")
    ap.add_argument("--transforms", default=DEFAULT_TRANSFORMS,
                    help=f"comma-separated keys (default: {DEFAULT_TRANSFORMS})")
    ap.add_argument("--severities", default="1,2,3,4,5",
                    help="comma-separated severity levels (default: 1,2,3,4,5)")
    ap.add_argument("--sample", type=int, default=200,
                    help="images per cell, balanced across classes (default: 200)")
    ap.add_argument("--max-side", type=int, default=768,
                    help="decode cap in pixels (default: 768)")
    ap.add_argument("--detector", "-d", default=None, help="registered detector name")
    ap.add_argument("--threshold", "-t", type=float, default=0.5)
    ap.add_argument("--out", "-o", default=None,
                    help=f"report path (default: <dir>/{DEFAULT_REPORT})")
    ap.add_argument("--list-transforms", action="store_true")
    return ap.parse_args(argv)


def build_sample(dataset, n: int):
    """Balanced subset of the labeled images, deterministic."""
    import random

    labeled = [i for i, it in enumerate(dataset.items) if it.label is not None]
    rng = random.Random(20260829)
    reals = [i for i in labeled if dataset.items[i].label == 0]
    ais = [i for i in labeled if dataset.items[i].label == 1]
    rng.shuffle(reals)
    rng.shuffle(ais)
    half = max(1, min(n // 2, min(len(reals), len(ais)) or n))
    picked = sorted(reals[:half] + ais[:half])
    return ([dataset.items[i].path for i in picked],
            [dataset.items[i].label for i in picked])


def load_image(path: str, max_side: int):
    img = Image.open(path)
    try:
        img.draft("RGB", (max_side, max_side))
    except Exception:
        pass
    img = img.convert("RGB")
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                         Image.BILINEAR)
    return img


def score_cell(detector, paths, labels, spec, severity, max_side):
    """Transform and re-score one (transform, severity) cell."""
    bs = max(1, int(getattr(detector, "batch_size", 16)))
    scores, kept = [], []
    for start in range(0, len(paths), bs):
        chunk = paths[start:start + bs]
        images, chunk_labels = [], []
        for j, p in enumerate(chunk):
            try:
                img = load_image(p, max_side)
                img = spec.apply(img, severity) if severity else img
                img._aigc_source = p            # hint the placeholder backend uses
                img._aigc_severity = severity
                images.append(img)
                chunk_labels.append(labels[start + j])
            except Exception as exc:
                runner.warn(f"skipped {os.path.basename(p)}: {exc}")
        if not images:
            continue
        scores.extend(float(s) for s in detector.predict_images(images))
        kept.extend(chunk_labels)
    return kept, scores


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_transforms:
        for spec in TRANSFORMS:
            levels = "  |  ".join(spec.label_for(i) for i in range(1, 6))
            print(f"{spec.key:12s} {spec.display_name}")
            print(f"{'':12s} {spec.description}")
            print(f"{'':12s} severities: {levels}")
        return 0

    if not args.directory:
        print("error: a labeled image directory is required "
              "(use --list-transforms to see the options)", file=sys.stderr)
        return 2

    dataset = runner.scan(args.directory, total_steps=4)
    if not dataset.has_labels:
        print("error: this sweep measures accuracy, so it needs ground-truth labels.\n"
              "       use real/ and ai/ subfolders, a labels.csv, or real_/ai_ prefixes.",
              file=sys.stderr)
        return 2

    keys = [k.strip() for k in args.transforms.split(",") if k.strip()]
    unknown = [k for k in keys if k not in TRANSFORMS_BY_KEY or k == "clean"]
    if unknown:
        print(f"error: unknown transform(s): {', '.join(unknown)}\n"
              f"       available: {', '.join(t.key for t in TRANSFORMS)}", file=sys.stderr)
        return 2
    severities = [int(s) for s in args.severities.split(",") if s.strip()]

    detector = runner.prepare_detector(args.detector, total_steps=4)
    paths, labels = build_sample(dataset, args.sample)

    cells = [("clean", 0)] + [(k, sv) for k in keys for sv in severities]
    runner.step(3, 4, f"sweeping {len(cells)} cells x {len(paths)} images "
                      f"({len(cells) * len(paths):,} scored images)")

    bag = _Bag(args.threshold)
    started = time.perf_counter()

    for i, (key, severity) in enumerate(cells, 1):
        spec = TRANSFORMS_BY_KEY[key]
        name = "clean baseline" if key == "clean" else \
               f"{spec.display_name}  |  {spec.label_for(severity)}"
        t0 = time.perf_counter()
        kept, scores = score_cell(detector, paths, labels, spec, severity, args.max_side)
        m = M.compute_metrics(kept, scores, args.threshold)

        if key == "clean":
            bag.robustness_baseline = m
            delta = ""
        else:
            bag.robustness[(key, severity)] = _Cell(key, severity, m)
            base = bag.robustness_baseline
            d = m.accuracy - base.accuracy if base else float("nan")
            delta = f"  d {'+' if d >= 0 else '-'}{abs(d) * 100:4.1f}pp" if d == d else ""

        runner.log(f"  [{i:>2}/{len(cells)}] {name:<44} "
                   f"acc {M.fmt(m.accuracy):>6}  auc {M.fmt(m.auc, pct=False):>5}"
                   f"{delta}   ({time.perf_counter() - t0:.1f}s)")

    elapsed = time.perf_counter() - started
    out = args.out or os.path.join(dataset.root, DEFAULT_REPORT)

    result = runner.RunResult(detector_name=detector.name,
                              detector_display=detector.display_name,
                              is_placeholder=detector.is_placeholder,
                              scores=[], source=dataset.root)
    export_robustness_json(out, dataset, result, bag)

    runner.step(4, 4, "results")
    base = bag.robustness_baseline
    runner.log(f"clean baseline   accuracy {M.fmt(base.accuracy)}  "
               f"AUC {M.fmt(base.auc, pct=False)}", indent=6)

    drops = sorted(((c.metrics.accuracy - base.accuracy, k, sv)
                    for (k, sv), c in bag.robustness.items()
                    if c.metrics.accuracy == c.metrics.accuracy))
    if drops:
        worst_d, worst_k, worst_sv = drops[0]
        spec = TRANSFORMS_BY_KEY[worst_k]
        runner.log(f"worst case       {spec.display_name}  |  {spec.label_for(worst_sv)}"
                   f"   {worst_d * 100:+.1f}pp", indent=6)
        mean = sum(d for d, _, _ in drops) / len(drops)
        runner.log(f"mean drop        {mean * 100:+.1f}pp over {len(drops)} cells", indent=6)
    runner.log(f"swept in {elapsed:.1f}s", indent=6)
    runner.log(f"report -> {os.path.abspath(out)}", indent=6)
    runner.log("")
    runner.log("visualise it:  python main.py"
               + ("" if os.path.dirname(out) == dataset.root else f"  {dataset.root}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

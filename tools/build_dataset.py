"""Mash multiple image datasets into one deduplicated real/ai corpus, then split.

Targets (``--dest``) keep the graded axes separated so held-out generators never
leak into training:

  _pool          training corpus (default)     -> `split` turns it into train/test
  heldout/<name> generalisation, NEVER trained  -> used directly as a test set
  robustness     platform-degradation validation

Workflow:

  1. INGEST from local folders (files already on disk) or from a Hugging Face
     parquet dataset (images embedded as bytes, optionally streamed + capped).
     Every image is decoded, content-hashed and RE-ENCODED to one JPEG quality
     so a detector can't cheat on per-class compression artifacts. Content
     duplicates (global, across every source and dest) are skipped.

  2. SPLIT the _pool 80/20 into data/{train,test}/{real,ai}.

Examples
--------
  # local folders: .../cifake/**/REAL are real, .../FAKE are ai
  python tools/build_dataset.py ingest cifake \
      --real "downloads/cifake/**/REAL" --ai "downloads/cifake/**/FAKE"

  # Hugging Face: Tiny-GenImage, label 0=real 1=fake -> training pool
  python tools/build_dataset.py ingest-hf tiny_genimage TheKernel01/Tiny-GenImage \
      --splits train validation --real-labels 0 --ai-labels 1

  # Hugging Face streamed + capped -> held-out (never trained)
  python tools/build_dataset.py ingest-hf openfake ComplexDataLab/OpenFake \
      --splits test --stream --limit-per-class 2000 --dest heldout/openfake \
      --label-col label --real-labels real 0 --ai-labels fake 1

  python tools/build_dataset.py split --ratio 0.8 --seed 0
  python tools/build_dataset.py stats
"""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
POOL = DATA / "_pool"
HASHES = DATA / ".hashes.txt"  # global: one image lands in exactly one dest

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
JPEG_QUALITY = 90


# --------------------------------------------------------------------------- #
# dedup
# --------------------------------------------------------------------------- #
def _load_seen() -> set[str]:
    return set(HASHES.read_text().split()) if HASHES.exists() else set()


def _append_hash(h: str) -> None:
    HASHES.parent.mkdir(parents=True, exist_ok=True)
    with HASHES.open("a") as fh:
        fh.write(h + "\n")


def _pixel_hash(im: Image.Image) -> str:
    """Hash decoded pixels, so the same picture re-encoded twice still dedupes."""
    rgb = im.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def _save_reencoded(im: Image.Image, dest_dir: Path, source: str, h: str, quality: int) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(dest_dir / f"{source}_{h[:12]}.jpg", "JPEG", quality=quality)


def _reencode_worker(job: tuple) -> tuple[str, bool] | None:
    """Runs in a pool process: bytes -> hashed, re-encoded JPEG on disk.

    Hash-named paths make writes idempotent, so parallel workers dedupe for free
    (identical picture -> identical filename) with no shared state. Returns
    (label, was_new) or None on a bad image.
    """
    raw, label, dest_dir_str, source, quality = job
    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        h = hashlib.sha256(im.tobytes()).hexdigest()
        path = Path(dest_dir_str) / f"{source}_{h[:12]}.jpg"
        if path.exists():
            return label, False
        im.save(path, "JPEG", quality=quality)
        return label, True
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# local-folder ingest
# --------------------------------------------------------------------------- #
def _iter_images(patterns: list[str], base: Path) -> list[Path]:
    out: list[Path] = []
    for pat in patterns or []:
        for m in base.glob(pat):
            if m.is_dir():
                out += [p for p in m.rglob("*") if p.suffix.lower() in IMG_EXTS]
            elif m.suffix.lower() in IMG_EXTS:
                out.append(m)
    return out


def ingest(args: argparse.Namespace) -> int:
    base = Path(args.base).resolve() if args.base else ROOT
    dest_root = DATA / args.dest
    seen = _load_seen()

    plan = [
        ("real", _iter_images((args.real or []) + (args.real_glob or []), base)),
        ("ai", _iter_images((args.ai or []) + (args.ai_glob or []), base)),
    ]
    if not any(paths for _, paths in plan):
        print("Nothing matched. Pass --real/--ai (dirs) or --real-glob/--ai-glob (files).")
        return 1

    added = {"real": 0, "ai": 0}
    dup = bad = 0
    for label, paths in plan:
        for p in paths:
            try:
                with Image.open(p) as im:
                    im.load()
                    h = _pixel_hash(im)
                    if h in seen:
                        dup += 1
                        continue
                    seen.add(h)
                    _append_hash(h)
                    _save_reencoded(im, dest_root / label, args.source, h, args.jpeg_quality)
                    added[label] += 1
            except Exception:
                bad += 1

    print(f"[{args.source} -> {args.dest}] real={added['real']} ai={added['ai']} "
          f"(skipped {dup} dup, {bad} unreadable)")
    return 0


# --------------------------------------------------------------------------- #
# Hugging Face parquet ingest
# --------------------------------------------------------------------------- #
def _shard_files(repo: str, split: str, num_workers: int, worker_id: int) -> list[str]:
    """Parquet files for a split that belong to this worker (round-robin by index)."""
    from huggingface_hub import list_repo_files

    parquet = sorted(
        f for f in list_repo_files(repo, repo_type="dataset")
        if f.endswith(".parquet") and split in f
    )
    mine = [f for i, f in enumerate(parquet) if i % num_workers == worker_id]
    return [f"hf://datasets/{repo}/{f}" for f in mine]


def _raw_bytes(cell) -> bytes | None:
    """Pull encoded bytes out of a datasets image cell (decode disabled)."""
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    if isinstance(cell, dict) and cell.get("bytes"):
        return cell["bytes"]
    return None


def ingest_hf(args: argparse.Namespace) -> int:
    import multiprocessing as mp

    from datasets import Image as HFImage
    from datasets import load_dataset

    dest_root = DATA / args.dest
    real_set = set(args.real_labels or [])
    ai_set = set(args.ai_labels or [])
    cap = args.limit_per_class
    jobs = args.jobs if args.jobs and args.jobs > 0 else (mp.cpu_count() or 1)
    for label in ("real", "ai"):
        (dest_root / label).mkdir(parents=True, exist_ok=True)
    added = {"real": 0, "ai": 0}
    bad = skip = 0

    forced = "real" if args.all_real else "ai" if args.all_ai else None

    def label_of(value) -> str | None:
        if forced:
            return forced
        s = str(value)
        if s in real_set or value in real_set:
            return "real"
        if s in ai_set or value in ai_set:
            return "ai"
        return None

    def rows():
        """Yield (raw_bytes, label) for on-label images, respecting the cap."""
        nonlocal skip
        for split in args.splits:
            if args.num_workers > 1:
                files = _shard_files(args.repo, split, args.num_workers, args.worker_id)
                if not files:
                    print(f"worker {args.worker_id}: no shards for split '{split}', skipping")
                    continue
                print(f"worker {args.worker_id}/{args.num_workers} split '{split}': "
                      f"{len(files)} shard(s): {[f.rsplit('/', 1)[-1] for f in files]}")
                ds = load_dataset("parquet", data_files=files, split="train", streaming=args.stream)
            else:
                ds = load_dataset(args.repo, split=split, streaming=args.stream)
            ds = ds.cast_column(args.image_col, HFImage(decode=False))  # keep bytes, skip decode
            for row in ds:
                if cap and submitted["real"] >= cap and submitted["ai"] >= cap:
                    return
                label = label_of(row.get(args.label_col))
                if label is None or (cap and submitted[label] >= cap):
                    skip += 1
                    continue
                raw = _raw_bytes(row.get(args.image_col))
                if raw is None:
                    skip += 1
                    continue
                submitted[label] += 1
                yield raw, label, str(dest_root / label), args.source, args.jpeg_quality

    submitted = {"real": 0, "ai": 0}

    def tally(result) -> None:
        nonlocal bad
        if result is None:
            bad += 1
            return
        label, was_new = result
        if was_new:
            added[label] += 1

    if jobs == 1:
        for job in rows():
            tally(_reencode_worker(job))
    else:
        with mp.Pool(jobs) as pool:
            for result in pool.imap_unordered(_reencode_worker, rows(), chunksize=16):
                tally(result)

    print(f"[{args.source} -> {args.dest}] real={added['real']} ai={added['ai']} "
          f"(jobs={jobs}, {bad} bad, {skip} off-label)")
    return 0


# --------------------------------------------------------------------------- #
# URL-list ingest (big real datasets ship URLs, not embedded bytes)
# --------------------------------------------------------------------------- #
def ingest_urls(args: argparse.Namespace) -> int:
    import concurrent.futures as cf

    import requests
    from datasets import load_dataset

    label = "real" if args.label == "real" else "ai"
    dest = DATA / args.dest / label
    dest.mkdir(parents=True, exist_ok=True)
    cap = args.limit_per_class
    threads = args.jobs if args.jobs and args.jobs > 0 else 16
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (dataset-pool)"

    def fetch(url: str) -> str:
        try:
            r = session.get(url, timeout=10)
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            h = hashlib.sha256(im.tobytes()).hexdigest()
            path = dest / f"{args.source}_{h[:12]}.jpg"
            if path.exists():
                return "dup"
            im.save(path, "JPEG", quality=args.jpeg_quality)
            return "ok"
        except Exception:
            return "bad"

    def urls():
        """Stream url column; keep only this worker's rows (round-robin by index)."""
        idx = 0
        for split in args.splits:
            ds = load_dataset(args.repo, split=split, streaming=True)
            for row in ds:
                take = args.num_workers <= 1 or idx % args.num_workers == args.worker_id
                idx += 1
                if take and row.get(args.url_col):
                    yield row[args.url_col]

    added = dup = bad = 0
    with cf.ThreadPoolExecutor(max_workers=threads) as ex:
        pending: set = set()
        src = urls()
        for u in src:
            pending.add(ex.submit(fetch, u))
            if len(pending) >= threads * 4:
                done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    r = f.result()
                    added += r == "ok"; dup += r == "dup"; bad += r == "bad"
                if cap and added >= cap:
                    break
        for f in pending:
            r = f.result()
            added += r == "ok"; dup += r == "dup"; bad += r == "bad"

    print(f"[{args.source} -> {args.dest}] {label}={added} "
          f"(threads={threads}, {dup} dup, {bad} failed/404)")
    return 0


# --------------------------------------------------------------------------- #
# split + stats
# --------------------------------------------------------------------------- #
def split(args: argparse.Namespace) -> int:
    import random
    import shutil

    rng = random.Random(args.seed)
    for label in ("real", "ai"):
        files = sorted(p for p in (POOL / label).glob("*") if p.suffix.lower() in IMG_EXTS)
        rng.shuffle(files)
        cut = int(len(files) * args.ratio)
        for name, group in (("train", files[:cut]), ("test", files[cut:])):
            dest = DATA / name / label
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            for p in group:
                shutil.copy2(p, dest / p.name)
        print(f"{label:4s}: {cut} train / {len(files) - cut} test")
    print(f"-> {DATA}/train and {DATA}/test written (ratio={args.ratio}, seed={args.seed})")
    return 0


def stats(_args: argparse.Namespace) -> int:
    def count(d: Path) -> int:
        return sum(1 for p in d.glob("*") if p.suffix.lower() in IMG_EXTS) if d.exists() else 0

    def show(label: str, base: Path) -> None:
        print(f"{label:22s} real={count(base / 'real'):>6} ai={count(base / 'ai'):>6}")

    show("pool", POOL)
    for s in ("train", "test"):
        show(s, DATA / s)
    heldout = DATA / "heldout"
    if heldout.exists():
        for d in sorted(p for p in heldout.iterdir() if p.is_dir()):
            show(f"heldout/{d.name}", d)
    if (DATA / "robustness").exists():
        show("robustness", DATA / "robustness")
    return 0


def merge(args: argparse.Namespace) -> int:
    """Fold another machine's pool export into this one. Hash-named files dedupe
    for free (identical picture -> identical filename), so this is copy-if-absent."""
    import shutil

    src_root = Path(args.src).resolve()
    dest_root = DATA / args.dest
    added = dup = 0
    for label in ("real", "ai"):
        src = src_root / label
        if not src.exists():
            continue
        dst = dest_root / label
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.glob("*"):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            target = dst / p.name
            if target.exists():
                dup += 1
            else:
                shutil.copy2(p, target)
                added += 1
    print(f"[merge {src_root} -> {args.dest}] copied {added}, skipped {dup} already-present")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dest", default="_pool", help="target under data/ (default _pool)")
    common.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)

    ing = sub.add_parser("ingest", parents=[common], help="add local folders/files into a dest")
    ing.add_argument("source")
    ing.add_argument("--base")
    ing.add_argument("--real", nargs="*")
    ing.add_argument("--ai", nargs="*")
    ing.add_argument("--real-glob", nargs="*", dest="real_glob")
    ing.add_argument("--ai-glob", nargs="*", dest="ai_glob")
    ing.set_defaults(func=ingest)

    hf = sub.add_parser("ingest-hf", parents=[common], help="add a Hugging Face parquet dataset")
    hf.add_argument("source")
    hf.add_argument("repo")
    hf.add_argument("--splits", nargs="+", default=["train"])
    hf.add_argument("--image-col", default="image")
    hf.add_argument("--label-col", default="label")
    hf.add_argument("--real-labels", nargs="*", default=[])
    hf.add_argument("--ai-labels", nargs="*", default=[])
    hf.add_argument("--all-real", action="store_true", help="whole repo is real (single-class)")
    hf.add_argument("--all-ai", action="store_true", help="whole repo is ai (single-class)")
    hf.add_argument("--stream", action="store_true")
    hf.add_argument("--limit-per-class", type=int, default=0)
    hf.add_argument("--num-workers", type=int, default=1, help="total laptops sharing the work")
    hf.add_argument("--worker-id", type=int, default=0, help="this laptop's index, 0..N-1")
    hf.add_argument("--jobs", type=int, default=0, help="re-encode processes (0 = all CPU cores)")
    hf.set_defaults(func=ingest_hf)

    ur = sub.add_parser("ingest-urls", parents=[common], help="download a single-class URL-list dataset")
    ur.add_argument("source")
    ur.add_argument("repo")
    ur.add_argument("--label", choices=["real", "ai"], required=True)
    ur.add_argument("--splits", nargs="+", default=["train"])
    ur.add_argument("--url-col", default="url")
    ur.add_argument("--limit-per-class", type=int, default=0)
    ur.add_argument("--num-workers", type=int, default=1)
    ur.add_argument("--worker-id", type=int, default=0)
    ur.add_argument("--jobs", type=int, default=16, help="concurrent download threads")
    ur.set_defaults(func=ingest_urls)

    mg = sub.add_parser("merge", parents=[common], help="fold another machine's pool export in")
    mg.add_argument("src", help="path to the other machine's pool dir (contains real/ and ai/)")
    mg.set_defaults(func=merge)

    sp = sub.add_parser("split", help="write train/test from the _pool")
    sp.add_argument("--ratio", type=float, default=0.8)
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(func=split)

    st = sub.add_parser("stats", help="count images in every dest")
    st.set_defaults(func=stats)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

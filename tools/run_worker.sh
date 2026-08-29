#!/usr/bin/env bash
# One laptop's share of the diverse real/ai pull.
#
# Usage:   tools/run_worker.sh <WHO> <NUM_WORKERS> [CAP_PER_CLASS_PER_SOURCE]
#   WHO         laptop name (brennen|travis|dylan|joe) or a raw index 0..N-1
#   NUM_WORKERS total laptops sharing the work
#   CAP         max images per class per source on THIS laptop (0 = take everything)
#
# Laptop roster:  brennen=0  travis=1  dylan=2  joe=3
#
# Rough total corpus size  ~=  15 class-slots  x  CAP  x  NUM_WORKERS
#   e.g. CAP=1500 on 4 laptops ~= 90k    CAP=4500 on 4 laptops ~= 270k
# (streamable ceiling of this mix is ~350-400k balanced; full GenImage 2.7M is
#  a 678GB split-zip that won't fit/shard, so it's intentionally not here.)
#
# Every laptop runs the SAME command but with its own name. Each pulls a
# disjoint set of shards, so the downloads run in parallel with zero overlap.
set -euo pipefail

WHO="${1:?laptop name (brennen|travis|dylan|joe) or index 0..N-1}"
N="${2:?num workers}"
CAP="${3:-0}"

case "$WHO" in
  brennen) WID=0 ;;
  travis)  WID=1 ;;
  dylan)   WID=2 ;;
  joe)     WID=3 ;;
  *)       WID="$WHO" ;;   # already a numeric index
esac

cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source venv/bin/activate
export HF_HUB_DISABLE_XET=1          # plain HTTPS; avoids the flaky Xet backend

CAPARG=()
[ "$CAP" != "0" ] && CAPARG=(--limit-per-class "$CAP")

run() {
  python tools/build_dataset.py ingest-hf "$@" \
    --stream --jobs 0 --num-workers "$N" --worker-id "$WID" "${CAPARG[@]}"
}

echo ">>> $WHO = worker $WID of $N  (cap=$CAP)  starting diverse pull"

# ---- REAL + the 8 GenImage generators (mixed repo, label-based) ----
run tinygenimage TheKernel01/Tiny-GenImage --splits train validation --real-labels 0 --ai-labels 1

# ---- SID_Set: OpenImages reals + modern synthetic (0=real, 1=synthetic; 2=tampered skipped) ----
run sidset saberzl/SID_Set --splits train validation --real-labels 0 --ai-labels 1

# ---- CIFAKE warm-up, 32x32 (labels REVERSED: 0=FAKE, 1=REAL) ----
run cifake dragonintelligence/CIFAKE-image-dataset --splits train test --real-labels 1 --ai-labels 0

# ---- REAL sources (single-class repos) ----
run bmreal  bitmind/bm-real    --splits train --all-real
run coco    bitmind/MS-COCO    --splits train --all-real
run ffhq    bitmind/ffhq-256   --splits train --all-real
run celeba  bitmind/celeb-a-hq --splits train --all-real

# ---- AI sources (single-class repos, modern generators) ----
run sdxl      bitmind/stable-diffusion-xl     --splits train --all-ai
run realvis   bitmind/realvis-xl              --splits train --all-ai
run mobius    bitmind/bm-mobius               --splits train --all-ai
run fluxceleb bitmind/celeb-a-hq___FLUX.1-dev --splits train --all-ai
run fluxffhq  bitmind/ffhq-256___FLUX.1-dev   --splits train --all-ai

echo ">>> worker $WID done. Pool is at data/_pool. Send it to the main laptop:"
echo "    rsync -a data/_pool/  <main-laptop>:~/Tik-Tok-Jam-2026/data/_pool/"
python tools/build_dataset.py stats

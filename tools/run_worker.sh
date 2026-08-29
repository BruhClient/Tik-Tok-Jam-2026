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
# CAP is the per-source ceiling on THIS laptop. Small curated sources self-limit
# (they give whatever they have, up to CAP); the two BULK sources dominate scale:
#   ELSA1M (AI, 1M embedded)  and  OpenImages (real, downloaded from URLs).
# So total across NUM_WORKERS laptops ~= (small sources' totals) + CAP*NUM_WORKERS*2.
#
#   CAP=15000  on 4 laptops  ~=  ~500-700k balanced   (~45-60 min)
#   CAP=60000  on 4 laptops  ~=  ~1.2M balanced        (~1.5h)
#   CAP=250000 on 4 laptops  ~=  ~2M balanced          (~2-4h, download-bound)
#
# OpenImages reals are URL downloads: expect ~20-30% to 404/timeout (handled).
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

# reals that ship as URL lists -> download them (I/O-bound, uses threads)
runurl() {
  python tools/build_dataset.py ingest-urls "$@" \
    --jobs 32 --num-workers "$N" --worker-id "$WID" "${CAPARG[@]}"
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

# ---- AI bulk: ELSA1M, 1M images across multiple diffusion models (embedded) ----
run elsa1m    elsaEU/ELSA1M_track1            --splits train --all-ai

# ---- REAL bulk: OpenImages V7 (URL list -> downloaded) to balance the AI bulk ----
runurl openimages bitmind/open-images-v7 --label real --splits train

echo ">>> worker $WID done. Pool is at data/_pool. Send it to the main laptop:"
echo "    rsync -a data/_pool/  <main-laptop>:~/Tik-Tok-Jam-2026/data/_pool/"
python tools/build_dataset.py stats

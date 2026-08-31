# AIGC Image Detector — TikTok Jam 2026

Detects AI-generated images, and measures how much of that detection survives
the things that actually happen to a picture on the way to a screen — JPEG
recompression, downscaling, blur, a screenshot of a repost.

**The deliverable is `detect.py`**: image directory in, `predictions.json` out.
`robustness.py` is the same model under a transform sweep, `calibrate.py` picks
the decision threshold for a false-positive budget, and `gui.py` is the same
pipeline behind a window. All four call into `app/`, so a number read in the
window is the number the CLI printed.

`training_process/` is the other half of the repo: the pipeline that produced
the model — dataset preparation, the degradation bank, CLIP feature caching,
head training, evaluation and the bundle export.

---

## Contents

- [AIGC Image Detector — TikTok Jam 2026](#aigc-image-detector--tiktok-jam-2026)
  - [Contents](#contents)
  - [Project overview](#project-overview)
  - [Setup and installation](#setup-and-installation)
  - [Commands](#commands)
    - [`detect.py`](#detectpy)
    - [`calibrate.py`](#calibratepy)
    - [`robustness.py`](#robustnesspy)
    - [`gui.py`](#guipy)
  - [What we trained](#what-we-trained)
    - [Datasets and their numbers](#datasets-and-their-numbers)
    - [The training objective](#the-training-objective)
    - [The shipped bundles](#the-shipped-bundles)
  - [Reproducing the results](#reproducing-the-results)
    - [A. Inference only (you have `models/bundle.pt`)](#a-inference-only-you-have-modelsbundlept)
    - [B. The full training pipeline, from scratch](#b-the-full-training-pipeline-from-scratch)
  - [Results](#results)
    - [On the sample set](#on-the-sample-set)
    - [Generalisation, by content category](#generalisation-by-content-category)
    - [Agreement between the training pipeline and this repo](#agreement-between-the-training-pipeline-and-this-repo)
    - [Known weak spots under degradation](#known-weak-spots-under-degradation)
  - [Output format](#output-format)
  - [The robustness sweep](#the-robustness-sweep)
  - [The window](#the-window)
  - [Repository layout](#repository-layout)
    - [Data folders](#data-folders)
  - [Reflection: what works, what does not, what I would do next](#reflection-what-works-what-does-not-what-i-would-do-next)
    - [What worked](#what-worked)
    - [Limitations, stated plainly](#limitations-stated-plainly)
    - [What I would improve given more time](#what-i-would-improve-given-more-time)
  - [Troubleshooting](#troubleshooting)
  - [Credits](#credits)

---

## Project overview

The problem: given a folder of images, return P(AI-generated) for each one, and
be honest about how far that number can be trusted.

The system is two halves that meet at a single file.

**The frozen half.** A stock `openai/clip-vit-large-patch14` vision tower
(~303M parameters) turns an image into a 768-d embedding. It is never
fine-tuned. Freezing is the design, not a shortcut: fine-tuning 303M parameters
on a corpus of this size would overfit away the general representation that
gives cross-generator robustness, and caching the embeddings once is what made
~20 training experiments affordable instead of three.

**The trained half.** A ~0.7M-parameter MLP head (768 → 512 → 256 → 1,
BatchNorm, dropout 0.3) is fitted on those cached embeddings, using a
worst-case-view objective over a bank of post-processing degradations.

The two travel together in one file, `models/bundle.pt` — tower config and
weights, head config and weights, feature standardisation, Platt coefficients,
and the calibrated operating point. That is what lets inference run with **no
network access** and no config that can drift out of sync with the weights.

Three ideas drove everything else:

1. **Shortcuts are the default failure, not overfitting.** Most public
   real-vs-AI datasets separate their classes on something other than
   generation — file format, image geometry, resolution history, subject
   matter. The failure is silent: validation accuracy goes *up*. Every
   `prepare_*.py` script in `training_process/` has an `--audit` mode that runs
   before the data is built.
2. **AUROC alone hides the deployment failure.** Under compression the score
   distribution slides, AUROC barely moves, and a threshold calibrated on clean
   data starts calling everything real. So both AUROC *and* accuracy at a fixed
   threshold are reported everywhere.
3. **The expensive error is calling a real photograph fake.** The shipped
   threshold is calibrated for 1% FPR on real images, not for peak accuracy.

---

## Setup and installation

Requires **Python 3.9+**. A CUDA GPU is used automatically when torch finds
one; CPU works and is the assumed case — CLIP ViT-L/14 scores roughly 5–15
images a second on CPU for training, but very slow for evaluation. 

```bash
git clone <this repo>
cd "Tik Tok Jam 2026"

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

`requirements.txt` covers inference and the GUI:

```
PyQt6>=6.5     numpy>=1.24     Pillow>=10.0
matplotlib>=3.7                scikit-learn>=1.3
torch>=2.0     transformers>=4.40
```

`transformers` is only used to *build* the CLIP tower — the weights ship inside
the bundle, so nothing is downloaded at run time.

**The model bundle is not in the repository.** At ~1.2 GB it is far past
GitHub's 100 MB per-file limit, and past the 1 GB of LFS storage a free account
gets, so it ships as a **GitHub Release asset** instead — no quota, no bandwidth
billing, and clones stay small. `models/*` is gitignored for that reason.

Download `bundle.pt` from the repository's
[Releases page](https://github.com/BruhClient/Tik-Tok-Jam-2026/releases) and put
it at `models/bundle.pt`:

```
models/
└── bundle.pt        # ~1.2 GB, from the latest release
```

Then confirm it is picked up — a backend with no `[no checkpoint …]` tag is
ready to run:

```bash
python detect.py --list-detectors
```

**Training additionally needs** (installed by `training_process/run_all.py`, or
by hand):

```bash
pip install torch torchvision transformers datasets pillow numpy scikit-learn matplotlib
```

**Verify the install** without any data or GPU:

```bash
cd training_process
python smoke_test.py            # syntax-checks every file, then runs the pipeline end to end
python gpu_testing.py           # diagnoses CUDA / driver / torch-build problems
```

`smoke_test.py` builds a tiny synthetic dataset with `Make_smoke_data.py` and
pushes it through embed → train → evaluate. It is a plumbing test, not a
model-quality test: it proves every script runs and writes its output.

---

## Commands

| | command | what you get |
| --- | --- | --- |
| **Score** | `python detect.py <dir>` | `predictions.json`, plus metrics if the folder is labeled |
| **Robustness** | `python robustness.py <dir>` | transform sweep → `robustness_report.json` (needs labels) |
| **Calibrate** | `python calibrate.py <dir>` | threshold recommendations per FPR budget (needs labels) |
| **Window** | `python gui.py [<dir> \| <preds.json>]` | upload, then insights or verdicts |

Everything prints progress to the terminal — in the GUI too.

### `detect.py`

```bash
python detect.py <dir>
python detect.py <dir> --best-threshold --report run_report.json
python detect.py <dir> --out results.json --weights models/bundle_cvar.pt
python detect.py <dir> --tta 4                   # average 4 views per image
python detect.py <dir> --require-labels          # fail if labels are missing
python detect.py --list-detectors
```

| flag | meaning |
| --- | --- |
| `--out, -o` | output JSON path (default `predictions.json`) |
| `--detector, -d` | registered backend name (default: the best one that has a checkpoint) |
| `--weights, -w` | checkpoint to load instead of the backend's default |
| `--threshold, -t` | decision threshold for the **printed** summary (default: the model's own operating point) |
| `--tta N` | test-time augmentation: average N views per image (1 = off; 2 adds a flip; up to 4). Slower, steadier |
| `--best-threshold` | also report the threshold that maximises F1 (labeled folders only) |
| `--require-labels` | exit 2 instead of silently falling back to scores-only |
| `--relative` | write paths relative to the input directory |
| `--report` | also write a metrics/timing report to a JSON path |
| `--quiet, -q` | suppress progress output |
| `--list-detectors` | print the registered backends and exit |

Exit codes: `0` success, `2` bad input (missing directory, unknown detector,
missing checkpoint, `--require-labels` with no labels).

Three backends are registered:

| `--detector` | what it is |
| --- | --- |
| `clip_head` | the real model — CLIP ViT-L/14 + trained MLP head, from `models/bundle.pt` |
| `ensemble` | several bundles over **one** shared tower; embeds once, runs every head, averages the *calibrated probabilities*. `--weights a.pt,b.pt` |
| `trained` | generic slot for any other TorchScript checkpoint at `models/model.pt` |

### `calibrate.py`

```bash
python calibrate.py <labeled_dir>
python calibrate.py <labeled_dir> --fpr 0.01 --out calibration.json
```

Scores a labeled folder and reports, side by side: the threshold the bundle
ships with, the lowest threshold that keeps FPR within 0.5 / 1 / 2 / 5 %, the
best-F1 threshold, and the best balanced-accuracy threshold. It never touches
the scores or refits the model — only the decision boundary, which is the only
lever that changes false positives without retraining.

### `robustness.py`

```bash
python robustness.py <dir>
python robustness.py <dir> --official
python robustness.py <dir> --transforms jpeg,blur --severities 1,3,5
python robustness.py <dir> --sample 400 --max-side 1024
python robustness.py --list-transforms
```

| flag | meaning |
| --- | --- |
| `--transforms` | comma-separated keys (default `jpeg,blur,rescale,crop,social`) |
| `--severities` | comma-separated levels 1–5 (default `1,2,3,4,5`) |
| `--official` | sweep the challenge's exact parameter table instead; writes `robustness_report_official.json` |
| `--sample` | images per cell, balanced across classes (default 200) |
| `--max-side` | decode cap in pixels (default 768) |
| `--tta N` | as above |
| `--detector, -d` / `--weights, -w` | as above |
| `--threshold, -t` | the **fixed** threshold accuracy is measured at |
| `--out, -o` | report path (default `<dir>/robustness_report.json`) |
| `--list-transforms` | print the transform keys and what each severity means |

### `gui.py`

```bash
python gui.py                     # start empty, pick a folder in the app
python gui.py <image_directory>   # score that folder on launch
python gui.py predictions.json    # open a finished result file
```

---

## What we trained

Only the head. The CLIP tower is frozen throughout.

### Datasets and their numbers

Every dataset below has a `prepare_*.py` in `training_process/` that pulls it,
audits it for shortcuts, and lays it out as `real/<source>/` + `fake/<generator>/`
— the structure `clipfeat.scan_dir()` reads, and the structure that gives
`train.py --holdout-group <generator>` something real to hold out.

| dataset | source | role | counts as configured | ###CHECK ON THIS
| --- | --- | --- | --- |
| **SID_Set** | `saberzl/SID_Set` (HF) — reals are OpenImages photos, fakes are full-synthetic | primary train / val / test | **25,000 per class** train · **3,000 per class** val · **4,000 per class** test (`--offset 25000`, so disjoint from train) |
| **SID_Set tampered** | same dataset, label 2 — real photos with a locally edited region + mask | **evaluation probe only, never trained on** | **2,000 per class** |
| **Kaggle AI-vs-Human** | `alessandrasala79/ai-vs-human-generated-dataset` — CSV manifest, `0=real / 1=AI` | reals + fakes | **3,000 per class**, 15% carved out for val |
| **Defactify / MS COCOAI** | `Rajarshi-Roy-research/Defactify_Image_Dataset` | semantically aligned pairs + per-generator labels | **3,000 per generator** train · **500 per generator** val, across **5 generators**: SD 3, SD 2.1, SDXL, DALL-E 3, MidJourney v6 |
| **MS COCO** | cocodataset.org | training reals **and** the benchmark reals, kept strictly apart | `train2017` **118k** images → training reals (capped, e.g. `--limit 6000`) · `val2017` **5k** → **held out as benchmark, never trained on** |
| **Unsplash Lite** | github.com/unsplash/datasets (TSV metadata + self-fetched images) | stylistic diversity in the *real* class | **~25k** rows available · **~3,000** downloaded, keyworded `portrait,person,food,city,animal` |
| **Generator pool** | flat local pool, split by `split_pool.py` | fakes by generator family | real **70,000** · SDXL **53,087** · FLUX_DEV **7,273** · FLUX_PRO **3,209** — capped to `--per-generator` (e.g. 3,000) so the fake set is not 83% SDXL |
| **Official validation benchmark** | COCO val2017 + DALL-E Advanced | held out; a preview of the hidden test set | **~4,998 real / ~8,843 fake** (≈1:1.77 — hence *balanced* accuracy, since calling everything fake scores 64%) |

**The corpus behind the shipped bundle** is a nine-group mix assembled from the
above: **21,400 training images — 7,150 real / 14,250 generated**. Reals came
from the original Kaggle set, Unsplash, and `ai-vs-human`; generated from a
GenImage-derived group, SDXL, FLUX_DEV, FLUX_PRO, DALL-E 3 and `ai-vs-human`.
Validation was held at exactly **50/50**, because the operating point is derived
from it and inherits any skew in it directly.

**The shortcut audits, and what they found.** Four kinds of shortcut turned up
across the candidate datasets, and each `prepare_*.py` documents the one it hit:

| shortcut | where it showed up | fix |
| --- | --- | --- |
| **Geometry** | SID_Set: every synthetic image is 1024×1024, reals are non-square OpenImages photos. `width == height ⇒ fake` scores **in the 90s** on the raw data | `--normalise-geometry`: random square crop + random resize applied to **both** classes (on by default in `prepare_sid.py`) |
| **Format** | PNG generations against JPEG photographs | every image re-encoded to JPEG on the way in (`augment.normalise_source`, random q85–98) |
| **Resolution history** | Defactify: reals are ~640×480 COCO, fakes are 1024×1024 SD/DALL-E | **not fixable at ingest** — equalising output size does not undo resampling history. Only a trusted external test set catches it |
| **Content distribution** | prompted, aesthetic subjects vs candid snapshots | only covered by mixing corpora — see the Unsplash result below |

**The two corpus findings that changed the model.** Both are recorded in the
scripts that implement them:

- *The real-image corpus dominates transfer far more than the synthetic side.*
  Four external datasets all failed to transfer, and all four differed on the
  **real** side: pool **0.555 AUROC**, COCOAI **0.762 / 0.787**, SID_Set
  **0.637**. The one merge that worked kept the original reals and borrowed only
  the fakes. That is why `setup_coco.py` exists — the organisers' benchmark
  draws its reals from COCO val2017, the same corpus that scored 0.787.
- *Stylistic diversity in the real class pays; corpus count does not.* Adding
  more conventional web photography (MS COCO, OpenImages) as reals made **every**
  content category worse. Adding ~3,000 professionally-shot Unsplash stock
  photographs made every one better: from **people 0.62, food 0.78, city 0.83,
  animals 0.87** up to the numbers in [Results](#results) — city **+0.157**,
  food **+0.192**, people **+0.199**, animals **+0.076** AUROC.

### The training objective

There is no generator network. What exists instead is a min-max game over the
**degradation space** — real adversarial training against the threat model
actually posed (post-processing), made cheap because the views are precomputed:

```
min_head  max_{t in T}  L(head(CLIP(t(x))), y)
```

The inner max is approximated by sampling *k* cached views per image and
backpropagating through the worst ones.

**`augment.py` keeps the banks strictly apart**, which is what makes the
robustness numbers mean something:

| bank | contents | used for |
| --- | --- | --- |
| `TRAIN_RANGES` | *continuous* ranges — jpeg q30–95, blur σ0–2.0, resize 0.25–1.0×, noise σ0–0.10, colour 0.80–1.20×, crop 0.75–1.0. Composed in physical order: crop → resize → colour → noise → blur → jpeg | **training only** |
| `EVAL_GRID` | the challenge's *discrete* settings — jpeg q90/70/50/30, blur σ0.5/1.0/2.0, resize 0.5×/0.25×, noise σ0.02/0.05/0.10, colour ±20%, crop 80% | **never trained on** → measures interpolation |
| `HELDOUT_GRID` | deliberately outside the training ranges — jpeg q20, blur σ3.0, resize 0.125×, noise σ0.15, crop 50%, **webp q50** (an unseen codec) | measures extrapolation |
| `CHAIN_GRID` | realistic redistribution chains — social repost, filtered share, thumbnail crop, low-light message, screenshot | the case that actually shows up |

**Training run defaults** (`train.py`):

| setting | value | why |
| --- | --- | --- |
| head | MLP 768 → 512 → 256 → 1, BatchNorm, dropout 0.3 | ~0.7M params on cached features |
| epochs / batch / lr / wd | 30 / 256 / 1e-3 / 1e-4, AdamW + cosine | seconds per run on cached features |
| views cached per image | **6** (view 0 always clean, 5 random degradation chains) | `embed.py --views 6` |
| `--adv-views k` | 3 sampled per image per step | the inner max |
| `--adv-mode` | **`cvar`** — mean of the worst half (vs `mean` = plain augmentation, `max` = worst view only) | |
| `--consistency` | 0.5 — pulls degraded-view logits toward the clean-view logit | |
| **model selection** | best **worst-view** AUROC, not clean AUROC | this is the whole point |
| `--max-fpr` | **0.01** — the threshold is the highest-recall point holding FPR on reals under 1% | |
| calibration | threshold **and** Platt fit on *pooled augmented* val scores, not clean ones | a threshold set on clean data drifts as compression shifts the distribution |

**The ablations that were run** (`run_all.py`, days 1–3): the geometry shortcut
(normalised vs raw), the min-max objective (`mean` / `cvar` / `max`),
augmentation off (`--adv-views 1 --consistency 0`), consistency off, head
capacity (`linear` vs `mlp`), the feature tap (`proj` vs `pooled`), and
preprocessing (`resize` vs `nativecrop`). Two are worth singling out:

- **`nativecrop` was tested and rejected.** The obvious hypothesis for the weak
  categories — that CLIP's 224px downsampling destroys the fine texture animal
  and nature subjects depend on — was tested by training on native-resolution
  224px crops instead of the whole image resized. It did not help animals, and
  it cost city and food ~0.065 AUROC each. Global scene composition carries more
  signal here than fine texture, which makes the resize an evidenced choice.
- **Leave-one-generator-out.** `--holdout-group FLUX_DEV` scored **0.8894**
  against 0.8963 / 0.8929 for generators that *were* in training — a 0.007 gap.
  Within the diffusion family, holding a generator out costs almost nothing.

### The shipped bundles

`export_bundle.py` writes plain data only — config dicts, state_dicts, numbers,
strings — so the result reloads with `weights_only=True` and needs no
HuggingFace cache and no network. Two bundles are present, and they are **not**
the same model; each carries its own calibration and its own operating point,
read out of the file at load time and printed in the run log:

| bundle | training run | raw logit threshold | platt a / b | **operating point (probability)** |
| --- | --- | --- | --- | --- |
| `models/bundle.pt` | `runs/k2m`, cvar, warm-started from `runs/dalle/head.pt` | −0.0296 | 1.3104 / 0.2205 | **0.545** |
| `models/bundle_cvar.pt` | `runs/cvar`, cvar, from scratch | 2.9944 | 1.2142 / −0.5958 | **0.954** |

Whatever sits at `models/bundle.pt` is the default everywhere. Point at the
other with `--weights models/bundle_cvar.pt`. Never hardcode an operating point
from this table into anything: read it from the bundle, which is what the app
does.

**Why the operating point must be read, never typed.** A bundle stores it as a
**raw logit**; everything user-facing is a probability. The conversion is *not*
`sigmoid(logit)` — it has to go through the bundle's own Platt coefficients:

```
P = sigmoid(platt_a * logit + platt_b)
```

For `bundle.pt` that is `sigmoid(1.3104 × −0.0296 + 0.2205) = 0.545`, where the
naive `sigmoid(−0.0296)` gives `0.493`. A 0.05 shift, in exactly the direction
that manufactures false positives, out of a system that otherwise looks entirely
plausible. `clip_head.py` does the full conversion at load and prints the result
as `operating point`.

**Constants pinned to the training pipeline.** Several values in
`app/detectors/clip_head.py` mirror `training_process/` and must not be "cleaned
up" independently of it — each is commented at the site with what it mirrors:

| constant / step | mirrors |
| --- | --- |
| `SOURCE_JPEG_Q = 92` | `augment.normalise_source` re-encoded **every** image before it was ever seen (random q85–98; 92 is the midpoint, pinned so inference is deterministic) |
| `subsampling=2` on that re-encode | `augment.op_jpeg` used 4:2:0 chroma subsampling |
| `RES = 224`, bicubic, centre crop | `clipfeat.py` preprocessing |
| `CLIP_MEAN` / `CLIP_STD` | CLIP's constants, **not** ImageNet's |
| `MAX_PIXELS = 200_000_000`, `Image.MAX_IMAGE_PIXELS = None` | the same decode cap the training loader used |
| `Head.net` layer indices | checkpoint keys are `net.0/1/4/5/8`, so the ReLU and Dropout must stay in place even though they hold no weights |

Scoring therefore reproduces training exactly, in this order:

```
JPEG re-encode at q92, 4:2:0
shortest side -> 224 (bicubic), centre crop 224
CLIP normalisation (not ImageNet — the constants differ)
tower -> pooler_output -> visual_projection       768-d
L2-normalise -> (x - mu) / sd
head -> logit -> sigmoid(platt_a * logit + platt_b)
```

---

## Reproducing the results

### A. Inference only (you have `models/bundle.pt`)

```bash
pip install -r requirements.txt

python detect.py <labeled_dir> --best-threshold --report run_report.json
python detect.py <labeled_dir> --weights models/bundle_cvar.pt --best-threshold

python robustness.py <labeled_dir>              # the graded in-house grid
python robustness.py <labeled_dir> --official   # the challenge's exact table

python calibrate.py <labeled_dir> --out calibration.json
python gui.py <labeled_dir>
```

A labeled folder is one holding `real/` and `ai/`. Labels are auto-detected in
this order, so that layout is a convention rather than a requirement:

1. **Subfolder** — `real/` vs `ai/` (also `authentic`, `natural`, `human`,
   `camera`, `genuine`, `0` / `aigc`, `fake`, `generated`, `synthetic`, `sd`,
   `midjourney`, `1`), matched at any depth, nearest folder wins.
2. **Manifest** — `labels.csv` / `labels.json` (also `manifest.*`,
   `ground_truth.csv`) in the root, with a path column (`image_path`, `path`,
   `file`, …) and a label column (`label`, `is_ai`, `y`, `target`, …).
3. **Filename prefix** — `real_*.jpg`, `ai_*.png`.

Anything unlabeled is still scored; a folder can be partly labeled, and the
metrics use the part that is. Any other folder is scored without metrics — that
is not an error. Pass `--require-labels` to make it one.

### B. The full training pipeline, from scratch

All commands run from `training_process/`. `run_all.py --day 1|2|3|all` executes
this whole sequence unattended, which is **not** how it should be used — Day 1's
audit is supposed to change what you do next.

```bash
cd training_process
pip install torch torchvision transformers datasets pillow numpy scikit-learn matplotlib
```

**Day 1 — plumbing and the shortcut check.**

```bash
# 1. Measure the shortcut BEFORE anything else (~2 min). This is a result.
python prepare_sid.py --audit --n 3000

# 2. Build the binary data. Geometry normalisation is ON by default.
python prepare_sid.py --split train      --out data/train --per-class 25000
python prepare_sid.py --split validation --out data/val   --per-class 3000
python prepare_sid.py --split train      --out data/test  --per-class 4000 --offset 25000

# 3. Tampered probe — evaluation only, never trained on.
python prepare_sid.py --split validation --out data/probe --per-class 2000 --tampered

# 4. Smoke test end to end on 100 images per class. Ignore the metrics.
python embed.py --data data/train --out cache/smoke_tr --views 2 --limit 100
python embed.py --data data/val   --out cache/smoke_va --views 2 --limit 100
python train.py --train cache/smoke_tr --val cache/smoke_va --out runs/smoke --epochs 3
python evaluate.py --data data/test --ckpt runs/smoke/head.pt --out reports/smoke --limit 50 --groups grid

# 5. ABLATION 1 — the geometry shortcut, quantified. The gap IS the finding.
python prepare_sid.py --split train      --out data/train_raw --per-class 3000 --no-normalise-geometry
python prepare_sid.py --split validation --out data/val_raw   --per-class 800  --no-normalise-geometry
python embed.py --data data/train_raw --out cache/raw_tr --views 2
python embed.py --data data/val_raw   --out cache/raw_va --views 2
python train.py --train cache/raw_tr --val cache/raw_va --out runs/raw --epochs 15
```

**Day 2 — the real model and robustness.**

```bash
# The only expensive step (~1h at 50k x 6). Everything after it is seconds.
python embed.py --data data/train --out cache/train --views 6 --num-workers 16
python embed.py --data data/val   --out cache/val   --views 6 --num-workers 16
python embed.py --data data/probe --out cache/probe --views 6 --num-workers 16

# ABLATION 3 — the min-max objective.
python train.py --train cache/train --val cache/val --out runs/mean --adv-mode mean
python train.py --train cache/train --val cache/val --out runs/cvar --adv-mode cvar
python train.py --train cache/train --val cache/val --out runs/max  --adv-mode max

# ABLATIONS 2 and 4 — augmentation and consistency.
python train.py --train cache/train --val cache/val --out runs/clean_only     --adv-views 1 --consistency 0
python train.py --train cache/train --val cache/val --out runs/no_consistency --adv-mode cvar --consistency 0

# ABLATION 5 — head capacity.
python train.py --train cache/train --val cache/val --out runs/linear --head linear

# Robustness grid on the winner.
python evaluate.py --data data/test --ckpt runs/cvar/head.pt --out reports/cvar
```

**Day 3 — the bonus probe and error analysis.**

```bash
python probe_tampered.py --probe data/probe --ckpt runs/cvar/head.pt \
                         --out reports/tampered --reference reports/cvar/scores.npz

python error_analysis.py --data data/test --ckpt runs/cvar/head.pt --out errors/test
# open errors/test/gallery.html and LOOK at the misses
```

**Adding the other corpora.** Each is audited first, then folded in:

```bash
# Kaggle AI-vs-Human (CSV manifest)
python prepare_kaggle.py --root <kaggle-path> --inspect
python prepare_kaggle.py --root <kaggle-path> --out data_kaggle2 --per-class 3000 --val-frac 0.15
python prepare_custom.py audit --train-root data_kaggle2/train

# Defactify / MS COCOAI (per-generator, semantically aligned)
python prepare_cocoai.py --audit --n 3000
python prepare_cocoai.py --split train      --out data_cocoai/train --per-generator 3000
python prepare_cocoai.py --split validation --out data_cocoai/val   --per-generator 500

# COCO — training reals and the benchmark, kept disjoint by the script itself
python setup_coco.py --download-val   --out-bench bench_coco
python setup_coco.py --download-train --out-train data_coco --limit 6000
python setup_coco.py --verify --out-train data_coco --out-bench bench_coco

# Unsplash reals (download the Lite TSVs from github.com/unsplash/datasets first)
python fetch_unsplash.py --tsv-dir <unsplash-lite> --inspect
python fetch_unsplash.py --tsv-dir <unsplash-lite> --out data_unsplash --limit 3000 \
                         --keywords portrait,person,food,city,animal

# A flat generator-separated pool, capped per generator
python split_pool.py --pool <pool-path> --out data_pool --per-generator 3000 --normalise-geometry

# Any local train/test tree with ai/ and real/ subfolders
python prepare_custom.py audit  --train-root <path>/train
python prepare_custom.py layout --train-root <path>/train --test-root <path>/test --out data --val-frac 0.15

# Sanity-check the loaders against your real data before spending GPU time
python data_loader.py --train-root <path>/train --test-root <path>/test
```

**Warm-starting rather than retraining** — how `bundle.pt` was made:

```bash
python train.py --train cache/train_k2 --val cache/val_k2 --out runs/k2m \
                --adv-mode cvar --init-from runs/dalle/head.pt
```

`--init-from` also **reuses that checkpoint's `mu`/`sd`** and discards the fresh
statistics from this run's data. Loading old weights while normalising with new
statistics silently feeds the warm-started head data in a different space than
it learned in, which is worse than starting fresh.

**Leave-one-generator-out:**

```bash
python train.py --train cache/train --val cache/val --out runs/logo_flux --holdout-group FLUX_DEV
```

**Threshold without retraining, and the final export:**

```bash
python recalibrate.py --val cache/val --ckpt runs/cvar/head.pt \
                      --out runs/cvar/head_balanced.pt --mode balanced   # or: acc | fpr

python benchmark_official.py --data bench --ckpt runs/cvar/head_balanced.pt
python benchmark_official.py --data bench --ckpt runs/cvar/head_balanced.pt --sweep

python export_bundle.py --ckpt runs/k2m/head.pt --out ../models/bundle.pt
```

`--sweep` re-derives the optimal threshold **on** the benchmark and reports what
you *would* have scored. That is a diagnostic — it separates "the model cannot
separate these classes" (AUROC low) from "the threshold is mis-set for this
distribution" (AUROC high, fixable). Reporting the swept number as a result
would be dishonest; report the deployed-threshold number.

---

## Results

### On the sample set

Measured with **`models/bundle_cvar.pt`** on a 200-image labeled folder
(100 real, 100 AI, clean):

| | |
| --- | --- |
| AUC | **0.921** |
| accuracy at that bundle's own threshold (`0.954`) | **86.0%** |
| recall (AI images caught) | **78%** |
| FPR (authentic images wrongly flagged) | **6%** |
| best-F1 threshold | `0.946` → **87.5%** accuracy |

That operating point trades about a point and a half of accuracy for the low
false-accusation rate it was calibrated for — which is the right trade for this
job. `models/bundle.pt`, the current default, is a later model and has not been
re-measured on this table.

### Generalisation, by content category

Benchmarked on content-category sets the head never trained on (~200 images
each, 50 generated / 150 real) plus a small ChatGPT set. Balanced accuracy is
the headline because those sets are 1:3 — labelling everything "real" scores 75%
raw.

| benchmark | AUROC | balanced acc | FPR at the deployed point |
| --- | --- | --- | --- |
| in-distribution test set | 0.9639 | — | — |
| city | 0.9885 | 0.9600 | 4.7% |
| food | 0.9844 | 0.9500 | 2.0% |
| animals | 0.9634 | 0.8786 | 18.9% |
| people | 0.8355 | 0.7510 | 24.5% |
| ChatGPT / GPT-4o | 0.7333 | 0.6278 | 44.4% |

**Recall is high everywhere; the false-positive rate is not.** Across those
categories recall stays between 76% and 98% — the model does catch generated
images. What swings by an order of magnitude is how many *authentic*
photographs it flags alongside them: 2% on food, 24.5% on people. No single
global threshold serves all four, which is what `calibrate.py` exists for.

**Cross-family generator transfer fails.** Within the diffusion family, holding
a generator out costs 0.007 AUROC. Across families it does not work at all:
GPT-4o images score 0.7333, and adding DALL·E 3 to training as the closest
available proxy did not fix it. Treat any generator architecturally unlike the
training set as **undetected until measured**.

The category sets are ~200 images and the ChatGPT set is 19, so differences
below about 0.03 AUROC are noise, and the ChatGPT figure establishes "not
working" rather than a number. Every number here is also a snapshot against
today's generators.

### Agreement between the training pipeline and this repo

The same `k2m` model scored through `training_process/evaluate.py` and through
`app/`:

| category | training-pipeline AUROC | this pipeline | delta |
| --- | --- | --- | --- |
| food | 0.9844 | 0.983 | 0.001 |
| animals | 0.9634 | 0.960 | 0.003 |
| city | 0.9885 | 0.990 | 0.002 |
| people | 0.8355 | 0.836 | 0.000 |

TPR and FPR matched exactly on food (0.920 / 0.020), including the
`subsampling=2` detail. The one deliberate divergence is `SOURCE_JPEG_Q`:
training drew a random quality in 85–98, this repo pins 92 so two scans of the
same image cannot disagree with each other. That is worth ≤0.004 AUROC, against
score margins of 3.2–6.0.

### Known weak spots under degradation

Three conditions stand out, and all three are high-FPR failures on inputs that
sit deliberately outside the training ranges:

| condition | accuracy | FPR |
| --- | --- | --- |
| heavy downscaling, 0.125× | 0.790 | — |
| thumbnail-then-crop chain | 0.800 | 31.9% |
| **WebP q50** — an unseen codec | 0.802 | 36.2% |

`rescale`, `social` and `webp` are the rows to read first in any sweep. That
grid was measured on the in-distribution test set only, so whether the weak
*content* categories degrade further under compression is unknown — running the
sweep on a labeled category folder is how to find out.

---

## Output format

`detect.py` writes what the problem statement asks for, plus a verdict that
makes the file readable on its own:

```json
[
  {"image_path": "C:/data/img_0001.jpg", "pred": 0.8731, "prediction": "fake"},
  {"image_path": "C:/data/img_0002.png", "pred": 0.0412, "prediction": "real"}
]
```

| field | what it is |
| --- | --- |
| `image_path` | absolute by default; relative to the input directory with `--relative` |
| `pred` | — P(AI-generated) in `[0, 1]`, rounded to 6 decimals |
| `prediction` | `"fake"` if the score is at or above the threshold, `"real"` below |

Every input image gets exactly one record, in sorted path order. An image that
cannot be decoded is still emitted — its score as `0.5`, the maximally
uncommitted value, and its `prediction` as `null` rather than a verdict, since
0.5 sits below the operating point and calling it `"real"` would report a failed
decode as an authentic photograph. Failures are counted and named in the
terminal summary rather than silently dropped, so the record count always
matches the file count.

**The scores are always raw.** The threshold moves `prediction` and nothing
else — `pred` is the same value whatever threshold is in
effect. That threshold is `--threshold` when given, otherwise the model's own
operating point; in the window it is the live slider value, so an export matches
what is on screen.

`--report` additionally writes a run report: dataset composition, label source,
detector, timing, and the full metrics block at the chosen threshold.

---

## The robustness sweep

`robustness.py` answers the question the accuracy number alone cannot: *does
this still work after the image has been through the internet?*

Each selected transform is applied **in memory** at five severities, the sample
is re-scored, and every cell is compared against a clean baseline measured
through the same pipeline. The sample is balanced across classes and seeded, so
two runs on the same folder are comparable.

| key | transform | severities 1 → 5 |
| --- | --- | --- |
| `jpeg` | JPEG recompression | q90 → q30 |
| `webp` | WebP recompression | q90 → q40 |
| `blur` | Gaussian blur | σ0.5 → σ3.0 |
| `rescale` | Downscale → upscale | 75% → 15% |
| `crop` | Center crop | keep 95% → keep 50% |
| `bright` | Brightness / contrast | ±5% → ±40% |
| `saturation` | Saturation shift | +15% → +100% |
| `noise` | Gaussian noise | σ2 → σ20 |
| `social` | Social repost combo (downscale + sharpen + JPEG) | pass 1 → pass 5 |
| `screenshot` | Screenshot resample (odd ratio + blur + re-encode) | level 1 → level 5 |

`--official` sweeps the challenge's exact table instead — JPEG q90/70/50/30,
blur σ0.5/1.0/2.0, resize 0.5×/0.25×, noise σ0.02/0.05/0.10, colour jitter ±20%,
centre crop 80% — so the report's cells line up one-to-one with the spec, and it
writes to `robustness_report_official.json` so a graded sweep is not clobbered.

Ordering matters and is deliberate: the detector's own `prepare_source()` runs
**before** the sweep's transform, because training conditioned the source first
and degraded second. Reversing them measures a pipeline the model was never
trained under.

The threat model is **incidental** degradation — what happens to a picture
between upload and screen. Nothing here is claimed against an adaptive adversary
optimising perturbations against this specific detector, which is a different
and much harder problem.

The report lands at `<dir>/robustness_report.json`, and `gui.py` picks up a
report sitting next to the data automatically.

---

## The window

Two screens.

**Screen 1 — upload.** You say what you are uploading, **Labelled data** or
**Unlabelled data**, then pick the folder — or drag it onto the window. Drop a
folder to score it, a `predictions.json` to open a finished run, or a handful of
loose images to use the folder they sit in; a folder is the unit this app
scores, so the whole of it is read either way. A background scan reports what is
actually in the folder and refuses to run a labelled job on one with no labels.
Choosing *Unlabelled data* ignores any labels that are there.

**Screen 2 — results.** Where it goes follows from that choice:

| you uploaded | you get |
| --- | --- |
| Labelled data | **Insights** — metric cards, score distribution, ROC and the confusion matrix — with **Images** (every prediction, filtered by all / FP / FN / AI / real, with a preview pane) and **Robustness** (transform picker, severity scale, degradation curve and cell table) behind header tabs |
| Unlabelled data | one verdict grid: every image badged AI or authentic, filterable, with no metrics — there is no truth to measure against |

The threshold slider sits in the header and re-reads every view live. **Best F1**
jumps to the F1-optimal threshold; **Reset** goes back to the model's own
operating point. **Export JSON** writes the same `predictions.json` the CLI
writes.

The one screen that shows unfinished state is the run itself, which gets a
working screen — loading a gigabyte-plus bundle off disk otherwise looks exactly
like a hang.

---

## Repository layout

```
detect.py                  CLI: any folder -> predictions.json (+ metrics)
robustness.py              CLI: transform sweep -> robustness_report.json
calibrate.py               CLI: threshold recommendations per FPR budget
gui.py                     the window: upload, insights, images, robustness
requirements.txt
models/                    bundle.pt (~1.2 GB) lives here - gitignored

app/                       INFERENCE
  runner.py                scan / load / score, with all terminal logging
  sweep.py                 robustness sweep core, shared by CLI and GUI
  workers.py               Qt threads so the window stays responsive
  dataset.py               directory scan + 3-way label inference
  metrics.py               accuracy/P/R/F1/AUC/AP/confusion/threshold search
  transforms.py            in-house grid x5 severities + the official grid
  export.py                predictions JSON/CSV + run report
  theme.py                 dark palette, Qt stylesheet, matplotlib rcParams
  detectors/
    base.py                plugin interface + registry
    clip_head.py           THE model: CLIP ViT-L/14 + trained MLP head
    ensemble.py            several bundles over one shared tower
    trained.py             generic slot for any other models/model.pt
  widgets/
    window.py              app shell: screens, header, threshold, run state
    upload.py              screen 1: declare the data, pick the folder
    loading.py             the working screen: spinner, phases, progress
    pages.py               the labeled results: insights, images, robustness
    gallery.py             the unlabeled result: a verdict per image
    table.py               predictions table model + score-bar delegate
    charts.py              histogram, ROC, PR, confusion, degradation
    components.py          cards, chips, badges, type scale

training_process/          TRAINING
  run_all.py               the whole pipeline, day 1 / 2 / 3
  smoke_test.py            syntax-check + end-to-end run on synthetic data
  Make_smoke_data.py       tiny synthetic dataset for that smoke test
  gpu_testing.py           CUDA / driver / torch-build diagnosis
  diagnose_weights.py      did CLIP's weights actually materialise on the GPU?

  prepare_sid.py           SID_Set  -> real/fake, with the geometry audit
  prepare_kaggle.py        Kaggle AI-vs-Human (CSV manifest)
  prepare_cocoai.py        Defactify / MS COCOAI, one folder per generator
  prepare_custom.py        any local train/test tree; `audit` and `layout`
  setup_coco.py            COCO train2017 reals + val2017 benchmark, disjoint
  fetch_unsplash.py        Unsplash TSV -> downloaded stock reals
  split_pool.py            flat generator pool -> capped, split, geometry-fixed
  data_loader.py           one call: raw folders -> train/val/test DataLoaders

  augment.py               the degradation bank: TRAIN_RANGES / EVAL_GRID /
                           HELDOUT_GRID / CHAIN_GRID
  clipfeat.py              CLIP preprocessing, ViewDataset, scan_dir, bundling
  embed.py                 cache CLIP features once, iterate on the head free
  train.py                 the head, the CVaR objective, Platt + threshold
  recalibrate.py           new threshold on the same scores, no retraining
  evaluate.py              robustness grid: AUROC and fixed-threshold accuracy
  benchmark_official.py    COCO val2017 + DALL-E Advanced, balanced accuracy
  probe_tampered.py        BONUS: locally-edited images, binned by region size
  error_analysis.py        every miss, as a self-contained HTML gallery
  export_bundle.py         head.pt + CLIP -> one portable bundle.pt
  detector.py              inference API used to validate the bundle
  load_bundle_example.py   the minimal load-and-score pattern
```

The dependency direction inside `app/` is one-way and worth preserving:
`widgets/` may import from `app/`, `app/` never imports from `widgets/`. That is
what keeps the CLI free of any Qt dependency — `detect.py`, `robustness.py` and
`calibrate.py` run fine on a machine where PyQt6 is not installed.

### Data folders

Both are gitignored — they are rebuilt, not cloned.

```
sample_data/               the folder the GUI offers by default: real/ + ai/
data/
├── train/  val/  test/    real/<source>/ + fake/<generator>/
└── probe/                 tampered images, evaluation only
cache/                     feats.npy + labels.npy + meta.json per split
runs/                      head.pt + history.json per training run
reports/                   evaluation grids
errors/                    error_analysis galleries
```

---

## Reflection: what works, what does not, what I would do next

### What worked

**Freezing the tower and caching the features was the right call under a
three-day deadline.** Encoding is the only expensive step (~1h for 50k × 6
views); after it, a head trains in seconds. That is what turned "we can afford
three experiments" into "we ran about twenty" — the geometry ablation, three
objective variants, augmentation and consistency ablations, head capacity,
feature tap, preprocessing, and leave-one-generator-out. Almost every finding
below exists because the experiment was cheap enough to actually run.

**Auditing before training caught real shortcuts.** `width == height ⇒ fake`
scores in the 90s on raw SID_Set. Had we trained on it and reported that, the
number would have looked excellent and meant nothing. Building `--audit` into
every `prepare_*.py` was the highest-leverage hour spent.

**Optimising for the worst view, not the clean one, is why the robustness
numbers hold up.** Model selection on worst-view AUROC and calibration on
*pooled augmented* validation scores — not clean ones — directly targets the
failure that sinks most detectors: AUROC holds under compression while accuracy
collapses because the threshold was set somewhere else.

**Separating train ranges from eval grids makes the numbers mean something.**
Training on continuous ranges and evaluating on the challenge's discrete grid
measures interpolation rather than memorisation; `HELDOUT_GRID` reports
extrapolation separately. When WebP q50 came in at 36.2% FPR, that was a real
measurement of an unseen codec, not a training artefact.

### Limitations, stated plainly

**The false-positive rate is the real problem, not recall.** Recall sits between
76% and 98% across content categories. FPR swings from 2% (food) to 24.5%
(people) at one global threshold. There is no single operating point that serves
all four domains, so any honest deployment needs per-domain calibration —
`calibrate.py` is the cheap version of that, not a solution to it.

**Cross-family generator transfer does not work.** Holding out FLUX_DEV costs
0.007 AUROC. GPT-4o images score 0.7333, and adding DALL·E 3 as the nearest
available proxy did not fix it. Any generator architecturally unlike the
training set should be assumed undetected until measured. The ChatGPT figure is
also only 19 images — it establishes "not working", not a number.

**The people and animals weakness is unexplained.** The obvious hypothesis was
tested and rejected: training on native-resolution crops instead of the resized
whole image did not help animals and cost city and food ~0.065 AUROC each. We
know it is not a resolution problem. We do not know what it is.

**The real-image corpus dominates transfer, and we only partly solved it.** Four
external datasets all failed to transfer (0.555, 0.637, 0.762, 0.787 AUROC), and
all four differed on the *real* side. Adding conventional web photography made
every category worse; adding professionally-shot stock made every one better.
That is a useful direction, but ~3,000 Unsplash images is a small patch on a
distribution problem, and it may simply be another distribution the model has
now overfitted to.

**A single held-out set is not a neutral arbiter.** One dataset scored 0.6246
standalone against its in-distribution test set and was rejected — then produced
the best model in the project (animals +0.076 AUROC) when judged on the content
categories instead. That test set was narrow in *generator*; the category sets
are narrow in *content*. Neither alone decides a dataset, and we have no
evaluation axis we fully trust.

**Every number here is a snapshot.** Category sets are ~200 images, so
differences below ~0.03 AUROC are noise. The robustness grid was measured on the
in-distribution test set only — whether the weak content categories degrade
*further* under compression is untested. And the whole thing is measured against
today's generators.

**Nothing is claimed against an adaptive adversary.** The threat model is
incidental degradation: what happens to a picture between upload and screen. A
perturbation optimised against this specific detector is a different and much
harder problem, and we have not touched it.

**Tampered images are a different task and mostly fail.** The head is trained
binary — real vs *fully* synthetic. Locally edited real images were never
trained on, and because CLIP resizes to 224×224, a 3% edited patch becomes
roughly 7×7 pixels. `probe_tampered.py` measures where that boundary sits rather
than pretending it works.

### What I would improve given more time

1. **Per-domain calibration as a first-class feature.** A lightweight content
   classifier in front of the head, routing to a per-domain threshold, would
   directly attack the 2%-to-24.5% FPR spread — which is a bigger real-world win
   than any AUROC point available elsewhere.
2. **Diagnose the people/animals gap properly.** Native-resolution crops were
   the one hypothesis tested. Next I would run `error_analysis.py` at scale on
   those two categories and actually look at the misses, then check whether the
   CLIP embedding itself separates them (a linear probe on frozen features
   against a face- and animal-heavy set) before assuming the head is at fault.
3. **Broaden the real class deliberately, not by volume.** The Unsplash result
   says stylistic diversity pays and corpus count does not. I would sample reals
   across *acquisition* axes — phone vs DSLR, flash vs ambient, heavily filtered
   vs untouched, screenshotted vs original — rather than adding another web
   photography corpus.
4. **Close the cross-family gap, or scope the claim.** Either add genuinely
   non-diffusion generators (autoregressive, GAN, GPT-4o-class) to training and
   re-run the leave-one-family-out test, or ship an explicit out-of-distribution
   detector that abstains rather than returning a confident 0.73-AUROC guess.
5. **Extend the robustness grid to the weak categories.** The whole grid was run
   on the in-distribution test set. Running it per content category is cheap and
   would tell us whether people and animals fail *worse* under compression —
   likely the most under-measured risk in the system.
6. **Ensemble properly.** `ensemble.py` already runs several heads over one
   shared tower for a few extra matmuls, but the members are two bundles from the
   same pipeline. Training deliberately decorrelated heads — different seeds,
   different corpus slices, different feature taps — is nearly free given the
   cached features and has not been tried.
7. **Get a trustworthy evaluation axis.** Every conclusion above is limited by
   ~200-image category sets and a 19-image ChatGPT set. Before tuning anything
   further I would build a larger, multi-generator, multi-content held-out set
   and re-verify which of these findings survive it.

---

## Troubleshooting

**The very first run after a clone fails at step 2/5.**

```
[2/5] loading detector
      CLIP ViT-L/14 + MLP head
      weights ...\models\bundle.pt
error: CLIP ViT-L/14 + MLP head needs a checkpoint, and none is at
       ...\models\bundle.pt
       put one there, pass --weights <file>, or pick another backend with --detector.
```

This is expected, not a bug. The bundle is ~1.2 GB, and `.gitignore` excludes it
(`models/*`, with only `models/.gitkeep` tracked), so a clone gets an empty
`models/` directory — the weights ship as a
[Release asset](https://github.com/BruhClient/Tik-Tok-Jam-2026/releases), not in
git. Download it to `models/bundle.pt`, or leave it wherever it lives and point
at it: `python detect.py <dir> --weights D:/somewhere/bundle.pt`.

The name in the message is the backend's *display* name
(`CLIP ViT-L/14 + MLP head`), not the value you would pass to `--detector`
(`clip_head`). `python detect.py --list-detectors` prints both, and tags every
backend that has no usable checkpoint:

```
clip_head    CLIP ViT-L/14 + MLP head
ensemble     CLIP tower + head ensemble  [no checkpoint at ]
trained      Trained model               [no checkpoint at models\model.pt]
default: clip_head
```

**Everything scores as AI, or everything scores as real.** Check the threshold.
The calibrated operating point is not `0.5` — it is whatever the bundle carries
(`0.545` for `bundle.pt`, `0.954` for `bundle_cvar.pt`), and the run log prints
it. If you passed `--threshold 0.5` you are asking a different question. Run
`--best-threshold`, or `calibrate.py`, to see where it should actually sit.

**High AUROC but low accuracy and low recall, with near-zero FPR.** That is the
1%-FPR operating point doing exactly what it was calibrated to do, not a
modelling failure. The fix is a different threshold on the *same* scores:
`calibrate.py` at inference, or `recalibrate.py --mode balanced` to bake it into
a checkpoint.

**`no labels found - scores only`** on a folder you believe is labeled. The
scanner tried subfolders, then a manifest, then filename prefixes. Check the
folder names against the token list above, or pass `--require-labels` to make it
an error rather than a shrug.

**Some images written as `0.5`.** Those failed to decode. The count and the first
five names print in the run summary; they are emitted anyway so the record count
matches the file count.

**The sweep refuses to run.** It measures accuracy, so it needs ground truth. An
unlabeled folder has nothing to be right or wrong about.

**Unicode errors on a Windows console.** Handled — `runner.py` reconfigures
stdout/stderr to UTF-8 at import, because a cp1252 console cannot encode a σ in a
transform label, let alone a CJK filename, and would otherwise kill a run
mid-way.

**Near-chance embeddings on GPU while CPU works.** Run
`training_process/diagnose_weights.py`. fp16 autocast and TF32 have both been
ruled out (three runs at different precisions gave *identical* AUROC
trajectories, and real precision differences always wobble the numbers). The
remaining candidate is `low_cpu_mem_usage=True` leaving parameters on the meta
device — the script checks for that directly, and also whether two clearly
different images produce meaningfully different embeddings. `embed.py --no-amp`
forces fp32 as a comparison.

**CUDA not detected at all.** `training_process/gpu_testing.py` distinguishes a
CPU-only torch wheel (`torch.version.cuda is None`) from a driver problem, and
runs a real matmul to confirm compiled kernels exist for the GPU's compute
capability.

---

## Credits

- **Vision tower** — `openai/clip-vit-large-patch14`, frozen, redistributed
  inside the bundle.
- **Datasets** — SID_Set (`saberzl/SID_Set`), Kaggle AI-vs-Human
  (`alessandrasala79/ai-vs-human-generated-dataset`), Defactify / MS COCOAI
  (`Rajarshi-Roy-research/Defactify_Image_Dataset`), MS COCO train2017 /
  val2017, Unsplash Lite, and a generator pool covering SDXL, FLUX_DEV and
  FLUX_PRO. Each remains under its own licence; none is redistributed here.
- **This repository** — the training pipeline (`training_process/`), inference
  and evaluation (`app/`), the robustness lab, and the window.

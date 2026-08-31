# AIGC Image Detector

Detects AI-generated images, and measures how much of that detection survives
the things that actually happen to a picture on the way to your screen — JPEG
recompression, downscaling, blur, a screenshot of a repost.

**The deliverable is `detect.py`**: image directory in, `predictions.json` out.
`robustness.py` is the same model under a transform sweep. `gui.py` is the same
pipeline behind a window. All three call into `app/`, so a number you read in
the window is the number the CLI printed.

There is **no training step in this repository**. The model is trained
elsewhere (see [Model provenance](#model-provenance)); this scores images with
it and tells you how far you can trust the answer. What this repo does own on
the data side is the corpus the training consumed — see
[Building the corpus](#building-the-corpus).

---

## Contents

- [Quick start](#quick-start)
- [Commands](#commands)
- [Model provenance](#model-provenance)
- [Building the corpus](#building-the-corpus)
- [The model](#the-model)
- [Where the model works and where it stops](#where-the-model-works-and-where-it-stops)
- [Two kinds of input](#two-kinds-of-input)
- [Output format](#output-format)
- [The robustness sweep](#the-robustness-sweep)
- [The window](#the-window)
- [Results on the sample set](#results-on-the-sample-set)
- [Plugging in a different model](#plugging-in-a-different-model)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)

---

## Quick start

```bash
pip install -r requirements.txt

# put the trained bundle here (see "Model provenance")
#   models/bundle.pt

python detect.py sample_data                 # -> predictions.json + metrics
python robustness.py sample_data             # -> robustness_report.json
python gui.py                                # the window
```

Requires Python 3.9+. A CUDA GPU is used automatically when torch finds one;
CPU works and is the assumed case — CLIP ViT-L/14 scores roughly 5–15 images a
second on a modern laptop CPU.

---

## Commands

| | command | what you get |
| --- | --- | --- |
| **Score** | `python detect.py <dir>` | `predictions.json`, plus metrics if the folder is labeled |
| **Robustness** | `python robustness.py <dir>` | transform sweep → `robustness_report.json` (needs labels) |
| **Window** | `python gui.py [<dir> \| <preds.json>]` | upload, then insights or verdicts |

Everything prints progress to the terminal — in the GUI too. The window shows
finished results; the one exception is the run itself, which gets a working
screen, because loading a gigabyte-plus bundle off disk otherwise looks exactly
like a hang.

### `detect.py`

```bash
python detect.py <dir>
python detect.py <dir> --best-threshold --report run_report.json
python detect.py <dir> --out results.json --weights models/bundle_cvar.pt
python detect.py <dir> --require-labels          # fail if labels are missing
python detect.py --list-detectors
```

| flag | meaning |
| --- | --- |
| `--out, -o` | output JSON path (default `predictions.json`) |
| `--detector, -d` | registered backend name (default: the best one that has a checkpoint) |
| `--weights, -w` | checkpoint to load instead of the backend's default |
| `--threshold, -t` | decision threshold for the **printed** summary (default: the model's own operating point) |
| `--best-threshold` | also report the threshold that maximises F1 (labeled folders only) |
| `--require-labels` | exit 2 instead of silently falling back to scores-only |
| `--relative` | write paths relative to the input directory |
| `--report` | also write a metrics/timing report to a JSON path |
| `--quiet, -q` | suppress progress output |
| `--list-detectors` | print the registered backends and exit |

Exit codes: `0` success, `2` bad input (missing directory, unknown detector,
missing checkpoint, `--require-labels` with no labels).

### `robustness.py`

```bash
python robustness.py <dir>
python robustness.py <dir> --transforms jpeg,blur --severities 1,3,5
python robustness.py <dir> --sample 400 --max-side 1024
python robustness.py --list-transforms
```

| flag | meaning |
| --- | --- |
| `--transforms` | comma-separated keys (default `jpeg,blur,rescale,crop,social`) |
| `--severities` | comma-separated levels 1–5 (default `1,2,3,4,5`) |
| `--sample` | images per cell, balanced across classes (default 200) |
| `--max-side` | decode cap in pixels (default 768) |
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

On the upload screen the folder can be dragged onto the window instead of
browsed for. Drop a folder to score it, a `predictions.json` to open a finished
run, or a handful of loose images to use the folder they sit in - a folder is
the unit this app scores, so the whole of it is read either way. Declaring
labelled or unlabelled first still gates the folder, exactly as it gates the
Browse button.

---

## Model provenance

The detector shipped here is **not trained in this repository**. The training
pipeline — feature extraction (`clipfeat.py`), the augmentation stack
(`augment.py`), the CVaR head objective and the calibration fit — lives in
Joe's upstream repository, and this project consumes its output.

> **Upstream training repository:** _add the link to Joe's repo here_

**What upstream actually trains is a head, and nothing else.** The CLIP
ViT-L/14 vision tower (~303M parameters) is frozen; a ~0.7M-parameter MLP
(768 -> 512 -> 256 -> 1, BatchNorm, dropout 0.3) is fitted on its cached
embeddings. Freezing is the point, not a shortcut: fine-tuning 303M parameters
on ~21k images would overfit away the general representation that gives
cross-generator robustness, and caching the embeddings once is what made ~20
training experiments affordable instead of three.

**The "adversarial" half is a degradation sampler, not a generator.** Upstream
draws k cached views per image from the post-processing space
(crop -> resize -> colour -> noise -> blur -> JPEG) and backpropagates through
the worst-scoring ones (`--adv-mode cvar`), then early-stops on the *worst*
view's AUROC rather than the clean one. `robustness.py` in this repo measures
exactly what that objective was built for, which is why the two belong
together.

The current upstream checkpoint is `runs/k2m/head_bal.pt`: 21,400 training
images (7,150 real / 14,250 generated) across nine source groups, with
validation held at exactly 50/50 because the operating point is derived from it
and inherits any skew in it directly.

What crosses the boundary is a single file: `models/bundle.pt`. Everything the
detector needs travels inside it (tower config and weights, head config and
weights, feature standardisation, Platt coefficients, operating point), which
is what lets this repo run with **no network access at inference time** and no
config that can drift out of sync with the weights.

Several constants in `app/detectors/clip_head.py` are pinned to that training
pipeline and must not be "cleaned up" independently of it — each one is
commented at the site with what it mirrors upstream:

| constant / step | mirrors upstream |
| --- | --- |
| `SOURCE_JPEG_Q = 92` | `augment.normalise_source` re-encoded **every** image before it was ever seen |
| `subsampling=2` on that re-encode | `augment.op_jpeg` used 4:2:0 chroma subsampling |
| `RES = 224`, bicubic, centre crop | `clipfeat.py` preprocessing |
| `CLIP_MEAN` / `CLIP_STD` | CLIP's constants, **not** ImageNet's |
| `MAX_PIXELS`, `Image.MAX_IMAGE_PIXELS = None` | the same decode cap the training loader used |
| `Head.net` layer indices | the checkpoint keys are `net.0/1/4/5/8`, so the ReLU and Dropout must stay in place even though they hold no weights |

Two bundles are present, and they are **not** the same model — each carries its
own calibration and its own operating point, read straight out of the file at
load time and printed in the run log:

| bundle | operating point | notes |
| --- | --- | --- |
| `models/bundle.pt` | `0.780` | the default — whatever sits at this path wins |
| `models/bundle_cvar.pt` | `0.954` | the CVaR/1%-FPR calibration the figures below were measured on |

Point at either with `--weights models/bundle_cvar.pt`. Never hardcode an
operating point from this table into anything: read it from the bundle, which is
what the app does.

**Why it has to be read rather than typed.** A bundle stores its operating
point as a **raw logit**; everything user-facing here is a probability. The
conversion is not `sigmoid(logit)` — it has to go through the bundle's own
Platt coefficients:

```
P = sigmoid(platt_a * logit + platt_b)
```

On upstream's `k2m` model that is `sigmoid(1.3104 * -0.0296 + 0.2205) = 0.545`,
where the naive `sigmoid(-0.0296)` gives `0.493`. A 0.05 shift, in exactly the
direction that manufactures false positives, out of a system that otherwise
looks entirely plausible. `clip_head.py` does the full conversion at load and
prints the result — that is the number the run log calls `operating point`.

---

## Building the corpus

The weights come from Joe's repo, but the corpus they were trained on is built
here, by `tools/build_dataset.py`. It pools a dozen public datasets into one
deduplicated `real/` + `ai/` tree and then splits it.

```bash
# local folders
python tools/build_dataset.py ingest cifake \
    --real "downloads/cifake/**/REAL" --ai "downloads/cifake/**/FAKE"

# a Hugging Face parquet dataset, streamed and capped, kept out of training
python tools/build_dataset.py ingest-hf openfake ComplexDataLab/OpenFake \
    --splits test --stream --limit-per-class 2000 --dest heldout/openfake \
    --real-labels real 0 --ai-labels fake 1

python tools/build_dataset.py split --ratio 0.8 --seed 0
python tools/build_dataset.py stats
```

| subcommand | what it does |
| --- | --- |
| `ingest` | pull in local folders or globs |
| `ingest-hf` | pull a Hugging Face parquet dataset (images embedded as bytes) |
| `ingest-urls` | download a single-class dataset that ships URLs, not images |
| `merge` | fold another machine's pool export in |
| `split` | write `data/{train,test}/{real,ai}` from the pool |
| `stats` | count every destination |

Three properties of the build matter more than the source list:

**Every image is re-encoded to one JPEG quality on the way in** (`--jpeg-quality`,
default 90). This is the same defence the detector's `prepare_source()` applies
at inference: if reals arrive as JPEG and generated images as PNG, a model
learns the container instead of the task. Normalising at ingest means the
per-class compression tell is gone before anything is trained on it.

**Deduplication is global and by pixel content, not by file.** Images are hashed
after decoding, so the same picture re-encoded twice still collides, and the
hash becomes the filename — which makes writes idempotent, lets parallel
workers dedupe with no shared state, and reduces `merge` to copy-if-absent. One
image lands in exactly one destination across every source.

**Destinations keep the evaluation axes apart**, which is what stops a
generalisation claim from being circular:

| `--dest` | role |
| --- | --- |
| `_pool` (default) | the training corpus — `split` turns it into `train`/`test` |
| `heldout/<name>` | generalisation check, **never trained on**; used directly as a test set |
| `robustness` | platform-degradation validation |

A held-out generator only means something if it was never in the pool, so the
separation is enforced at ingest time rather than remembered later.

### The shortcuts this is defending against

Upstream audited every candidate dataset before training on it and found that
**four of five contained a shortcut** — some property that separates the two
classes for reasons unrelated to generation. The failure is silent: validation
accuracy goes *up*, not down. Four kinds turned up, and they map onto what an
ingest step can and cannot fix:

| shortcut | how it showed up | fixed at ingest? |
| --- | --- | --- |
| **Format** — PNG generations vs JPEG photographs | one set was 100% PNG fakes against 100% JPEG reals | **yes** — the single re-encode quality above |
| **Geometry** — generators emit squares, cameras do not | `width == height` alone scored 98.6% on one dataset | upstream only, by random square crop + resize on *both* classes |
| **Resolution history** — 256px reals against 1024px fakes | an upscaled 256px image stays soft; a downscaled 1024px one does not | **no** — equalising the output size does not undo the resampling |
| **Content distribution** — prompted, aesthetic subjects vs candid snapshots | one real corpus becomes a proxy for "real" | **no** — only covered, by mixing corpora |

The third is the cautionary one. A 130k-image pool passed its audit, trained
cleanly, and scored **0.5547 AUROC — chance** on a held-out set from a different
source. *Equalising a measurable property does not equalise the process that
produced it*, and only a test set from an unrelated source catches the
difference. That is why `heldout/` is a separate ingest destination here rather
than a slice taken out of the pool afterwards.

The fourth has a positive result attached, and it is worth reading before adding
another source to `run_worker.sh`. Adding *more conventional web photography* as
reals (MS COCO, OpenImages) made every content category worse; adding ~3,000
professionally-shot stock photographs made every one better — city +0.157, food
+0.192, people +0.199 AUROC. It is **stylistic** diversity in the real class
that pays, not corpus count. More of the same kind of real image only deepens a
region already occupied.

One methodological warning comes with it. Upstream rejected a dataset for
scoring 0.6246 standalone against its in-distribution test set, then reinstated
it when judged on the content categories instead — where it produced the best
model in the project (animals +0.076 AUROC). A single held-out set is not a
neutral arbiter if its own composition is narrow along an axis you are not
controlling for: that test set was narrow in *generator*, the category sets are
narrow in *content*. Neither alone decides a dataset.

### Pulling it in parallel

`tools/run_worker.sh` is the roster version of the above: every laptop runs the
same command under its own name, takes a disjoint round-robin slice of each
dataset's shards, and pulls with no overlap.

```bash
tools/run_worker.sh <brennen|travis|dylan|joe> <num_workers> [cap_per_class_per_source]
tools/run_worker.sh travis 4 15000
```

It walks the full source list — Tiny-GenImage and SID_Set for a labeled mix,
CIFAKE as a warm-up, `bitmind` reals (MS-COCO, FFHQ, CelebA-HQ) against
`bitmind` generators (SDXL, RealVis-XL, Mobius, FLUX.1-dev), then the two bulk
sources that actually set the scale: **ELSA1M** for AI and **OpenImages V7**
for real. Roughly 500–700k balanced images across four laptops at `CAP=15000`.
OpenImages reals are URL downloads, so expect 20–30% to 404 or time out; that
is handled and counted, not fatal. Each worker finishes by printing the `rsync`
line that sends its pool to the main machine, where `merge` folds it in.

`tools/predict_dir.py` is a headless `detect.py` against the same registry and
writer — useful on a worker box with no display.

> **Note on `CIFAKE`:** its labels are reversed relative to every other source
> (`0=FAKE`, `1=REAL`), which is why `run_worker.sh` passes
> `--real-labels 1 --ai-labels 0` for that one repo and nowhere else.

---

## The model

`app/detectors/clip_head.py` is the real backend: a **frozen CLIP ViT-L/14**
vision tower plus a **small MLP head** trained on top of its embeddings. Only
the head was trained — the tower is stock `openai/clip-vit-large-patch14`,
loaded from the bundle rather than downloaded.

Put the bundle at `models/bundle.pt` and it becomes the default backend
everywhere — CLI, GUI and sweep — with nothing to flip. Point elsewhere with
`--weights <file>`.

Scoring reproduces the training pipeline exactly, in this order:

```
JPEG re-encode at q92, 4:2:0                      <- not an optimisation; see below
shortest side -> 224 (bicubic), centre crop 224
CLIP normalisation (not ImageNet — the constants differ)
tower -> pooler_output -> visual_projection       768-d
L2-normalise -> (x - mu) / sd
head -> logit -> sigmoid(platt_a * logit + platt_b)
```

**Why the JPEG re-encode is first.** Training re-encoded every image, of both
classes, before the model ever saw it — otherwise the head learns "real photos
arrive as JPEG, generated ones arrive as PNG" instead of learning the task.
`mu`, `sd`, the Platt coefficients and the threshold were therefore all fitted
on re-encoded features. A pristine PNG scored without that step is
off-distribution. The midpoint quality (92) is used rather than a random draw
in 85–98, so inference is deterministic.

**The threshold is not 0.5.** Training used `pos_weight` and a CVaR objective,
both of which distort the output scale, so the head's raw logits sit high. The
Platt coefficients — fitted on pooled augmented validation scores — make the
output readable as a probability, and the bundle carries the operating point it
was calibrated for (`0.954` for `bundle_cvar.pt`, chosen for **1% FPR on real
images**; `0.780` for `bundle.pt`). `detect.py` and the window both read it out
of the file and adopt it automatically — the run log prints it as
`operating point 0.780` — **Reset** on the slider goes back to it, and
`--threshold` overrides it. **The JSON is always raw scores either way** — the
threshold only ever changes what gets *printed* or *drawn*.

**What the architecture can and cannot see.** CLIP resizes to 224 before the
patch embedding, which is why these features survive JPEG and blur so well and
why they cannot see subtle resampling traces. The flip side shows up in the
sweep: heavy compression shifts *both* classes' scores upward, so a fixed
threshold drifts even where AUROC holds. Read the Robustness page with that in
mind.

---

## Where the model works and where it stops

Upstream benchmarks the head on content-category sets it never trained on
(~200 images each, 50 generated / 150 real) plus a small ChatGPT set. Balanced
accuracy is the headline because those sets are 1:3 — labelling everything
"real" scores 75% raw.

| benchmark | AUROC | balanced acc | FPR at the deployed point |
| --- | --- | --- | --- |
| in-distribution test set | 0.9639 | — | — |
| city | 0.9885 | 0.9600 | 4.7% |
| food | 0.9844 | 0.9500 | 2.0% |
| animals | 0.9634 | 0.8786 | 18.9% |
| people | 0.8355 | 0.7510 | 24.5% |
| ChatGPT / GPT-4o | 0.7333 | 0.6278 | 44.4% |

Three things there change how you should read anything this repo prints.

**Recall is high everywhere; the false-positive rate is not.** Across those
categories recall stays between 76% and 98% — the model does catch generated
images. What swings by an order of magnitude is how many *authentic*
photographs it flags alongside them: 2% on food, 24.5% on people. No single
global threshold serves all four, so the honest deployment answer is per-domain
calibration — and on a labeled folder `--best-threshold` is the cheap version
of it.

**Cross-family generator transfer fails.** Holding a whole generator out of
training costs almost nothing *within* the diffusion family: held-out FLUX_DEV
scored 0.8894 against 0.8963 / 0.8929 for generators that were in training, a
0.007 gap. Across families it does not work at all. GPT-4o images score 0.7333
AUROC, and adding DALL·E 3 to training as the closest available proxy did not
fix it. Treat any generator architecturally unlike the training set as
**undetected until measured**. (That ChatGPT figure is 19 images; it
establishes "not working", not a number.)

**The weak categories are not a resolution problem.** The obvious hypothesis —
that CLIP's 224px downsampling destroys the fine texture animal and nature
subjects depend on — was tested directly upstream, by training a full model on
native-resolution 224px crops instead of the whole image resized. It did not
help animals, and it cost city and food ~0.065 AUROC each. Global scene
composition carries more signal here than fine texture, which makes the resize
in `clip_head.py` an evidenced choice rather than an unexamined default. The
animals and people weakness itself remains unexplained.

The category sets are ~200 images and the ChatGPT set is 19, so differences
below about 0.03 AUROC are noise. Every number is also a snapshot against
today's generators.

---

## Two kinds of input

**From the terminal the folder decides**; in the window you say which you are
uploading and it holds you to it. Either way the two inputs are the same.

**Labeled** — one folder holding `real/` and `ai/`:

```
my_data/
├── real/    authentic photographs
└── ai/      generated images
```

Both classes are pooled into a single evaluation set, scored by the same model
at the same threshold, and you get accuracy, AUC, F1, FPR, the confusion counts
and the charts. `--best-threshold` also reports where F1 peaks, which is
usually not 0.50.

**Unlabeled** — any other folder. Scores only; there is no truth to measure
against, so the metric panels say so instead of showing zeros. This is not an
error. Pass `--require-labels` if you expected labels and want it to fail
loudly when the subfolders turn out to be misnamed.

Labels are auto-detected in this order, so the `real/` + `ai/` layout is a
convention, not a requirement:

1. **Subfolder** — `real/` vs `ai/` (also `authentic`, `natural`, `human`,
   `camera`, `genuine`, `0` / `aigc`, `fake`, `generated`, `synthetic`, `sd`,
   `midjourney`, `1`), matched at any depth, nearest folder wins.
2. **Manifest** — `labels.csv` / `labels.json` (also `manifest.*`,
   `ground_truth.csv`) in the root, with a path column (`image_path`, `path`,
   `file`, …) and a label column (`label`, `is_ai`, `y`, `target`, …). Paths
   resolve absolute, relative-to-manifest, or by basename.
3. **Filename prefix** — `real_*.jpg`, `ai_*.png`.

Anything that yields no label stays unlabeled and is still scored; a folder can
be partly labeled, and the metrics simply use the part that is.

---

## Output format

`detect.py` writes exactly what the problem statement asks for:

```json
[
  {"image_path": "C:/data/img_0001.jpg", "pred": 0.8731},
  {"image_path": "C:/data/img_0002.png", "pred": 0.0412}
]
```

`pred` is P(AI-generated) in `[0, 1]`, rounded to 6 decimals. Every input image
gets exactly one record, in sorted path order. An image that cannot be decoded
is still emitted — as `0.5`, the maximally uncommitted score — and is counted
and named in the terminal summary rather than silently dropped, so the record
count always matches the file count.

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

Ordering matters and is deliberate: the detector's own `prepare_source()` runs
**before** the sweep's transform, because training conditioned the source first
and degraded second. Reversing them measures a pipeline the model was never
trained under.

**Known weak spots, from upstream's own grid.** Three conditions stand out, and
all three are high-FPR failures on inputs that sit deliberately outside the
training ranges: heavy downscaling (0.790 accuracy at 0.125x), a
thumbnail-then-crop chain (0.800 accuracy, 31.9% FPR), and **WebP at q50**
(0.802, 36.2% FPR) — an unseen codec. `rescale`, `social` and `webp` are the
rows to read first. Upstream measured that grid on its in-distribution test set
only, so whether the weak *content* categories degrade further under
compression and blur is unknown; running this sweep on a labeled category
folder is how you find out.

The threat model is **incidental** degradation — what happens to a picture
between upload and screen. Nothing here is claimed against an adaptive
adversary optimising perturbations against this specific detector, which is a
different and much harder problem.

The report lands at `<dir>/robustness_report.json`, and `gui.py` picks up a
report sitting next to the data automatically.

---

## The window

Two screens.

**Screen 1 — upload.** You say what you are uploading, **Labelled data** or
**Unlabelled data**, then pick the folder. A background scan reports what is
actually in it and refuses to run a labelled job on a folder with no labels.
Choosing *Unlabelled data* ignores any labels that are there, so what you asked
for is what you get.

**Screen 2 — results.** Where it goes follows from that choice:

| you uploaded | you get |
| --- | --- |
| Labelled data | **Insights** — metric cards, score distribution, ROC and the confusion matrix — with **Images** (every prediction, filtered by all / FP / FN / AI / real, with a preview pane) and **Robustness** (transform picker, severity scale, degradation curve and cell table) behind header tabs |
| Unlabelled data | one verdict grid: every image badged AI or authentic, filterable, with no metrics — there is no truth to measure against |

The threshold slider sits in the header and re-reads every view live. **Best
F1** jumps to the F1-optimal threshold; **Reset** goes back to the model's own
operating point. **Export JSON** writes the same `predictions.json` the CLI
writes.

---

## Results on the sample set

Measured with **`models/bundle_cvar.pt`** on `sample_data/` (100 real, 100 AI,
clean):

| | |
| --- | --- |
| AUC | **0.921** |
| accuracy at that bundle's own threshold (`0.954`) | **86.0%** |
| recall (AI images caught) | **78%** |
| FPR (authentic images wrongly flagged) | **6%** |
| best-F1 threshold | `0.946` → **87.5%** accuracy |

That operating point trades about a point and a half of accuracy for the low
false-accusation rate it was calibrated for — which is the right trade for this
job. Calling a real photograph fake is the expensive error.

`models/bundle.pt`, the current default, is a later model and has not been
re-measured on this table. Reproduce it for whichever bundle you are shipping:

```bash
python detect.py sample_data --best-threshold --report run_report.json
python detect.py sample_data --weights models/bundle_cvar.pt --best-threshold
```

### Agreement with the upstream pipeline

Upstream scored its `k2m` model both ways — through its own evaluation code and
through this repo's — on the four content categories:

| category | upstream benchmark AUROC | this pipeline | delta |
| --- | --- | --- | --- |
| food | 0.9844 | 0.983 | 0.001 |
| animals | 0.9634 | 0.960 | 0.003 |
| city | 0.9885 | 0.990 | 0.002 |
| people | 0.8355 | 0.836 | 0.000 |

TPR and FPR matched exactly on food (0.920 / 0.020), including the
`subsampling=2` detail. The one deliberate divergence is `SOURCE_JPEG_Q`:
training drew a random quality in 85-98, this repo pins 92 so that two scans of
the same image cannot disagree with each other. That is worth <=0.004 AUROC,
against score margins of 3.2-6.0.

---

## Plugging in a different model

**The easy path.** Save it as TorchScript and drop it at `models/model.pt`:

```python
torch.jit.save(torch.jit.script(model), "models/model.pt")
```

`app/detectors/trained.py` picks it up. It ranks below `clip_head` while the
bundle is present; pick it explicitly with `--detector trained`, or the
**Weights** field on the upload screen.

Check the constants at the top of `trained.py` against how the model was
actually trained: `INPUT_SIZE`, `MEAN`/`STD`, and `AI_CLASS_INDEX`. It handles
a 1-logit sigmoid head and a 2-logit softmax head automatically.

| checkpoint shape | works? |
| --- | --- |
| TorchScript | yes — architecture travels with the weights |
| pickled `nn.Module` | yes, if the defining class is importable here |
| bare `state_dict` | fill in `build_model()` — a tensor dict alone doesn't say what to build |

**The full path.** For a model that needs its own preprocessing, add a detector
— see `app/detectors/base.py`:

```python
from .base import Detector, register

@register
class MyModel(Detector):
    name = "my_model"
    display_name = "EfficientNet-B0 + FFT head"
    description = "Trained on ..."
    requires_weights = True
    default_weights = "models/my_model.pt"

    def load(self):
        self.model = ...                      # called once, lazily, off the GUI thread

    def predict_batch(self, paths):
        return [float(p) for p in ...]        # 0.0 = authentic, 1.0 = AI
```

Import it in `app/detectors/__init__.py` and it appears in the CLI and the
picker automatically. Two optional overrides are worth knowing:

- `predict_images(images)` — score already-decoded PIL images. The base class
  round-trips through temp files so a path-only detector still works; override
  it and the robustness sweep stops touching the disk.
- `prepare_source(img)` — normalise a freshly decoded image before any
  degradation is applied, for a model whose training conditioned the source.

Set `default_threshold` in `load()` if your checkpoint carries a calibrated
operating point; the CLI and the window both adopt it.

---

## Repository layout

```
detect.py                  CLI: any folder -> predictions.json (+ metrics)
robustness.py              CLI: transform sweep -> robustness_report.json
gui.py                     the window: upload, insights, images, robustness
requirements.txt
models/                    bundle.pt (the trained model) lives here — gitignored
app/
  runner.py                scan / load / score, with all terminal logging
  sweep.py                 robustness sweep core, shared by CLI and GUI
  workers.py               Qt threads so the window stays responsive
  dataset.py               directory scan + 3-way label inference
  metrics.py               accuracy/P/R/F1/AUC/AP/confusion/threshold search
  transforms.py            post-processing transforms x 5 severities
  export.py                predictions JSON/CSV + run report
  theme.py                 dark palette, Qt stylesheet, matplotlib rcParams
  detectors/
    base.py                plugin interface + registry
    clip_head.py           THE model: CLIP ViT-L/14 + trained MLP head
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
tools/
  build_dataset.py         pool public datasets -> deduplicated real/ai corpus
  run_worker.sh            one laptop's disjoint share of the pull
  predict_dir.py           headless detect.py, for a worker box with no display
```

The dependency direction is one-way and worth preserving: `widgets/` may import
from `app/`, `app/` never imports from `widgets/`. That is what keeps the CLI
free of any Qt dependency — `detect.py` and `robustness.py` run fine on a
machine where PyQt6 is not installed.

### Sample dataset

`sample_data/` is the folder the GUI offers by default — 100 authentic and 100
generated images in the labeled layout. It is gitignored, so it is not in a
fresh clone:

```
sample_data/
├── real/
└── ai/
```

`data/` is the much larger tree `tools/build_dataset.py` writes, in the same
labeled layout at every level. Also gitignored — it is rebuilt, not cloned:

```
data/
├── _pool/            everything ingested, deduplicated       -> real/ ai/
├── train/  test/     the 80/20 split of the pool             -> real/ ai/
├── heldout/<name>/   never trained on                        -> real/ ai/
├── robustness/       degradation validation                  -> real/ ai/
└── .hashes.txt       the global content-hash ledger
```

Delete `.hashes.txt` and a re-ingest will re-add everything it already has;
keep it with the pool.

---

## Troubleshooting

**`error: <detector> needs a checkpoint, and none is at .../models/bundle.pt`**
The bundle is gitignored and is not in a clone. Put it at `models/bundle.pt`,
or pass `--weights <file>`. `python detect.py --list-detectors` shows which
backends can currently run.

**Everything scores as AI, or everything scores as real.** Check the threshold.
The calibrated operating point is not `0.5` — it is whatever the bundle carries
(`0.780` / `0.954` for the two here), and the run log prints it as
`operating point 0.780`. If you passed `--threshold 0.5` you are asking a
different question. Run `--best-threshold` to see where F1 actually peaks on
your data.

**`no labels found - scores only`** on a folder you believe is labeled. The
scanner tried subfolders, then a manifest, then filename prefixes. Check the
folder names against the token list above, or pass `--require-labels` to make
it an error rather than a shrug.

**Some images written as `0.5`.** Those failed to decode. The count and the
first five names print in the run summary; they are emitted anyway so the
record count matches the file count.

**The sweep refuses to run.** It measures accuracy, so it needs ground truth.
An unlabeled folder has nothing to be right or wrong about.

**Unicode errors on a Windows console.** Handled — `runner.py` reconfigures
stdout/stderr to UTF-8 at import, because a cp1252 console cannot encode a σ in
a transform label, let alone a CJK filename, and would otherwise kill a run
mid-way.

---

## Credits

- Training pipeline, weights and calibration — Joe's upstream repository (see
  [Model provenance](#model-provenance)); current checkpoint
  `runs/k2m/head_bal.pt`.
- Vision tower — `openai/clip-vit-large-patch14`, frozen, redistributed inside
  the bundle.
- Training corpus — the shipped bundle carries upstream's own nine-group mix
  (reals: an original Kaggle set, Unsplash, `ai-vs-human`; generated: a
  GenImage-derived group, SDXL, FLUX_DEV, FLUX_PRO, DALL-E 3, `ai-vs-human`),
  which is **not** the pool built in this repo. The pool below is what
  `tools/build_dataset.py` assembles for retraining:
  Tiny-GenImage, SID_Set, CIFAKE, ELSA1M, OpenImages V7, and the `bitmind`
  real/generator collections (MS-COCO, FFHQ, CelebA-HQ, SDXL, RealVis-XL,
  Mobius, FLUX.1-dev). Each remains under its own licence.
- This repository — inference, evaluation, the robustness lab and the window.

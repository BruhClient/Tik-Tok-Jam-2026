# AIGC Image Detector

Detects AI-generated images, and measures how much of that detection survives
the things that actually happen to a picture on the way to your screen — JPEG
recompression, downscaling, blur, a screenshot of a repost.

**The deliverable is `detect.py`**: image directory in, `predictions.json` out.
`robustness.py` is the same model under a transform sweep. `gui.py` is the same
pipeline behind a window. All three call into `app/`, so a number you read in
the window is the number the CLI printed.

There is **no training step in this repository**. The model is trained
in `training_process/` (see [Model provenance](#model-provenance)); this scores images with
it and tells you how far you can trust the answer.

---

## About this project

### How our solution addresses the problem

Our model trains specifically on **augmented images** so it is not fooled by
post-processing. We apply JPEG recompression, Gaussian blur, downscaling,
noise, social-repost chains and more during training — the model is penalised
on the worst-case augmented view via a CVaR objective, making it robust to the
exact things that happen to an image on its way to your screen.

We use **CLIP ViT-L/14** as a frozen feature extractor. Before embedding, every
image is re-encoded as JPEG at a fixed quality — this neutralises format-based
shortcuts (real photos tend to arrive as JPEG, generated ones as PNG) and brings
both classes to the same distribution. Only a small MLP head is trained on top
of these features, which means the tower's generalist representations carry the
weight and the head adapts to the detection task.

Because of limited time and datasets, we report **AUC** as our primary metric
rather than accuracy. The optimal decision threshold shifts between different
evaluated image sets, so a fixed accuracy figure is misleading. AUC measures
the model's discriminative power independently of any threshold choice.
Accuracy becomes meaningful once a threshold is fixed for a specific deployment
context — `--best-threshold` shows where F1 peaks on any labeled folder.

### Development tools

- Visual Studio Code

### Models and APIs

- **CLIP ViT-L/14** (`openai/clip-vit-large-patch14`) — frozen vision tower,
  redistributed inside `bundle.pt`

### Libraries and frameworks

| library | role |
| --- | --- |
| PyTorch | model training and inference |
| Hugging Face Transformers | CLIP model loading (`CLIPVisionModelWithProjection`) |
| Hugging Face Datasets | downloading training datasets from HuggingFace Hub |
| Pillow | image loading, JPEG re-encoding, augmentation ops |
| scikit-learn | AUC, average precision, calibration metrics |
| NumPy | numerical operations throughout |
| PyQt6 | desktop GUI |
| matplotlib | score histograms, ROC, PR, confusion matrix, degradation curves |

### Datasets and assets

**Training data**

| dataset | source | notes |
| --- | --- | --- |
| SID_Set | [`saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set) (HuggingFace) | primary source — real OpenImages photos vs fully synthetic images; geometry-normalised to remove the square-equals-fake shortcut |
| MS COCOAI / Defactify | [`Rajarshi-Roy-research/Defactify_Image_Dataset`](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) (HuggingFace) | semantically-aligned real/fake pairs (SD3, SDXL, DALL·E 3, MidJourney) — removes content/framing shortcuts |
| AI vs Human Generated | [Kaggle](https://www.kaggle.com/datasets/alessandrasala79/ai-vs-human-generated-dataset) | additional real/AI pairs with CSV manifest |
| 130k Real vs Fake Faces | [Kaggle](https://www.kaggle.com/datasets/shreyanshpatel1/130k-real-vs-fake-face) | large face-specific real/AI dataset |
| Unsplash Lite | [github.com/unsplash/datasets](https://github.com/unsplash/datasets) | professionally-shot stock photos added to broaden the real-image distribution |

**Benchmark / validation data**

| dataset | source | notes |
| --- | --- | --- |
| MS COCO val2017 | [cocodataset.org](http://images.cocodataset.org) | held-out real images for the official benchmark; never used during training |
| DALL·E Advanced | organiser-provided | synthetic half of the official validation benchmark |

---

## Contents

- [About this project](#about-this-project)
- [Quick start](#quick-start)
- [Downloading the model](#downloading-the-model)
- [Commands](#commands)
- [Model provenance](#model-provenance)
- [The model](#the-model)
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

# 1. Download bundle.pt from the Releases page (link below)
# 2. Move it into the models/ folder in this repo
# 3. Run:
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

---

## Downloading the model

The trained bundle (`bundle.pt`, ~1.1 GB) is too large to store in the
repository and is distributed as a GitHub Release asset.

1. Go to the **[Releases page](https://github.com/BruhClient/Tik-Tok-Jam-2026/releases/tag/Model_for_tiktokTechJam2026)**
2. Download `bundle.pt`
3. Move or drag it into the `models/` folder in this repo

Once it is there, `detect.py`, `robustness.py` and `gui.py` will all pick it
up automatically with no further configuration.

---

## Model provenance

The detector shipped here is **not trained in this repository**. The training
pipeline — feature extraction (`clipfeat.py`), the augmentation stack
(`augment.py`), the CVaR head objective and the calibration fit — lives in
`training_process/`, and this inference repo consumes its output.

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

The bundle carries its own calibration and operating point, read straight out
of the file at load time and printed in the run log as `operating point 0.XXX`.
Never hardcode an operating point: read it from the bundle, which is what the
app does.

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
output readable as a probability, and the bundle carries its calibrated
operating point (printed in the run log as `operating point 0.XXX`). `detect.py`
and the window both read it out of the file and adopt it automatically —
**Reset** on the slider goes back to it, and `--threshold` overrides it.
**The JSON is always raw scores either way** — the threshold only ever changes
what gets *printed* or *drawn*.

**What the architecture can and cannot see.** CLIP resizes to 224 before the
patch embedding, which is why these features survive JPEG and blur so well and
why they cannot see subtle resampling traces. The flip side shows up in the
sweep: heavy compression shifts *both* classes' scores upward, so a fixed
threshold drifts even where AUROC holds. Read the Robustness page with that in
mind.

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

The report lands at `<dir>/robustness_report.json`, and `gui.py` picks up a
report sitting next to the data automatically.

---

## The window

Two screens.

**Screen 1 — upload.** You say what you are uploading, a **labeled dataset** or
**just images**, then pick the folder. A background scan reports what is
actually in it and refuses to run a labeled job on a folder with no labels.
Choosing *just images* ignores any labels that are there, so what you asked for
is what you get.

**Screen 2 — results.** Where it goes follows from that choice:

| you uploaded | you get |
| --- | --- |
| a labeled dataset | **Insights** — metric cards, score distribution, ROC and the confusion matrix — with **Images** (every prediction, filtered by all / FP / FN / AI / real, with a preview pane) and **Robustness** (transform picker, severity scale, degradation curve and cell table) behind header tabs |
| just images | one verdict grid: every image badged AI or authentic, filterable, with no metrics — there is no truth to measure against |

The threshold slider sits in the header and re-reads every view live. **Best
F1** jumps to the F1-optimal threshold; **Reset** goes back to the model's own
operating point. **Export JSON** writes the same `predictions.json` the CLI
writes.

---

## Results on the sample set

Measured with **`models/bundle.pt`** on `sample_data/` (100 real, 100 AI, clean):

| | |
| --- | --- |
| AUC | **0.921** |
| recall (AI images caught) | **78%** |
| FPR (authentic images wrongly flagged) | **6%** |

The model is calibrated for low false-accusation rate — calling a real
photograph fake is the expensive error. Reproduce the numbers with:

```bash
python detect.py sample_data --best-threshold --report run_report.json
```

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
  [Model provenance](#model-provenance)).
- Vision tower — `openai/clip-vit-large-patch14`, frozen, redistributed inside
  the bundle.
- This repository — inference, evaluation, the robustness lab and the window.

# AIGC Image Detector

Detects AI-generated images and reports how well that detection survives
realistic post-processing.

**The deliverable is `detect.py`** — image directory in, `predictions.json`
out. `gui.py` is the same pipeline behind a window. Work happens in scripts
that log to the terminal; the window shows finished results.

There is no training step here. The model is trained elsewhere; this scores
images with it.

---

## Commands

```bash
pip install -r requirements.txt
```

| | command | what you get |
| --- | --- | --- |
| **Score** | `python detect.py <dir>` | `predictions.json`, plus metrics if the folder is labeled |
| **Robustness** | `python robustness.py <dir>` | transform sweep → `robustness_report.json` (needs labels) |
| **Window** | `python gui.py [<dir> \| <preds.json>]` | upload, then insights or verdicts |

```bash
python detect.py <dir> --best-threshold --report run_report.json
python detect.py <dir> --out results.json --weights models/best.pt
python detect.py <dir> --require-labels          # fail if labels are missing
python robustness.py <dir> --transforms jpeg,blur --severities 1,3,5
python detect.py --list-detectors
python robustness.py --list-transforms
```

Progress always prints to the terminal — in the GUI too. The window shows
finished results, never a loading bar.

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

1. **Subfolder** — `real/` vs `ai/` (also `authentic`, `natural`, `human`, `0` /
   `aigc`, `fake`, `generated`, `synthetic`, `1`), at any depth.
2. **Manifest** — `labels.csv` / `labels.json` in the root, with a path column
   (`image_path`, `path`, `file`, …) and a label column (`label`, `is_ai`, …).
3. **Filename prefix** — `real_*.jpg`, `ai_*.png`.

## Output format

`detect.py` writes exactly what the problem statement asks for:

```json
[
  {"image_path": "C:/data/img_0001.jpg", "pred": 0.8731},
  {"image_path": "C:/data/img_0002.png", "pred": 0.0412}
]
```

`pred` is P(AI-generated) in `[0, 1]`. Scores are always raw floats — the
threshold only affects what gets *printed*, never what gets written.

## The model

`app/detectors/clip_head.py` is the real backend: a **frozen CLIP ViT-L/14**
vision tower plus a **small MLP head** trained on top of its embeddings. Only
the head was trained — the tower is stock `openai/clip-vit-large-patch14`.

Put the bundle at `models/bundle.pt` and it becomes the default backend
everywhere — CLI, GUI and sweep — with nothing to flip. Point elsewhere with
`--weights <file>`.

Everything the model needs travels in that one file, so nothing is downloaded
at run time: the tower's config and weights, the head's config and weights, the
feature standardisation (`mu`/`sd`), the Platt coefficients and the operating
point. Scoring reproduces the training pipeline exactly:

```
shortest side -> 224 (bicubic), centre crop 224
CLIP normalisation (not ImageNet — the constants differ)
tower -> pooler_output -> visual_projection            768-d
L2-normalise -> (x - mu) / sd
head -> logit -> sigmoid(platt_a * logit + platt_b)
```

**The threshold is not 0.5.** Training used `pos_weight` and a CVaR objective,
both of which distort the output scale, so the head's raw logits sit high. The
Platt coefficients fitted on pooled augmented validation scores make the output
readable as a probability, and the bundle carries the operating point that was
chosen for **1% FPR on real images** — about `0.954`. `detect.py` and the
window both adopt it automatically; **Reset** on the slider goes back to it,
and `--threshold` overrides it. The JSON is always raw scores either way.

On `sample_data/` (100 real, 100 AI, clean): **AUC 0.921**, and at the bundle's
own threshold **86.0% accuracy, 78% recall, 6% FPR**. `--best-threshold` finds
87.5% at `0.946`, so the shipped operating point is trading a little accuracy
for the low false-accusation rate it was calibrated for — which is the right
trade for this job.

CLIP resizes to 224 before the patch embedding, which is why these features
survive JPEG and blur so well and why they cannot see subtle resampling traces.
The flip side shows up in the sweep: heavy compression shifts *both* classes'
scores upward, so a fixed threshold drifts even where AUROC holds. Read the
Robustness page with that in mind.

## Plugging in a different model

Save it as TorchScript and drop it at `models/model.pt`:

```python
torch.jit.save(torch.jit.script(model), "models/model.pt")
```

`app/detectors/trained.py` picks it up. It ranks below `clip_head` while the
bundle is present; pick it explicitly with `--detector trained`, or the
**Weights** field on the Run page.

Check the constants at the top of `trained.py` against how the model was
actually trained: `INPUT_SIZE`, `MEAN`/`STD`, and `AI_CLASS_INDEX`. It handles
a 1-logit sigmoid head and a 2-logit softmax head automatically.

Other checkpoint shapes:

| shape | works? |
| --- | --- |
| TorchScript | yes — architecture travels with the weights |
| pickled `nn.Module` | yes, if the defining class is importable here |
| bare `state_dict` | fill in `build_model()` — a tensor dict alone doesn't say what to build |

For a model that needs its own preprocessing, add a detector instead — see
`app/detectors/base.py`. Any class with `@register` shows up in the picker
automatically. Override `predict_images()` so the robustness sweep can score
transformed images in memory instead of round-tripping through temp files.

## What each piece does

**`detect.py`** — the deliverable. Scans recursively, skips non-images,
survives undecodable files (they become `0.5` and are counted), writes the
JSON, and adds the metrics block when labels exist.

**`robustness.py`** — applies transforms in memory at five severities each
(JPEG, WebP, blur, downscale→upscale, crop, brightness/contrast, saturation,
noise, social-repost combo, screenshot resample), re-scores the sample, and
compares against a clean baseline measured through the same pipeline.

**`gui.py`** — two screens. On the first you say what you are uploading, a
**labeled dataset** or **just images**, then pick the folder; a background scan
reports what is actually in it and refuses to run a labeled job on a folder
with no labels. Choosing *just images* ignores any labels that are there, so
what you asked for is what you get.

Where the second screen goes follows from that choice:

| you uploaded | you get |
| --- | --- |
| a labeled dataset | **Insights** — metric cards, score distribution, ROC and the confusion matrix — with **Images** (every prediction, filtered by all / FP / FN / AI / real) and **Robustness** (transform picker and degradation curve) behind header tabs |
| just images | one verdict grid: every image badged AI or authentic, filterable, with no metrics — there is no truth to measure against |

The threshold slider sits in the header and re-reads every view live.
A `robustness_report.json` sitting next to the data is loaded automatically.

## Placeholder backends

Two stubs ship so the pipeline runs end to end before a checkpoint exists.

| Name | What it is |
| --- | --- |
| `random` | Deterministic hash-based scores with a mild label-aware bias. |
| `heuristic` | FFT high-frequency ratio, noise residual, saturation, detail uniformity and JPEG-quantisation evidence with hand-set weights. A baseline, not a trained model. |

Both set `is_placeholder = True`, which drives the terminal warning and the
window's amber dot. They stop being the default the moment real weights exist.

## Sample dataset

`sample_data/` is the folder the GUI offers by default — 100 authentic and 100
generated images in the labeled layout:

```
sample_data/
├── real/
└── ai/
```

## Layout

```
detect.py                  CLI: any folder -> predictions.json (+ metrics)
robustness.py              CLI: transform sweep -> robustness_report.json
gui.py                     the window: run, overview, images, robustness
models/                    bundle.pt (the trained model) lives here
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
    heuristic_stub.py      placeholder
    random_stub.py         placeholder
  widgets/
    window.py              app shell: upload screen, results header, run state
    upload.py              screen 1: declare the data, pick the folder
    pages.py               the labeled results: insights, images, robustness
    gallery.py             the unlabeled result: a verdict per image
    table.py               predictions table model + score-bar delegate
    charts.py              histogram, ROC, PR, confusion, degradation
    components.py          cards, chips, badges, type scale
```

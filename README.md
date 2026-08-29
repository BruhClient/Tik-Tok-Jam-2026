# AIGC Image Detector

Detects AI-generated images and reports how well that detection survives
realistic post-processing.

**The deliverable is `predict.py`** — image directory in, `predictions.json`
out. `main.py` is a viewer for looking at what came out. Scripts do the work
and log to the terminal; the window only draws finished results.

**The detection model is not implemented.** Two clearly-marked placeholder
backends ship so the whole pipeline runs end to end today. Everything else —
scanning, label inference, transforms, metrics, charts, exports — is real.

---

## Commands

```bash
pip install -r requirements.txt
```

| | command |
| --- | --- |
| **prod** | `python predict.py <image_dir>` |
| **dev** | `python main.py` |

```bash
# PROD - any directory, no labels or structure needed
python predict.py "C:\path\to\images"                    # -> predictions.json
python predict.py <dir> --out results.json --detector heuristic
python predict.py <dir> --report run_report.json --threshold 0.4
python predict.py --list-detectors

# DEV - score the dev dataset, then open the viewer
python main.py                        # sample_data/
python main.py <image_dir>            # any other directory
python main.py predictions.json       # visualise a finished result file

# ROBUSTNESS - how far accuracy falls under post-processing
python robustness.py <image_dir>
python robustness.py <dir> --transforms jpeg,blur,rescale --severities 1,3,5 --sample 200
python robustness.py --list-transforms
```

## Output format

`predict.py` writes exactly what the problem statement asks for:

```json
[
  {"image_path": "C:/data/test/img_0001.jpg", "pred": 0.8731},
  {"image_path": "C:/data/test/img_0002.png", "pred": 0.0412}
]
```

`pred` is P(AI-generated) in `[0, 1]`. Scores are always raw floats — the
threshold only affects what gets *printed*, never what gets written.

## What each piece does

**`predict.py`** — scans recursively, skips non-images, survives undecodable
files (they become `0.5` and are counted in the log), writes the JSON. Needs no
labels and no folder structure. If the directory happens to be labeled it also
prints accuracy / AUC / F1 / FPR; on a blind set that section is skipped.

**`main.py`** — runs the same pipeline, then opens one window: metric cards, a
live threshold slider, a filterable score table (all / false positives / false
negatives / flagged AI / flagged real), an image preview, and four charts —
score distribution, ROC, confusion matrix, and the robustness degradation
curve. Nothing loads inside the window; it opens on a finished result.

**`robustness.py`** — applies nine transforms in memory at five severities each
(JPEG, WebP, blur, downscale→upscale, crop, brightness/contrast, saturation,
noise, social-repost combo, screenshot resample), re-scores the sample, and
compares against a clean baseline measured through the same pipeline. Prints a
per-cell table and writes `robustness_report.json` next to the dataset —
`main.py` finds it automatically and draws it.

## Ground-truth labels

Only needed to measure accuracy. Auto-detected in this order:

1. **Subfolder** — `real/` vs `ai/` (also `authentic`, `natural`, `human`, `0` /
   `aigc`, `fake`, `generated`, `synthetic`, `1`), at any depth. So
   `test/ai/x.png` and `train/real/y.jpg` both resolve correctly.
2. **Manifest** — `labels.csv` / `labels.json` in the root, with a path column
   (`image_path`, `path`, `file`, …) and a label column (`label`, `is_ai`, …).
3. **Filename prefix** — `real_*.jpg`, `ai_*.png`.

No labels → scores only, and the metric panels say so instead of showing zeros.

## Dev dataset

`python main.py` with no argument opens `sample_data/`, resolved relative to
`main.py` (not the shell's working directory). Both splits load pooled:

```
sample_data/
├── train/real/  train/ai/
└── test/real/   test/ai/
```

Change `DEV_DATA_DIR` at the top of `main.py` to target one split. Metrics
pooled across data a model was fitted to read optimistically — use
`python main.py sample_data/test` for the honest number.

## Plugging in the real model

Create `app/detectors/real_model.py`:

```python
from .base import Detector, register

@register
class MyModel(Detector):
    name = "my_model"
    display_name = "EfficientNet-B0 + FFT head"
    description = "Trained on ..."
    is_placeholder = False        # removes the placeholder warnings
    batch_size = 32

    def load(self):
        self.model = ...          # called once, lazily

    def predict_batch(self, paths):
        return [...]              # one float in [0, 1] per path

    def predict_images(self, images):
        return [...]              # PIL images; used by robustness.py
```

Add `from . import real_model` to `app/detectors/__init__.py`. All three
entry points pick it up automatically, and it becomes the default detector
(real backends sort ahead of placeholders).

`predict_images` matters: the robustness sweep transforms images **in memory**
and never touches disk. The base class falls back to temp files without it.

## Placeholder backends

| Name | What it is |
| --- | --- |
| `random` | Deterministic hash-based scores with a mild label-aware bias. |
| `heuristic` | FFT high-frequency ratio, noise residual, saturation, detail uniformity and JPEG-quantisation evidence with hand-set weights. A baseline, not a trained model. |

Both set `is_placeholder = True`, which drives the terminal warning and the
window's amber badge.

## Layout

```
predict.py                 PROD: any directory -> predictions.json
main.py                    DEV: score, then open the viewer
robustness.py              transform sweep -> robustness_report.json
app/
  runner.py                scan / load / score, with all terminal logging
  dataset.py               directory scan + 3-way label inference
  metrics.py               accuracy/P/R/F1/AUC/AP/confusion/threshold search
  transforms.py            9 post-processing transforms x 5 severities
  export.py                predictions JSON/CSV, run + robustness reports
  theme.py                 dark palette, Qt stylesheet, matplotlib rcParams
  detectors/               plugin interface + placeholder backends
  widgets/
    window.py              the single results window
    charts.py              histogram, ROC, PR, confusion, degradation
    components.py          cards, chips, badges
```

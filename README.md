# AIGC Image Detector — evaluation console

Desktop front end (PyQt6) for an AI-generated-image detector: bulk-load an image
directory, score it, inspect per-image confidences, measure accuracy against
ground truth, and stress-test robustness under realistic post-processing.

**The detection model is not implemented.** Two clearly-marked placeholder
backends ship so every screen, chart and export path works end to end today.
Everything else — dataset scanning, label inference, threading, transforms,
metrics, charts, exports — is real and final.

---

## Run it

```bash
pip install -r requirements.txt

python main.py                       # then Load directory / Ctrl+O / drag-drop
python main.py sample_data           # or open a directory straight away
python main.py "C:\path\to\images"
```

Drop your own images into `sample_data/real/` and `sample_data/ai/` to have a
fixture that loads with one command, or point the app at any directory.

## Screens

| Tab | What it does |
| --- | --- |
| **Dataset** | Load (or drag & drop) a directory, see label counts, browse a lazy thumbnail grid, inspect any image. |
| **Results** | Per-image score table, live decision threshold, accuracy / AUC / F1 / FPR, score histogram, ROC, PR, confusion matrix, threshold sweep, exports. |
| **Robustness lab** | Re-scores the set under JPEG, blur, rescale, crop, colour, noise, "social repost" and screenshot transforms at five severities, charts the fall-off against a clean baseline. |

Shortcuts: `Ctrl+O` load directory · `Ctrl+R` run detection.

## Ground-truth labels

Auto-detected in this order (override in the Dataset tab's *Labels* combo):

1. **Subfolder** — `real/` vs `ai/` (also `authentic`, `natural`, `human`, `0` /
   `aigc`, `fake`, `generated`, `synthetic`, `1`), at any depth.
2. **Manifest** — `labels.csv` / `labels.json` in the root, or any file picked
   via *Load manifest…*. Needs a path column (`image_path`, `path`, `file`, …)
   and a label column (`label`, `is_ai`, `y`, …). Paths may be absolute,
   relative to the manifest, or bare filenames.
3. **Filename prefix** — `real_*.jpg`, `ai_*.png`, regex `^(real|ai|fake)[_-]`.

No labels found → the app still scores everything, and the metric panels say so
instead of showing fake zeros.

## Output format

*Export predictions.json* writes exactly what the problem statement asks for:

```json
[
  {"image_path": "C:/data/test/img_0001.jpg", "pred": 0.8731},
  {"image_path": "C:/data/test/img_0002.png", "pred": 0.0412}
]
```

`pred` is P(AI-generated) in `[0, 1]`. Also available: CSV with predicted/true
labels, a run report (metrics + timings + detector identity), and a robustness
report (per transform × severity, with deltas against the clean baseline).

The same output is available headlessly:

```bash
python tools/predict_dir.py <image_dir> --out predictions.json --detector heuristic
python tools/predict_dir.py --list-detectors
```

## Plugging in the real model

Create `app/detectors/real_model.py`:

```python
from .base import Detector, register

@register
class MyModel(Detector):
    name = "my_model"
    display_name = "EfficientNet-B0 + FFT head"
    description = "Trained on ..."
    is_placeholder = False        # removes the amber warning banners
    batch_size = 32

    def load(self):
        self.model = ...          # called once, lazily, off the GUI thread

    def predict_batch(self, paths):
        return [...]              # one float in [0, 1] per path

    def predict_images(self, images):
        return [...]              # PIL images; used by the robustness sweep
```

Then add `from . import real_model` to `app/detectors/__init__.py`. It appears
in the toolbar picker automatically — no UI changes needed.

`predict_images` matters: the robustness lab transforms images **in memory** and
never touches disk. The base class falls back to temp files if you skip it.

## Placeholder backends

| Name | What it is |
| --- | --- |
| `random` | Deterministic hash-based scores with a mild label-aware bias, plus a simulated per-image cost so progress/cancel are observable. |
| `heuristic` | FFT high-frequency ratio, noise residual, saturation, detail uniformity and JPEG-quantisation evidence with hand-set weights. A baseline, not a trained model. |

Both set `is_placeholder = True`, which is what drives the amber banners on the
Results and Robustness tabs.

## Layout

```
main.py                    entry point
app/
  theme.py                 dark palette, Qt stylesheet, matplotlib rcParams
  state.py                 AppState + RunResult, all signals
  dataset.py               directory scan + 3-way label inference
  metrics.py               accuracy/P/R/F1/AUC/AP/confusion/threshold search
  transforms.py            9 post-processing transforms × 5 severities
  export.py                predictions JSON/CSV, run + robustness reports
  workers.py               thumbnail pool, inference thread, sweep thread
  detectors/               plugin interface + placeholder backends
  widgets/                 main window, three tabs, grid, charts, components
tools/
  predict_dir.py           headless directory -> predictions.json
```

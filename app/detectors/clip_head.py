"""CLIP ViT-L/14 + adversarially-trained MLP head - the real detector.

The model is two halves. A frozen CLIP vision tower turns an image into a
768-d embedding, and a small MLP trained on top of those embeddings decides
whether the image is AI-generated. Only the head was trained; the tower is
stock `openai/clip-vit-large-patch14`.

Everything travels in one file, `models/bundle.pt`, so nothing is downloaded
at run time and nothing has to match a config kept somewhere else:

    clip_config / clip_state_dict   the frozen tower, its config and weights
    head_state_dict / head_config   the trained MLP
    mu, sd                          per-dimension feature standardisation
    platt_a, platt_b                logit -> calibrated probability
    threshold                       the 1%-FPR operating point, in logit space
    feature, preproc, l2            the preprocessing the head was trained for

Scoring reproduces the training pipeline exactly, in this order:

    JPEG re-encode at q92                                  (SOURCE_JPEG_Q)
    resize shortest side to 224 (bicubic) -> centre crop 224
    CLIP normalisation (not ImageNet - the constants differ)
    vision tower -> pooler_output -> visual_projection      (feature="proj")
    L2-normalise -> (x - mu) / sd
    head -> logit -> sigmoid(platt_a * logit + platt_b)

That first step is easy to mistake for an optimisation and delete. Training
re-encoded every image, of both classes, before it was ever seen - so mu, sd,
the Platt coefficients and the threshold were all fitted on re-encoded
features. A pristine PNG scored without it is off-distribution.

That last step matters. Training used pos_weight and a CVaR objective, both of
which distort the output scale, so a raw sigmoid(logit) is a confidence number
and not a probability. The Platt coefficients were fitted on pooled augmented
validation scores, which is what makes `pred` safe to show as "% likely AI".
"""

from __future__ import annotations

import io
import os

import numpy as np
from PIL import Image

from .base import Detector, register

# --------------------------------------------------------------------------- #
# Constants that must match the training pipeline (clipfeat.py).
# --------------------------------------------------------------------------- #

RES = 224
CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Training re-encoded every image at a random JPEG quality in 85-98
# (augment.normalise_source) so the head could not learn "real photos are
# JPEG, generated ones are PNG" instead of learning the actual task. That
# means every feature the head ever saw came from a re-encoded image, so
# scoring has to do it too. The midpoint is used rather than a random draw:
# inference has to be deterministic and not depend on file ordering.
SOURCE_JPEG_Q = 92

# PIL opens anything; a corrupt header can claim gigapixels. Same cap the
# training loader used, so the two agree on what counts as unreadable.
# PIL's own decompression-bomb guard trips at 178M px, well under that cap,
# and _open would turn it into a silent NaN - so disable it (as clipfeat.py
# does) and let the explicit check be the one that decides.
MAX_PIXELS = 200_000_000
Image.MAX_IMAGE_PIXELS = None

DEFAULT_WEIGHTS = os.path.join("models", "bundle.pt")


_HEAD_CLS = None


def _head_class(torch):
    """The trained Head class, defined lazily so importing this module is free.

    Nothing here may drift from training: the checkpoint keys are net.0/1/4/5/8,
    which pins both the attribute name `net` and the layer indices - the ReLU
    and Dropout have to stay in place even though they hold no weights. Dropout
    is inert in eval() anyway.
    """
    global _HEAD_CLS
    if _HEAD_CLS is not None:
        return _HEAD_CLS

    nn = torch.nn

    class Head(nn.Module):
        def __init__(self, dim, kind="mlp", hidden=512, dropout=0.3):
            super().__init__()
            if kind == "linear":
                self.net = nn.Linear(dim, 1)
            elif kind == "mlp":
                self.net = nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(inplace=False),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden // 2),
                    nn.BatchNorm1d(hidden // 2),
                    nn.ReLU(inplace=False),
                    nn.Dropout(dropout),
                    nn.Linear(hidden // 2, 1),
                )
            else:
                raise ValueError(f"unknown head kind {kind!r}")

        def forward(self, x):
            return self.net(x).squeeze(-1)        # (B, D) -> (B,)

    _HEAD_CLS = Head
    return Head


@register
class ClipHeadDetector(Detector):
    name = "clip_head"
    display_name = "CLIP ViT-L/14 + MLP head"
    description = (
        "Frozen CLIP ViT-L/14 embeddings scored by an MLP head trained "
        "adversarially against post-processing (JPEG, blur, resize, noise). "
        f"Loads the self-contained bundle at {DEFAULT_WEIGHTS}; the output is "
        "Platt-calibrated, so it reads as a probability."
    )
    requires_weights = True
    default_weights = DEFAULT_WEIGHTS
    # CLIP ViT-L/14 is ~300M params. 16 keeps CPU memory sane and the progress
    # bar moving; a GPU is nowhere near saturated, but it is not the bottleneck.
    batch_size = 16

    def __init__(self):
        super().__init__()
        self.clip = None
        self.head = None
        self.device = None
        self._torch = None
        # filled from the bundle at load()
        self.mu = self.sd = None
        self.platt_a, self.platt_b = 1.0, 0.0
        self.threshold_logit = 0.0
        self.feature = "proj"
        self.preproc = "resize"
        self.l2 = True

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        path = self.resolve_weights(self.weights)
        if not path or not os.path.isfile(path):
            raise SystemExit(
                f"error: no bundle at {os.path.abspath(path or DEFAULT_WEIGHTS)}\n"
                "       put bundle.pt there, or pass --weights <file>.")

        try:
            import torch
        except ImportError:                       # pragma: no cover - env-dependent
            raise SystemExit("error: torch is required to run this detector "
                             "(pip install -r requirements.txt)")
        try:
            from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection
        except ImportError:                       # pragma: no cover - env-dependent
            raise SystemExit("error: transformers is required to run this detector "
                             "(pip install -r requirements.txt)")

        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.note(f"torch {torch.__version__} · device {str(self.device).upper()}"
                  + ("" if self.device.type == "cuda" else " (no CUDA device)"))

        # By far the longest step, and torch.load gives no callback to hang a
        # percentage on - so say how much is coming instead of pretending.
        size_gb = os.path.getsize(path) / 1e9
        self.note(f"reading {os.path.basename(path)} ({size_gb:.2f} GB) from disk")
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        missing = {"clip_config", "clip_state_dict", "head_state_dict",
                   "head_dim", "mu", "sd"} - set(bundle)
        if missing:
            raise SystemExit(
                f"error: {path} is not a detector bundle - missing "
                f"{', '.join(sorted(missing))}")

        self.feature = bundle.get("feature", "proj")
        self.preproc = bundle.get("preproc", "resize")
        self.l2 = bool(bundle.get("l2", True))
        self.platt_a = float(bundle.get("platt_a", 1.0))
        self.platt_b = float(bundle.get("platt_b", 0.0))
        self.threshold_logit = float(bundle.get("threshold", 0.0))

        # Build the tower from the bundled config rather than by name: no
        # network access, and no chance of drifting from the weights.
        name = bundle.get("clip_model_name", "CLIP vision tower")
        self.note(f"building {name}")
        cfg = CLIPVisionConfig(**bundle["clip_config"])
        clip = CLIPVisionModelWithProjection(cfg)
        self.note(f"loading {len(bundle['clip_state_dict']):,} tower tensors")
        clip.load_state_dict(bundle["clip_state_dict"], strict=True)
        n_params = sum(p.numel() for p in clip.parameters())
        self.note(f"tower ready · {n_params / 1e6:.0f}M frozen parameters"
                  f" · moving to {str(self.device).upper()}")
        clip.eval().to(self.device)
        for p in clip.parameters():
            p.requires_grad_(False)
        self.clip = clip

        hc = bundle.get("head_config") or {}
        hidden = int(hc.get("hidden", 512))
        self.note(f"loading {hc.get('head', 'mlp')} head "
                  f"{int(bundle['head_dim'])} -> {hidden} -> {hidden // 2} -> 1")
        head = _head_class(torch)(
            int(bundle["head_dim"]), hc.get("head", "mlp"),
            hidden, float(hc.get("dropout", 0.3)))
        head.load_state_dict(bundle["head_state_dict"], strict=True)
        head.eval().to(self.device)
        self.head = head

        self.mu = torch.as_tensor(bundle["mu"], dtype=torch.float32).to(self.device)
        self.sd = torch.as_tensor(bundle["sd"], dtype=torch.float32).to(self.device)

        # The bundle's threshold lives in logit space; the scores we hand back
        # are Platt-calibrated, so put the operating point in the same units.
        # It is not 0.5 and it is not supposed to be: it was chosen for 1% FPR
        # on real images, which pushes it well up the scale.
        self.default_threshold = 1.0 / (
            1.0 + np.exp(-(self.platt_a * self.threshold_logit + self.platt_b)))
        self.note(f"ready · feature={self.feature} · preproc={self.preproc}"
                  f" · operating point {self.default_threshold:.3f}")

    def unload(self) -> None:
        self.clip = None
        self.head = None
        self.mu = self.sd = None

    # -- inference ---------------------------------------------------------
    def predict_batch(self, paths: list) -> list:
        images, keep = [], []
        for i, p in enumerate(paths):
            img = self._open(p)
            if img is not None:
                images.append(img)
                keep.append(i)

        scores = [float("nan")] * len(paths)      # unreadable files stay NaN
        for i, s in zip(keep, self.predict_images(images)):
            scores[i] = s
        return scores

    def predict_images(self, images: list) -> list:
        """Score decoded PIL images. The robustness sweep calls this directly."""
        if not images:
            return []
        self.ensure_loaded()
        torch = self._torch

        batch = torch.from_numpy(
            np.stack([self._to_clip_array(img) for img in images])).to(self.device)

        with torch.no_grad():
            feats = self._embed(batch)
            if self.l2:
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-8)
            feats = (feats - self.mu) / self.sd
            logits = self.head(feats)             # (B,), the head squeezes
            probs = torch.sigmoid(self.platt_a * logits + self.platt_b)

        return [round(float(p), 6) for p in probs.float().cpu()]

    def _embed(self, batch):
        """Tower forward, honouring whichever feature the head was trained on."""
        out = self.clip.vision_model(pixel_values=batch)
        pooled = out.pooler_output
        if self.feature == "pooled":
            return pooled.float()
        proj = self.clip.visual_projection(pooled)
        if self.feature == "proj":
            return proj.float()
        if self.feature == "both":
            return self._torch.cat([pooled, proj], dim=-1).float()
        raise ValueError(f"unknown feature {self.feature!r}")

    # -- preprocessing -----------------------------------------------------
    def prepare_source(self, img: Image.Image) -> Image.Image:
        """Re-encode as JPEG, exactly as training did to every image.

        Mirrors augment.op_jpeg. subsampling=2 (4:2:0) is not incidental -
        it is what the training re-encode used, and chroma subsampling is
        precisely the kind of high-frequency damage this model is asked to
        see through. The sweep calls this before its own transforms, so a
        degraded image is re-encoded first and degraded second, as in training.
        """
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=SOURCE_JPEG_Q,
                                subsampling=2)
        buf.seek(0)
        out = Image.open(buf)
        out.load()
        return out.convert("RGB")

    def _open(self, path: str):
        """Decode at full resolution - no size cap.

        Deliberately not Detector.open_image: that pre-shrinks to 1024 with a
        bilinear filter, and the model was trained on a bicubic resize straight
        from the source. Two resamples are not the same as one.
        """
        try:
            img = Image.open(path)
            w, h = img.size
            if w * h > MAX_PIXELS:
                return None
            return self.prepare_source(img.convert("RGB"))
        except Exception:
            return None

    def _to_clip_array(self, img: Image.Image) -> np.ndarray:
        """Shortest side to 224 (bicubic), centre crop, CLIP normalisation."""
        img = img.convert("RGB")
        w, h = img.size
        s = RES / min(w, h)
        img = img.resize((max(RES, int(round(w * s))), max(RES, int(round(h * s)))),
                         Image.Resampling.BICUBIC)
        w, h = img.size
        left, top = (w - RES) // 2, (h - RES) // 2
        img = img.crop((left, top, left + RES, top + RES))

        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        arr = arr - np.array(CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
        arr /= np.array(CLIP_STD, dtype=np.float32).reshape(3, 1, 1)
        return arr

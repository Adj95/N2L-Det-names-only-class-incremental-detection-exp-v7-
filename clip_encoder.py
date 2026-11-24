"""clip_encoder.py -- crop preprocessing (numpy/PIL) + a thin open_clip wrapper.

Everything except `ClipEncoder` is torch-free and unit-tested. `ClipEncoder` is deliberately
tiny (it only loads weights and calls encode_image / encode_text) so there is little untested
surface; it is exercised by the Colab smoke test `step2_colab_smoke.py`.
"""
from __future__ import annotations

import re
import warnings
from typing import List

import numpy as np
from PIL import Image

from boxes import pad_boxes

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def crop_resize(img: Image.Image, boxes_xyxy, size: int = 224) -> np.ndarray:
    """Squash-resize each box region to size x size (bicubic) like the official CIFAR224
    pipeline (Resize((224,224), BICUBIC)). Returns uint8 (n, size, size, 3)."""
    img = img.convert("RGB")
    W, H = img.size
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    out = np.zeros((len(boxes), size, size, 3), dtype=np.uint8)
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        ix1, iy1 = int(np.floor(max(x1, 0))), int(np.floor(max(y1, 0)))
        ix2, iy2 = int(np.ceil(min(x2, W))), int(np.ceil(min(y2, H)))
        ix2, iy2 = max(ix2, ix1 + 1), max(iy2, iy1 + 1)       # never an empty crop
        ix1, iy1 = min(ix1, W - 1), min(iy1, H - 1)
        ix2, iy2 = min(max(ix2, ix1 + 1), W), min(max(iy2, iy1 + 1), H)
        out[k] = np.asarray(img.crop((ix1, iy1, ix2, iy2)).resize((size, size), Image.BICUBIC))
    return out


def crops_for_image(img: Image.Image, boxes_xyxy, pad_frac: float = 0.1, size: int = 224):
    """Pad boxes by `pad_frac` (context), then crop+resize. Returns uint8 (n,size,size,3)."""
    W, H = img.size
    return crop_resize(img, pad_boxes(boxes_xyxy, pad_frac, W, H), size)


def to_clip_input(crops_u8: np.ndarray) -> np.ndarray:
    """uint8 (n,H,W,3) -> float32 (n,3,H,W) normalised with CLIP mean/std."""
    x = crops_u8.astype(np.float32) / 255.0
    x = (x - CLIP_MEAN) / CLIP_STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


_QG_MISMATCH = re.compile(r"QuickGELU mismatch.*pretrained tag .*?\(quick_gelu=(True|False)\)", re.S)


def create_clip(open_clip, model_name: str, pretrained: str):
    """open_clip.create_model_and_transforms, but never silently with the wrong activation.

    Some weights (e.g. 'openai') were trained with QuickGELU while the plain 'ViT-B-16' config uses GELU;
    open_clip only WARNS about it and then computes slightly wrong features. We read the warning, rebuild the
    model with the activation the weights expect, and raise if a mismatch remains."""
    def build(qg):
        kw = {} if qg is None else {"force_quick_gelu": qg}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, **kw)
        bad = [str(w.message) for w in caught if "QuickGELU mismatch" in str(w.message)]
        for w in caught:                                      # pass every other warning through untouched
            if "QuickGELU mismatch" not in str(w.message):
                warnings.warn(w.message, w.category)
        return out, bad

    out, bad = build(None)
    if bad:
        m = _QG_MISMATCH.search(bad[0])
        if not m:
            raise RuntimeError(f"unparseable QuickGELU mismatch warning: {bad[0]}")
        out, bad = build(m.group(1) == "True")
        if bad:
            raise RuntimeError(f"QuickGELU mismatch persists for {model_name}/{pretrained}: {bad[0]}")
    return out


class ClipEncoder:
    """open_clip ViT-B/16. `pretrained` e.g. 'openai' or 'laion400m_e32' (the weights the
    official N2L CIFAR config uses). Returns RAW (un-normalised) float32 numpy features."""

    def __init__(self, pretrained: str = "openai", model_name: str = "ViT-B-16",
                 device: str = None, fp16: bool = True):
        import open_clip, torch  # noqa: lazy -- only needed on a GPU box
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and self.device == "cuda"
        self.model, _, _ = create_clip(open_clip, model_name, pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval().to(self.device)
        if self.fp16:
            self.model.half()
        self.dim = int(self.model.visual.output_dim)

    def encode_text(self, texts: List[str], batch: int = 256) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch):
                tok = self.tokenizer(list(texts[i:i + batch])).to(self.device)
                out.append(self.model.encode_text(tok).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

    def encode_images_u8(self, crops_u8: np.ndarray, batch: int = 256) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.no_grad():
            for i in range(0, len(crops_u8), batch):
                x = torch.from_numpy(to_clip_input(crops_u8[i:i + batch])).to(self.device)
                if self.fp16:
                    x = x.half()
                out.append(self.model.encode_image(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)

"""prompts.py -- prompt templates and CLIP text embeddings (encoder is injected)."""
from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np

# first template is the one the official N2L code uses for pseudo-labelling
DET_TEMPLATES = [
    "a photo of a {}.", "a cropped photo of a {}.", "a close-up photo of a {}.",
    "a low resolution photo of a {}.", "a blurry photo of a {}.", "a bad photo of a {}.",
    "a good photo of a {}.", "a photo of a small {}.", "a photo of a large {}.",
    "a photo of the {}.",
]

# Heuristic 'background' prompts (full sentences, no {}). Quality is measured in step 4
# against GT and is an ablation row -- do not treat as given.
BG_PROMPTS = [
    "a photo of the background.", "a photo of a wall.", "a photo of the sky.",
    "a photo of grass.", "a photo of the ground.", "a photo of a road.",
    "a blurry photo with no clear object.", "a photo of a small part of an object.",
]


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)


def _ensemble(texts_per_item: Sequence[Sequence[str]], encode_text: Callable) -> np.ndarray:
    """Official ensembling: normalise each prompt embedding, average, renormalise."""
    flat = [t for ts in texts_per_item for t in ts]
    emb = l2norm(encode_text(flat))
    out, k = [], 0
    for ts in texts_per_item:
        out.append(emb[k:k + len(ts)].mean(axis=0))
        k += len(ts)
    return l2norm(np.stack(out))


def class_embeddings(names: Sequence[str], encode_text: Callable,
                     templates: Sequence[str] = DET_TEMPLATES) -> np.ndarray:
    """(K, D) unit-norm text embeddings, one per class name."""
    if not names:
        return np.zeros((0, 0))
    return _ensemble([[t.format(n) for t in templates] for n in names], encode_text)


def background_embedding(encode_text: Callable, prompts: Sequence[str] = BG_PROMPTS) -> np.ndarray:
    """(1, D) unit-norm embedding for the background pseudo-class."""
    return _ensemble([list(prompts)], encode_text)


def zero_shot_logits(image_feats: np.ndarray, text_emb: np.ndarray, scale: float = 100.0):
    """scale * cos(image, text). image_feats are RAW features (normalised here)."""
    return scale * l2norm(image_feats) @ np.asarray(text_emb, dtype=np.float64).T

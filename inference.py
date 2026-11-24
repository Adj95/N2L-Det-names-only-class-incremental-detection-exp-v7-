"""inference.py -- turn per-box head scores into detections.

fused[:, 0] is background; fused[:, 1+j] is the j-th learned class (class_order[j]).
Detection score for class c on a box = fused score; background only competes inside the
softmax / ridge, it is never output.
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np
from scipy.special import softmax

from boxes import batched_nms


@dataclasses.dataclass
class InferConfig:
    fusion_weight: float = 0.8     # weight on the ridge head (official N2L); 1-w on zero-shot CLIP
    topk_classes: int = 3          # candidate classes per box
    min_score: float = 0.001
    nms_iou: float = 0.5
    max_dets: int = 100
    yolo_gamma: float = 0.5        # multiply by (proposal score)**gamma (localisation quality); 0 = off
    ridge_norm: bool = False       # clip ridge scores at 0 and renormalise each row to sum 1 before fusing
    geo_fusion: bool = False       # geometric instead of arithmetic fusion: p^(1-w) * r^w (rows renormalised)
    # --- experimental rescoring knobs (N2L-Det-exp, round 1). Defaults = OFF = identical to the frozen pipeline.
    gamma_new: float = -1.0        # >=0: YOLO-score exponent for classes the detector was NOT trained on (base keeps yolo_gamma)
    clip_obj: float = 0.0          # >0: also multiply by (1 - background prob of the fused row) ** clip_obj
    yolo_floor: float = 0.0        # >0: proposal scores below this are raised to it before the exponent


def fuse(zs_logits: np.ndarray, ridge_scores: Optional[np.ndarray], cfg: InferConfig) -> np.ndarray:
    """(1-w)*softmax(zero-shot logits) + w*ridge scores; zero-shot only if ridge_scores is None."""
    p = softmax(np.asarray(zs_logits, dtype=np.float64), axis=1)
    if ridge_scores is None:
        return p
    r = np.asarray(ridge_scores, dtype=np.float64)
    if r.shape != p.shape:
        raise ValueError(f"ridge scores {r.shape} vs zero-shot {p.shape}")
    if cfg.ridge_norm:
        r = np.clip(r, 0.0, None)
        tot = r.sum(axis=1, keepdims=True)
        r = np.where(tot > 0, r / np.where(tot > 0, tot, 1.0), p)          # an all-zero row falls back to the zero-shot row
    if cfg.geo_fusion:
        g = p ** (1.0 - cfg.fusion_weight) * np.clip(r, 1e-6, None) ** cfg.fusion_weight
        return g / g.sum(axis=1, keepdims=True)
    return (1.0 - cfg.fusion_weight) * p + cfg.fusion_weight * r


def detect(boxes: np.ndarray, yolo_scores: np.ndarray, fused: np.ndarray,
           class_order: Sequence[int], cfg: InferConfig, base_classes: Optional[Sequence[int]] = None):
    """Returns (boxes (m,4), scores (m,), class idx (m,)) after class-wise NMS, sorted by score.
    base_classes (the classes the frozen detector was trained on) is needed only when cfg.gamma_new >= 0."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    fg = np.asarray(fused, dtype=np.float64)[:, 1:]
    if fg.shape[1] != len(class_order):
        raise ValueError(f"{fg.shape[1]} class columns but class_order has {len(class_order)}")
    n, K = fg.shape
    if n == 0 or K == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64)
    k = min(cfg.topk_classes, K)
    cols = np.argsort(-fg, axis=1, kind="stable")[:, :k]              # (n,k)
    sc = np.take_along_axis(fg, cols, axis=1)
    y = np.clip(np.asarray(yolo_scores, dtype=np.float64), 1e-6, 1.0)
    if cfg.yolo_floor > 0:
        y = np.maximum(y, min(float(cfg.yolo_floor), 1.0))
    if cfg.gamma_new >= 0:
        if base_classes is None:
            raise ValueError("gamma_new >= 0 needs base_classes (the classes the detector was trained on)")
        is_base = np.isin(np.asarray(class_order, dtype=np.int64)[cols], np.asarray(list(base_classes), dtype=np.int64))
        sc = sc * y[:, None] ** np.where(is_base, cfg.yolo_gamma, cfg.gamma_new)
    elif cfg.yolo_gamma > 0:
        sc = sc * (y ** cfg.yolo_gamma)[:, None]
    if cfg.clip_obj > 0:
        obj = np.clip(1.0 - np.asarray(fused, dtype=np.float64)[:, 0], 0.0, 1.0)
        sc = sc * (obj ** cfg.clip_obj)[:, None]
    bi = np.repeat(np.arange(n), k)
    cols, sc = cols.reshape(-1), sc.reshape(-1)
    ok = sc > cfg.min_score
    bi, cols, sc = bi[ok], cols[ok], sc[ok]
    if not len(sc):
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64)
    labels = np.asarray(class_order, dtype=np.int64)[cols]
    keep = batched_nms(boxes[bi], sc, labels, cfg.nms_iou, max_keep=cfg.max_dets)
    return boxes[bi][keep], sc[keep], labels[keep]

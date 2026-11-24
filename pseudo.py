"""pseudo.py -- turn CLIP crop features into pseudo-label logits for N2L (step 4).

Head column layout (fixed for the whole run):   col 0 = background,  col 1+j = j-th LEARNED class
(`class_order[j]`). New tasks append columns, so the layout never shifts.

Ground truth appears here ONLY in `gt_targets` / `label_quality`, which are diagnostics. Nothing
that feeds training imports them (tests enforce this).
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Sequence

import numpy as np
from scipy.special import softmax

from boxes import iou_matrix
from prompts import l2norm


@dataclasses.dataclass
class PseudoConfig:
    scale: float = 100.0      # CLIP logit scale
    use_bg: bool = True       # include the background prompt as column 0
    tau_pos: float = 0.0      # drop a sample if its top class is a real class with softmax prob < tau_pos
    tau_bg: float = 0.0       # ... or the background with prob < tau_bg
    use_loc: bool = False     # weight samples by the proposal (YOLO) confidence
    loc_pow: float = 1.0
    loc_floor: float = 0.05   # never exactly 0 so low-score boxes still count a little
    min_yolo: float = 0.0     # drop proposals whose detector score is below this (hard objectness filter; 0 = off)
    class_topfrac: float = 1.0  # per pseudo-class, keep only this top fraction by confidence (1 = all)


def class_to_col(class_order: Sequence[int]) -> Dict[int, int]:
    return {int(c): j + 1 for j, c in enumerate(class_order)}


def build_label_logits(feats_raw: np.ndarray, names_emb: np.ndarray, bg_emb: Optional[np.ndarray],
                       cfg: PseudoConfig) -> np.ndarray:
    """(n, 1+K) logits [bg, names...] (or (n,K) when use_bg=False)."""
    T = np.asarray(names_emb, dtype=np.float64)
    if cfg.use_bg:
        if bg_emb is None:
            raise ValueError("use_bg=True needs bg_emb")
        T = np.vstack([np.asarray(bg_emb, dtype=np.float64).reshape(1, -1), T])
    return cfg.scale * l2norm(feats_raw) @ T.T


def build_task_inputs(feats_raw: np.ndarray, yolo_scores: np.ndarray, names_emb: np.ndarray,
                      bg_emb: Optional[np.ndarray], cfg: PseudoConfig):
    """Returns (keep_mask (n,), logits (n_keep, C), base_weights (n_keep,) or None).
    Ignored samples are REMOVED (not zero-weighted): n and class counts feed the inter-class
    weights, so zero-weight rows would silently distort them."""
    logits = build_label_logits(feats_raw, names_emb, bg_emb, cfg)
    p = softmax(logits, axis=1)
    top = p.argmax(1)
    ptop = p[np.arange(len(p)), top]
    is_bg = (top == 0) if cfg.use_bg else np.zeros(len(top), bool)
    drop = np.where(is_bg, ptop < cfg.tau_bg, ptop < cfg.tau_pos)
    keep = ~drop
    if cfg.min_yolo > 0:
        keep &= np.asarray(yolo_scores, dtype=np.float64) >= cfg.min_yolo
    if cfg.class_topfrac < 1.0:
        for g in np.unique(top[keep]):
            idx = np.flatnonzero(keep & (top == g))
            n_keep = max(1, int(np.ceil(cfg.class_topfrac * len(idx))))
            if n_keep < len(idx):
                worst = idx[np.argsort(-ptop[idx], kind="stable")[n_keep:]]
                keep[worst] = False
    w = None
    if cfg.use_loc:
        s = np.clip(np.asarray(yolo_scores, dtype=np.float64), 0, 1) ** cfg.loc_pow
        w = (cfg.loc_floor + (1 - cfg.loc_floor) * s)[keep]
    return keep, logits[keep], w


# ------------------------------------------------------------------ diagnostics only
def gt_targets(prop_boxes: np.ndarray, gt_boxes: np.ndarray, gt_cls: np.ndarray,
               col_of_class: Dict[int, int], pos_iou: float = 0.5, neg_iou: float = 0.4):
    """Head column a perfect labeller would give each proposal: best-IoU GT's column if IoU>=pos_iou,
    0 (background) if max IoU < neg_iou, -1 (ambiguous) in between. GT classes not in
    col_of_class (not learned yet) count as ambiguous when they overlap (-1), never as background."""
    n = len(prop_boxes)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    if len(gt_boxes) == 0:
        return np.zeros(n, np.int64), np.zeros(n)
    iou = iou_matrix(prop_boxes, gt_boxes)
    j = iou.argmax(1)
    best = iou[np.arange(n), j]
    out = np.full(n, -1, np.int64)
    out[best < neg_iou] = 0
    pos = best >= pos_iou
    for i in np.flatnonzero(pos):
        out[i] = col_of_class.get(int(gt_cls[j[i]]), -1)
    return out, best


def label_quality(pred_cols: np.ndarray, true_cols: np.ndarray) -> dict:
    """Compare pseudo-labels with GT-derived targets on proposals where the target is determinate
    (>=0). Reports overall accuracy and bg / foreground precision & recall."""
    pred_cols, true_cols = np.asarray(pred_cols), np.asarray(true_cols)
    m = true_cols >= 0
    p, t = pred_cols[m], true_cols[m]
    n = len(t)
    if n == 0:
        return {"n": 0}
    fg_t, fg_p = t > 0, p > 0
    f = lambda a, b: float(a / b) if b else float("nan")
    return {
        "n": int(n), "acc": float(np.mean(p == t)),
        "bg_precision": f(np.sum((p == 0) & (t == 0)), np.sum(p == 0)),
        "bg_recall": f(np.sum((p == 0) & (t == 0)), np.sum(t == 0)),
        "fg_precision": f(np.sum(fg_p & fg_t & (p == t)), np.sum(fg_p)),   # right class, not just 'object'
        "fg_recall": f(np.sum(fg_p & fg_t & (p == t)), np.sum(fg_t)),
        "frac_bg_truth": float(np.mean(t == 0)),
    }

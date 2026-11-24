"""boxes.py -- box geometry + NMS in plain numpy. All boxes are xyxy, float, pixels."""
from __future__ import annotations

import numpy as np


def xywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def xyxy_to_xywh(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def box_area(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def iou_matrix(a, b):
    """(n,4) x (m,4) -> (n,m). Degenerate (zero-area) boxes get IoU 0 with everything."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def clip_boxes(boxes, w, h):
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, h)
    return b


def pad_boxes(boxes, frac, w, h, min_size=2.0):
    """Grow each box by `frac` of its own width/height on every side, clip to the image,
    and guarantee at least `min_size` px in each dimension (when the image allows)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    b[:, 0] -= frac * bw; b[:, 2] += frac * bw
    b[:, 1] -= frac * bh; b[:, 3] += frac * bh
    b = clip_boxes(b, w, h)
    for lo, hi, lim in ((0, 2, w), (1, 3, h)):
        short = (b[:, hi] - b[:, lo]) < min_size
        if short.any():
            c = (b[short, lo] + b[short, hi]) / 2
            new_lo = np.clip(c - min_size / 2, 0, max(lim - min_size, 0))
            b[short, lo] = new_lo
            b[short, hi] = np.minimum(new_lo + min_size, lim)
    return b


def nms(boxes, scores, iou_thr, max_keep=None):
    """Greedy NMS. Returns kept indices sorted by descending score (stable on ties)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores differ in length")
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if max_keep is not None and len(keep) >= max_keep:
            break
        if order.size == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thr]
    return np.array(keep, dtype=np.int64)


def batched_nms(boxes, scores, labels, iou_thr, max_keep=None):
    """Class-wise NMS: boxes only suppress boxes with the SAME label."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        keep.extend(idx[nms(boxes[idx], scores[idx], iou_thr)].tolist())
    keep = np.array(sorted(keep, key=lambda i: (-scores[i], i)), dtype=np.int64)
    return keep if max_keep is None else keep[:max_keep]

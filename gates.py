"""gates.py -- measurements that decide whether the next build step is worth doing."""
from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np
from PIL import Image

from boxes import iou_matrix
from clip_encoder import crops_for_image
from prompts import zero_shot_logits


def gt_crop_zeroshot_eval(items: Iterable[Tuple[Image.Image, np.ndarray, np.ndarray]],
                          encode_u8_fn: Callable, text_emb: np.ndarray, pad_frac: float = 0.1,
                          class_subset=None):
    """GATE (E0 ceiling): zero-shot CLIP accuracy on GROUND-TRUTH-box crops.
    items: (PIL image, gt boxes xyxy (n,4), gt class idx (n,)).  text_emb rows are the
    candidate classes in order; if class_subset is given, only those columns compete and only
    GT objects of those classes are scored.  Returns dict(acc, n, per_class_acc)."""
    cols = np.arange(text_emb.shape[0]) if class_subset is None else np.asarray(class_subset)
    T = text_emb[cols]
    correct = np.zeros(text_emb.shape[0]); total = np.zeros(text_emb.shape[0])
    for img, boxes, cls in items:
        cls = np.asarray(cls)
        sel = np.isin(cls, cols)
        if not sel.any():
            continue
        feats = encode_u8_fn(crops_for_image(img, boxes[sel], pad_frac))
        pred = cols[zero_shot_logits(feats, T).argmax(1)]
        for c, p in zip(cls[sel], pred):
            total[c] += 1; correct[c] += (c == p)
    n = int(total.sum())
    per = np.where(total > 0, correct / np.maximum(total, 1), np.nan)
    return {"acc": float(correct.sum() / n) if n else float("nan"), "n": n, "per_class": per}


def box_recall(gt_boxes: np.ndarray, gt_cls: np.ndarray, prop_boxes: np.ndarray, classes,
               iou_thrs=(0.5, 0.7)):
    """GATE 1 building block: for GT objects of `classes`, the fraction that has at least one
    proposal with IoU >= thr (no one-to-one matching; this is 'could any box be right')."""
    sel = np.isin(gt_cls, list(classes))
    g = gt_boxes[sel]
    out = {"n_gt": int(sel.sum())}
    if not len(g):
        return {**out, **{f"hit@{t}": 0 for t in iou_thrs}}
    best = iou_matrix(g, prop_boxes).max(axis=1) if len(prop_boxes) else np.zeros(len(g))
    return {**out, **{f"hit@{t}": int((best >= t).sum()) for t in iou_thrs}}


def recall_over_dataset(proposals, gt, groups, iou_thrs=(0.5, 0.7)):
    """GATE 1 over a whole set of images.
    proposals: image_id -> boxes (n,4);  gt: image_id -> (boxes, class idx);
    groups: {'base': [...class idx...], 'new': [...]}.
    Returns {group: {n_gt, recall@0.5, recall@0.7}} and 'boxes_per_image'."""
    agg = {g: {"n_gt": 0, **{f"hit@{t}": 0 for t in iou_thrs}} for g in groups}
    n_box = 0
    for iid, (gb, gc) in gt.items():
        pb = np.asarray(proposals.get(iid, np.zeros((0, 4))))
        n_box += len(pb)
        for g, cls in groups.items():
            r = box_recall(gb, gc, pb, cls, iou_thrs)
            for k, v in r.items():
                agg[g][k] += v
    out = {}
    for g, a in agg.items():
        out[g] = {"n_gt": a["n_gt"],
                  **{f"recall@{t}": (a[f"hit@{t}"] / a["n_gt"] if a["n_gt"] else float("nan"))
                     for t in iou_thrs}}
    out["boxes_per_image"] = n_box / max(len(gt), 1)
    return out

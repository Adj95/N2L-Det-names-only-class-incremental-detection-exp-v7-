"""synth.py -- a synthetic detection world at the FEATURE level (no images, no models).

Used (1) by the unit tests of the incremental runner and (2) by `run_pipeline.py selftest` as a
pre-flight check on Colab before any GPU time is spent.

World: C classes with random prototypes; every proposal's 'CLIP feature' is its prototype
(object / background / half-half for partial overlaps) times `scale` plus N(0,1) noise; the
'text embedding' of class c is its prototype plus `text_noise` noise (the teacher's quality).
"""
from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Sequence

import numpy as np

import coco_utils as cu
from boxes import iou_matrix
from feature_cache import CacheWriter, FeatureCache
from prompts import l2norm


@dataclasses.dataclass
class SynthWorld:
    tasks: List[List[int]]
    train_cache: FeatureCache
    train_ids: List[List[int]]
    train_gt: Dict[int, tuple]
    eval_cache: FeatureCache
    eval_ids: List[int]
    eval_coco: dict
    text_label: np.ndarray
    text_eval: np.ndarray
    bg_label: np.ndarray
    bg_eval: np.ndarray
    dim: int


def _boxes_for_objects(rng, n, canvas=300):
    boxes = []
    for _ in range(n * 10):
        w, h = rng.uniform(40, 110, size=2)
        x, y = rng.uniform(0, canvas - w), rng.uniform(0, canvas - h)
        b = np.array([x, y, x + w, y + h])
        if all(iou_matrix(b[None], np.array([o]))[0, 0] < 0.05 for o in boxes):
            boxes.append(b)
        if len(boxes) == n:
            break
    return np.array(boxes).reshape(-1, 4)


def make_feature_world(out_dir: str, tasks: Sequence[Sequence[int]], dim: int = 64,
                       imgs_per_task: int = 60, n_eval: int = 80, sep: float = 1.0,
                       text_noise: float = 0.5, scale: float = 5.0, p_old: float = 0.4,
                       seed: int = 0, canvas: int = 300) -> SynthWorld:
    rng = np.random.default_rng(seed)
    tasks = [list(t) for t in tasks]
    C = max(max(t) for t in tasks) + 1
    P = rng.normal(scale=sep, size=(C, dim)); bgp = rng.normal(scale=sep, size=(dim,))
    text_label = l2norm(P + rng.normal(scale=text_noise, size=P.shape))
    text_eval = l2norm(P + rng.normal(scale=text_noise, size=P.shape))
    bg_text = l2norm(bgp + rng.normal(scale=text_noise, size=(1, dim)))

    def make_image(classes):
        gt_b = _boxes_for_objects(rng, len(classes), canvas)
        classes = np.array(classes[:len(gt_b)])
        boxes, feats, scores = [], [], []
        for b, c in zip(gt_b, classes):
            for _ in range(2):                                      # two good proposals per object
                jit = b + rng.normal(scale=0.04 * (b[2] - b[0]), size=4)
                jit[2] = max(jit[2], jit[0] + 5); jit[3] = max(jit[3], jit[1] + 5)
                boxes.append(jit); feats.append(P[c] * scale + rng.normal(size=dim))
        for _ in range(4):                                          # background-ish proposals
            w, h = rng.uniform(20, 90, size=2)
            x, y = rng.uniform(0, canvas - w), rng.uniform(0, canvas - h)
            b = np.array([x, y, x + w, y + h])
            iou = iou_matrix(b[None], gt_b).max() if len(gt_b) else 0.0
            if iou >= 0.5:
                continue
            j = iou_matrix(b[None], gt_b).argmax() if len(gt_b) else 0
            mix = 0.0 if iou < 0.3 else 0.5
            proto = (1 - mix) * bgp + mix * (P[classes[j]] if len(gt_b) else bgp)
            boxes.append(b); feats.append(proto * scale + rng.normal(size=dim))
        boxes = np.array(boxes).reshape(-1, 4)
        iou_best = iou_matrix(boxes, gt_b).max(1) if len(gt_b) else np.zeros(len(boxes))
        scores = np.clip(0.2 + 0.7 * iou_best + rng.normal(scale=0.05, size=len(boxes)), 0.01, 0.99)
        return gt_b, classes, boxes, np.array(feats), scores

    os.makedirs(out_dir, exist_ok=True)
    # ---- train images, per task (an image may contain old-class objects)
    wtr = CacheWriter(os.path.join(out_dir, "train"), dim)
    train_ids, train_gt, iid = [], {}, 0
    for t, cls_t in enumerate(tasks):
        ids_t = []
        older = [c for tt in tasks[:t] for c in tt]
        for _ in range(imgs_per_task):
            k = int(rng.integers(1, 4))
            cls = [int(rng.choice(cls_t))] + [int(rng.choice(cls_t)) for _ in range(k - 1)]
            if older and rng.random() < p_old:
                cls.append(int(rng.choice(older)))
            iid += 1
            gb, gc, b, f, s = make_image(cls)
            wtr.add_image(iid, b, s, f)
            train_gt[iid] = (gb, gc); ids_t.append(iid)
        train_ids.append(ids_t)
    wtr.save()
    # ---- eval images: any class
    allc = [c for t in tasks for c in t]
    wev = CacheWriter(os.path.join(out_dir, "eval"), dim)
    eval_ids, images, anns = [], [], []
    for _ in range(n_eval):
        k = int(rng.integers(1, 4))
        cls = [int(rng.choice(allc)) for _ in range(k)]
        iid += 1
        gb, gc, b, f, s = make_image(cls)
        wev.add_image(iid, b, s, f)
        eval_ids.append(iid)
        images.append({"id": iid, "file_name": f"{iid}.jpg", "width": canvas, "height": canvas})
        for bb, c in zip(gb, gc):
            anns.append({"image_id": iid, "category_id": cu.COCO_IDS[int(c)], "iscrowd": 0,
                         "bbox": [float(bb[0]), float(bb[1]), float(bb[2] - bb[0]), float(bb[3] - bb[1])]})
    wev.save()
    return SynthWorld(tasks, FeatureCache(os.path.join(out_dir, "train")), train_ids, train_gt,
                      FeatureCache(os.path.join(out_dir, "eval")), eval_ids,
                      {"images": images, "annotations": anns},
                      np.vstack([text_label, np.zeros((80 - C, dim))]) if C < 80 else text_label,
                      np.vstack([text_eval, np.zeros((80 - C, dim))]) if C < 80 else text_eval,
                      bg_text, bg_text, dim)

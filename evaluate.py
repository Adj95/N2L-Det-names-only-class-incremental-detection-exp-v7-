"""evaluate.py -- COCO-style mAP via pycocotools, restricted to a set of images and classes."""
from __future__ import annotations

import contextlib
import io
from typing import Dict, List, Optional, Sequence

import numpy as np

from boxes import xyxy_to_xywh
from coco_utils import COCO_IDS, COCO_NAMES


def make_coco_gt(coco: dict, image_ids: Sequence[int]):
    """pycocotools COCO object over the given images (crowd annotations kept so they are
    handled the standard way)."""
    from pycocotools.coco import COCO
    ids = set(int(i) for i in image_ids)
    anns = []
    for k, a in enumerate(a for a in coco["annotations"] if a["image_id"] in ids):
        a = dict(a)
        a.setdefault("id", k + 1)
        a.setdefault("iscrowd", 0)
        a.setdefault("area", float(a["bbox"][2] * a["bbox"][3]))
        anns.append(a)
    for k, a in enumerate(anns):                       # guarantee unique ids
        a["id"] = k + 1
    ds = {"images": [im for im in coco["images"] if im["id"] in ids], "annotations": anns,
          "categories": [{"id": i, "name": n} for i, n in zip(COCO_IDS, COCO_NAMES)]}
    gt = COCO()
    with contextlib.redirect_stdout(io.StringIO()):
        gt.dataset = ds
        gt.createIndex()
    return gt


def dets_to_coco(image_id: int, boxes_xyxy, scores, cls_idx) -> List[dict]:
    b = xyxy_to_xywh(boxes_xyxy)
    return [{"image_id": int(image_id), "category_id": int(COCO_IDS[int(c)]),
             "bbox": [float(v) for v in bb], "score": float(s)}
            for bb, s, c in zip(b, np.asarray(scores).reshape(-1), np.asarray(cls_idx).reshape(-1))]


def coco_eval(gt, dets: List[dict], image_ids: Sequence[int], class_idx: Sequence[int],
              max_dets: int = 100) -> Dict:
    """AP@[.5:.95], AP50, AP75 over `class_idx` and `image_ids`; plus per-class AP50.
    Classes with no GT in these images give nan AP (not 0)."""
    from pycocotools.cocoeval import COCOeval
    class_idx = [int(c) for c in class_idx]
    cat_ids = [COCO_IDS[c] for c in class_idx]
    keep, img_set = set(cat_ids), set(int(i) for i in image_ids)
    dets = [d for d in dets if d["category_id"] in keep and d["image_id"] in img_set]
    n_gt = sum(1 for a in gt.dataset["annotations"]
               if a["category_id"] in keep and a["image_id"] in img_set and not a["iscrowd"])
    out = {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan"), "n_gt": n_gt,
           "per_class_AP50": {c: float("nan") for c in class_idx}}
    if n_gt == 0:
        return out
    if not dets:
        out.update(AP=0.0, AP50=0.0, AP75=0.0)
        out["per_class_AP50"] = {c: 0.0 for c in class_idx}
        return out
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        E = COCOeval(gt, dt, "bbox")
        E.params.imgIds = sorted(int(i) for i in image_ids)
        E.params.catIds = cat_ids
        E.params.maxDets = [1, 10, max_dets]
        E.evaluate(); E.accumulate(); E.summarize()
    out.update(AP=float(E.stats[0]), AP50=float(E.stats[1]), AP75=float(E.stats[2]))
    prec = E.eval["precision"]                          # [T,R,K,A,M]
    # COCOeval sorts and de-duplicates params.catIds, so the K axis follows ascending category id, NOT the order
    # of class_idx. Look each class up by its id (class_idx in task order, e.g. VOC, is not id-sorted).
    k_of = {int(cid): k for k, cid in enumerate(E.params.catIds)}
    for c in class_idx:
        p = prec[0, :, k_of[COCO_IDS[c]], 0, 2]
        p = p[p > -1]
        out["per_class_AP50"][c] = float(p.mean()) if p.size else float("nan")
    return out

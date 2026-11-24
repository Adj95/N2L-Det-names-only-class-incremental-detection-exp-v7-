"""baselines.py -- reference methods scored with the SAME images, folds and COCOeval as our runs.

  yolo_world : open-vocabulary detector (ultralytics YOLO-World), zero-shot on all class names at once.
               Not incremental and not trained on our base classes; it is a *reference point*, not a competitor under
               our protocol. CAVEAT for the paper: YOLO-World was pre-trained on Objects365 / GQA / Flickr30k, so the
               "new" classes are not novel for it.

The plain-ridge-on-pseudo-labels baseline (no N2L refinement, no bi-level weights) lives in pipeline.ablation_configs
as B_plain_ridge, because it shares the frozen proposals and features with every other run.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import incremental as inc
from evaluate import coco_eval, dets_to_coco
from yolo_tools import results_to_arrays


class WorldDetector:
    """YOLO-World with a fixed class list. __call__(PIL) -> (boxes xyxy, scores, class position in `names`)."""

    def __init__(self, names: Sequence[str], weights: str = "yolov8s-worldv2.pt", conf: float = 0.001, iou: float = 0.7,
                 max_det: int = 100, imgsz: int = 640, device=None, model=None):
        if model is None:
            from ultralytics import YOLOWorld               # lazy: GPU box only
            model = YOLOWorld(weights)
        model.set_classes(list(names))
        self.model, self.conf, self.iou, self.max_det, self.imgsz, self.device = model, conf, iou, max_det, imgsz, device

    def __call__(self, img: Image.Image):
        res = self.model.predict(source=np.asarray(img.convert("RGB"))[..., ::-1], conf=self.conf, iou=self.iou,
                                 max_det=self.max_det, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return results_to_arrays(res, img.width, img.height)


def score_detector(detector: Callable, images: Iterable[Tuple[int, Image.Image]], eval_ids: Sequence[int], eval_gt,
                   tasks, n_folds: int = 5, fold_seed: int = 123, name: str = "yolo_world", max_dets: int = 100) -> Dict:
    """Run `detector` (PIL -> boxes, scores, position in class_order) on the images and score it like run_incremental's
    final task. Class position j means class_order[j], where class_order = the tasks' classes concatenated."""
    class_order = [int(c) for t in tasks for c in t]
    dets: List[dict] = []
    for iid, img in images:
        b, s, c = detector(img)
        if len(b):
            order = np.argsort(-np.asarray(s), kind="stable")[:max_dets]
            dets += dets_to_coco(iid, np.asarray(b)[order], np.asarray(s)[order],
                                 np.asarray(class_order)[np.asarray(c, dtype=np.int64)[order]])
    r = coco_eval(eval_gt, dets, eval_ids, class_order)
    pc = r["per_class_AP50"]
    mean = lambda cs: float(np.nanmean([pc[k] for k in cs])) if cs else float("nan")
    old = [c for t in tasks[:-1] for c in t]
    row = {"task": len(tasks) - 1, "n_seen": len(class_order), "AP": r["AP"], "AP50": r["AP50"], "AP75": r["AP75"],
           "AP50_base": mean(tasks[0]), "AP50_old": mean(old), "AP50_new": mean(tasks[-1])}
    out = {"variant": name, "curve": [row], "diag": [], "A_bar_AP": r["AP"], "final_AP": r["AP"], "final_AP50": r["AP50"],
           "base_AP50_drop": float("nan")}
    folds = inc.image_folds(eval_ids, n_folds, fold_seed)
    if folds:
        out["final_folds"] = inc.fold_scores(eval_gt, dets, folds, class_order, list(tasks[-1]))
    return out

"""incremental.py -- the class-incremental experiment on CACHED features.

Variants (all share frozen proposals + frozen CLIP features, so they are directly comparable):
  'zeroshot'   E0 : CLIP text vs crop, no training at all
  'ours'       E5 : N2L head on pseudo-labels (names only, no GT labels)
  'gt_ceiling' E6 : same head trained on GROUND-TRUTH labels (upper bound for frozen YOLO+CLIP+ridge)
Ground truth is read only by 'gt_ceiling', by the diagnostics, and by the evaluator.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional

import numpy as np

import n2l_core as nc
import pseudo as ps
from boxes import iou_matrix
from evaluate import coco_eval, dets_to_coco
from feature_cache import FeatureCache
from heads import make_learner
from inference import InferConfig, detect, fuse
from prompts import l2norm


@dataclasses.dataclass
class RunConfig:
    tasks: List[List[int]]
    variant: str = "ours"                         # 'ours' | 'zeroshot' | 'gt_ceiling'
    pseudo: ps.PseudoConfig = dataclasses.field(default_factory=ps.PseudoConfig)
    n2l: nc.N2LConfig = dataclasses.field(
        default_factory=lambda: nc.N2LConfig(pca_rule="energy", pca_rho=0.95, pca_k_floor="classes"))
    infer: InferConfig = dataclasses.field(default_factory=InferConfig)
    expand: int = 0                               # frozen random ReLU expansion width (0 = off)
    head: str = "ridge"                           # 'ridge' (N2L) | 'proto' (prototype head on the same pseudo-labels)
    seed: int = 0
    gt_pos_iou: float = 0.5
    gt_neg_iou: float = 0.4
    n_folds: int = 5                              # final-task AP is also computed on this many disjoint image folds (0 = off)
    fold_seed: int = 123


@dataclasses.dataclass
class Bundle:
    train_cache: FeatureCache
    train_ids: List[List[int]]
    train_gt: Dict[int, tuple]                    # image_id -> (boxes, class idx); diagnostics / ceiling only
    eval_cache: FeatureCache
    eval_ids: List[int]
    eval_gt: object                               # pycocotools COCO over eval_ids
    text_label: np.ndarray                        # (80,D) first-template text embeddings (pseudo-labelling)
    text_eval: np.ndarray                         # (80,D) template-ensemble embeddings (inference)
    bg_label: np.ndarray                          # (1,D)
    bg_eval: np.ndarray                           # (1,D)


def _true_cols(feats_gather_img_idx, boxes, ids, gt, col_of_class, pos_iou, neg_iou):
    """GT-derived head column for each gathered proposal (-1 ambiguous). Diagnostics / ceiling only."""
    out = np.full(len(boxes), -1, np.int64)
    for k, iid in enumerate(ids):
        m = feats_gather_img_idx == k
        if not m.any():
            continue
        gb, gc = gt.get(iid, (np.zeros((0, 4)), np.zeros(0, int)))
        out[m] = ps.gt_targets(boxes[m], gb, gc, col_of_class, pos_iou, neg_iou)[0]
    return out


def _eval_scores(Xe, class_order, learner, B: Bundle, cfg: RunConfig):
    T = np.vstack([B.bg_eval.reshape(1, -1), B.text_eval[class_order]])
    zs = cfg.pseudo.scale * l2norm(Xe) @ T.T
    if not cfg.pseudo.use_bg:
        zs[:, 0] = -1e9                              # background prompt switched off
    ridge = learner.scores(Xe) if (learner is not None and cfg.variant != "zeroshot") else None
    return fuse(zs, ridge, cfg.infer)


def image_folds(ids, n_folds: int, seed: int = 123) -> List[List[int]]:
    """Deterministic disjoint folds of image ids (sizes differ by at most 1). [] if folding is off or impossible."""
    ids = sorted(int(i) for i in ids)
    if n_folds < 2 or len(ids) < 2 * n_folds:
        return []
    perm = np.random.default_rng(seed).permutation(len(ids))
    return [sorted(ids[j] for j in perm[k::n_folds]) for k in range(n_folds)]


def fold_scores(gt, dets, folds, class_order, new_classes) -> Dict:
    """AP / AP50 / new-class AP50 of the same detections on each image fold (for mean +- sd and paired CIs)."""
    out = {"AP": [], "AP50": [], "AP50_new": [], "n_images": []}
    for ids in folds:
        r = coco_eval(gt, dets, ids, class_order)
        new = [r["per_class_AP50"][c] for c in new_classes]
        out["AP"].append(r["AP"]); out["AP50"].append(r["AP50"])
        out["AP50_new"].append(float(np.nanmean(new)) if new and not np.all(np.isnan(new)) else float("nan"))
        out["n_images"].append(len(ids))
    return out


def make_detections(eval_ids, e_img, Be, Se, fused, class_order, infer: InferConfig, base_classes=None) -> List[dict]:
    """Per-image detection (top-k classes, YOLO^gamma, class-wise NMS) -> COCO result dicts."""
    dets: List[dict] = []
    for k, iid in enumerate(eval_ids):
        m = e_img == k
        b, s, c = detect(Be[m], Se[m], fused[m], class_order, infer, base_classes)
        dets += dets_to_coco(iid, b, s, c)
    return dets


def run_incremental(B: Bundle, cfg: RunConfig, log: Callable = print, probe: Optional[Callable] = None) -> Dict:
    """probe(ctx) (optional, read-only diagnostics) is called after every task with the learner, the pseudo-labels,
    the GT-derived targets, and the fused eval scores. It must not change the run; tests enforce that."""
    if cfg.variant not in ("ours", "zeroshot", "gt_ceiling"):
        raise ValueError(f"unknown variant {cfg.variant!r}")
    D = B.train_cache.meta["feat_dim"]
    fmap = nc.RandomReLUExpansion(D, cfg.expand, seed=cfg.seed) if cfg.expand else None
    learner = None
    if cfg.variant != "zeroshot":
        learner = make_learner(cfg, D, fmap)
    rng = np.random.default_rng(cfg.seed)
    Xe, e_img, Be, Se, _ = B.eval_cache.gather(B.eval_ids)
    class_order: List[int] = []
    curve, diag = [], []
    for t, cls_t in enumerate(cfg.tasks):
        class_order += list(cls_t)
        col_of = ps.class_to_col(class_order)
        info = {"task": t, "classes": list(cls_t)}
        train_ctx = {}
        if cfg.variant != "zeroshot":
            X, imgidx, Bx, Sx, _ = B.train_cache.gather(B.train_ids[t])
            n_new = len(cls_t) + (1 if t == 0 else 0)
            true = _true_cols(imgidx, Bx, B.train_ids[t], B.train_gt, col_of, cfg.gt_pos_iou, cfg.gt_neg_iou)
            train_ctx = {"X": X, "imgidx": imgidx, "boxes": Bx, "yolo": Sx, "true": true, "col_of": col_of}
            if cfg.variant == "ours":
                keep, logits, w = ps.build_task_inputs(X, Sx, B.text_label[class_order], B.bg_label, cfg.pseudo)
                gate = (np.asarray(Sx)[keep] >= cfg.n2l.refine_gate) if cfg.n2l.refine_gate > 0 else None
                res = learner.add_task(X[keep], logits, n_new, rng=rng, base_weights=w, gate=gate,
                                       bg_col=0 if cfg.pseudo.use_bg else None)
                off = learner.num_classes - logits.shape[1]
                train_ctx.update(keep=keep, logits=logits, weights=w, result=res, col_offset=off)
                info.update(n=int(len(X)), kept=int(keep.sum()), k=int(res.k),
                            zero_shot_labels=ps.label_quality(res.label_history[0] + off, true[keep]),
                            refined_labels=ps.label_quality(res.labels + off, true[keep]),
                            used_labels=ps.label_quality((res.labels + off)[res.weights > 0], true[keep][res.weights > 0]),
                            n_used=int((res.weights > 0).sum()))
            else:                                                       # 'gt_ceiling'
                ok = true >= 0
                labels = true[ok]
                w = nc.bilevel_weights(labels, np.zeros(len(labels)), learner.num_classes + n_new, rng,
                                       intra=False, inter=True)
                learner.add_labeled(X[ok], labels, n_new, w)
                info.update(n=int(len(X)), kept=int(ok.sum()))
        fused = _eval_scores(Xe, class_order, learner, B, cfg)
        dets = make_detections(B.eval_ids, e_img, Be, Se, fused, class_order, cfg.infer, cfg.tasks[0])
        r = coco_eval(B.eval_gt, dets, B.eval_ids, class_order)
        old = [c for tt in cfg.tasks[:t] for c in tt]
        mean = lambda cs: float(np.nanmean([r["per_class_AP50"][c] for c in cs])) if cs else float("nan")
        row = {"task": t, "n_seen": len(class_order), "AP": r["AP"], "AP50": r["AP50"],
               "AP75": r["AP75"], "AP50_base": mean(cfg.tasks[0]), "AP50_old": mean(old),
               "AP50_new": mean(cls_t),
               "per_class_AP50": {int(c): float(r["per_class_AP50"][c]) for c in class_order}}   # exp v7: for forgetting
        curve.append(row); diag.append(info)
        if probe is not None:
            probe({"task": t, "class_order": list(class_order), "learner": learner, "cfg": cfg, "bundle": B,
                   "info": info, "fused": fused, "eval": (Xe, e_img, Be, Se),
                   "train": train_ctx if cfg.variant != "zeroshot" else None, "row": row})
        log(f"[{cfg.variant}] task {t + 1}/{len(cfg.tasks)} seen={len(class_order)} "
            f"AP={row['AP']:.3f} AP50={row['AP50']:.3f} base={row['AP50_base']:.3f} new={row['AP50_new']:.3f}"
            + (f" | kept {info['kept']}/{info['n']} k={info.get('k', '-')}" if 'kept' in info else ""))
    folds = image_folds(B.eval_ids, cfg.n_folds, cfg.fold_seed)
    out = {"variant": cfg.variant, "curve": curve, "diag": diag,
           "A_bar_AP": float(np.mean([r["AP"] for r in curve])), "final_AP": curve[-1]["AP"],
           "final_AP50": curve[-1]["AP50"],
           "base_AP50_drop": curve[0]["AP50_base"] - curve[-1]["AP50_base"],
           "forgetting": forgetting(curve, cfg.tasks)}
    if folds:
        out["final_folds"] = fold_scores(B.eval_gt, dets, folds, class_order, list(cfg.tasks[-1]))
    return out


def forgetting(curve, tasks) -> Dict:
    """exp v7, pre-registered 19 Nov. Uses the per-class AP50 stored at every step.
    old_class: mean over the FIRST task's classes of (AP50 right after task 1 - AP50 after the last task).
    avg (Chaudhry et al. 2018): for every class learned before the last task, max over the steps from when it was learned up
    to the step before last of its AP50, minus its AP50 after the last task; mean over those classes.
    by_task[k]: the same, averaged over the classes of task k only (k < last). Classes with no ground truth (NaN AP) are skipped.
    Positive = forgotten; negative = the class improved."""
    T = len(tasks)
    nan = float("nan")
    if T < 2 or any("per_class_AP50" not in r for r in curve):
        return {"old_class": nan, "avg": nan, "by_task": []}
    ap = lambda s, c: float(curve[s]["per_class_AP50"].get(c, curve[s]["per_class_AP50"].get(str(c), nan)))
    first = [ap(0, c) - ap(T - 1, c) for c in tasks[0]]
    per_task, every = [], []
    for k in range(T - 1):
        fs = [max(ap(s, c) for s in range(k, T - 1)) - ap(T - 1, c) for c in tasks[k]]
        fs = [f for f in fs if f == f]
        per_task.append(float(np.mean(fs)) if fs else nan)
        every += fs
    first = [f for f in first if f == f]
    return {"old_class": float(np.mean(first)) if first else nan, "avg": float(np.mean(every)) if every else nan,
            "by_task": per_task}


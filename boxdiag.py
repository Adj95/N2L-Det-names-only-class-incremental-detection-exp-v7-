"""boxdiag.py -- N2L-Det-exp round 1: WHERE do the new classes lose AP? Read-only analysis (diagnostics only).

Three measurements, all on the tune or eval split (never the final split):

 A. coverage(): for each ground-truth object, is there a proposal on it (IoU >= 0.5 / 0.7) among the top-K proposals of its image,
    for K = 5 .. all? Split base vs new classes, per class, and by object size. Also: how confident the detector is on the best box
    that covers an object (a covered object with a tiny YOLO score is found but buried by the score^gamma weighting).
 B. naming(): on proposals that cover a real object, what does the fused head call it? (right class / another class of the same group /
    a class of the other group / background wins). Separates "named wrong" from "not found".
 C. oracle_ladder(): replace one part of the pipeline at a time by a perfect one and score AP50 (base / new):
        ref          the run as configured
        O1_names     every proposal that covers a real object (IoU >= 0.5) gets its TRUE class (one-hot); background boxes unchanged
        O2_no_bg     ... and every proposal that covers nothing is removed
        O3_rank      ... and the remaining boxes are scored by their IoU with the object instead of the YOLO score
    The AP still missing at O3 is what the boxes themselves cost (objects with no proposal, poor localisation).
    The step sizes say which part is worth fixing first.

Ground truth enters only here, for measurement; nothing in this file feeds back into training or inference.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence

import numpy as np

import coco_utils as cu
import incremental as inc
from boxes import iou_matrix
from diagnose import gt_from_pycoco
from evaluate import coco_eval

KS = (5, 10, 20, 30, 40, 60, 100, 150, 200, 300)
SIZES = (("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, float("inf")))


def groups_of(tasks: List[List[int]]) -> Dict[str, List[int]]:
    """base = classes of task 0 (the detector was trained on them); new = every later class; plus one group per later task."""
    g = {"base": list(tasks[0]), "new": [c for t in tasks[1:] for c in t]}
    if len(tasks) > 2:
        for i, t in enumerate(tasks[1:], 1):
            g[f"task{i}"] = list(t)
    return g


# ------------------------------------------------------------------ A. coverage
def coverage(proposals: Dict[int, tuple], gt: Dict[int, tuple], tasks, ks: Sequence[int] = KS,
             thrs: Sequence[float] = (0.5, 0.7)) -> Dict:
    """proposals: image_id -> (boxes (n,4), scores (n,)); gt: image_id -> (boxes, class idx).
    Returns per group and per class: n_gt, recall@thr at each K (top-K proposals by score in that image; 'all' = every proposal),
    recall@0.5 by object size, and statistics of the YOLO score / rank of the best covering proposal."""
    groups = groups_of(tasks)
    used = sorted({c for t in tasks for c in t})
    max_n = max((len(p[0]) for p in proposals.values()), default=0)
    kk = sorted({k for k in ks if k < max_n}) + ["all"]
    rows = []                                       # one row per GT object
    for iid, (gb, gc) in gt.items():
        gb, gc = np.asarray(gb, float).reshape(-1, 4), np.asarray(gc, np.int64).reshape(-1)
        sel = np.isin(gc, used)
        gb, gc = gb[sel], gc[sel]
        if not len(gb):
            continue
        pb, ps_ = proposals.get(iid, (np.zeros((0, 4)), np.zeros(0)))
        pb, ps_ = np.asarray(pb, float).reshape(-1, 4), np.asarray(ps_, float).reshape(-1)
        order = np.argsort(-ps_, kind="stable")
        rank = np.empty(len(order), np.int64); rank[order] = np.arange(len(order))      # 0 = most confident in its image
        M = iou_matrix(gb, pb) if len(pb) else np.zeros((len(gb), 0))
        area = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
        for j in range(len(gb)):
            r = {"cls": int(gc[j]), "area": float(area[j])}
            for t in thrs:
                hit = M[j] >= t if M.shape[1] else np.zeros(0, bool)
                for k in kk:
                    r[(t, k)] = bool(hit.any()) if k == "all" else bool((hit & (rank < k)).any())
            hit5 = M[j] >= 0.5 if M.shape[1] else np.zeros(0, bool)
            if hit5.any():
                r["best_score"] = float(ps_[hit5].max())
                r["best_rank"] = int(rank[hit5].min())
            rows.append(r)

    def summ(sub: List[dict]) -> Dict:
        n = len(sub)
        out = {"n_gt": n}
        if not n:
            return out
        for t in thrs:
            out[f"recall@{t:g}"] = {str(k): float(np.mean([r[(t, k)] for r in sub])) for k in kk}
        out["recall@0.5_by_size"] = {}
        for name, lo, hi in SIZES:
            s = [r for r in sub if lo <= r["area"] < hi]
            out["recall@0.5_by_size"][name] = {"n": len(s), "recall": float(np.mean([r[(0.5, "all")] for r in s])) if s else float("nan")}
        sc = np.array([r["best_score"] for r in sub if "best_score" in r])
        rk = np.array([r["best_rank"] for r in sub if "best_rank" in r])
        if len(sc):
            q = np.quantile(sc, [0.25, 0.5, 0.75])
            out["best_cover_score"] = {"q25": float(q[0]), "median": float(q[1]), "q75": float(q[2]),
                                       "frac_below_0.01": float((sc < 0.01).mean()), "frac_below_0.05": float((sc < 0.05).mean()),
                                       "frac_below_0.1": float((sc < 0.1).mean())}
            out["best_cover_rank"] = {"median": float(np.median(rk)), "frac_rank_ge_20": float((rk >= 20).mean())}
        return out

    res = {"ks": [str(k) for k in kk], "groups": {g: summ([r for r in rows if r["cls"] in set(cs)]) for g, cs in groups.items()},
           "per_class": {cu.prompt_names([c])[0]: summ([r for r in rows if r["cls"] == c]) for c in used},
           "boxes_per_image": float(np.mean([len(p[0]) for p in proposals.values()])) if proposals else 0.0,
           "n_images": len(gt)}
    return res


def cache_proposals(cache, ids) -> Dict[int, tuple]:
    out = {}
    for i in ids:
        d = cache.image(i)
        out[int(i)] = (d["boxes"], d["scores"])
    return out


# ------------------------------------------------------------------ helpers for B and C
def _match(Be, e_img, ids, gt):
    """best IoU of every eval proposal with any GT object of its image, and that object's class (-1 if none)."""
    best = np.zeros(len(Be)); cls = np.full(len(Be), -1, np.int64)
    for k, iid in enumerate(ids):
        m = np.flatnonzero(e_img == k)
        gb, gc = gt.get(int(iid), (np.zeros((0, 4)), np.zeros(0, np.int64)))
        if not len(m) or not len(gb):
            continue
        M = iou_matrix(Be[m], gb)
        j = M.argmax(1)
        best[m] = M[np.arange(len(m)), j]
        cls[m] = np.asarray(gc)[j]
    return best, cls


def _capture_last(B: inc.Bundle, cfg: inc.RunConfig) -> dict:
    box = {}

    def probe(ctx):
        if ctx["task"] == len(cfg.tasks) - 1:
            box.update(fused=ctx["fused"].copy(), eval=ctx["eval"], class_order=list(ctx["class_order"]))

    inc.run_incremental(B, dataclasses.replace(cfg, n_folds=0), log=lambda *_: None, probe=probe)
    return box


def naming(fused, class_order, best_iou, true_cls, tasks, pos_iou: float = 0.5) -> Dict:
    groups = {"base": set(tasks[0]), "new": {c for t in tasks[1:] for c in t}}
    order = np.asarray(class_order, np.int64)
    fg = fused[:, 1:]
    top = order[fg.argmax(1)]
    bg_wins = fused[:, 0] > fg.max(1)
    out = {}
    for g, cs in groups.items():
        m = (best_iou >= pos_iou) & np.isin(true_cls, list(cs))
        n = int(m.sum())
        if not n:
            out[g] = {"n": 0}
            continue
        right = top[m] == true_cls[m]
        same = np.isin(top[m], list(cs)) & ~right
        other = ~np.isin(top[m], list(cs))
        out[g] = {"n": n, "right": float(right.mean()), "wrong_same_group": float(same.mean()),
                  "wrong_other_group": float(other.mean()), "background_beats_all_classes": float(bg_wins[m].mean())}
    return out


def _group_ap(r, tasks):
    m = lambda cs: float(np.nanmean([r["per_class_AP50"][c] for c in cs])) if cs else float("nan")
    return {"AP50": r["AP50"], "AP": r["AP"], "AP50_base": m(tasks[0]), "AP50_new": m([c for t in tasks[1:] for c in t])}


def oracle_ladder(B: inc.Bundle, cfg: inc.RunConfig, pos_iou: float = 0.5, n_folds: int = 10) -> Dict:
    cap = _capture_last(B, cfg)
    Xe, e_img, Be, Se = cap["eval"]
    order, fused = cap["class_order"], cap["fused"]
    gt = gt_from_pycoco(B.eval_gt, B.eval_ids)
    best, tcls = _match(Be, e_img, B.eval_ids, gt)
    col = {c: j + 1 for j, c in enumerate(order)}
    real = (best >= pos_iou) & np.isin(tcls, order)
    f1 = fused.copy()
    f1[real] = 0.0
    f1[np.flatnonzero(real), [col[int(c)] for c in tcls[real]]] = 1.0
    f2 = f1.copy(); f2[~real] = 0.0
    # O3: rank by localisation quality. gamma=1 on the IoU, no extra rescoring, so the score IS the IoU.
    i3 = dataclasses.replace(cfg.infer, yolo_gamma=1.0, gamma_new=-1.0, clip_obj=0.0, yolo_floor=0.0)
    stages = {"ref": (fused, Se, cfg.infer), "O1_names": (f1, Se, cfg.infer), "O2_no_bg": (f2, Se, cfg.infer),
              "O3_rank": (f2, np.where(real, best, 1e-6), i3)}
    folds = inc.image_folds(B.eval_ids, n_folds, cfg.fold_seed)
    out = {}
    for name, (f, s, ic) in stages.items():
        dets = inc.make_detections(B.eval_ids, e_img, Be, s, f, order, ic, cfg.tasks[0])
        r = coco_eval(B.eval_gt, dets, B.eval_ids, order)
        out[name] = {**_group_ap(r, cfg.tasks), "per_class_AP50": {cu.prompt_names([c])[0]: r["per_class_AP50"][c] for c in order}}
        if folds:
            fs = inc.fold_scores(B.eval_gt, dets, folds, order, [c for t in cfg.tasks[1:] for c in t])
            out[name]["folds_AP50"] = fs["AP50"]; out[name]["folds_AP50_new"] = fs["AP50_new"]
    out["naming"] = naming(fused, order, best, tcls, cfg.tasks, pos_iou)
    return out


def run_boxdiag(B: inc.Bundle, tasks, cfgs: Dict[str, inc.RunConfig], gt: Optional[Dict] = None, n_folds: int = 10) -> Dict:
    gt = gt if gt is not None else gt_from_pycoco(B.eval_gt, B.eval_ids)
    res = {"coverage": coverage(cache_proposals(B.eval_cache, B.eval_ids), gt, tasks), "oracle": {}}
    for name, cfg in cfgs.items():
        res["oracle"][name] = oracle_ladder(B, cfg, n_folds=n_folds)
    return res


# ------------------------------------------------------------------ report
def _f(v, d=3):
    return "-" if v is None or v != v else f"{v:.{d}f}"


def format_coverage(cov: Dict, title: str) -> List[str]:
    ks = cov["ks"]
    L = [f"### {title} ({cov['n_images']} images, {cov['boxes_per_image']:.1f} proposals per image)", "",
         "Recall = share of real objects with at least one proposal at IoU >= thr among the top-K proposals of their image.", "",
         "| group | n objects | thr | " + " | ".join(f"K={k}" for k in ks) + " |", "|---|---|---|" + "---|" * len(ks)]
    rows = list(cov["groups"].items()) + [(n, v) for n, v in cov["per_class"].items()]
    for g, s in rows:
        if not s.get("n_gt"):
            continue
        for t in ("0.5", "0.7"):
            L.append(f"| {g} | {s['n_gt']} | {t} | " + " | ".join(_f(s[f'recall@{t}'][k]) for k in ks) + " |")
    L += ["", "Covered objects: YOLO score and rank (0 = most confident box of the image) of the best covering proposal.", "",
          "| group | median score | q25 | q75 | share < 0.01 | share < 0.05 | share < 0.1 | median rank | share rank >= 20 | recall@0.5 small / medium / large |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g, s in rows:
        if "best_cover_score" not in s:
            continue
        b, r, z = s["best_cover_score"], s["best_cover_rank"], s["recall@0.5_by_size"]
        sz = " / ".join(f"{_f(z[n]['recall'], 2)} (n={z[n]['n']})" for n, _, _ in SIZES)
        L.append(f"| {g} | {_f(b['median'])} | {_f(b['q25'])} | {_f(b['q75'])} | {_f(b['frac_below_0.01'], 2)} | "
                 f"{_f(b['frac_below_0.05'], 2)} | {_f(b['frac_below_0.1'], 2)} | {r['median']:.0f} | {_f(r['frac_rank_ge_20'], 2)} | {sz} |")
    return L


def format_report(res: Dict, split: str) -> str:
    L = [f"# Box diagnosis on the {split.upper()} images (diagnostic only; nothing here changes the method)", ""]
    L += format_coverage(res["coverage"], "A. Proposal coverage (the cached proposals the pipeline uses)") + [""]
    for name, o in res["oracle"].items():
        L += [f"## {name}", "", "### B. What the head calls proposals that cover a real object (IoU >= 0.5)", "",
              "| objects of | n proposals | right class | wrong, same group | wrong, other group | background prob beats every class |",
              "|---|---|---|---|---|---|"]
        for g, s in o["naming"].items():
            if s.get("n"):
                L.append(f"| {g} | {s['n']} | {_f(s['right'])} | {_f(s['wrong_same_group'])} | {_f(s['wrong_other_group'])} | "
                         f"{_f(s['background_beats_all_classes'])} |")
        L += ["", "### C. Oracle ladder (each row makes one more part perfect; pooled AP50 over all images)", "",
              "| stage | AP50 | AP50 base | AP50 new | step on new | AP |", "|---|---|---|---|---|---|"]
        prev = None
        for st in ("ref", "O1_names", "O2_no_bg", "O3_rank"):
            s = o[st]
            step = "-" if prev is None else f"{s['AP50_new'] - prev:+.3f}"
            L.append(f"| {st} | {_f(s['AP50'])} | {_f(s['AP50_base'])} | {_f(s['AP50_new'])} | {step} | {_f(s['AP'])} |")
            prev = s["AP50_new"]
        L += ["", "Read: O1 - ref = cost of naming real objects wrong; O2 - O1 = cost of background boxes scored as objects; "
              "O3 - O2 = cost of the YOLO score ranking boxes badly; 1 - O3 (roughly) = cost of objects with no good proposal.", ""]
    return "\n".join(L)


def format_propdiag(res: Dict) -> str:
    L = ["# Proposer coverage (diagnostic only: proposals are NOT used by the pipeline here)", ""]
    for name, r in res.items():
        L += format_coverage(r["coverage"], f"{name}  (conf {r['settings']['conf']}, NMS IoU {r['settings']['nms_iou']}, "
                                            f"max_det {r['settings']['max_det']}, {r['seconds_per_image']:.3f} s/image)") + [""]
    L += ["Compare rows at the same K. If recall keeps rising past K=60 (what the pipeline keeps), more boxes per image can help; "
          "if a different proposer covers new classes far better at the same K, the frozen base detector is the bottleneck."]
    return "\n".join(L)

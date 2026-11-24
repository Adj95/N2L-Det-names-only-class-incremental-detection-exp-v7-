"""diagnose.py -- WHY does the learned head help (or not) over zero-shot CLIP?  Read-only analysis on cached features.

Runs E5 (ours), E5 without refinement and E0 once each with a probe attached and reports, per task:

 A. pseudo-label quality vs ground truth (diagnostic use only), and how it varies with CLIP's confidence:
    "what-if" table = accuracy / kept fraction if we dropped pseudo-labels below a confidence threshold.
 B. the most common pseudo-label confusions (true class -> pseudo-label).
 C. box classification accuracy on eval proposals that overlap a GT box (IoU>=0.5), for zero-shot CLIP, the ridge head
    alone, and the fused scores: this isolates *classification* from localisation / objectness effects.

Ground truth enters only through the evaluator-style matching in this file; nothing here feeds back into training.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.special import softmax

import coco_utils as cu
import incremental as inc
import n2l_core as nc
from boxes import iou_matrix
from prompts import l2norm

BINS = (0.0, 0.5, 0.7, 0.9, 0.99, 1.0000001)
THRESHOLDS = (0.0, 0.5, 0.7, 0.9, 0.95, 0.99)


def gt_from_pycoco(gt, image_ids: Sequence[int]) -> Dict[int, tuple]:
    """image_id -> (xyxy (n,4), class idx (n,)) from a pycocotools COCO object (non-crowd, w,h > 1)."""
    want = set(int(i) for i in image_ids)
    acc: Dict[int, list] = {i: [] for i in want}
    for a in gt.dataset["annotations"]:
        if a["image_id"] in want and not a.get("iscrowd", 0) and a["bbox"][2] > 1 and a["bbox"][3] > 1:
            x, y, w, h = a["bbox"]
            acc[a["image_id"]].append((x, y, x + w, y + h, cu.ID2IDX[a["category_id"]]))
    out = {}
    for i, rows in acc.items():
        arr = np.array(rows, dtype=float).reshape(-1, 5)
        out[i] = (arr[:, :4], arr[:, 4].astype(np.int64))
    return out


def calibration(true_cols, pred_cols, ptop) -> Dict:
    """Pseudo-label accuracy by CLIP-confidence bin, and the what-if for each drop threshold.
    Only proposals whose GT-derived target is determinate (>=0) count."""
    m = np.asarray(true_cols) >= 0
    t, p, c = np.asarray(true_cols)[m], np.asarray(pred_cols)[m], np.asarray(ptop)[m]
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = (c >= lo) & (c < hi)
        rows.append({"lo": lo, "n": int(s.sum()), "acc": float((p[s] == t[s]).mean()) if s.any() else float("nan")})
    whatif = []
    for tau in THRESHOLDS:
        s = c >= tau
        whatif.append({"tau": tau, "kept": float(s.mean()) if len(s) else float("nan"),
                       "acc": float((p[s] == t[s]).mean()) if s.any() else float("nan")})
    return {"n": int(m.sum()), "bins": rows, "whatif": whatif}


def confusions(true_cols, pred_cols, col_names: Sequence[str], top: int = 8) -> List[dict]:
    """Most frequent wrong (true, predicted) pairs among determinate proposals (column 0 = background)."""
    m = np.asarray(true_cols) >= 0
    t, p = np.asarray(true_cols)[m], np.asarray(pred_cols)[m]
    wrong = t != p
    pairs, counts = np.unique(np.stack([t[wrong], p[wrong]], 1), axis=0, return_counts=True) if wrong.any() else \
        (np.zeros((0, 2), int), np.zeros(0, int))
    order = np.argsort(-counts, kind="stable")[:top]
    return [{"true": col_names[pairs[i, 0]], "pred": col_names[pairs[i, 1]], "n": int(counts[i]),
             "share_of_true": float(counts[i] / max(1, (t == pairs[i, 0]).sum()))} for i in order]


def head_vs_zeroshot(ctx: dict, gt_by_img: Dict[int, tuple], iou: float = 0.5) -> Dict:
    """Class accuracy of zero-shot / head-only / fused scores on eval proposals overlapping a seen-class GT box."""
    B, order = ctx["bundle"], ctx["class_order"]
    Xe, e_img, Be, _ = ctx["eval"]
    cfg = ctx["cfg"]
    T = np.vstack([B.bg_eval.reshape(1, -1), B.text_eval[order]])
    zs = softmax(cfg.pseudo.scale * l2norm(Xe) @ T.T, axis=1)
    learner = ctx["learner"]
    head = learner.scores(Xe) if (learner is not None and cfg.variant != "zeroshot") else None
    fused = ctx["fused"]
    cls_true = np.full(len(Be), -1, np.int64)
    for k, iid in enumerate(B.eval_ids):
        m = np.flatnonzero(e_img == k)
        gb, gc = gt_by_img.get(iid, (np.zeros((0, 4)), np.zeros(0, int)))
        if not len(m) or not len(gb):
            continue
        io = iou_matrix(Be[m], gb)
        j, best = io.argmax(1), io.max(1)
        ok = best >= iou
        cls_true[m[ok]] = gc[j[ok]]
    seen = np.isin(cls_true, order)
    cls_true = np.where(seen, cls_true, -1)
    sel = cls_true >= 0
    o = np.asarray(order)
    pick = lambda S: o[np.argmax(S[:, 1:], axis=1)] if S is not None else None
    new = np.isin(cls_true, ctx["info"]["classes"])
    out = {"n_boxes": int(sel.sum())}
    for name, S in (("zeroshot", zs), ("head_only", head), ("fused", fused)):
        if S is None:
            continue
        hit = pick(S) == cls_true
        out[name] = {"acc": float(hit[sel].mean()) if sel.any() else float("nan"),
                     "acc_new": float(hit[sel & new].mean()) if (sel & new).any() else float("nan"),
                     "acc_old": float(hit[sel & ~new].mean()) if (sel & ~new).any() else float("nan")}
    out["n_new"], out["n_old"] = int((sel & new).sum()), int((sel & ~new).sum())
    return out


def run_diagnosis(bundle: inc.Bundle, tasks, log=print) -> Dict:
    gt_img = gt_from_pycoco(bundle.eval_gt, bundle.eval_ids)
    names = ["background"] + cu.prompt_names()
    out: Dict[str, dict] = {}
    n2l_norefine = nc.N2LConfig(pca_rule="energy", pca_rho=0.95, pca_k_floor="classes", n_rounds=0)
    variants = {"E5_ours": inc.RunConfig(tasks=tasks, n_folds=0),
                "E5_no_refine": inc.RunConfig(tasks=tasks, n2l=n2l_norefine, n_folds=0),
                "E0_zeroshot": inc.RunConfig(tasks=tasks, variant="zeroshot", n_folds=0)}
    for vname, cfg in variants.items():
        per_task: List[dict] = []

        def probe(ctx, per_task=per_task):
            rec = {"task": ctx["task"], "classes": ctx["info"]["classes"],
                   "head_vs_zeroshot": head_vs_zeroshot(ctx, gt_img)}
            tr = ctx["train"]
            if tr is not None and "logits" in tr:
                keep, logits, off = tr["keep"], tr["logits"], tr["col_offset"]
                true = tr["true"][keep]
                p = softmax(logits, axis=1)
                pred0 = p.argmax(1) + off
                cn = [names[0]] + [names[1 + c] for c in ctx["class_order"]]
                rec["pseudo_calibration"] = calibration(true, pred0, p.max(1))
                rec["pseudo_confusions"] = confusions(true, pred0, cn)
                rec["label_quality_zeroshot"] = ctx["info"]["zero_shot_labels"]
                rec["label_quality_refined"] = ctx["info"]["refined_labels"]
                rec["kept"], rec["n"], rec["k"] = ctx["info"]["kept"], ctx["info"]["n"], ctx["info"]["k"]
            per_task.append(rec)

        res = inc.run_incremental(bundle, cfg, log=log, probe=probe)
        out[vname] = {"tasks": per_task, "final_AP50": res["final_AP50"], "final_AP": res["final_AP"]}
    return out


def _f(v, p=3):
    return "n/a" if v is None or v != v else f"{v:.{p}f}"


def format_report(d: Dict) -> str:
    L = ["# Diagnosis", ""]
    L += ["## C. Box classification on eval proposals overlapping a GT box (IoU>=0.5)", "",
          "| variant | task | boxes | zero-shot | head only | fused | fused - zero-shot |", "|---|---|---|---|---|---|---|"]
    for v, r in d.items():
        for t in r["tasks"]:
            h = t["head_vs_zeroshot"]
            zs, hd, fu = h["zeroshot"]["acc"], h.get("head_only", {}).get("acc"), h["fused"]["acc"]
            L.append(f"| {v} | {t['task'] + 1} | {h['n_boxes']} | {_f(zs)} | {_f(hd)} | {_f(fu)} | {_f(fu - zs)} |")
    L += ["", "Positive 'fused - zero-shot' = the learned head classifies matched boxes better than CLIP alone.", ""]
    for t in d["E5_ours"]["tasks"]:
        if "pseudo_calibration" not in t:
            continue
        c = t["pseudo_calibration"]
        L += [f"## A. Pseudo-label calibration, task {t['task'] + 1}  ({c['n']} determinate boxes; "
              f"kept {t['kept']}/{t['n']}, PCA k={t['k']})", "",
              "| CLIP confidence >= | boxes | pseudo-label accuracy |", "|---|---|---|"]
        L += [f"| {b['lo']:.2f} | {b['n']} | {_f(b['acc'])} |" for b in c["bins"]]
        L += ["", "| drop labels below | kept fraction | accuracy of kept |", "|---|---|---|"]
        L += [f"| {w['tau']:.2f} | {_f(w['kept'])} | {_f(w['acc'])} |" for w in c["whatif"]]
        q0, q1 = t["label_quality_zeroshot"], t["label_quality_refined"]
        L += ["", f"Label accuracy: zero-shot {_f(q0.get('acc'))} -> after N2L refinement {_f(q1.get('acc'))}; "
              f"fg precision {_f(q0.get('fg_precision'))} -> {_f(q1.get('fg_precision'))}; "
              f"bg precision {_f(q0.get('bg_precision'))} -> {_f(q1.get('bg_precision'))}.", "",
              "### B. Most common wrong pseudo-labels", "", "| true | pseudo-label | boxes | share of that class |", "|---|---|---|---|"]
        L += [f"| {x['true']} | {x['pred']} | {x['n']} | {_f(x['share_of_true'])} |" for x in t["pseudo_confusions"]]
        L.append("")
    return "\n".join(L)

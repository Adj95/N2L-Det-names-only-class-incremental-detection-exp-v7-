"""variants.py -- independent single-change experiments on the TUNED E5, judged on the held-out TUNE images.

Every variant changes exactly ONE thing relative to `E5_tuned` (the settings chosen by `tune`), so no result depends on
another. They are scored on the tune images (never the eval or final images), with paired image-fold differences against
E5_tuned. E5_tuned itself was tuned on these images, so it has a small advantage the variants do not: a variant that still
wins is a real candidate; one that loses by a little may only have lost the tuning advantage.

Caveat printed with every report: with N variants the best-looking difference is optimistically biased (best of N).
A candidate must be confirmed on images it was not chosen on (the eval or final split) before it is claimed.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Tuple

import incremental as inc
import pipeline as pl
import tune as tn


def _p(**kw):
    return lambda c: dataclasses.replace(c, pseudo=dataclasses.replace(c.pseudo, **kw))


def _n(**kw):
    return lambda c: dataclasses.replace(c, n2l=dataclasses.replace(c.n2l, **kw))


def _i(**kw):
    return lambda c: dataclasses.replace(c, infer=dataclasses.replace(c.infer, **kw))


# name -> (group, one-line description, modifier of the tuned E5 config)
VARIANTS: Dict[str, Tuple[str, str, Callable]] = {}
for v in (0.05, 0.1, 0.2, 0.3):
    VARIANTS[f"min_yolo_{v:g}"] = ("pseudo-label filter", f"drop training boxes with detector score < {v:g}", _p(min_yolo=v))
for f in (0.9, 0.7, 0.5):
    VARIANTS[f"topfrac_{f:g}"] = ("pseudo-label filter", f"per pseudo-class keep the {f:.0%} most confident boxes", _p(class_topfrac=f))
for a in (0.25, 0.5, 1.0):
    VARIANTS[f"soft_{a:g}"] = ("soft labels", f"ridge targets = {1 - a:g} x one-hot + {a:g} x zero-shot softmax", _n(soft=a))
VARIANTS["ridge_norm"] = ("fusion", "ridge scores clipped at 0 and renormalised per box before fusing", _i(ridge_norm=True))
VARIANTS["geo_fusion"] = ("fusion", "geometric instead of arithmetic CLIP/ridge fusion", _i(geo_fusion=True))
VARIANTS["geo_fusion_ridge_norm"] = ("fusion", "both of the above", _i(geo_fusion=True, ridge_norm=True))
VARIANTS["head_proto"] = ("head", "prototype (class-mean cosine) head on the same pseudo-labels instead of ridge",
                          lambda c: dataclasses.replace(c, head="proto"))

# --- N2L-Det-exp round 1: rescoring (inference only; no retraining). Applied to BOTH tuned E5 and tuned zero-shot E0, because a
# better score for new-class boxes is not specific to our head: the paper comparison is E5 vs E0 with the same rescoring.
RESCORE: Dict[str, Tuple[str, str, Callable]] = {}
for g in (0.0, 0.1):
    RESCORE[f"gnew_{g:g}"] = ("rescoring", f"YOLO-score exponent {g:g} for classes the detector was not trained on (base keeps the tuned one)",
                              _i(gamma_new=g))
for b in (0.5, 1.0):
    RESCORE[f"clipobj_{b:g}"] = ("rescoring", f"score x (1 - CLIP/head background prob) ** {b:g}", _i(clip_obj=b))
for f in (0.05, 0.1):
    RESCORE[f"yfloor_{f:g}"] = ("rescoring", f"proposal scores below {f:g} raised to {f:g} before the exponent", _i(yolo_floor=f))
for name, v in RESCORE.items():
    VARIANTS[name] = v

# --- N2L-Det-exp round 2: detection-aware N2L refinement (training only; E0 is unaffected). Each changes ONE thing in the tuned
# E5's refinement. The frozen tune.json keeps refinement ON (n_rounds = default 3), so these act on the refinement it really uses.
REFINE: Dict[str, Tuple[str, str, Callable]] = {
    "rf_off": ("refinement", "no refinement rounds (n_rounds = 0): zero-shot labels go straight to the head (reference)", _n(n_rounds=0)),
    "rf_gate_0.1": ("refinement", "only boxes with detector score >= 0.1 may change label; the rest keep the zero-shot label", _n(refine_gate=0.1)),
    "rf_gate_0.3": ("refinement", "only boxes with detector score >= 0.3 may change label; the rest keep the zero-shot label", _n(refine_gate=0.3)),
    "rf_anchor_0.5": ("refinement", "each round's scores = 0.5 x ridge + 0.5 x zero-shot softmax (labels cannot drift far)", _n(anchor=0.5)),
    "rf_to_bg": ("refinement", "refinement may only turn a label into background (a background filter), never into another class", _n(refine_mode="to_bg")),
    "rf_agree": ("refinement", "keep a box only if refinement and zero-shot agree on its label (others get weight 0)", _n(refine_mode="agree")),
}
for name, v in REFINE.items():
    VARIANTS[name] = v

# --- N2L-Det-exp v4: forgetting baselines (ABLATIONS for the paper, never candidates). The recursive ridge head keeps every
# old task's statistics; these rows throw them away to show what that memory is worth. 'headonly' rows switch off the fusion
# with zero-shot CLIP (fusion_weight 1), because zero-shot CLIP alone never forgets and would hide the head's forgetting.
def _both(n2l: dict, infer: dict):
    return lambda c: dataclasses.replace(c, n2l=dataclasses.replace(c.n2l, **n2l), infer=dataclasses.replace(c.infer, **infer))


FORGET: Dict[str, Tuple[str, str, Callable]] = {
    "mem_none": ("forgetting", "head re-fit on the current task only (old statistics thrown away); pseudo-labels may still name old classes", _n(memory="none")),
    "mem_newonly": ("forgetting", "head re-fit on the current task only AND its boxes may only be named new classes or background (fine-tune-on-new-data)", _n(memory="none_newonly")),
    "headonly": ("forgetting", "reference for the next row: no fusion with zero-shot CLIP (fusion_weight 1), full memory", _i(fusion_weight=1.0)),
    "headonly_mem_newonly": ("forgetting", "no fusion with zero-shot CLIP AND fine-tune-on-new-data (the classic forgetting setting)", _both({"memory": "none_newonly"}, {"fusion_weight": 1.0})),
}
for name, v in FORGET.items():
    VARIANTS[name] = v
E0_PREFIX = "E0__"                                                    # E0__<name> = the same change applied to E0_zeroshot_tuned

REFERENCES = ("E0_zeroshot_tuned", "E5_tuned", "E6_gt_ceiling_tuned")


def variant_configs(tasks, t) -> Dict[str, inc.RunConfig]:
    base = pl.ablation_configs(tasks, tn._load(t))
    out = {n: base[n] for n in REFERENCES}
    for name, (_, _, mod) in VARIANTS.items():
        out[name] = mod(base["E5_tuned"])
    for name, (_, _, mod) in RESCORE.items():
        out[E0_PREFIX + name] = mod(base["E0_zeroshot_tuned"])
    return out


def run_variants(bundle: inc.Bundle, t, tasks, only=None, n_folds: int = 10, log=print) -> Dict:
    cfgs = variant_configs(tasks, t)
    names = list(cfgs) if not only else list(dict.fromkeys(list(REFERENCES) + list(only)))
    unknown = [n for n in names if n not in cfgs]
    if unknown:
        raise ValueError(f"unknown variants {unknown}; known: {list(cfgs)}")
    if any(n in REFINE for n in names) and cfgs["E5_tuned"].n2l.n_rounds == 0:
        log("  WARNING: E5_tuned has n_rounds = 0 (no refinement), so the rf_* variants cannot change anything except via their"
            " own setting; read them as 'refinement switched on in a new way', not as a change to the refinement in use.")
    log(f"  E5_tuned refinement in use: n_rounds={cfgs['E5_tuned'].n2l.n_rounds}, use_inter={cfgs['E5_tuned'].n2l.use_inter}, "
        f"use_intra={cfgs['E5_tuned'].n2l.use_intra}, pca_rule={cfgs['E5_tuned'].n2l.pca_rule}, pca_rho={cfgs['E5_tuned'].n2l.pca_rho}, "
        f"lam={cfgs['E5_tuned'].n2l.lam}")
    out = {}
    for n in names:
        out[n] = inc.run_incremental(bundle, dataclasses.replace(cfgs[n], n_folds=n_folds), log=lambda *_: None)
        log(f"  {n:26s} AP50={out[n]['final_AP50']:.4f}  AP={out[n]['final_AP']:.4f}")
    return out


def format_variants(res: Dict, folds_path: str, n_folds: int, on: str = "tune") -> str:
    import compare_runs as cr
    folds = cr.load_folds(folds_path)
    where = {"tune": "TUNE", "eval": "EVAL", "final": "FINAL"}.get(on, on.upper())
    L = [f"# Single-change variants of the tuned E5, scored on the {where} images ({n_folds} image folds)", "",
         "Each row differs from `E5_tuned` in one setting. Differences are paired over image folds with 95% intervals.", ""]
    e0_rows = [n for n in folds if n.startswith(E0_PREFIX)]
    e5_rows = [n for n in folds if not n.startswith(E0_PREFIX)]
    for metric in ("AP50", "AP50_new"):
        L += [f"## {metric}, difference vs E5_tuned", "", cr.format_rows(cr.compare(folds, "E5_tuned", metric, runs=e5_rows), "E5_tuned", metric), ""]
    if e0_rows:
        for metric in ("AP50", "AP50_new"):
            rows = cr.compare(folds, "E0_zeroshot_tuned", metric, runs=["E0_zeroshot_tuned"] + e0_rows)
            L += [f"## {metric}, the same rescoring applied to zero-shot E0, difference vs E0_zeroshot_tuned", "",
                  cr.format_rows(rows, "E0_zeroshot_tuned", metric), ""]
        L += ["Read a rescoring row against BOTH tables: if it lifts E0 as much as E5, it is a better inference rule for any CLIP "
              "scorer, not a gain of our head (report it, and compare E5 vs E0 with the rule applied to both).", ""]
    if any("forgetting" in r for r in res.values()):
        nt = max(len(r.get("forgetting", {}).get("by_task", [])) for r in res.values())
        f3 = lambda x: "-" if x != x else f"{x:+.3f}"
        L += ["## Forgetting (AP50; positive = forgotten, negative = improved)", "",
              "old-class = task-1 classes right after task 1 minus after the last task; average = Chaudhry et al. over every class "
              "learned before the last task (best earlier AP50 minus last); by task = the same per task group.", "",
              "| run | old-class | average | " + " | ".join(f"task {k + 1}" for k in range(nt)) + " |",
              "|---|---|---|" + "---|" * nt]
        for n, r in res.items():
            fo = r.get("forgetting", {})
            bt = list(fo.get("by_task", [])) + [float("nan")] * (nt - len(fo.get("by_task", [])))
            L.append(f"| {n} | {f3(fo.get('old_class', float('nan')))} | {f3(fo.get('avg', float('nan')))} | "
                     + " | ".join(f3(x) for x in bt) + " |")
        L += [""]
    L += ["## What each variant changes", "", "| variant | group | change |", "|---|---|---|"]
    L += [f"| {n} | {g} | {d} |" for n, (g, d, _) in VARIANTS.items() if n in res]
    L += [f"| {E0_PREFIX}{n} | {g} (on E0) | {d} |" for n, (g, d, _) in RESCORE.items() if E0_PREFIX + n in res]
    n_var = sum(n in VARIANTS or n.startswith(E0_PREFIX) for n in res)
    if on == "tune":
        best = max((n for n in res if n in VARIANTS), key=lambda n: res[n]["final_AP50"], default=None)
        L += ["", f"{n_var} variants were compared: the best-looking difference"
              f"{' (' + best + ')' if best else ''} is optimistically biased (best of N), and `E5_tuned` was tuned on these same "
              "images. Treat a winner as a CANDIDATE: confirm it on the eval images (and finally the final split) before claiming it."]
    elif on == "final":
        L += ["", f"FINAL SPLIT: {n_var} row(s) fixed before looking, frozen tune.json, scored once on images never used for any "
              "choice. Report every row as it is."]
    else:
        L += ["", f"CONFIRMATION run: {n_var} variant(s) chosen beforehand on the tune images and scored here on the eval images, which "
              "neither the variants nor `E5_tuned` were tuned on (the eval images were used only by diagnostics). Count it as confirmed only if the "
              "AP50 interval excludes 0 with a positive sign AND AP50_new does not drop; a variant that fails here is dropped. "
              "The eval images are still dev data: the final split is the only untouched one."]
    return "\n".join(L)

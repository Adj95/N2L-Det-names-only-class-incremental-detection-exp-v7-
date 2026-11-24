"""tune.py -- choose hyper-parameters on HELD-OUT tune images, then report once on the eval images.

Protocol (so the final numbers are not tuned on the images they are reported on):
  1. `fetch --n_tune N` draws N images disjoint from the eval images; `extract --splits tune` caches them.
  2. `tune` searches on the tune images only:
       stage 1  training settings (pseudo-label thresholds, N2L rounds, ridge strength, PCA energy, bi-level weights)
                -- random search, trial 0 = the current default, trial 1 = default without refinement;
       stage 2  inference settings (ridge/CLIP fusion weight, YOLO^gamma, top-k classes) -- a grid evaluated on the
                already trained head, for E5 *and, separately, for the zero-shot baseline E0*, so the comparison
                between them stays fair (otherwise only our method gets tuned).
  3. `run --tuned results/tune.json` adds E5_tuned and E0_zeroshot_tuned to the table on the eval images.

Ties go to the earlier (simpler) candidate, so a tuned setting must beat the default on the tune images to be chosen.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
from typing import Dict, List, Optional, Sequence

import numpy as np

import incremental as inc
import n2l_core as nc
import pseudo as ps
from evaluate import coco_eval
from inference import InferConfig

TRAIN_SPACE: Dict[str, Dict[str, list]] = {
    "pseudo": {"tau_pos": [0.0, 0.3, 0.5, 0.7, 0.9], "tau_bg": [0.0, 0.5, 0.7, 0.9], "use_loc": [False, True]},
    "n2l": {"n_rounds": [0, 1, 2, 3], "lam": [0.01, 0.1, 1.0, 10.0], "pca_rho": [0.8, 0.9, 0.95, 0.99],
            "use_intra": [True, False], "use_inter": [True, False]},
}
FUSION = [0.2, 0.5, 0.8, 1.0]
GAMMA = [0.25, 0.5, 0.75, 1.0]
TOPK = [1, 3, 5]
OBJECTIVES = ("final_AP50", "final_AP", "A_bar_AP", "final_AP50_base")
# final_AP50_base (exp v6): mean AP50 over the FIRST task's (base) classes after the last task. Its value depends only on
# base-class ground truth (COCOeval scores each class on its own GT and its own detections), so tuning with it uses no
# new-class labels at all: the label-free-for-new-classes tuning check.


# ------------------------------------------------------------------ configs <-> plain dicts (for json)
def default_params() -> dict:
    d = inc.RunConfig(tasks=[[0]])
    return {"pseudo": dataclasses.asdict(d.pseudo), "n2l": dataclasses.asdict(d.n2l)}


def sample_params(rng: np.random.Generator) -> dict:
    p = default_params()
    for grp, space in TRAIN_SPACE.items():
        for k, vals in space.items():
            p[grp][k] = vals[int(rng.integers(len(vals)))]
    return p


def cfg_from(tasks, params: dict, infer: Optional[dict] = None, variant: str = "ours") -> inc.RunConfig:
    ic = InferConfig(**infer) if infer else InferConfig()
    return inc.RunConfig(tasks=tasks, variant=variant, infer=ic,
                         pseudo=ps.PseudoConfig(**params["pseudo"]) if params else ps.PseudoConfig(),
                         n2l=nc.N2LConfig(**params["n2l"]) if params else inc.RunConfig(tasks=tasks).n2l)


def candidate_params(n_trials: int, seed: int) -> List[dict]:
    """Trial 0 = default, trial 1 = default with refinement off, then random draws (no duplicates)."""
    rng = np.random.default_rng(seed)
    first = default_params()
    off = json.loads(json.dumps(first)); off["n2l"]["n_rounds"] = 0
    out, seen = [], set()
    for p in [first, off] + [sample_params(rng) for _ in range(max(0, 50 * n_trials))]:
        key = json.dumps(p, sort_keys=True)
        if key not in seen:
            seen.add(key); out.append(p)
        if len(out) >= max(2, n_trials):
            break
    return out


# ------------------------------------------------------------------ search
def infer_grid(bundle: inc.Bundle, cfg: inc.RunConfig, infers: Sequence[InferConfig]) -> List[dict]:
    """Train once, then score every inference setting on the final task (no retraining)."""
    rows: List[dict] = []

    def probe(ctx):
        if ctx["task"] != len(cfg.tasks) - 1:
            return
        Xe, e_img, Be, Se = ctx["eval"]
        order = ctx["class_order"]
        for ic in infers:
            c2 = dataclasses.replace(cfg, infer=ic)
            fused = inc._eval_scores(Xe, order, ctx["learner"], bundle, c2)
            dets = inc.make_detections(bundle.eval_ids, e_img, Be, Se, fused, order, ic, cfg.tasks[0])
            r = coco_eval(bundle.eval_gt, dets, bundle.eval_ids, order)
            new_c = [r["per_class_AP50"][c] for c in cfg.tasks[-1]]
            base_c = [r["per_class_AP50"][c] for c in cfg.tasks[0]]
            rows.append({"infer": dataclasses.asdict(ic), "final_AP50": r["AP50"], "final_AP": r["AP"],
                         "AP50_new": float(np.nanmean(new_c)) if not np.all(np.isnan(new_c)) else float("nan"),
                         "final_AP50_base": float(np.nanmean(base_c)) if not np.all(np.isnan(base_c)) else float("nan")})

    inc.run_incremental(bundle, dataclasses.replace(cfg, n_folds=0), log=lambda *_: None, probe=probe)
    return rows


def _best(rows: List[dict], objective: str) -> int:
    """Index of the best row; first one wins ties (earlier = simpler)."""
    vals = [r[objective] if r[objective] == r[objective] else -1.0 for r in rows]
    return int(np.argmax(vals))


def run_search(bundle: inc.Bundle, tasks, n_trials: int = 30, seed: int = 0, objective: str = "final_AP50",
               log=print) -> dict:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    quiet = lambda *_: None
    trials = []
    for i, params in enumerate(candidate_params(n_trials, seed)):
        r = inc.run_incremental(bundle, dataclasses.replace(cfg_from(tasks, params), n_folds=0), log=quiet)
        trials.append({"trial": i, "params": params, "final_AP50": r["final_AP50"], "final_AP": r["final_AP"],
                       "A_bar_AP": r["A_bar_AP"],
                       "final_AP50_base": r["curve"][-1]["AP50_base"] if "curve" in r else float("nan")})
        log(f"  stage 1  trial {i:2d}  AP50={r['final_AP50']:.4f}  AP={r['final_AP']:.4f}  "
            f"{_short(params)}")
    b = _best(trials, objective)
    best_params = trials[b]["params"]
    # stage 2: inference grids, E5 on its best training settings, and E0 on its own
    e5_infers = [InferConfig(fusion_weight=f, yolo_gamma=g) for f, g in itertools.product(FUSION, GAMMA)]
    g5 = infer_grid(bundle, cfg_from(tasks, best_params), e5_infers)
    k5 = _best(g5, objective)
    topk5 = infer_grid(bundle, cfg_from(tasks, best_params),
                       [InferConfig(**{**g5[k5]["infer"], "topk_classes": k}) for k in TOPK])
    g5 += topk5
    k5 = _best(g5, objective)
    e0_infers = [InferConfig(yolo_gamma=g, topk_classes=k) for g, k in itertools.product(GAMMA, TOPK)]
    g0 = infer_grid(bundle, inc.RunConfig(tasks=tasks, variant="zeroshot"), e0_infers)
    k0 = _best(g0, objective)
    default_e5 = trials[0]
    out = {"objective": objective, "n_tune_images": len(bundle.eval_ids), "seed": seed, "n_trials": len(trials),
           "stage1": trials, "best_train_trial": b, "stage2_E5": g5, "stage2_E0": g0,
           "E5": {**best_params, "infer": g5[k5]["infer"]}, "E0": {"infer": g0[k0]["infer"]},
           "tune_scores": {"E5_default": default_e5[objective], "E5_tuned": g5[k5][objective],
                           "E0_default": _default_e0(bundle, tasks, objective), "E0_tuned": g0[k0][objective]}}
    log(f"  tune-split {objective}: E5 default {out['tune_scores']['E5_default']:.4f} -> tuned "
        f"{out['tune_scores']['E5_tuned']:.4f};  E0 default {out['tune_scores']['E0_default']:.4f} -> tuned "
        f"{out['tune_scores']['E0_tuned']:.4f}")
    return out


def _default_e0(bundle, tasks, objective):
    r = inc.run_incremental(bundle, inc.RunConfig(tasks=tasks, variant="zeroshot", n_folds=0), log=lambda *_: None)
    return r["curve"][-1]["AP50_base"] if objective == "final_AP50_base" else r[objective]


def _short(p: dict) -> str:
    d, q = default_params(), []
    for grp in ("pseudo", "n2l"):
        q += [f"{k}={v}" for k, v in p[grp].items() if v != d[grp][k]]
    return ", ".join(q) or "(default)"


def tuned_configs(path_or_dict, tasks) -> Dict[str, inc.RunConfig]:
    """E5_tuned / E0_zeroshot_tuned RunConfigs from a tune.json (path or loaded dict)."""
    t = path_or_dict
    if isinstance(t, str):
        with open(t) as f:
            t = json.load(f)
    e5 = {"pseudo": t["E5"]["pseudo"], "n2l": t["E5"]["n2l"]}
    return {"E5_tuned": cfg_from(tasks, e5, t["E5"]["infer"]),
            "E0_zeroshot_tuned": inc.RunConfig(tasks=tasks, variant="zeroshot", infer=InferConfig(**t["E0"]["infer"])),
            # ground-truth-label ceiling with E5's tuned ridge strength and inference settings (the pseudo-label and
            # refinement settings do not apply to it), so the ceiling is tuned as hard as the methods it bounds
            "E6_gt_ceiling_tuned": cfg_from(tasks, e5, t["E5"]["infer"], variant="gt_ceiling")}


# ------------------------------------------------------------------ which tuned setting carries the gain?
def _load(t):
    if isinstance(t, str):
        with open(t) as f:
            return json.load(f)
    return t


def changed_settings(t) -> List[str]:
    """Dotted names ('n2l.n_rounds', 'infer.yolo_gamma', ...) of the tuned E5 settings that differ from the defaults."""
    t = _load(t)
    d, di = default_params(), dataclasses.asdict(InferConfig())
    out = [f"{g}.{k}" for g in ("pseudo", "n2l") for k, v in t["E5"][g].items() if v != d[g][k]]
    return out + [f"infer.{k}" for k, v in t["E5"]["infer"].items() if v != di[k]]


def loo_configs(t, tasks) -> Dict[str, inc.RunConfig]:
    """Reference rows + 'tuned minus one changed setting' rows, all on the same eval images.

    E5_default, E5_tuned, E5_tuned_only_train (tuned training, default inference), E5_tuned_only_infer (default
    training, tuned inference), E0_zeroshot_tuned, and E5_tuned_minus_<group.key> for every changed setting:
    a row clearly BELOW E5_tuned means that setting carries part of the gain; a row equal to it means it does not."""
    t = _load(t)
    d, di = default_params(), dataclasses.asdict(InferConfig())
    e5 = {"pseudo": t["E5"]["pseudo"], "n2l": t["E5"]["n2l"]}
    out = {"E0_zeroshot_tuned": tuned_configs(t, tasks)["E0_zeroshot_tuned"],
           "E6_gt_ceiling_tuned": tuned_configs(t, tasks)["E6_gt_ceiling_tuned"],
           "E5_default": cfg_from(tasks, d), "E5_tuned": cfg_from(tasks, e5, t["E5"]["infer"]),
           "E5_tuned_only_train": cfg_from(tasks, e5), "E5_tuned_only_infer": cfg_from(tasks, d, t["E5"]["infer"])}
    for name in changed_settings(t):
        g, k = name.split(".")
        if g == "infer":
            out[f"E5_tuned_minus_{name}"] = cfg_from(tasks, e5, {**t["E5"]["infer"], k: di[k]})
        else:
            p = json.loads(json.dumps(e5)); p[g][k] = d[g][k]
            out[f"E5_tuned_minus_{name}"] = cfg_from(tasks, p, t["E5"]["infer"])
    # settings that belong together are also reverted TOGETHER (a single reversion can be hidden by the others)
    ch = changed_settings(t)
    for gname, members in GROUPS.items():
        hit = [m for m in members if m in ch]
        if len(hit) < 2:
            continue                                                   # one member = the single row above
        p = json.loads(json.dumps(e5)); inf = dict(t["E5"]["infer"])
        for m in hit:
            g, k = m.split(".")
            if g == "infer":
                inf[k] = di[k]
            else:
                p[g][k] = d[g][k]
        out[f"E5_tuned_minus_group.{gname}"] = cfg_from(tasks, p, inf)
    return out


GROUPS = {"n2l_machinery": ["n2l.n_rounds", "n2l.use_intra", "n2l.use_inter"],      # refinement + bi-level weights
          "pseudo_filters": ["pseudo.tau_pos", "pseudo.tau_bg", "pseudo.use_loc"],
          "inference": ["infer.fusion_weight", "infer.topk_classes", "infer.yolo_gamma"]}


def run_loo(bundle: inc.Bundle, t, tasks, log=print, n_folds: int = 5) -> Dict:
    out = {}
    for name, cfg in loo_configs(t, tasks).items():
        out[name] = inc.run_incremental(bundle, dataclasses.replace(cfg, n_folds=n_folds), log=lambda *_: None)
        log(f"  {name:38s} AP50={out[name]['final_AP50']:.4f}  AP={out[name]['final_AP']:.4f}")
    return out


# ------------------------------------------------------------------ where does the head help? gamma x fusion weight
SWEEP_GAMMAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
SWEEP_WEIGHTS = (0.0, 0.25, 0.5, 0.8, 1.0)


def run_sweep(bundle: inc.Bundle, tasks, t, gammas=SWEEP_GAMMAS, weights=SWEEP_WEIGHTS, log=print) -> Dict:
    """Zero-shot E0 and the tuned-training E5 head scored over a grid of YOLO^gamma x ridge/CLIP fusion weight.
    DESCRIPTIVE (it is scored on the eval images, so it must not be used to choose settings): it shows where the head helps.
    fusion weight 0 = CLIP only inside the E5 pipeline, 1 = head only."""
    t = _load(t)
    e5 = {"pseudo": t["E5"]["pseudo"], "n2l": t["E5"]["n2l"]}
    e5i = t["E5"]["infer"]
    e0 = infer_grid(bundle, inc.RunConfig(tasks=tasks, variant="zeroshot"),
                    [InferConfig(**{**t["E0"]["infer"], "yolo_gamma": g}) for g in gammas])
    grid = [(g, w) for g in gammas for w in weights]
    e5r = infer_grid(bundle, cfg_from(tasks, e5), [InferConfig(**{**e5i, "yolo_gamma": g, "fusion_weight": w}) for g, w in grid])
    out = {"gammas": list(gammas), "weights": list(weights),
           "E0": [{"gamma": g, "AP50": r["final_AP50"], "AP50_new": r["AP50_new"]} for g, r in zip(gammas, e0)],
           "E5": [{"gamma": g, "weight": w, "AP50": r["final_AP50"], "AP50_new": r["AP50_new"]} for (g, w), r in zip(grid, e5r)]}
    log(f"  sweep: {len(e0)} zero-shot + {len(e5r)} head settings scored")
    return out


def format_sweep(res: Dict) -> str:
    L = ["# YOLO^gamma x fusion weight on the eval images (descriptive; NOT for choosing settings)", "",
         "gamma = exponent on the proposal score (0 = ignore it); weight = share of the ridge head in the fused score "
         "(0 = CLIP only, 1 = head only). The E0 column is zero-shot CLIP with the same gamma.", ""]
    for metric in ("AP50", "AP50_new"):
        L += [f"## {metric}", "", "| gamma | E0 zero-shot | " + " | ".join(f"head w={w:g}" for w in res["weights"]) + " | best head - E0 |",
              "|---|---|" + "---|" * (len(res["weights"]) + 1)]
        for e0 in res["E0"]:
            g = e0["gamma"]
            vals = [next(r[metric] for r in res["E5"] if r["gamma"] == g and r["weight"] == w) for w in res["weights"]]
            best = max(vals)
            L.append(f"| {g:g} | {e0[metric]:.3f} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {best - e0[metric]:+.3f} |")
        L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ which classes gain?
def final_eval(bundle: inc.Bundle, cfg: inc.RunConfig) -> Dict:
    """Full COCO evaluation (incl. per-class AP50) of the model after the LAST task."""
    box = {}

    def probe(ctx):
        if ctx["task"] == len(cfg.tasks) - 1:
            Xe, e_img, Be, Se = ctx["eval"]
            dets = inc.make_detections(bundle.eval_ids, e_img, Be, Se, ctx["fused"], ctx["class_order"], cfg.infer, cfg.tasks[0])
            box["r"] = coco_eval(bundle.eval_gt, dets, bundle.eval_ids, ctx["class_order"])

    inc.run_incremental(bundle, dataclasses.replace(cfg, n_folds=0), log=lambda *_: None, probe=probe)
    return box["r"]


PERCLASS_RUNS = ("E0_zeroshot_tuned", "E5_tuned", "E6_gt_ceiling_tuned")


def run_perclass(bundle: inc.Bundle, tasks, t, log=print) -> Dict:
    import coco_utils as cu
    import pipeline as pl
    cfgs = pl.ablation_configs(tasks, _load(t))
    res = {n: final_eval(bundle, cfgs[n])["per_class_AP50"] for n in PERCLASS_RUNS}
    names = cu.prompt_names()
    out = {"tasks": [list(x) for x in tasks],
           "classes": [{"class": int(c), "name": names[c], "task": k, **{n: res[n][c] for n in PERCLASS_RUNS}}
                       for k, cl in enumerate(tasks) for c in cl]}
    log(f"  per-class AP50 for {len(out['classes'])} classes")
    return out


def format_perclass(res: Dict) -> str:
    rows = res["classes"]
    last = len(res["tasks"]) - 1
    L = ["# Per-class AP50 on the eval images (new classes first)", "",
         "| class | task | E0 tuned | E5 tuned | E6 ceiling (tuned) | E5 - E0 | share of E6 - E0 gained |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["task"] != last, r["task"], -(r["E5_tuned"] - r["E0_zeroshot_tuned"]))):
        d, head = r["E5_tuned"] - r["E0_zeroshot_tuned"], r["E6_gt_ceiling_tuned"] - r["E0_zeroshot_tuned"]
        share = f"{d / head:.0%}" if abs(head) > 0.01 else "-"
        L.append(f"| {r['name']} | {'new' if r['task'] == last else 'base'} | {r['E0_zeroshot_tuned']:.3f} | {r['E5_tuned']:.3f} | "
                 f"{r['E6_gt_ceiling_tuned']:.3f} | {d:+.3f} | {share} |")
    for label, sel in (("new", lambda r: r["task"] == last), ("base", lambda r: r["task"] != last)):
        c = [r for r in rows if sel(r)]
        g = [r["E5_tuned"] - r["E0_zeroshot_tuned"] for r in c]
        L.append(f"| **mean {label}** | {label} | {np.mean([r['E0_zeroshot_tuned'] for r in c]):.3f} | {np.mean([r['E5_tuned'] for r in c]):.3f} | "
                 f"{np.mean([r['E6_gt_ceiling_tuned'] for r in c]):.3f} | {np.mean(g):+.3f} | |")
    better = sum(1 for r in rows if r["task"] == last and r["E5_tuned"] > r["E0_zeroshot_tuned"])
    n_new = sum(1 for r in rows if r["task"] == last)
    L += ["", f"E5 tuned is above E0 tuned on {better} of {n_new} new classes. Single classes have few images: read the pattern, not single rows."]
    return "\n".join(L)


# ------------------------------------------------------------------ variation from the TRAINING images
def subsample_train_ids(train_ids: Sequence[Sequence[int]], frac: float, seed: int) -> List[List[int]]:
    """Random subset (without replacement, sorted) of each task's training images; same fraction in every task."""
    if not 0 < frac <= 1:
        raise ValueError("train_frac must be in (0, 1]")
    rng = np.random.default_rng(seed)
    out = []
    for ids in train_ids:
        ids = sorted(int(i) for i in ids)
        n = max(1, int(round(frac * len(ids))))
        out.append(sorted(int(i) for i in rng.choice(ids, size=min(n, len(ids)), replace=False)))
    return out


SEED_RUNS = ("E0_zeroshot_tuned", "E5_ours", "E5_tuned", "B_plain_ridge", "E6_gt_ceiling_tuned")


def run_seeds(bundle: inc.Bundle, tasks, t, n_seeds: int = 5, train_frac: float = 0.8, seed0: int = 0, log=print) -> Dict:
    """Re-train each method on `n_seeds` different random subsets of the cached training images and score on the SAME eval
    images. Shows how much the result depends on WHICH training images were used (not on the eval images: that is what
    the folds measure, and not on re-training the detector or on the class split). Training-free E0 is run once."""
    import pipeline as pl
    if n_seeds < 2:
        raise ValueError("need at least 2 seeds")
    cfgs = pl.ablation_configs(tasks, t)
    quiet = lambda *_: None
    rows = []
    e0 = inc.run_incremental(bundle, dataclasses.replace(cfgs["E0_zeroshot_tuned"], n_folds=0), log=quiet)
    for k in range(n_seeds):
        sub = subsample_train_ids(bundle.train_ids, train_frac, seed0 + k)
        b = dataclasses.replace(bundle, train_ids=sub)
        row = {"seed": seed0 + k, "n_train": [len(x) for x in sub]}
        for name in SEED_RUNS:
            r = e0 if name == "E0_zeroshot_tuned" else inc.run_incremental(b, dataclasses.replace(cfgs[name], n_folds=0), log=quiet)
            row[name] = {"AP50": r["final_AP50"], "AP": r["final_AP"], "AP50_new": r["curve"][-1]["AP50_new"]}
        rows.append(row)
        log(f"  seed {seed0 + k}: " + "  ".join(f"{n}={row[n]['AP50']:.4f}" for n in SEED_RUNS))
    return {"n_seeds": n_seeds, "train_frac": train_frac, "seed0": seed0, "per_seed": rows}


def format_seeds(res: Dict, ref: str = "E0_zeroshot_tuned") -> str:
    import compare_runs as cr
    L = [f"# {res['n_seeds']} random subsets of the training images ({res['train_frac']:.0%} of each task), same eval images", ""]
    for metric in ("AP50", "AP50_new"):
        L += [f"| run | {metric} mean | sd over seeds | diff vs {ref} | 95% CI (+-) |", "|---|---|---|---|---|"]
        for name in SEED_RUNS:
            v = [r[name][metric] for r in res["per_seed"]]
            m, sd, _ = cr.mean_sd(v)
            if name == ref:
                L.append(f"| {name} | {m:.3f} | {sd:.3f} | - | - |"); continue
            d, ci, _ = cr.paired_ci(v, [r[ref][metric] for r in res["per_seed"]])
            L.append(f"| {name} | {m:.3f} | {sd:.3f} | {d:+.3f} | {ci:.3f} |")
        L.append("")
    L.append("Only the training images vary here (detector, class split, eval images fixed); the fold intervals in `compare_*.md` "
             "measure the eval-image variation. Both are small-sample intervals: indicative only. The reference run does not vary with "
             "the seed (zero-shot CLIP uses no training images), so the interval here only reflects how stable the head is, "
             "NOT whether the gain is significant: use the fold intervals for that.")
    return "\n".join(L)


def format_report(t: dict) -> str:
    s = t["tune_scores"]
    L = [f"# Tuning on {t['n_tune_images']} held-out images (objective {t['objective']}, seed {t['seed']})", "",
         f"| | default | tuned |", "|---|---|---|",
         f"| E5 | {s['E5_default']:.4f} | {s['E5_tuned']:.4f} |", f"| E0 zero-shot | {s['E0_default']:.4f} | {s['E0_tuned']:.4f} |", "",
         "Chosen E5 settings: " + _short({"pseudo": t["E5"]["pseudo"], "n2l": t["E5"]["n2l"]}) +
         f"; infer {t['E5']['infer']}", f"Chosen E0 inference settings: {t['E0']['infer']}", "",
         "## Stage 1 (best 10 of {})".format(len(t["stage1"])), "", "| trial | AP50 | AP | settings changed from default |", "|---|---|---|---|"]
    if t["objective"] == "final_AP50_base":
        L[-4] = L[-4].replace("Stage 1 (best", "Stage 1, sorted by all-class AP50 (selection used base-class AP50 only; best")
    for r in sorted(t["stage1"], key=lambda r: -r["final_AP50"])[:10]:
        L.append(f"| {r['trial']} | {r['final_AP50']:.4f} | {r['final_AP']:.4f} | {_short(r['params'])} |")
    L += ["", "Reading it: if the best stage-1 row is barely above trial 0 (default) the settings do not matter; "
          "if trial 1 (no refinement) is near the top, refinement is not helping.", ""]
    return "\n".join(L)

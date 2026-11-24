"""pipeline.py -- stage functions that glue the tested modules together.

Every stage takes a `models` object (real: real_models.RealModels; fake: fake_models.FakeModels):
    models.dim, models.encode_text(list[str]) -> (n,D), models.encode_images_u8(uint8 crops) -> (n,D),
    models.propose(PIL) -> (boxes xyxy, scores)
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

import coco_utils as cu
import evaluate as ev
import incremental as inc
import n2l_core as nc
import prompts as pr
import pseudo as ps
from extract_features import extract_cache
from feature_cache import FeatureCache
from gates import gt_crop_zeroshot_eval, recall_over_dataset
from inference import InferConfig


def iter_images(coco: dict, ids: Sequence[int], img_dir: str):
    by_id = {im["id"]: im for im in coco["images"]}
    for i in ids:
        with Image.open(os.path.join(img_dir, by_id[i]["file_name"])) as im:
            yield i, im.convert("RGB")


LABEL_TEMPLATES = 1          # how many prompt templates the PSEUDO-LABEL text uses (1 = official N2L; 0 = all of them)


def set_label_templates(n: int) -> None:
    global LABEL_TEMPLATES
    n = int(n)
    if not 0 <= n <= len(pr.DET_TEMPLATES):
        raise ValueError(f"label_templates must be in 0..{len(pr.DET_TEMPLATES)} (0 = all), got {n}")
    LABEL_TEMPLATES = n


def text_assets(models) -> Dict[str, np.ndarray]:
    """label text = the first LABEL_TEMPLATES templates (default 1, as the official pseudo-labelling); eval text = full ensemble."""
    names = cu.prompt_names()
    lt = pr.DET_TEMPLATES[:LABEL_TEMPLATES] if LABEL_TEMPLATES else pr.DET_TEMPLATES
    return {"text_label": pr.class_embeddings(names, models.encode_text, lt),
            "text_eval": pr.class_embeddings(names, models.encode_text, pr.DET_TEMPLATES),
            "bg_label": pr.background_embedding(models.encode_text),
            "bg_eval": pr.background_embedding(models.encode_text)}


# ------------------------------------------------------------------ Gate E0
def stage_gate_clip(models, coco: dict, img_dir: str, ids: Sequence[int], tasks) -> Dict:
    """Zero-shot CLIP accuracy on GT-box crops: all classes, and within each task's classes."""
    T = text_assets(models)["text_eval"]
    gt = cu.gt_by_image(coco, ids)
    items = lambda: ((im, *gt[i]) for i, im in iter_images(coco, ids, img_dir))
    used = sorted({c for cls in tasks for c in cls})           # candidates = the classes this run actually uses
    subset = None if len(used) == T.shape[0] else used          # (all 80 for COCO 40+40: unchanged)
    out = {"all": gt_crop_zeroshot_eval(items(), models.encode_images_u8, T, class_subset=subset)["acc"]}
    for t, cls in enumerate(tasks):
        out[f"task{t}_within"] = gt_crop_zeroshot_eval(items(), models.encode_images_u8, T,
                                                       class_subset=cls)["acc"]
    return out


# ------------------------------------------------------------------ extraction
def stage_extract(models, coco: dict, ids: Sequence[int], img_dir: str, out_dir: str,
                  pad_frac: float = 0.1, max_boxes: int = 100) -> str:
    return extract_cache(iter_images(coco, ids, img_dir), models.propose, models.encode_images_u8,
                         out_dir, models.dim, pad_frac=pad_frac, max_boxes=max_boxes)


def stage_gate_recall(cache_dir: str, coco: dict, ids: Sequence[int], tasks) -> Dict:
    """GATE 1: how many GT objects of each task have a frozen-proposer box at IoU>=0.5/0.7."""
    cache = FeatureCache(cache_dir)
    gt = cu.gt_by_image(coco, ids)
    props = {i: cache.image(i)["boxes"] for i in ids}
    groups = {f"task{t}": cls for t, cls in enumerate(tasks)}
    return recall_over_dataset(props, gt, groups)


# ------------------------------------------------------------------ the experiment
def build_bundle(models, train_cache_dir: str, train_ids: List[List[int]], coco_train: dict,
                 eval_cache_dir: str, eval_ids: List[int], coco_val: dict) -> inc.Bundle:
    T = text_assets(models)
    tr, evc = FeatureCache(train_cache_dir), FeatureCache(eval_cache_dir)
    dims = {"text": T["text_label"].shape[1], "train features": tr.meta["feat_dim"],
            "eval features": evc.meta["feat_dim"]}
    if len(set(dims.values())) != 1:
        raise ValueError(f"embedding dimensions disagree {dims}: features and text must come from the same CLIP model")
    all_tr = sorted({i for ids in train_ids for i in ids})
    return inc.Bundle(tr, train_ids, cu.gt_by_image(coco_train, all_tr), evc, list(eval_ids),
                      ev.make_coco_gt(coco_val, eval_ids), **T)


def ablation_configs(tasks: List[List[int]], tuned=None) -> Dict[str, inc.RunConfig]:
    """E0 / E5 / E5 ablations / E6. Each differs from E5 in exactly one setting.
    `tuned` (a tune.json path or dict) adds E5_tuned and E0_zeroshot_tuned: settings chosen on held-out tune images."""
    R = lambda **kw: inc.RunConfig(tasks=tasks, **kw)
    n2l = lambda **kw: nc.N2LConfig(**{"pca_rule": "energy", "pca_rho": 0.95, "pca_k_floor": "classes", **kw})
    out = {
        # InferConfig's default yolo_gamma is 0.5 (it helped in every COCO run); *_g0 / *_g1 are the variants
        "E0_zeroshot": R(variant="zeroshot"),
        "E0_zeroshot_g0": R(variant="zeroshot", infer=InferConfig(yolo_gamma=0.0)),
        "E0_zeroshot_g1": R(variant="zeroshot", infer=InferConfig(yolo_gamma=1.0)),
        "E5_ours": R(),
        "E5_no_refine": R(n2l=n2l(n_rounds=0)),
        "E5_no_weights": R(n2l=n2l(use_intra=False, use_inter=False)),
        "E5_inter_only": R(n2l=n2l(use_intra=False)),
        "E5_no_bg": R(pseudo=ps.PseudoConfig(use_bg=False)),
        "E5_ignore_band": R(pseudo=ps.PseudoConfig(tau_pos=0.5, tau_bg=0.5)),
        "E5_loc_weight": R(pseudo=ps.PseudoConfig(use_loc=True)),
        "E5_k_energy80_nofloor": R(n2l=n2l(pca_rho=0.8, pca_k_floor="none")),
        "E5_k_energy99": R(n2l=n2l(pca_rho=0.99)),
        "E5_k_abs_official": R(n2l=n2l(pca_rule="abs_official", pca_theta=10.0, pca_k_floor="none")),
        "E5_expand2000": R(expand=2000),
        "E5_fusion_0.5": R(infer=InferConfig(fusion_weight=0.5)),
        "E5_ridge_only": R(infer=InferConfig(fusion_weight=1.0)),
        "E5_yolo_gamma0": R(infer=InferConfig(yolo_gamma=0.0)),
        "E5_yolo_gamma1": R(infer=InferConfig(yolo_gamma=1.0)),
        # baseline: ridge on the raw CLIP pseudo-labels -- no N2L refinement, no bi-level weights (the simplest self-training)
        "B_plain_ridge": R(n2l=n2l(n_rounds=0, use_intra=False, use_inter=False)),
        "E6_gt_ceiling": R(variant="gt_ceiling"),
        "E6_gt_ceiling_g0": R(variant="gt_ceiling", infer=InferConfig(yolo_gamma=0.0)),
        "E6_gt_ceiling_g1": R(variant="gt_ceiling", infer=InferConfig(yolo_gamma=1.0)),
    }
    if tuned is not None:
        import tune as tn
        out.update(tn.tuned_configs(tuned, tasks))
    return out


def run_ablation(bundle: inc.Bundle, tasks, only: Optional[Sequence[str]] = None, log=print, tuned=None) -> Dict:
    cfgs = ablation_configs(tasks, tuned)
    names = list(cfgs) if only is None else list(only)
    unknown = [n for n in names if n not in cfgs]
    if unknown:
        raise KeyError(f"unknown ablation names {unknown}; choose from {list(cfgs)}")
    return {n: inc.run_incremental(bundle, cfgs[n], log) for n in names}


def format_table(results: Dict) -> str:
    rows = ["| run | A-bar AP | final AP | final AP50 | base AP50 drop |", "|---|---|---|---|---|"]
    for n, r in results.items():
        rows.append(f"| {n} | {r['A_bar_AP']:.3f} | {r['final_AP']:.3f} | {r['final_AP50']:.3f} | "
                    f"{r['base_AP50_drop']:+.3f} |")
    return "\n".join(rows)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_results(results: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=1, default=_json_default)

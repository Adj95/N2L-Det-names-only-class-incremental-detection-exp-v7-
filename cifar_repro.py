"""cifar_repro.py -- reproduce the official N2L CIFAR-100 B0Inc10 protocol with n2l_core.

The library part (this file) is torch-free and tested on synthetic features. The Colab driver
(`run_pipeline.py repro-cifar`) extracts the CLIP features and calls `run_cifar_protocol`.

Protocol, copied from the official code (read, not executed):
  * class order = np.random.seed(1993); np.random.permutation(100)  (DataManager, shuffle=True)
  * 10 tasks x 10 classes; training uses the 500 images/class of the current task, unlabeled
  * pseudo-labels: softmax-free argmax of 100*cos(img, text) over the CURRENT task's names,
    text from the FIRST template only ("a photo of a {}.")
  * refinement: centred PCA keep sigma>=10 (+1 quirk), 3 rounds, lam=0.1, inter-weights only;
    final fit: bi-level weights (gauss std .5 / min .25), frozen random 512->5000 ReLU expansion
  * eval after each task on test images of ALL classes seen so far; logit =
    0.2*softmax(100*cos(img, ensemble-of-18-templates)) + 0.8*ridge score
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

import numpy as np
from scipy.special import softmax

import n2l_core as nc
from prompts import l2norm


def official_class_order(n_classes: int = 100, seed: int = 1993) -> List[int]:
    np.random.seed(seed)
    return np.random.permutation(n_classes).tolist()


def remap_targets(y: np.ndarray, order: Sequence[int]) -> np.ndarray:
    """Official _map_new_class_index: new label = position of the original class in `order`."""
    pos = {c: i for i, c in enumerate(order)}
    return np.array([pos[int(v)] for v in y])


def load_official_text_assets(repo_dir: str, key: str = "cifar224"):
    with open(os.path.join(repo_dir, "utils", "labels.json")) as f:
        names = json.load(f)[key]
    with open(os.path.join(repo_dir, "utils", "templates.json")) as f:
        templates = json.load(f)[key]
    return names, templates


def run_cifar_protocol(train_feats: np.ndarray, train_y: np.ndarray,
                       test_feats: np.ndarray, test_y: np.ndarray,
                       label_text_emb: np.ndarray, eval_text_emb: np.ndarray,
                       init_cls: int = 10, increment: int = 10, expand: int = 5000,
                       cfg: nc.N2LConfig = None, fusion_weight: float = 0.8,
                       seed: int = 1993, log=print) -> Dict:
    """train_y/test_y must ALREADY be in the shuffled order (use remap_targets).
    label_text_emb : (C,D) unit-norm, FIRST-template embeddings, rows in shuffled order
    eval_text_emb  : (C,D) unit-norm, template-ensemble embeddings, rows in shuffled order
    Returns per-task accuracies for the fused and ridge-only heads, and zero-shot."""
    cfg = cfg or nc.N2LConfig(lam=0.1, n_rounds=3, pca_rule="abs_official", pca_theta=10.0,
                              gau_std=0.5, gau_min=0.25, seed=seed)
    C = label_text_emb.shape[0]
    D = train_feats.shape[1]
    fmap = nc.RandomReLUExpansion(D, expand, seed=seed) if expand else None
    learner = nc.N2LLearner(D, cfg, feature_map=fmap, feat_dim=expand or D)
    bounds = [(0, init_cls)]
    while bounds[-1][1] < C:
        bounds.append((bounds[-1][1], min(bounds[-1][1] + increment, C)))
    res = {"fused": [], "ridge": [], "zeroshot": [], "k": [], "label_acc": []}
    rng = np.random.default_rng(seed)
    for t, (lo, hi) in enumerate(bounds):
        m = (train_y >= lo) & (train_y < hi)
        X = train_feats[m]
        zs = 100.0 * l2norm(X) @ label_text_emb[lo:hi].T            # official pseudo-label logits
        r = learner.add_task(X, zs, hi - lo, rng=rng)
        res["k"].append(r.k)
        res["label_acc"].append(float(np.mean(r.labels + lo == train_y[m])))
        seen = test_y < hi
        Xt, yt = test_feats[seen], test_y[seen]
        zs_t = softmax(100.0 * l2norm(Xt) @ eval_text_emb[:hi].T, axis=1)
        ridge = learner.scores(Xt)
        fused = (1 - fusion_weight) * zs_t + fusion_weight * ridge
        res["fused"].append(float(np.mean(fused.argmax(1) == yt)))
        res["ridge"].append(float(np.mean(ridge.argmax(1) == yt)))
        res["zeroshot"].append(float(np.mean(zs_t.argmax(1) == yt)))
        log(f"task {t + 1}/{len(bounds)} classes {lo}-{hi}: k={r.k} pseudo-label acc "
            f"{res['label_acc'][-1]:.3f} | seen-class top1: zero-shot {res['zeroshot'][-1]:.4f} "
            f"ridge {res['ridge'][-1]:.4f} fused {res['fused'][-1]:.4f}")
    for key in ("fused", "ridge", "zeroshot"):
        res[f"A_bar_{key}"] = float(np.mean(res[key]))
        res[f"A_B_{key}"] = res[key][-1]
    return res

"""heads.py -- alternative heads trained on the SAME pseudo-labels as the N2L ridge head (a 'simple baseline' check).

ProtoLearner: one prototype (weighted mean of L2-normalised features) per head column, scores = softmax(scale * cosine).
Recursive like the ridge head: a task only adds sums, so earlier tasks need no stored features. No refinement,
no bi-level weights: labels are the zero-shot argmax (after the same pseudo-label filters), weights are `base_weights`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import softmax

import n2l_core as nc
from prompts import l2norm


class ProtoLearner:
    def __init__(self, in_dim: int, scale: float = 100.0):
        self.in_dim, self.scale = int(in_dim), float(scale)
        self.S = np.zeros((0, self.in_dim))             # (K, D) weighted feature sums
        self.cnt = np.zeros(0)                          # (K,) total weight per column
        self.results: list = []

    @property
    def num_classes(self) -> int:
        return self.S.shape[0]

    def add_task(self, X, zs_logits, n_new_classes: int, rng=None, base_weights=None, **_refine_kw) -> nc.N2LResult:
        L = nc._as2d(zs_logits, "zs_logits")
        X = l2norm(nc._as2d(X, "X"))
        if len(X) != len(L):
            raise ValueError("X and zs_logits disagree on n")
        self.S = np.vstack([self.S, np.zeros((n_new_classes, self.in_dim))])
        self.cnt = np.concatenate([self.cnt, np.zeros(n_new_classes)])
        if L.shape[1] > self.num_classes:
            raise ValueError("zs_logits has more columns than the head has classes")
        off = self.num_classes - L.shape[1]
        labels = L.argmax(axis=1)
        w = np.ones(len(X)) if base_weights is None else np.asarray(base_weights, dtype=np.float64)
        if w.shape != (len(X),) or np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("base_weights must be finite, >= 0, shape (n,)")
        np.add.at(self.S, labels + off, X * w[:, None])
        np.add.at(self.cnt, labels + off, w)
        ent = nc.prediction_entropy(L)
        res = nc.N2LResult(labels=labels, weights=w, entropy=ent, k=0, label_history=[labels.copy()], basis=None)
        self.results.append(res)
        return res

    def scores(self, X) -> np.ndarray:
        if self.num_classes == 0:
            raise RuntimeError("no task added yet")
        X = l2norm(nc._as2d(X, "X"))
        P = l2norm(self.S)
        z = self.scale * X @ P.T
        z[:, self.cnt <= 0] = -np.inf                   # a column nobody was labelled with cannot win
        if np.all(np.isinf(z)):
            return np.zeros_like(z)
        return softmax(z, axis=1)

    def add_labeled(self, *a, **k):
        raise NotImplementedError("the prototype head is only defined for pseudo-labels ('ours')")


def make_learner(cfg, dim: int, feature_map=None):
    """The head named by cfg.head: 'ridge' (N2L, default) or 'proto'."""
    if cfg.head == "ridge":
        return nc.N2LLearner(dim, cfg.n2l, feature_map=feature_map, feat_dim=cfg.expand or dim)
    if cfg.head == "proto":
        if cfg.expand:
            raise ValueError("head='proto' does not use the random feature expansion (expand must be 0)")
        if cfg.variant == "gt_ceiling":
            raise ValueError("head='proto' is only defined for variant 'ours'")
        return ProtoLearner(dim, scale=cfg.pseudo.scale)
    raise ValueError(f"unknown head {cfg.head!r}")

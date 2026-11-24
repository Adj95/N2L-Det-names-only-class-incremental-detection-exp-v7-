"""
n2l_core.py -- Step 1 of N2L-Det: the closed-form Naming-to-Learn core.

Pure numpy/scipy (no torch), so it runs and is unit-tested anywhere.
It is a re-implementation of the training logic in the official N2L code
(github.com/zhoujiahuan1991/ICLR2026-N2L, models/naming2learn.py), with the
deviations listed in DEVIATIONS below.

What lives here
---------------
  RidgeAccumulator   A = sum m f f^T, C = sum m f y^T, W = (A + lam I)^-1 C,
                     with zero-padded columns when new classes arrive.
  PCABasis           centred PCA of the features + the rule that picks k.
  bilevel_weights    inter-class balance x intra-class entropy-ranked Gaussian.
  refine_task_labels the 3-round SVD/ridge pseudo-label refinement.
  N2LLearner         ties it together: add_task(features, zero-shot logits).
  fuse_scores        0.2*softmax(zero-shot) + 0.8*ridge score (official fusion).

DEVIATIONS from the official code (all switchable, all covered by tests)
---------------------------------------------------------------------
  1. float64 + Cholesky solve instead of float32 explicit inverse.
  2. mask_empty_classes=True: a class with no samples in a refinement round
     cannot win the argmax with an all-zero score column (official code lets it).
  3. PCA rule "abs_official" reproduces the official k (including its
     off-by-one, see official_k). Other rules ("abs", "scaled", "energy") are
     ours, needed because the official threshold is tied to n and feature scale.
  4. Labels may range over ALL seen classes (+ background) instead of only the
     current task's classes: N2LLearner right-aligns the label columns.
  5. Optional base_weights (e.g. objectness) multiply the bi-level weights.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import numpy as np
from scipy import linalg as sla
from scipy.special import softmax, xlogy


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _as2d(x, name):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN/Inf")
    return x


def prediction_entropy(logits: np.ndarray) -> np.ndarray:
    """Shannon entropy of softmax(logits) per row. Safe for zero probabilities."""
    p = softmax(np.asarray(logits, dtype=np.float64), axis=1)
    return -xlogy(p, p).sum(axis=1)


class RandomReLUExpansion:
    """Frozen random feature map f -> ReLU(f W), W ~ U(-1/sqrt(in), 1/sqrt(in)).

    Same init as the official `nn.Linear(512, expand, bias=False)` + ReLU.
    Never trained; must be the SAME object for every task and at inference.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        bound = 1.0 / np.sqrt(in_dim)
        self.W = rng.uniform(-bound, bound, size=(in_dim, out_dim))
        self.in_dim, self.out_dim = in_dim, out_dim

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(X, dtype=np.float64) @ self.W, 0.0)


# --------------------------------------------------------------------------
# ridge accumulator  (A, C, W)
# --------------------------------------------------------------------------
class RidgeAccumulator:
    """Recursive closed-form ridge regression.

        A  (d x d) = sum_i m_i f_i f_i^T
        C  (d x K) = sum_i m_i f_i y_i^T
        W  (d x K) = (A + lam I)^-1 C

    New classes append zero columns to C. Adding data and re-solving gives
    exactly the W you would get by fitting all data jointly (tested).
    """

    def __init__(self, dim: int, lam: float = 0.1):
        if lam <= 0:
            raise ValueError("lam must be > 0 (A alone may be singular)")
        self.dim, self.lam = int(dim), float(lam)
        self.A = np.zeros((self.dim, self.dim))
        self.C = np.zeros((self.dim, 0))
        self.n_samples = 0

    @property
    def num_classes(self) -> int:
        return self.C.shape[1]

    def add_classes(self, n_new: int) -> None:
        if n_new < 0:
            raise ValueError("n_new must be >= 0")
        self.C = np.hstack([self.C, np.zeros((self.dim, n_new))])

    def update(self, F, targets, weights=None) -> None:
        """Add a batch. `targets` is int labels (n,) or a target matrix (n, K).

        Zero-weight rows are allowed (they contribute nothing).
        """
        F = _as2d(F, "F")
        n = F.shape[0]
        if F.shape[1] != self.dim:
            raise ValueError(f"F has dim {F.shape[1]}, expected {self.dim}")
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
        if w.shape != (n,) or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite, non-negative, shape (n,)")
        # ---- validate EVERYTHING before mutating, so a bad batch cannot corrupt A/C
        t = np.asarray(targets)
        if t.ndim == 1:
            t = t.astype(np.int64)
            if t.shape != (n,):
                raise ValueError("labels must have shape (n,)")
            if n and (t.min() < 0 or t.max() >= self.num_classes):
                raise ValueError(
                    f"label out of range [0,{self.num_classes}): "
                    f"min={t.min()}, max={t.max()} (did you call add_classes?)")
            Y = None
        else:
            Y = _as2d(t, "Y")
            if Y.shape != (n, self.num_classes):
                raise ValueError(f"Y shape {Y.shape}, expected {(n, self.num_classes)}")
        # ---- apply
        Fw = F * w[:, None]
        self.A += F.T @ Fw
        if Y is None:
            for c in np.unique(t):
                self.C[:, c] += Fw[t == c].sum(axis=0)
        else:
            self.C += Fw.T @ Y
        self.n_samples += n

    def solve(self) -> np.ndarray:
        """W = (A + lam I)^-1 C via Cholesky (A + lam I is SPD)."""
        if self.num_classes == 0:
            return np.zeros((self.dim, 0))
        M = self.A + self.lam * np.eye(self.dim)
        return sla.solve(M, self.C, assume_a="pos")


# --------------------------------------------------------------------------
# PCA basis + rule for choosing k
# --------------------------------------------------------------------------
def official_k(sv: np.ndarray, theta: float) -> int:
    """Exactly what the official svd_torch does: loop until the first singular
    value < theta, then keep components[: index+1].  => (#sv>=theta) + 1 when some
    sv < theta (one extra component), and all components if none is below."""
    sv = np.asarray(sv)
    below = np.flatnonzero(sv < theta)
    return len(sv) if below.size == 0 else int(below[0]) + 1


def choose_k(sv, rule="abs_official", theta=10.0, n=None, n_ref=None, rho=0.8) -> int:
    """sv: singular values of the CENTRED feature matrix, descending.

    abs_official : official rule (sigma >= theta, plus its off-by-one)
    abs          : #{sigma >= theta}, at least 1
    scaled       : #{sigma >= theta*sqrt(n/n_ref)}, at least 1. Keeps the per-sample
                   variance threshold fixed when the number of samples n changes.
    energy       : smallest k with cumsum(sigma^2)/sum(sigma^2) >= rho
    """
    sv = np.asarray(sv, dtype=np.float64)
    if rule == "abs_official":
        return official_k(sv, theta)
    if rule == "abs":
        return max(1, int(np.sum(sv >= theta)))
    if rule == "scaled":
        if n is None or n_ref is None:
            raise ValueError("rule='scaled' needs n and n_ref")
        return max(1, int(np.sum(sv >= theta * np.sqrt(n / n_ref))))
    if rule == "energy":
        e = np.cumsum(sv ** 2)
        if e[-1] <= 0:
            return 1
        return int(min(len(sv), np.searchsorted(e / e[-1], rho - 1e-12) + 1))
    raise ValueError(f"unknown rule {rule!r}")


class MomentAccumulator:
    """Streaming n, sum(x), sum(x x^T): lets PCA run on millions of boxes without
    holding the feature matrix (needed for detection)."""

    def __init__(self, dim: int):
        self.n = 0
        self.s1 = np.zeros(dim)
        self.s2 = np.zeros((dim, dim))

    def update(self, X):
        X = _as2d(X, "X")
        self.n += X.shape[0]
        self.s1 += X.sum(axis=0)
        self.s2 += X.T @ X


@dataclasses.dataclass
class PCABasis:
    mean: np.ndarray            # (d,)
    components_all: np.ndarray  # (d, r) all principal directions, strongest first
    singular_values: np.ndarray
    k: int                      # number of directions actually used
    n: int

    @property
    def components(self) -> np.ndarray:
        return self.components_all[:, :self.k]

    def project(self, X) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mean) @ self.components

    def with_k(self, k: int) -> "PCABasis":
        k = int(min(max(k, 1), self.components_all.shape[1]))
        return dataclasses.replace(self, k=k)

    @classmethod
    def fit(cls, X, method="auto", **rule_kw) -> "PCABasis":
        X = _as2d(X, "X")
        n = X.shape[0]
        if method == "auto":
            method = "svd" if n <= 50_000 else "moments"
        if method == "svd":
            mu = X.mean(axis=0)
            _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
            k = choose_k(s, n=n, **rule_kw)
            return cls(mu, Vt.T.copy(), s, k, n)
        acc = MomentAccumulator(X.shape[1])
        for i in range(0, n, 20_000):
            acc.update(X[i:i + 20_000])
        return cls.from_moments(acc, **rule_kw)

    @classmethod
    def from_moments(cls, acc: MomentAccumulator, **rule_kw) -> "PCABasis":
        mu = acc.s1 / acc.n
        S = acc.s2 - acc.n * np.outer(mu, mu)           # centred scatter = Xc^T Xc
        S = (S + S.T) / 2
        evals, evecs = np.linalg.eigh(S)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        s = np.sqrt(np.clip(evals, 0, None))
        k = choose_k(s, n=acc.n, **rule_kw)
        return cls(mu, evecs.copy(), s, k, acc.n)


# --------------------------------------------------------------------------
# bi-level weights
# --------------------------------------------------------------------------
def intra_class_gaussian_weights(n, rng, std=0.5, wmin=0.25) -> np.ndarray:
    """N(1, std^2) draws, clamped to [wmin, 2-wmin], rescaled to mean 1
    (official generate_mixture_weights). Strictly positive."""
    w = np.clip(rng.normal(1.0, std, size=n), wmin, 2.0 - wmin)
    return w * n / w.sum()


def bilevel_weights(labels, entropy, num_classes, rng, intra=True, inter=True,
                    std=0.5, wmin=0.25, base_weights=None) -> np.ndarray:
    """m_i = m_intra * m_inter * base.

    intra: inside each pseudo-class, sort the Gaussian weights descending and give
           the largest to the lowest-entropy (most confident) samples.
    inter: n / (N_c * C_present) -- every present class gets total weight n/C_present.
    """
    labels = np.asarray(labels, dtype=np.int64)
    entropy = np.asarray(entropy, dtype=np.float64)
    n = labels.shape[0]
    m = np.ones(n)
    counts = np.bincount(labels, minlength=num_classes)
    present = np.flatnonzero(counts)
    C = len(present)
    for c in present:
        idx = np.flatnonzero(labels == c)
        if intra:
            order = idx[np.argsort(entropy[idx], kind="stable")]   # confident first
            g = np.sort(intra_class_gaussian_weights(len(idx), rng, std, wmin))[::-1]
            m[order] = g
        if inter:
            m[idx] *= n / (counts[c] * C)
    if base_weights is not None:
        b = np.asarray(base_weights, dtype=np.float64)
        if b.shape != (n,) or np.any(b < 0) or not np.all(np.isfinite(b)):
            raise ValueError("base_weights must be finite, >= 0, shape (n,)")
        m = m * b
    return m


# --------------------------------------------------------------------------
# label refinement
# --------------------------------------------------------------------------
@dataclasses.dataclass
class N2LConfig:
    lam: float = 0.1
    n_rounds: int = 3
    pca_rule: str = "abs_official"
    pca_theta: float = 10.0
    pca_n_ref: Optional[float] = None
    pca_rho: float = 0.8
    gau_std: float = 0.5
    gau_min: float = 0.25
    use_intra: bool = True
    use_inter: bool = True
    mask_empty_classes: bool = True
    pca_k_floor: str = "none"     # 'classes': k >= (#classes present in the round-0 labels) - 1
    soft: float = 0.0             # ridge targets = (1-soft)*one-hot(final label) + soft*softmax(zero-shot logits); 0 = hard labels
    # --- N2L-Det-exp round 2: detection-aware refinement. Defaults = OFF = the frozen refinement exactly.
    refine_gate: float = 0.0      # >0: only boxes with detector score >= this may change label; the rest keep the zero-shot label
    anchor: float = 0.0           # >0: each round's scores = (1-anchor)*ridge + anchor*softmax(zero-shot) (labels cannot drift far)
    refine_mode: str = "full"     # 'full' | 'to_bg' (refinement may only turn a label into background) | 'agree' (keep the label
                                  #  only where refinement and zero-shot agree; disagreeing boxes get weight 0)
    # --- N2L-Det-exp v4: forgetting baselines (ablation only). Default 'full' = the recursive head keeps all old statistics.
    memory: str = "full"          # 'full' | 'none' (statistics reset at every later task: the head is re-fit on the current
                                  #  task only) | 'none_newonly' ('none' + the current task's boxes may only be named a NEW
                                  #  class or background: the classic fine-tune-on-new-data setting)
    seed: int = 0


@dataclasses.dataclass
class N2LResult:
    labels: np.ndarray            # final pseudo-labels (local columns)
    weights: np.ndarray           # final bi-level weights
    entropy: np.ndarray
    k: int                        # PCA dimension used
    label_history: list           # labels after round 0 (zero-shot), 1, ..., n_rounds
    basis: PCABasis


def _rule_kwargs(cfg: N2LConfig) -> dict:
    return dict(rule=cfg.pca_rule, theta=cfg.pca_theta, n_ref=cfg.pca_n_ref, rho=cfg.pca_rho)


def refine_task_labels(X, zs_logits, cfg: N2LConfig, rng=None,
                       base_weights=None, gate=None, bg_col=None) -> N2LResult:
    """Progressive label refinement (N2L, Sec. 'label refinement').

    X          (n, d)  frozen-encoder features of the unlabeled task data
    zs_logits  (n, K)  zero-shot scores over the label set (e.g. 100*cos(img,text))
    Rounds: label = argmax(logits); fit weighted ridge in the PCA space (inter-class
    weights only); logits <- ridge prediction.  After n_rounds, final labels get the
    full bi-level weights.
    """
    X = _as2d(X, "X")
    L = _as2d(zs_logits, "zs_logits")
    if X.shape[0] != L.shape[0]:
        raise ValueError("X and zs_logits disagree on n")
    rng = np.random.default_rng(cfg.seed) if rng is None else rng
    n, K = L.shape

    basis = PCABasis.fit(X, **_rule_kwargs(cfg))
    if cfg.pca_k_floor == "classes":
        # C classes span a (C-1)-dim subspace of centred features: truncating below that cannot
        # represent the label space at all (observed: energy-80% kept k=4 for 9 classes).
        basis = basis.with_k(max(basis.k, len(np.unique(L.argmax(axis=1))) - 1))
    elif cfg.pca_k_floor != "none":
        raise ValueError(f"unknown pca_k_floor {cfg.pca_k_floor!r}")
    Z = basis.project(X)

    if cfg.refine_mode not in ("full", "to_bg", "agree"):
        raise ValueError(f"unknown refine_mode {cfg.refine_mode!r}")
    if cfg.refine_mode == "to_bg" and bg_col is None:
        raise ValueError("refine_mode='to_bg' needs bg_col (the background column of zs_logits)")
    if not 0.0 <= cfg.anchor <= 1.0:
        raise ValueError("anchor must be in [0, 1]")
    zs_labels = L.argmax(axis=1)
    zs_prob = softmax(L, axis=1) if cfg.anchor > 0 else None
    if gate is not None:
        gate = np.asarray(gate, dtype=bool)
        if gate.shape != (n,):
            raise ValueError("gate must be a boolean mask of shape (n,)")

    logits = L
    history = []
    for r in range(cfg.n_rounds + 1):
        labels = logits.argmax(axis=1)
        if r > 0:                                                  # round 0 IS the zero-shot labelling: nothing to restrict
            if gate is not None:
                labels = np.where(gate, labels, zs_labels)
            if cfg.refine_mode == "to_bg":
                labels = np.where(labels == bg_col, labels, zs_labels)
        history.append(labels.copy())
        ent = prediction_entropy(logits)
        if r == cfg.n_rounds:
            break
        w_ref = bilevel_weights(labels, ent, K, rng, intra=False, inter=cfg.use_inter,
                                base_weights=base_weights)
        acc = RidgeAccumulator(basis.k, cfg.lam)
        acc.add_classes(K)
        acc.update(Z, labels, w_ref)
        logits = Z @ acc.solve()
        if cfg.anchor > 0:
            logits = (1.0 - cfg.anchor) * logits + cfg.anchor * zs_prob
        if cfg.mask_empty_classes:
            empty = np.bincount(labels, minlength=K) == 0
            logits = logits.copy()
            logits[:, empty] = -np.inf
    weights = bilevel_weights(labels, ent, K, rng, intra=cfg.use_intra, inter=cfg.use_inter,
                              std=cfg.gau_std, wmin=cfg.gau_min, base_weights=base_weights)
    if cfg.refine_mode == "agree":
        weights = np.where(labels == zs_labels, weights, 0.0)
    return N2LResult(labels, weights, ent, basis.k, history, basis)


# --------------------------------------------------------------------------
# the incremental learner
# --------------------------------------------------------------------------
class N2LLearner:
    """Frozen features in -> analytic head out. No gradients anywhere.

    add_task(X, zs_logits, n_new_classes):
        zs_logits columns are RIGHT-ALIGNED to the head's columns, so
          * official classification: zs_logits has only the new task's classes
          * our detection design   : zs_logits has all seen classes (+ background)
        both work.
    Features are pushed through `feature_map` and accumulated in chunks of `chunk` rows, so
    memory stays O(chunk * feat_dim) however many boxes a task has.
    """

    def __init__(self, in_dim: int, cfg: Optional[N2LConfig] = None,
                 feature_map: Optional[Callable] = None, feat_dim: Optional[int] = None,
                 chunk: int = 20000):
        self.cfg = cfg or N2LConfig()
        self.feature_map = feature_map
        self.acc = RidgeAccumulator(feat_dim or in_dim, self.cfg.lam)
        self.in_dim = in_dim
        self.chunk = int(chunk)
        self.W: Optional[np.ndarray] = None
        self.results: list = []

    @property
    def num_classes(self) -> int:
        return self.acc.num_classes

    def _phi(self, X):
        X = _as2d(X, "X")
        return X if self.feature_map is None else self.feature_map(X)

    def _accumulate(self, X, labels, weights) -> None:
        n = len(X)
        for i in range(0, n, self.chunk):
            sl = slice(i, i + self.chunk)
            self.acc.update(self._phi(X[sl]), labels[sl], weights[sl])

    def add_task(self, X, zs_logits, n_new_classes: int, rng=None,
                 base_weights=None, gate=None, bg_col=None) -> N2LResult:
        L = _as2d(zs_logits, "zs_logits")
        if self.cfg.memory not in ("full", "none", "none_newonly"):
            raise ValueError(f"unknown memory {self.cfg.memory!r}")
        later_task = len(self.results) > 0
        if later_task and self.cfg.memory != "full":                  # forget everything learned so far (ablation)
            self.acc.A[:] = 0.0
            self.acc.C[:] = 0.0
            self.acc.n_samples = 0
        self.acc.add_classes(n_new_classes)
        if L.shape[1] > self.acc.num_classes:
            raise ValueError("zs_logits has more columns than the head has classes")
        if later_task and self.cfg.memory == "none_newonly":            # old classes cannot be pseudo-labels any more
            L = L.copy()
            start = 1 if bg_col == 0 else 0
            stop = L.shape[1] - n_new_classes
            if stop > start:
                L[:, start:stop] = -1e9
        res = refine_task_labels(X, L, self.cfg, rng=rng, base_weights=base_weights, gate=gate, bg_col=bg_col)
        offset = self.acc.num_classes - L.shape[1]
        targets = res.labels + offset
        if self.cfg.soft > 0:
            if not 0.0 <= self.cfg.soft <= 1.0:
                raise ValueError("soft must be in [0, 1]")
            Y = np.zeros((len(L), self.acc.num_classes))
            Y[np.arange(len(L)), targets] = 1.0 - self.cfg.soft
            Y[:, offset:] += self.cfg.soft * softmax(L, axis=1)
            targets = Y
        self._accumulate(_as2d(X, "X"), targets, res.weights)
        self.W = self.acc.solve()
        self.results.append(res)
        return res

    def add_labeled(self, X, labels, n_new_classes: int, weights=None) -> None:
        """Supervised variant (GT-label ceiling): accumulate given GLOBAL column labels, no
        refinement, no pseudo-labelling."""
        X = _as2d(X, "X")
        labels = np.asarray(labels, dtype=np.int64)
        w = np.ones(len(X)) if weights is None else np.asarray(weights, dtype=np.float64)
        self.acc.add_classes(n_new_classes)
        self._accumulate(X, labels, w)
        self.W = self.acc.solve()

    def scores(self, X) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("no task added yet")
        X = _as2d(X, "X")
        return np.concatenate([self._phi(X[i:i + self.chunk]) @ self.W
                               for i in range(0, len(X), self.chunk)]) if len(X) else \
            np.zeros((0, self.W.shape[1]))


def fuse_scores(zs_logits, ridge_scores, fusion_weight: float = 0.8) -> np.ndarray:
    """Official fusion: (1-w)*softmax(zero-shot logits) + w*ridge score.
    zs_logits are the scaled (100*cos) logits over the SAME classes as ridge_scores."""
    zs = np.asarray(zs_logits, dtype=np.float64)
    rs = np.asarray(ridge_scores, dtype=np.float64)
    if zs.shape != rs.shape:
        raise ValueError(f"shape mismatch {zs.shape} vs {rs.shape}")
    return (1.0 - fusion_weight) * softmax(zs, axis=1) + fusion_weight * rs

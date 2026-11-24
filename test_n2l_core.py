"""Step-1 validation of n2l_core.py.  Run:  python -m pytest -q test_n2l_core.py

Every test targets a specific claim the N2L-Det design relies on.
"""
import numpy as np
import pytest
from scipy.special import softmax

import n2l_core as nc
from n2l_core import (N2LConfig, N2LLearner, PCABasis, RandomReLUExpansion,
                      RidgeAccumulator, bilevel_weights, choose_k, fuse_scores,
                      official_k, prediction_entropy, refine_task_labels)


# ---------------------------------------------------------------- synthetic data
def make_blobs(n_per_class, classes, d=64, sep=3.0, seed=0, class_means=None):
    """Gaussian blobs; class means shared across calls via class_means."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in classes:
        X.append(class_means[c] + rng.normal(size=(n_per_class, d)))
        y.append(np.full(n_per_class, c))
    return np.vstack(X), np.concatenate(y)


def class_means(K, d=64, sep=3.0, seed=123):
    return np.random.default_rng(seed).normal(scale=sep, size=(K, d))


def noisy_zs_logits(y_local, K, p_noise, rng, strength=6.0):
    """Simulated CLIP zero-shot logits whose argmax is wrong with prob p_noise
    (random wrong class, independent of the features)."""
    n = len(y_local)
    pred = y_local.copy()
    flip = rng.random(n) < p_noise
    pred[flip] = (y_local[flip] + rng.integers(1, K, size=flip.sum())) % K
    L = rng.normal(size=(n, K))
    L[np.arange(n), pred] += strength
    return L, pred


# ---------------------------------------------------------------- RidgeAccumulator
def test_solve_matches_explicit_inverse():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(200, 12)); y = rng.integers(0, 4, 200); w = rng.uniform(0.2, 2, 200)
    acc = RidgeAccumulator(12, lam=0.1); acc.add_classes(4); acc.update(F, y, w)
    Y = np.eye(4)[y]
    W_ref = np.linalg.inv(F.T @ (w[:, None] * F) + 0.1 * np.eye(12)) @ (F.T @ (w[:, None] * Y))
    np.testing.assert_allclose(acc.solve(), W_ref, rtol=1e-9, atol=1e-10)


def test_recursive_equals_joint_with_new_class_padding():
    """CORE CLAIM: task-by-task accumulation == fitting all data jointly."""
    rng = np.random.default_rng(1)
    d, tasks = 20, [(0, 3), (3, 7), (7, 12)]            # class ranges
    data = []
    for lo, hi in tasks:
        n = 150
        F = rng.normal(size=(n, d)) + rng.normal(size=(1, d))
        # labels mostly new classes, but ~20% old-class objects in new images
        y = rng.integers(lo, hi, n)
        old = rng.random(n) < 0.2
        if lo > 0:
            y[old] = rng.integers(0, lo, old.sum())
        data.append((F, y, rng.uniform(0.25, 2.0, n)))

    inc = RidgeAccumulator(d, 0.1)
    for (lo, hi), (F, y, w) in zip(tasks, data):
        inc.add_classes(hi - lo)
        inc.update(F, y, w)
    W_inc = inc.solve()

    joint = RidgeAccumulator(d, 0.1); joint.add_classes(12)
    joint.update(np.vstack([t[0] for t in data]), np.concatenate([t[1] for t in data]),
                 np.concatenate([t[2] for t in data]))
    np.testing.assert_allclose(W_inc, joint.solve(), rtol=1e-9, atol=1e-10)
    assert W_inc.shape == (d, 12)


def test_order_and_batch_split_invariance():
    rng = np.random.default_rng(2)
    F = rng.normal(size=(300, 10)); y = rng.integers(0, 5, 300); w = rng.uniform(0.3, 2, 300)
    a = RidgeAccumulator(10); a.add_classes(5); a.update(F, y, w)
    perm = rng.permutation(300)
    b = RidgeAccumulator(10); b.add_classes(5)
    for chunk in np.array_split(perm, 7):
        b.update(F[chunk], y[chunk], w[chunk])
    np.testing.assert_allclose(a.solve(), b.solve(), rtol=1e-9, atol=1e-10)


def test_new_columns_are_zero_and_old_C_untouched_when_no_old_labels():
    rng = np.random.default_rng(3)
    acc = RidgeAccumulator(8); acc.add_classes(3)
    acc.update(rng.normal(size=(50, 8)), rng.integers(0, 3, 50))
    C_before = acc.C.copy()
    acc.add_classes(2)
    assert acc.num_classes == 5
    np.testing.assert_array_equal(acc.C[:, :3], C_before)
    assert np.all(acc.C[:, 3:] == 0)
    acc.update(rng.normal(size=(50, 8)), rng.integers(3, 5, 50))     # only new labels
    np.testing.assert_array_equal(acc.C[:, :3], C_before)            # old C bit-identical
    assert np.any(acc.C[:, 3:] != 0)


def test_soft_target_matrix_equals_label_path():
    rng = np.random.default_rng(4)
    F = rng.normal(size=(80, 6)); y = rng.integers(0, 3, 80); w = rng.uniform(0.5, 1.5, 80)
    a = RidgeAccumulator(6); a.add_classes(3); a.update(F, y, w)
    b = RidgeAccumulator(6); b.add_classes(3); b.update(F, np.eye(3)[y], w)
    np.testing.assert_allclose(a.solve(), b.solve(), atol=1e-12)


def test_zero_weight_rows_are_inert():
    rng = np.random.default_rng(5)
    F = rng.normal(size=(60, 5)); y = rng.integers(0, 2, 60)
    a = RidgeAccumulator(5); a.add_classes(2); a.update(F[:40], y[:40])
    w = np.r_[np.ones(40), np.zeros(20)]
    b = RidgeAccumulator(5); b.add_classes(2); b.update(F, y, w)
    np.testing.assert_allclose(a.solve(), b.solve(), atol=1e-12)


def test_input_validation():
    acc = RidgeAccumulator(4); acc.add_classes(2)
    good = np.ones((3, 4))
    with pytest.raises(ValueError): acc.update(good * np.nan, [0, 1, 0])
    with pytest.raises(ValueError): acc.update(good, [0, 1, 2])            # label >= K
    with pytest.raises(ValueError): acc.update(good, [0, -1, 0])
    with pytest.raises(ValueError): acc.update(good, [0, 1, 0], [1, -1, 1])  # negative weight
    with pytest.raises(ValueError): acc.update(np.ones((3, 5)), [0, 1, 0])   # wrong dim
    with pytest.raises(ValueError): acc.update(good, np.ones((3, 5)))        # Y wrong width
    with pytest.raises(ValueError): RidgeAccumulator(4, lam=0)
    # a failed update must not have corrupted state (A, C untouched, counter unchanged)
    assert acc.n_samples == 0
    assert np.all(acc.A == 0) and np.all(acc.C == 0)


# ---------------------------------------------------------------- PCA / choosing k
def test_official_k_reproduces_the_official_quirk():
    assert official_k([20, 15, 12, 9, 8], 10) == 4      # 3 above + 1 extra
    assert official_k([20, 15], 10) == 2                # none below -> all
    assert official_k([5, 4], 10) == 1                  # first already below -> 1
    # our clean rule differs by exactly the extra component
    assert choose_k([20, 15, 12, 9, 8], "abs", theta=10) == 3


def test_energy_rule():
    sv = np.sqrt([400, 225, 144, 81, 64])               # cum frac .438 .684 .841 .930 1
    assert choose_k(sv, "energy", rho=0.8) == 3
    assert choose_k(sv, "energy", rho=0.95) == 5
    assert choose_k(sv, "energy", rho=0.01) == 1
    assert choose_k(np.zeros(4), "energy") == 1


def test_scaled_rule_is_invariant_to_sample_count_but_abs_is_not():
    """The official absolute threshold changes k when you just have more data;
    the scaled rule does not. This is the detection (n ~ 1e5+) motivation."""
    rng = np.random.default_rng(6)
    n, d = 400, 12
    scales = np.array([10, 8, 6, 4, 2, 1, .5, .5, .5, .5, .5, .5])
    X = rng.normal(size=(n, d)) * scales
    X4 = np.vstack([X] * 4)                              # same distribution, 4x samples, sigma x2
    k_abs_small = PCABasis.fit(X, rule="abs", theta=100).k
    k_abs_big = PCABasis.fit(X4, rule="abs", theta=100).k
    k_sc_big = PCABasis.fit(X4, rule="scaled", theta=100, n_ref=n).k
    assert k_abs_small == 3
    assert k_abs_big == 4                                # drifts with n
    assert k_sc_big == k_abs_small                       # does not


def test_moments_pca_equals_svd_pca():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(500, 16)) * np.linspace(10, 1, 16) + rng.normal(size=(1, 16)) * 5
    a = PCABasis.fit(X, method="svd", rule="abs", theta=30)
    b = PCABasis.fit(X, method="moments", rule="abs", theta=30)
    assert a.k == b.k
    np.testing.assert_allclose(a.singular_values, b.singular_values, rtol=1e-7)
    np.testing.assert_allclose(a.mean, b.mean, atol=1e-12)
    # components agree up to sign
    cos = np.abs(np.sum(a.components * b.components, axis=0))
    np.testing.assert_allclose(cos, 1.0, atol=1e-6)


def test_projection_is_centred_and_orthonormal():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(300, 10)) + 7.0
    b = PCABasis.fit(X, rule="abs", theta=1)
    Z = b.project(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(b.components.T @ b.components, np.eye(b.k), atol=1e-10)


# ---------------------------------------------------------------- weights
def test_entropy_is_safe_and_correct():
    e = prediction_entropy(np.array([[1000.0, 0, 0], [0, 0, 0]]))
    assert np.all(np.isfinite(e))
    np.testing.assert_allclose(e, [0.0, np.log(3)], atol=1e-9)


def test_intra_gaussian_weights_positive_mean_one_clamped():
    rng = np.random.default_rng(9)
    w = nc.intra_class_gaussian_weights(100000, rng)
    assert np.all(w > 0)
    assert abs(w.mean() - 1) < 1e-12
    # clamp [0.25, 1.75] then rescale by ~1 => range stays close
    assert w.min() >= 0.25 / 1.75 and w.max() <= 1.75 / 0.25
    assert 0.2 < w.min() < 0.3 and 1.7 < w.max() < 1.8


def test_bilevel_weights_inter_class_totals_and_intra_ranking():
    rng = np.random.default_rng(10)
    labels = np.r_[np.zeros(60, int), np.ones(30, int), np.full(10, 3)]     # class 2 absent
    ent = rng.random(100)
    m = bilevel_weights(labels, ent, 5, rng)
    C_present = 3
    for c in (0, 1, 3):                                   # every present class sums to n/C
        np.testing.assert_allclose(m[labels == c].sum(), 100 / C_present, rtol=1e-9)
    assert np.all(m > 0)
    for c in (0, 1, 3):                                   # lower entropy => weight not smaller
        idx = np.flatnonzero(labels == c)
        order = idx[np.argsort(ent[idx])]
        assert np.all(np.diff(m[order]) <= 1e-12)
    # switches
    m_inter = bilevel_weights(labels, ent, 5, rng, intra=False)
    np.testing.assert_allclose(m_inter[labels == 0], 100 / (60 * 3))
    m_intra = bilevel_weights(labels, ent, 5, rng, inter=False)
    np.testing.assert_allclose(m_intra[labels == 1].mean(), 1.0)


def test_base_weights_multiply_and_validate():
    rng = np.random.default_rng(11)
    labels = rng.integers(0, 3, 50); ent = rng.random(50)
    b = rng.uniform(0, 1, 50)
    m0 = bilevel_weights(labels, ent, 3, np.random.default_rng(0))
    m1 = bilevel_weights(labels, ent, 3, np.random.default_rng(0), base_weights=b)
    np.testing.assert_allclose(m1, m0 * b)
    with pytest.raises(ValueError):
        bilevel_weights(labels, ent, 3, rng, base_weights=-b)


# ---------------------------------------------------------------- label refinement
def _noisy_task(p_noise, K=10, n_per=300, seed=0):
    means = class_means(K, seed=123)
    X, y = make_blobs(n_per, range(K), seed=seed, class_means=means)
    rng = np.random.default_rng(seed + 1)
    L, pred0 = noisy_zs_logits(y, K, p_noise, rng)
    return X, y, L, pred0


def test_refinement_removes_random_label_noise():
    """CLAIM: random (feature-independent) label noise is voted away by the
    low-rank ridge; the labels get closer to the truth."""
    X, y, L, pred0 = _noisy_task(0.30)
    n = len(y)
    cfg = N2LConfig(pca_rule="abs", pca_theta=float(np.sqrt(3 * n)))      # keeps ~K-1 class dims
    res = refine_task_labels(X, L, cfg)
    acc0, acc3 = (pred0 == y).mean(), (res.labels == y).mean()
    assert 0.65 < acc0 < 0.75                                              # sanity of the setup
    assert acc3 > 0.95, (acc0, acc3)
    assert 5 <= res.k <= 20
    assert len(res.label_history) == cfg.n_rounds + 1
    np.testing.assert_array_equal(res.label_history[0], L.argmax(1))


def test_without_low_rank_truncation_noise_is_not_removed():
    """Ablation sanity: if we keep ALL directions (k=d) the ridge can memorise noise
    far more, so refinement helps much less. Shows the truncation is what does the work."""
    X, y, L, pred0 = _noisy_task(0.30, n_per=5, seed=3)         # n=50 < d=64 => interpolation possible
    n = len(y)
    low = refine_task_labels(X, L, N2LConfig(pca_rule="abs", pca_theta=float(np.sqrt(3 * n)), seed=1))
    full = refine_task_labels(X, L, N2LConfig(pca_rule="energy", pca_rho=1.0, lam=1e-9, seed=1))
    # full-rank, ~no ridge: the fit reproduces the noisy labels it was given
    assert (full.labels == pred0).mean() > 0.9
    assert (low.labels == y).mean() > (full.labels == y).mean() + 0.1


def test_systematic_errors_survive_refinement():
    """Honest limit from the 'cons': a consistent confusion (every sample of class 0
    labelled 1) is NOT fixed by refinement."""
    K = 4
    means = class_means(K, seed=5)
    X, y = make_blobs(300, range(K), seed=1, class_means=means)
    rng = np.random.default_rng(0)
    pred = y.copy(); pred[y == 0] = 1                         # systematic confusion
    L = rng.normal(size=(len(y), K)); L[np.arange(len(y)), pred] += 6
    cfg = N2LConfig(pca_rule="abs", pca_theta=float(np.sqrt(3 * len(y))))
    res = refine_task_labels(X, L, cfg)
    assert (res.labels[y == 0] == 1).mean() > 0.9


def test_empty_class_mask_property():
    """With mask_empty_classes, a class with no samples at round r cannot appear at r+1."""
    rng = np.random.default_rng(12)
    for seed in range(30):
        X = np.random.default_rng(seed).normal(size=(120, 16))
        L = np.random.default_rng(seed + 100).normal(size=(120, 6))
        L[:, 5] -= 50                                          # class 5 never wins at round 0
        res = refine_task_labels(X, L, N2LConfig(pca_rule="energy", pca_rho=0.9, seed=seed))
        for a, b in zip(res.label_history[:-1], res.label_history[1:]):
            assert set(np.unique(b)) <= set(np.unique(a))


def test_empty_class_mask_blocks_a_real_failure_case():
    """Zero-weight samples (e.g. low-objectness boxes) make the bias-free linear fit
    extrapolate to NEGATIVE scores on both populated classes there; an empty class (score
    exactly 0) would then win the argmax. The mask must prevent that."""
    rng = np.random.default_rng(0)
    n = 600
    X = rng.normal(size=(n, 4)); X[:, 0] *= 3
    lab = (X[:, 1] > 0).astype(int)
    L = np.full((n, 3), -50.0); L[np.arange(n), lab] = 5      # class 2 is empty at round 0
    bw = (X[:, 0] > 0).astype(float)                           # half the samples get weight 0
    kw = dict(pca_rule="energy", pca_rho=0.999, n_rounds=1)
    off = refine_task_labels(X, L, N2LConfig(mask_empty_classes=False, **kw), base_weights=bw)
    on = refine_task_labels(X, L, N2LConfig(mask_empty_classes=True, **kw), base_weights=bw)
    assert (off.label_history[1] == 2).sum() > 50              # the failure exists...
    assert (on.label_history[1] == 2).sum() == 0               # ...and the mask removes it


def test_refinement_fit_uses_inter_weights_only_and_final_fit_uses_both(monkeypatch):
    """Wiring check against the official code: the 3 refinement ridge fits use only the
    inter-class balance; the Gaussian (intra) weights appear only in the final weights."""
    calls = []
    real = nc.bilevel_weights

    def spy(labels, entropy, num_classes, rng, intra=True, inter=True, **kw):
        calls.append((intra, inter))
        return real(labels, entropy, num_classes, rng, intra=intra, inter=inter, **kw)

    monkeypatch.setattr(nc, "bilevel_weights", spy)
    X, y, L, _ = _noisy_task(0.2, K=4, n_per=60, seed=2)
    refine_task_labels(X, L, N2LConfig(pca_rule="energy", pca_rho=0.8, n_rounds=3))
    assert calls == [(False, True)] * 3 + [(True, True)]


def test_refinement_is_deterministic_for_a_seed():
    X, y, L, _ = _noisy_task(0.2, K=5, n_per=100, seed=4)
    cfg = N2LConfig(pca_rule="energy", pca_rho=0.8, seed=7)
    a, b = refine_task_labels(X, L, cfg), refine_task_labels(X, L, cfg)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.weights, b.weights)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        refine_task_labels(np.ones((5, 3)), np.ones((6, 2)), N2LConfig())


# ---------------------------------------------------------------- full incremental learner
def _incremental_run(feature_map=None, feat_dim=None, full_width_logits=False, seed=0):
    d, per_task, T = 64, 5, 3
    K = per_task * T
    means = class_means(K, seed=321)
    cfg = N2LConfig(pca_rule="abs", pca_theta=float(np.sqrt(3 * 1500)), seed=seed)
    L = N2LLearner(d, cfg, feature_map=feature_map, feat_dim=feat_dim)
    rng = np.random.default_rng(seed)
    record = []
    for t in range(T):
        cls = range(t * per_task, (t + 1) * per_task)
        X, y = make_blobs(300, cls, seed=10 + t, class_means=means)
        if full_width_logits:                                  # our detection design
            width = (t + 1) * per_task
            if t > 0:                                          # new images also contain OLD-class objects
                Xo, yo = make_blobs(60, range(0, t * per_task), seed=50 + t, class_means=means)
                X, y = np.vstack([X, Xo]), np.concatenate([y, yo])
            Lg, _ = noisy_zs_logits(y, width, 0.25, rng)
        else:                                                  # official: current classes only
            Lg, _ = noisy_zs_logits(y - t * per_task, per_task, 0.25, rng)
        res = L.add_task(X, Lg, per_task, rng=np.random.default_rng(seed + t))
        record.append((X, res, Lg.shape[1]))
    Xt, yt = make_blobs(100, range(K), seed=99, class_means=means)
    return L, record, Xt, yt


def test_incremental_learner_end_to_end_accuracy():
    L, record, Xt, yt = _incremental_run()
    pred = L.scores(Xt).argmax(1)
    assert L.num_classes == 15 and L.W.shape == (64, 15)
    assert (pred == yt).mean() > 0.90
    # old classes still recognised after the last task (no forgetting)
    old = yt < 5
    assert (pred[old] == yt[old]).mean() > 0.85


@pytest.mark.parametrize("full_width", [False, True])
def test_learner_equals_joint_fit_on_the_recorded_labels_and_weights(full_width):
    """Wiring check: offsets / padding / right-alignment are correct, so the
    learner's W is bit-for-bit what one joint ridge on all recorded
    (feature, label, weight) triples gives."""
    L, record, _, _ = _incremental_run(full_width_logits=full_width)
    K = L.num_classes
    joint = RidgeAccumulator(64, L.cfg.lam); joint.add_classes(K)
    # label columns are right-aligned against the classes known *after* that task
    for t, (X, res, width) in enumerate(record):
        offset = 5 * (t + 1) - width
        joint.update(X, res.labels + offset, res.weights)
    np.testing.assert_allclose(L.W, joint.solve(), rtol=1e-9, atol=1e-10)


def test_full_width_labels_can_land_on_old_classes():
    """Detection design: new images may contain old-class objects."""
    L, record, _, _ = _incremental_run(full_width_logits=True)
    X, res, width = record[2]
    assert width == 15
    # rows 0..1499 are new-class objects; rows 1500..2099 are true old-class objects (10 x 60)
    old_rows = slice(1500, 2100)
    assert (res.labels[old_rows] < 10).mean() > 0.9        # labelled with OLD names, in OLD columns
    assert (res.labels[:1500] >= 10).mean() > 0.9          # genuinely new objects get new names


def test_random_relu_expansion_deterministic_nonneg_and_usable():
    f1, f2 = RandomReLUExpansion(64, 300, seed=1), RandomReLUExpansion(64, 300, seed=1)
    x = np.random.default_rng(0).normal(size=(5, 64))
    np.testing.assert_array_equal(f1(x), f2(x))
    assert f1(x).shape == (5, 300) and f1(x).min() >= 0
    L, _, Xt, yt = _incremental_run(feature_map=f1, feat_dim=300)
    assert L.W.shape == (300, 15)
    assert (L.scores(Xt).argmax(1) == yt).mean() > 0.85


def test_fuse_scores():
    zs = np.array([[10.0, 0.0, 0.0]]); rs = np.array([[0.2, 0.9, 0.1]])
    f = fuse_scores(zs, rs, 0.8)
    np.testing.assert_allclose(f, 0.2 * softmax(zs, axis=1) + 0.8 * rs)
    assert fuse_scores(zs, rs, 0.0).argmax() == 0 and fuse_scores(zs, rs, 1.0).argmax() == 1
    with pytest.raises(ValueError):
        fuse_scores(zs, rs[:, :2])


def test_learner_guards():
    L = N2LLearner(8)
    with pytest.raises(RuntimeError):
        L.scores(np.ones((2, 8)))
    with pytest.raises(ValueError):                       # more logit columns than classes
        L.add_task(np.random.default_rng(0).normal(size=(30, 8)), np.zeros((30, 5)), 3)


# ---------------------------------------------------------------- chunking / supervised variant
def test_chunked_accumulation_equals_unchunked():
    d, per = 32, 4
    means = class_means(per * 2, d=d, seed=5)
    cfg = N2LConfig(pca_rule="energy", pca_rho=0.8, seed=3)
    fmap = RandomReLUExpansion(d, 90, seed=2)
    out = []
    for chunk in (10**9, 77):
        L = N2LLearner(d, cfg, feature_map=fmap, feat_dim=90, chunk=chunk)
        for t in range(2):
            X, y = make_blobs(100, range(t * per, (t + 1) * per), d=d, seed=t, class_means=means)
            Lg, _ = noisy_zs_logits(y - t * per, per, 0.2, np.random.default_rng(t))
            L.add_task(X, Lg, per, rng=np.random.default_rng(t))
        out.append(L)
    np.testing.assert_allclose(out[0].W, out[1].W, rtol=1e-9, atol=1e-10)
    Xt, _ = make_blobs(10, range(8), d=d, seed=9, class_means=means)
    np.testing.assert_allclose(out[0].scores(Xt), out[1].scores(Xt), rtol=1e-9, atol=1e-10)
    assert out[1].scores(np.zeros((0, d))).shape == (0, 8)


def test_add_labeled_equals_joint_and_validates():
    rng = np.random.default_rng(0)
    d = 12
    L = N2LLearner(d, N2LConfig(lam=0.1), chunk=50)
    Xs, ys, ws = [], [], []
    for lo, hi in ((0, 3), (3, 5)):
        X = rng.normal(size=(120, d)); y = rng.integers(0, hi, 120); w = rng.uniform(0.3, 2, 120)
        L.add_labeled(X, y, hi - lo, w)
        Xs.append(X); ys.append(y); ws.append(w)
    joint = RidgeAccumulator(d, 0.1); joint.add_classes(5)
    joint.update(np.vstack(Xs), np.concatenate(ys), np.concatenate(ws))
    np.testing.assert_allclose(L.W, joint.solve(), rtol=1e-9, atol=1e-10)
    with pytest.raises(ValueError):                       # label beyond the head
        L.add_labeled(rng.normal(size=(3, d)), [0, 1, 9], 0)


# ---------------------------------------------------------------- k floor (found by the step-5 runner)
def test_pcabasis_with_k_and_components_view():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 10)) * np.linspace(10, 1, 10)
    b = PCABasis.fit(X, rule="abs", theta=60)
    assert b.components_all.shape == (10, 10) and b.components.shape == (10, b.k)
    k0 = b.k
    b2 = b.with_k(k0 + 1)
    assert b2.k == k0 + 1 and b2.components.shape == (10, k0 + 1) and b.k == k0      # original untouched
    np.testing.assert_array_equal(b2.components, b.components_all[:, :k0 + 1])
    assert b.with_k(0).k == 1 and b.with_k(99).k == 10                      # clamped


def test_k_floor_prevents_collapse_of_the_label_space():
    """Energy-based truncation can keep fewer dimensions than the label space needs; the floor
    k >= classes-1 restores them."""
    K, d = 12, 64
    means = class_means(K, d=d, sep=0.6, seed=7)
    X, y = make_blobs(150, range(K), d=d, seed=1, class_means=means)
    L, _ = noisy_zs_logits(y, K, 0.15, np.random.default_rng(0))
    # a dominant nuisance direction soaks up the energy budget
    X[:, 0] += np.random.default_rng(2).normal(scale=25, size=len(X))
    base = dict(pca_rule="energy", pca_rho=0.8, seed=0)
    plain = refine_task_labels(X, L, N2LConfig(**base))
    floor = refine_task_labels(X, L, N2LConfig(pca_k_floor="classes", **base))
    assert plain.k < K - 1                                                  # precondition: rule alone is too low
    assert floor.k >= K - 1
    assert (floor.labels == y).mean() > (plain.labels == y).mean() + 0.05
    with pytest.raises(ValueError):
        refine_task_labels(X, L, N2LConfig(pca_k_floor="bogus", **base))

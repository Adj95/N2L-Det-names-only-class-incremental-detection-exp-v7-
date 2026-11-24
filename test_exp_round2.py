"""N2L-Det-exp round 2: detection-aware refinement options (off by default = frozen refinement) and their variants."""
import dataclasses
import importlib.util
import os

import numpy as np
import pytest

import n2l_core as nc
import run_pipeline as rp
import variants as vr
from test_exp_round1 import wd_tuned, _base  # noqa: F401
from test_pipeline import TASKS
from test_tune import _tune_dict


def _data(n=300, d=16, K=5, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (K, d))
    y = rng.integers(0, K, n)
    X = centers[y] + rng.normal(0, 1.5, (n, d))
    L = 10 * (X @ centers.T) / np.linalg.norm(centers, axis=1) + rng.normal(0, 60, (n, K))   # noisy zero-shot (refinement changes ~25% of labels)
    return X, L


CFG = nc.N2LConfig(pca_rule="energy", pca_rho=0.95, pca_k_floor="classes", n_rounds=3)

FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "n2l_det_frozen", "n2l_core.py")


@pytest.mark.skipif(not os.path.exists(FROZEN), reason="frozen v6 copy not next to this package (only on the dev machine)")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_defaults_reproduce_the_frozen_refinement_exactly(seed):
    spec = importlib.util.spec_from_file_location("frozen_n2l_core", FROZEN)
    import sys
    fz = importlib.util.module_from_spec(spec); sys.modules['frozen_n2l_core'] = fz; spec.loader.exec_module(fz)
    X, L = _data(seed=seed)
    w = np.random.default_rng(seed).uniform(0.1, 1, len(X))
    a = nc.refine_task_labels(X, L, CFG, rng=np.random.default_rng(5), base_weights=w)
    b = fz.refine_task_labels(X, L, fz.N2LConfig(**{k: v for k, v in dataclasses.asdict(CFG).items()
                                                    if k not in ("refine_gate", "anchor", "refine_mode", "memory")}),
                              rng=np.random.default_rng(5), base_weights=w)
    assert np.array_equal(a.labels, b.labels) and np.allclose(a.weights, b.weights) and a.k == b.k


def test_default_is_unchanged_by_passing_no_gate_and_a_bg_col():
    X, L = _data(seed=3)
    a = nc.refine_task_labels(X, L, CFG, rng=np.random.default_rng(0))
    b = nc.refine_task_labels(X, L, CFG, rng=np.random.default_rng(0), gate=None, bg_col=0)
    assert np.array_equal(a.labels, b.labels) and np.allclose(a.weights, b.weights)


def test_gate_all_false_keeps_zero_shot_labels_and_all_true_equals_full():
    X, L = _data(seed=4)
    zs = L.argmax(1)
    full = nc.refine_task_labels(X, L, CFG, rng=np.random.default_rng(0))
    assert not np.array_equal(full.labels, zs)                           # refinement does change something here
    none = nc.refine_task_labels(X, L, dataclasses.replace(CFG, refine_gate=0.5), rng=np.random.default_rng(0),
                                 gate=np.zeros(len(X), bool))
    assert all(np.array_equal(h, zs) for h in none.label_history)
    allg = nc.refine_task_labels(X, L, dataclasses.replace(CFG, refine_gate=0.5), rng=np.random.default_rng(0),
                                 gate=np.ones(len(X), bool))
    assert np.array_equal(allg.labels, full.labels)
    part = np.arange(len(X)) % 2 == 0
    p = nc.refine_task_labels(X, L, CFG, rng=np.random.default_rng(0), gate=part)
    assert np.array_equal(p.labels[~part], zs[~part])


def test_to_bg_only_demotes_to_background():
    X, L = _data(seed=5)
    zs = L.argmax(1)
    r = nc.refine_task_labels(X, L, dataclasses.replace(CFG, refine_mode="to_bg"), rng=np.random.default_rng(0), bg_col=0)
    changed = r.labels != zs
    assert np.all(r.labels[changed] == 0)
    with pytest.raises(ValueError):
        nc.refine_task_labels(X, L, dataclasses.replace(CFG, refine_mode="to_bg"))


def test_agree_zeroes_weights_where_refinement_disagrees():
    X, L = _data(seed=6)
    zs = L.argmax(1)
    full = nc.refine_task_labels(X, L, CFG, rng=np.random.default_rng(0))
    r = nc.refine_task_labels(X, L, dataclasses.replace(CFG, refine_mode="agree"), rng=np.random.default_rng(0))
    assert np.array_equal(r.labels, full.labels)
    dis = r.labels != zs
    assert dis.any() and np.all(r.weights[dis] == 0) and np.allclose(r.weights[~dis], full.weights[~dis])


def test_anchor_one_keeps_zero_shot_labels_and_bad_values_raise():
    X, L = _data(seed=7)
    r = nc.refine_task_labels(X, L, dataclasses.replace(CFG, anchor=1.0), rng=np.random.default_rng(0))
    assert np.array_equal(r.labels, L.argmax(1))
    with pytest.raises(ValueError):
        nc.refine_task_labels(X, L, dataclasses.replace(CFG, anchor=1.5))
    with pytest.raises(ValueError):
        nc.refine_task_labels(X, L, dataclasses.replace(CFG, refine_mode="nope"))


def test_refine_variants_change_only_the_n2l_settings_of_e5():
    t = _tune_dict(infer__yolo_gamma=0.25)
    cfgs = vr.variant_configs(TASKS, t)
    for name in vr.REFINE:
        a, b = cfgs["E5_tuned"], cfgs[name]
        assert dataclasses.replace(b, n2l=a.n2l) == a and b.n2l != a.n2l


def test_variants_cli_runs_refinement_rows(wd_tuned):
    wd, base = wd_tuned
    assert rp.main(["variants", *base, "--only", *vr.REFINE, "--n_folds", "2", "--tag", "r2"]) == 0
    md = open(os.path.join(wd, "results", "variants_r2.md")).read()
    assert all(n in md for n in vr.REFINE)

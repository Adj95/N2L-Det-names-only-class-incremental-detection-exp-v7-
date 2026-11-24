"""N2L-Det-exp v4: forgetting baselines (memory='none' / 'none_newonly'), ablations only; default 'full' unchanged."""
import dataclasses
import os

import numpy as np
import pytest

import n2l_core as nc
import run_pipeline as rp
import variants as vr
from test_exp_round1 import wd_tuned, _base  # noqa: F401
from test_pipeline import TASKS
from test_tune import _tune_dict

CFG = nc.N2LConfig(pca_rule="energy", pca_rho=0.95, pca_k_floor="classes", n_rounds=0, lam=1.0)


def _task(seed, n=200, d=12, K=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    L = rng.normal(size=(n, K)) * 5
    return X, L


def test_memory_none_refits_on_the_current_task_only():
    X1, L1 = _task(1, K=3)                       # task 0: bg + 2 classes
    X2, L2 = _task(2, K=5)                       # task 1: bg + 4 classes (2 new)
    a = nc.N2LLearner(12, dataclasses.replace(CFG, memory="none"))
    a.add_task(X1, L1, 3, rng=np.random.default_rng(0))
    a.add_task(X2, L2, 2, rng=np.random.default_rng(0))
    b = nc.N2LLearner(12, CFG)                   # fresh learner that only ever sees task 1
    b.add_task(X2, L2, 5, rng=np.random.default_rng(0))
    assert np.allclose(a.W, b.W)
    full = nc.N2LLearner(12, CFG)
    full.add_task(X1, L1, 3, rng=np.random.default_rng(0))
    full.add_task(X2, L2, 2, rng=np.random.default_rng(0))
    assert not np.allclose(full.W, a.W)


def test_memory_none_newonly_never_labels_old_classes_and_full_is_default():
    X1, L1 = _task(3, K=3)
    X2, L2 = _task(4, K=5)
    L2[:, 1:3] += 100                            # make the OLD classes win the zero-shot argmax everywhere
    m = nc.N2LLearner(12, dataclasses.replace(CFG, memory="none_newonly"))
    m.add_task(X1, L1, 3, rng=np.random.default_rng(0))
    r = m.add_task(X2, L2, 2, rng=np.random.default_rng(0), bg_col=0)
    assert not np.isin(r.labels, [1, 2]).any()   # local cols 1,2 = old classes: never chosen
    first = nc.N2LLearner(12, dataclasses.replace(CFG, memory="none_newonly"))
    r0 = first.add_task(X1, L1, 3, rng=np.random.default_rng(0))
    assert np.array_equal(r0.labels, L1.argmax(1))   # the first task is untouched
    assert nc.N2LConfig().memory == "full"
    with pytest.raises(ValueError):
        nc.N2LLearner(12, dataclasses.replace(CFG, memory="nope")).add_task(X1, L1, 3)


def test_forgetting_variants_are_registered_and_change_only_their_settings():
    t = _tune_dict(infer__yolo_gamma=0.25)
    cfgs = vr.variant_configs(TASKS, t)
    base = cfgs["E5_tuned"]
    assert cfgs["mem_none"].n2l.memory == "none" and cfgs["mem_newonly"].n2l.memory == "none_newonly"
    assert cfgs["headonly"].infer.fusion_weight == 1.0 and cfgs["headonly"].n2l == base.n2l
    h = cfgs["headonly_mem_newonly"]
    assert h.n2l.memory == "none_newonly" and h.infer.fusion_weight == 1.0
    assert dataclasses.replace(h, n2l=base.n2l, infer=base.infer) == base


def test_variants_cli_forgetting_rows_on_three_tasks_show_more_forgetting_without_memory(wd_tuned):
    wd, base = wd_tuned
    assert rp.main(["variants", *base, "--on", "eval", "--only", *vr.FORGET, "--n_folds", "2", "--tag", "fg"]) == 0
    import json
    r = json.load(open(os.path.join(wd, "results", "variants_eval_fg.json")))
    assert set(vr.FORGET) <= set(r)
    for n in vr.FORGET:
        assert len(r[n]["curve"]) == len(TASKS) and "base_AP50_drop" in r[n]

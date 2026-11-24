"""Step-4 validation: pseudo-label construction and GT diagnostics. Run: python -m pytest -q test_step4.py"""
import inspect

import numpy as np
import pytest
from scipy.special import softmax

import pseudo as ps
from prompts import l2norm


def _world(seed=0):
    rng = np.random.default_rng(seed)
    names = l2norm(rng.normal(size=(3, 16)))
    bg = l2norm(rng.normal(size=(1, 16)))
    return rng, names, bg


def test_class_to_col():
    assert ps.class_to_col([5, 2, 9]) == {5: 1, 2: 2, 9: 3}


def test_label_logits_layout_and_scale():
    rng, names, bg = _world()
    f = rng.normal(size=(4, 16)) * 9
    cfg = ps.PseudoConfig()
    L = ps.build_label_logits(f, names, bg, cfg)
    assert L.shape == (4, 4)
    np.testing.assert_allclose(L[:, 0], 100 * l2norm(f) @ bg[0])                  # col 0 is background
    np.testing.assert_allclose(L[:, 2], 100 * l2norm(f) @ names[1])               # col 1+j is class j
    L2 = ps.build_label_logits(f, names, None, ps.PseudoConfig(use_bg=False))
    assert L2.shape == (4, 3)
    with pytest.raises(ValueError):
        ps.build_label_logits(f, names, None, cfg)


def test_ignore_band_hand_case():
    rng, names, bg = _world(1)
    # sample 0: exactly a class-1 text vector (confident class); sample 1: exactly bg; sample 2: midpoint of two
    f = np.vstack([names[1], bg[0], l2norm(names[0] + names[2])]) * 5.0
    cfg_all = ps.PseudoConfig(scale=30.0)
    keep, logits, w = ps.build_task_inputs(f, np.ones(3), names, bg, cfg_all)
    assert keep.all() and logits.shape == (3, 4) and w is None
    p = softmax(logits, axis=1)
    assert p.argmax(1)[0] == 2 and p.argmax(1)[1] == 0                             # class idx1 -> col 2 ; bg -> col 0
    top = p.max(1)
    thr = (top[2] + min(top[0], top[1])) / 2                                       # between the ambiguous and the confident
    keep, logits, _ = ps.build_task_inputs(f, np.ones(3), names, bg, ps.PseudoConfig(scale=30.0, tau_pos=thr, tau_bg=thr))
    assert keep.tolist() == [True, True, False] and logits.shape == (2, 4)         # ambiguous one REMOVED, rows aligned
    # tau_pos only affects classes, tau_bg only the background
    k2, *_ = ps.build_task_inputs(f, np.ones(3), names, bg, ps.PseudoConfig(scale=30.0, tau_pos=1.01, tau_bg=0.0))
    assert k2.tolist() == [False, True, False]
    k3, *_ = ps.build_task_inputs(f, np.ones(3), names, bg, ps.PseudoConfig(scale=30.0, tau_pos=0.0, tau_bg=1.01))
    assert k3.tolist() == [True, False, True]


def test_loc_weights_follow_yolo_score_and_align_with_kept_rows():
    rng, names, bg = _world(2)
    f = rng.normal(size=(5, 16))
    cfg = ps.PseudoConfig(use_loc=True, loc_pow=1.0, loc_floor=0.1)
    s = np.array([1.0, 0.5, 0.0, 2.0, -1.0])                                       # out-of-range values are clipped
    keep, logits, w = ps.build_task_inputs(f, s, names, bg, cfg)
    np.testing.assert_allclose(w, 0.1 + 0.9 * np.array([1, 0.5, 0, 1, 0]))
    assert w.min() >= 0.1 and len(w) == len(logits)
    # with dropping, weights must stay aligned to the kept rows (rows 1 and 3 are ambiguous midpoints)
    mid = lambda a, b: l2norm(names[a] + names[b])
    g = np.vstack([names[0], mid(0, 1), bg[0], mid(1, 2)]) * 3.0
    s2 = np.array([0.9, 0.8, 0.7, 0.6])
    probs = softmax(ps.build_label_logits(g, names, bg, ps.PseudoConfig(scale=30.0)), axis=1).max(1)
    thr = (max(probs[1], probs[3]) + min(probs[0], probs[2])) / 2
    cfg2 = ps.PseudoConfig(use_loc=True, loc_floor=0.0, tau_pos=thr, tau_bg=thr, scale=30.0)
    keep2, lg2, w2 = ps.build_task_inputs(g, s2, names, bg, cfg2)
    assert keep2.tolist() == [True, False, True, False]                           # precondition: rows really dropped
    np.testing.assert_allclose(w2, [0.9, 0.7])


def test_gt_targets_hand_cases():
    gt_b = np.array([[0, 0, 10, 10], [100, 100, 120, 120.0]])
    gt_c = np.array([7, 3])
    col = {7: 1, 3: 2}                                                              # class 3 -> col 2, class 7 -> col 1
    props = np.array([[0, 0, 10, 10],          # IoU 1.0 with gt0                  -> col 1
                      [100, 100, 120, 120],    # IoU 1.0 with gt1                  -> col 2
                      [50, 50, 60, 60],        # no overlap                        -> bg 0
                      [0, 0, 10, 15.0],        # IoU 10/15=.667 -> still positive  -> col 1
                      [0, 0, 10, 24.0],        # IoU 100/240=.417 -> ambiguous     -> -1
                      ])
    t, best = ps.gt_targets(props, gt_b, gt_c, col)
    assert t.tolist() == [1, 2, 0, 1, -1]
    np.testing.assert_allclose(best[[0, 1, 2]], [1, 1, 0])
    # a GT of a class that has not been learned yet is ambiguous, never background
    t2, _ = ps.gt_targets(props[:1], gt_b, gt_c, {3: 2})
    assert t2.tolist() == [-1]
    assert ps.gt_targets(props, np.zeros((0, 4)), np.zeros(0, int), col)[0].tolist() == [0] * 5
    assert ps.gt_targets(np.zeros((0, 4)), gt_b, gt_c, col)[0].size == 0


def test_label_quality_hand_numbers():
    true = np.array([0, 0, 0, 0, 1, 1, 2, 2, -1, -1])
    pred = np.array([0, 0, 1, 0, 1, 2, 2, 0, 1, 0])
    q = ps.label_quality(pred, true)
    assert q["n"] == 8                                                              # ambiguous (-1) excluded
    assert abs(q["acc"] - 5 / 8) < 1e-12                                            # correct: idx 0,1,3,4,6
    assert abs(q["bg_precision"] - 3 / 4) < 1e-12                                   # pred bg: idx 0,1,3,7 ; right: 0,1,3
    assert abs(q["bg_recall"] - 3 / 4) < 1e-12
    assert abs(q["fg_precision"] - 2 / 4) < 1e-12                                   # pred fg: idx 2,4,5,6 ; right class: 4,6
    assert abs(q["fg_recall"] - 2 / 4) < 1e-12                                      # true fg: 4,5,6,7
    assert abs(q["frac_bg_truth"] - 4 / 8) < 1e-12
    assert ps.label_quality(np.array([1]), np.array([-1])) == {"n": 0}


def test_training_inputs_cannot_see_ground_truth():
    """build_* functions must not accept anything that looks like GT."""
    for fn in (ps.build_label_logits, ps.build_task_inputs):
        params = " ".join(inspect.signature(fn).parameters)
        assert "gt" not in params and "truth" not in params and "target" not in params

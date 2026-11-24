"""Validate inference.py. Run: python -m pytest -q test_step6_infer.py"""
import numpy as np
import pytest
from scipy.special import softmax

import inference as inf
from boxes import iou_matrix

C = inf.InferConfig


def test_fuse_formula_and_shapes():
    zs = np.array([[2.0, 0.0, -1.0]]); r = np.array([[0.1, 0.8, 0.3]])
    f = inf.fuse(zs, r, C(fusion_weight=0.8))
    np.testing.assert_allclose(f, 0.2 * softmax(zs, axis=1) + 0.8 * r)
    np.testing.assert_allclose(inf.fuse(zs, None, C()), softmax(zs, axis=1))
    np.testing.assert_allclose(inf.fuse(zs, r, C(fusion_weight=0.0)), softmax(zs, axis=1))
    np.testing.assert_allclose(inf.fuse(zs, r, C(fusion_weight=1.0)), r)
    with pytest.raises(ValueError):
        inf.fuse(zs, r[:, :2], C())


def test_detect_hand_case_background_never_output_and_class_mapping():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60.0]])
    #            bg    c0    c1        (class_order maps columns 1,2 -> classes 40, 7)
    fused = np.array([[0.9, 0.05, 0.05],
                      [0.1, 0.80, 0.10],      # same place as box 0, same top class
                      [0.2, 0.10, 0.70]])
    b, s, c = inf.detect(boxes, np.ones(3), fused, [40, 7], C(topk_classes=1, nms_iou=0.5, min_score=0.0))
    assert c.tolist() == [40, 7]
    np.testing.assert_allclose(s, [0.8, 0.7])                       # box0 (0.05) suppressed by box1 (0.80)
    np.testing.assert_allclose(b, [[1, 1, 11, 11], [50, 50, 60, 60]])


def test_topk_and_class_wise_nms():
    boxes = np.array([[0, 0, 10, 10.0]])
    fused = np.array([[0.1, 0.5, 0.3, 0.1]])
    _, s, c = inf.detect(boxes, np.ones(1), fused, [1, 2, 3], C(topk_classes=2, min_score=0.0))
    assert c.tolist() == [1, 2]
    np.testing.assert_allclose(s, [0.5, 0.3])                         # same box, two classes: both kept
    _, s1, c1 = inf.detect(boxes, np.ones(1), fused, [1, 2, 3], C(topk_classes=1, min_score=0.0))
    assert c1.tolist() == [1]
    _, s9, c9 = inf.detect(boxes, np.ones(1), fused, [1, 2, 3], C(topk_classes=99, min_score=0.0))
    assert len(c9) == 3                                                              # k clipped to #classes


def test_min_score_max_dets_gamma_and_empty():
    boxes = np.array([[i * 20, 0, i * 20 + 10, 10.0] for i in range(5)])
    fused = np.array([[0.0, 0.1 * (i + 1)] for i in range(5)])
    _, s, _ = inf.detect(boxes, np.ones(5), fused, [3], C(min_score=0.25))
    np.testing.assert_allclose(s, [0.5, 0.4, 0.3])
    _, s, _ = inf.detect(boxes, np.ones(5), fused, [3], C(max_dets=2, min_score=0.0))
    np.testing.assert_allclose(s, [0.5, 0.4])
    ys = np.array([1.0, 1.0, 1.0, 1.0, 0.01])                                        # box 4 had the top class score
    _, s, c = inf.detect(boxes, ys, fused, [3], C(yolo_gamma=1.0, min_score=0.0, max_dets=1))
    assert s.tolist() == [pytest.approx(0.4)]                                        # 0.5*0.01 < 0.4*1
    for z in (inf.detect(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 2)), [1], C()),
              inf.detect(boxes, np.ones(5), np.zeros((5, 1)), [], C()),
              inf.detect(boxes, np.ones(5), np.zeros((5, 2)), [1], C(min_score=0.1))):
        assert z[0].shape == (0, 4) and len(z[1]) == 0 and len(z[2]) == 0
    with pytest.raises(ValueError):
        inf.detect(boxes, np.ones(5), fused, [3, 4], C())


def _reference_detect(boxes, fused, class_order, cfg):
    fg = fused[:, 1:]
    cands = []
    for i in range(len(boxes)):
        order = sorted(range(fg.shape[1]), key=lambda j: (-fg[i, j], j))[:cfg.topk_classes]
        for j in order:
            if fg[i, j] > cfg.min_score:
                cands.append((fg[i, j], i, class_order[j]))
    cands.sort(key=lambda t: (-t[0], t[1], t[2]))
    keep = []
    for s, i, c in cands:
        if all(not (c2 == c and iou_matrix(boxes[i:i+1], boxes[i2:i2+1])[0, 0] > cfg.nms_iou) for _, i2, c2 in keep):
            keep.append((s, i, c))
    return keep[:cfg.max_dets]


@pytest.mark.parametrize("seed", range(6))
def test_detect_matches_bruteforce_reference(seed):
    rng = np.random.default_rng(seed)
    n, K = 40, 5
    xy = rng.uniform(0, 60, size=(n, 2)); wh = rng.uniform(10, 40, size=(n, 2))
    boxes = np.hstack([xy, xy + wh])
    fused = rng.dirichlet(np.ones(K + 1) * 0.5, size=n)
    order = [10, 3, 77, 21, 5]
    cfg = C(topk_classes=3, min_score=0.05, nms_iou=0.5, max_dets=25)
    b, s, c = inf.detect(boxes, np.ones(n), fused, order, cfg)
    ref = _reference_detect(boxes, fused, order, cfg)
    assert len(s) == len(ref)
    np.testing.assert_allclose(s, [r[0] for r in ref])
    assert c.tolist() == [r[2] for r in ref]
    for bb, r in zip(b, ref):
        np.testing.assert_allclose(bb, boxes[r[1]])

"""Validate evaluate.py (the metric) against hand computations and an independent AP implementation.
Run: python -m pytest -q test_step6_eval.py"""
import numpy as np
import pytest

import coco_utils as cu
import evaluate as ev
from boxes import iou_matrix


def mk_coco(gt):
    """gt: {image_id: [(x1,y1,x2,y2,class_idx,iscrowd)]}"""
    imgs = [{"id": i, "file_name": f"{i}.jpg", "width": 400, "height": 400} for i in gt]
    anns = []
    for i, rows in gt.items():
        for (x1, y1, x2, y2, c, crowd) in rows:
            anns.append({"image_id": i, "category_id": cu.COCO_IDS[c], "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "iscrowd": crowd})
    return {"images": imgs, "annotations": anns}


def run(gt, dets, classes, image_ids=None):
    coco = mk_coco(gt)
    ids = list(gt) if image_ids is None else image_ids
    g = ev.make_coco_gt(coco, ids)
    d = []
    for i, rows in dets.items():
        if rows:
            b = np.array([r[:4] for r in rows], float); s = [r[4] for r in rows]; c = [r[5] for r in rows]
            d += ev.dets_to_coco(i, b, s, c)
    return ev.coco_eval(g, d, ids, classes)


def test_perfect_detections_give_ap_1():
    gt = {1: [(10, 10, 60, 60, 0, 0), (100, 100, 200, 150, 5, 0)], 2: [(0, 0, 50, 50, 5, 0)]}
    dets = {1: [(10, 10, 60, 60, .9, 0), (100, 100, 200, 150, .8, 5)], 2: [(0, 0, 50, 50, .7, 5)]}
    r = run(gt, dets, [0, 5])
    assert r["AP"] == pytest.approx(1.0) and r["AP50"] == pytest.approx(1.0) and r["AP75"] == pytest.approx(1.0)
    assert r["per_class_AP50"] == {0: pytest.approx(1.0), 5: pytest.approx(1.0)}


def test_ranking_matters_hand_values():
    gt = {1: [(0, 0, 50, 50, 3, 0)]}
    tp_first = {1: [(0, 0, 50, 50, .9, 3), (200, 200, 250, 250, .8, 3)]}
    fp_first = {1: [(0, 0, 50, 50, .8, 3), (200, 200, 250, 250, .9, 3)]}
    assert run(gt, tp_first, [3])["AP50"] == pytest.approx(1.0)
    assert run(gt, fp_first, [3])["AP50"] == pytest.approx(0.5)          # precision 1/2 at recall 1


def test_iou_threshold_behaviour():
    gt = {1: [(0, 0, 100, 100, 0, 0)]}
    dets = {1: [(0, 0, 100, 60, .9, 0)]}                                   # IoU 0.6
    r = run(gt, dets, [0])
    assert r["AP50"] == pytest.approx(1.0) and r["AP75"] == pytest.approx(0.0)
    assert r["per_class_AP50"][0] == pytest.approx(1.0)                    # per-class value is the IoU-0.5 one
    assert 0 < r["AP"] < 1


def test_class_subset_restricts_gt_and_detections():
    gt = {1: [(0, 0, 50, 50, 1, 0), (100, 100, 150, 150, 2, 0)]}
    dets = {1: [(0, 0, 50, 50, .9, 1), (100, 100, 150, 150, .9, 2), (300, 300, 350, 350, .99, 2)]}
    only1 = run(gt, dets, [1])
    assert only1["AP50"] == pytest.approx(1.0) and only1["n_gt"] == 1       # class-2 FP must not hurt class 1
    both = run(gt, dets, [1, 2])
    assert both["per_class_AP50"][1] == pytest.approx(1.0)
    assert both["per_class_AP50"][2] == pytest.approx(0.5) and both["AP50"] == pytest.approx(0.75)


def test_missing_gt_and_missing_dets():
    gt = {1: [(0, 0, 50, 50, 1, 0)]}
    r = run(gt, {1: []}, [1]); assert r["AP50"] == 0.0 and r["AP"] == 0.0
    r2 = run(gt, {1: [(0, 0, 50, 50, .9, 1)]}, [7])                         # no GT for class 7
    assert np.isnan(r2["AP50"]) and r2["n_gt"] == 0
    r3 = run(gt, {1: [(0, 0, 50, 50, .9, 1)]}, [1, 7])
    assert np.isnan(r3["per_class_AP50"][7]) and r3["per_class_AP50"][1] == pytest.approx(1.0)


def test_crowd_regions_are_not_counted_as_missed_objects():
    gt = {1: [(0, 0, 50, 50, 0, 0), (100, 100, 300, 300, 0, 1)]}            # crowd region
    dets = {1: [(0, 0, 50, 50, .9, 0), (120, 120, 160, 160, .8, 0)]}        # 2nd det lies inside the crowd
    r = run(gt, dets, [0])
    assert r["n_gt"] == 1 and r["AP50"] == pytest.approx(1.0)               # not penalised as a false positive


def test_detections_on_unselected_images_are_ignored_not_fatal():
    gt = {1: [(0, 0, 50, 50, 0, 0)], 2: [(0, 0, 50, 50, 0, 0)]}
    dets = {1: [(0, 0, 50, 50, .9, 0)], 2: [(300, 300, 350, 350, .99, 0)]}   # image 2 is not evaluated
    assert run(gt, dets, [0], image_ids=[1])["AP50"] == pytest.approx(1.0)


def test_gt_built_on_more_images_than_evaluated():
    """The GT object may cover more images than a given evaluation call (we reuse one GT across tasks)."""
    coco = mk_coco({1: [(0, 0, 50, 50, 0, 0)], 2: [(0, 0, 50, 50, 0, 0)]})
    g = ev.make_coco_gt(coco, [1, 2])
    d = ev.dets_to_coco(1, np.array([[0, 0, 50, 50.0]]), [0.9], [0])
    assert ev.coco_eval(g, d, [1], [0])["AP50"] == pytest.approx(1.0)
    assert ev.coco_eval(g, d, [1, 2], [0])["AP50"] == pytest.approx(51 / 101)


def test_only_selected_images_are_scored():
    gt = {1: [(0, 0, 50, 50, 0, 0)], 2: [(0, 0, 50, 50, 0, 0)]}
    dets = {1: [(0, 0, 50, 50, .9, 0)], 2: []}
    assert run(gt, dets, [0], image_ids=[1])["AP50"] == pytest.approx(1.0)
    assert run(gt, dets, [0], image_ids=[1, 2])["AP50"] == pytest.approx(51 / 101)   # recall caps at .5: 51 of 101 thresholds at precision 1


def _ref_ap50(gt_by_img, dets):
    """Independent single-class COCO-style AP@0.5: greedy match in score order, 101-pt interpolation."""
    n_gt = sum(len(v) for v in gt_by_img.values())
    order = sorted(dets, key=lambda d: -d[1])
    used = {i: np.zeros(len(v), bool) for i, v in gt_by_img.items()}
    tp = []
    for img, s, box in order:
        g = gt_by_img.get(img, np.zeros((0, 4)))
        ok = False
        if len(g):
            iou = iou_matrix(np.array([box]), g)[0]
            iou[used[img]] = -1
            j = int(iou.argmax())
            if iou[j] >= 0.5:
                used[img][j] = True; ok = True
        tp.append(ok)
    tp = np.array(tp, float); fp = 1 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    rec = ctp / n_gt; prec = ctp / (ctp + cfp)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        idx = np.searchsorted(rec, t, side="left")
        ap += prec[idx] if idx < len(prec) else 0.0
    return ap / 101


@pytest.mark.parametrize("seed", range(6))
def test_matches_independent_reference_on_random_data(seed):
    rng = np.random.default_rng(seed)
    gt, dets = {}, {}
    gt_by_img, ref_dets = {}, []
    for img in range(1, 9):
        n = rng.integers(1, 5)
        xy = rng.uniform(0, 250, size=(n, 2)); wh = rng.uniform(20, 120, size=(n, 2))
        boxes = np.hstack([xy, xy + wh])
        gt[img] = [(*b, 4, 0) for b in boxes]
        gt_by_img[img] = boxes
        rows = []
        for b in boxes:                                                    # noisy copies + junk
            for _ in range(rng.integers(0, 3)):
                jb = b + rng.normal(scale=8, size=4)
                if jb[2] > jb[0] + 2 and jb[3] > jb[1] + 2:
                    rows.append((*jb, float(rng.random()), 4))
        for _ in range(rng.integers(0, 4)):
            xy2 = rng.uniform(0, 300, size=2); wh2 = rng.uniform(10, 90, size=2)
            rows.append((*xy2, *(xy2 + wh2), float(rng.random()), 4))
        dets[img] = rows
        ref_dets += [(img, r[4], np.array(r[:4])) for r in rows]
    got = run(gt, dets, [4])["AP50"]
    assert got == pytest.approx(_ref_ap50(gt_by_img, ref_dets), abs=1e-9)


def test_dets_to_coco_format_and_category_mapping():
    d = ev.dets_to_coco(7, np.array([[10, 20, 40, 60.0]]), [0.5], [cu.ID2IDX[72]])
    assert d == [{"image_id": 7, "category_id": 72, "bbox": [10.0, 20.0, 30.0, 40.0], "score": 0.5}]
    assert ev.dets_to_coco(1, np.zeros((0, 4)), [], []) == []


def test_per_class_ap_is_labelled_by_class_even_when_class_idx_is_not_sorted_by_category_id():
    """Regression: COCOeval sorts catIds, so per-class AP used to be assigned by list position (labels shuffled for VOC order)."""
    import coco_utils as cu
    from pycocotools.coco import COCO
    ci = [58, 6]                                     # potted plant (id 64) listed before train (id 7)
    ids = [cu.COCO_IDS[c] for c in ci]
    gt = COCO()
    gt.dataset = {"images": [{"id": 1, "width": 100, "height": 100}, {"id": 2, "width": 100, "height": 100}],
                  "annotations": [{"id": 1, "image_id": 1, "category_id": ids[1], "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0},
                                  {"id": 2, "image_id": 2, "category_id": ids[0], "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0}],
                  "categories": [{"id": i, "name": str(i)} for i in ids]}
    gt.createIndex()
    dets = [{"image_id": 1, "category_id": ids[1], "bbox": [10, 10, 30, 30], "score": 0.9}]      # only the train is found
    r = ev.coco_eval(gt, dets, [1, 2], ci)
    assert r["per_class_AP50"][6] == pytest.approx(1.0) and r["per_class_AP50"][58] == pytest.approx(0.0)

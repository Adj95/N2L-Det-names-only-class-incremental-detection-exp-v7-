import json
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest

import coco_utils as cu
import diagnose as dg
import evaluate as ev
import run_pipeline as rp
from inference import InferConfig
import incremental as inc


def test_calibration_hand_computed():
    true = np.array([1, 2, 0, -1, 3, 3])
    pred = np.array([1, 1, 0, 9, 3, 2])
    ptop = np.array([.95, .6, .99, .5, .4, .8])
    c = dg.calibration(true, pred, ptop)
    assert c["n"] == 5                                              # the -1 (ambiguous) box is excluded
    by = {round(b["lo"], 2): b for b in c["bins"]}
    assert (by[0.0]["n"], by[0.0]["acc"]) == (1, 1.0)              # p=.4  right
    assert (by[0.5]["n"], by[0.5]["acc"]) == (1, 0.0)              # p=.6  wrong
    assert (by[0.7]["n"], by[0.7]["acc"]) == (1, 0.0)              # p=.8  wrong
    assert (by[0.9]["n"], by[0.9]["acc"]) == (1, 1.0)              # p=.95 right
    assert (by[0.99]["n"], by[0.99]["acc"]) == (1, 1.0)            # p=.99 right
    w = {x["tau"]: x for x in c["whatif"]}
    assert w[0.0]["kept"] == 1.0 and w[0.0]["acc"] == pytest.approx(3 / 5)
    assert w[0.9]["kept"] == pytest.approx(2 / 5) and w[0.9]["acc"] == 1.0
    assert w[0.99]["kept"] == pytest.approx(1 / 5)


def test_calibration_empty_is_nan_not_crash():
    c = dg.calibration(np.array([-1, -1]), np.array([0, 0]), np.array([.9, .9]))
    assert c["n"] == 0 and all(b["n"] == 0 for b in c["bins"])


def test_confusions_ranked_and_normalised():
    names = ["background", "cat", "dog", "cow"]
    true = np.array([1, 1, 1, 2, 2, 3, 0, -1])
    pred = np.array([2, 2, 1, 1, 2, 3, 0, 1])
    out = dg.confusions(true, pred, names)
    assert out[0] == {"true": "cat", "pred": "dog", "n": 2, "share_of_true": pytest.approx(2 / 3)}
    assert out[1]["true"] == "dog" and out[1]["pred"] == "cat" and out[1]["n"] == 1
    assert len(out) == 2                                            # only wrong pairs
    assert dg.confusions(np.array([1]), np.array([1]), names) == []


def test_gt_from_pycoco_filters_crowd_and_tiny():
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [{"image_id": 1, "category_id": cu.COCO_IDS[3], "bbox": [10, 10, 20, 30], "iscrowd": 0},
                            {"image_id": 1, "category_id": cu.COCO_IDS[4], "bbox": [0, 0, 50, 50], "iscrowd": 1},
                            {"image_id": 1, "category_id": cu.COCO_IDS[5], "bbox": [0, 0, 1, 40], "iscrowd": 0}]}
    g = dg.gt_from_pycoco(ev.make_coco_gt(coco, [1]), [1, 2])
    b, c = g[1]
    assert b.tolist() == [[10, 10, 30, 40]] and c.tolist() == [3] and len(g[2][1]) == 0


class StubLearner:
    def __init__(self, scores): self._s = scores
    def scores(self, X): return self._s


def make_ctx(Xe, head_scores, fused, variant="ours"):
    # two seen classes: 0 -> text [1,0], 1 -> text [0,1]; background text [0.7,0.7] (never the argmax among classes)
    B = NS(bg_eval=np.array([[1.0, 1.0]]), text_eval=np.eye(2), eval_ids=[7])
    cfg = inc.RunConfig(tasks=[[0], [1]], variant=variant)
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [0, 0, 10, 10.5], [200, 200, 210, 210]], float)
    e_img = np.zeros(4, np.int64)
    return {"bundle": B, "class_order": [0, 1], "eval": (Xe, e_img, boxes, np.ones(4)), "cfg": cfg,
            "learner": StubLearner(head_scores) if head_scores is not None else None, "fused": fused,
            "info": {"classes": [1]}}


def test_head_vs_zeroshot_counts_only_matched_seen_boxes_and_splits_new_old():
    # GT in image 7: class0 at box0, class1 at box1, class 5 (never seen) at the 4th box region
    gt = {7: (np.array([[0, 0, 10, 10], [50, 50, 60, 60], [200, 200, 210, 210]], float), np.array([0, 1, 5]))}
    Xe = np.array([[1.0, 0.0],      # box0: CLIP says class0 (correct)
                   [1.0, 0.0],      # box1: CLIP says class0 (WRONG, truth class1)
                   [0.0, 1.0],      # box2 duplicates box0's GT: CLIP says class1 (wrong, truth class0)
                   [1.0, 0.0]])     # box3 overlaps only the unseen class 5 -> excluded
    head = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]], float)    # head: box0->c0 ok, box1->c1 ok, box2->c0 ok
    fused = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0]], float)   # fused: ok, ok, wrong
    r = dg.head_vs_zeroshot(make_ctx(Xe, head, fused), gt)
    assert r["n_boxes"] == 3 and r["n_new"] == 1 and r["n_old"] == 2        # box1 is the only class-1 ('new') box
    assert r["zeroshot"]["acc"] == pytest.approx(1 / 3)
    assert r["head_only"]["acc"] == pytest.approx(1.0)
    assert r["fused"]["acc"] == pytest.approx(2 / 3)
    assert r["zeroshot"]["acc_new"] == 0.0 and r["head_only"]["acc_new"] == 1.0
    assert r["zeroshot"]["acc_old"] == pytest.approx(0.5)


def test_head_vs_zeroshot_without_a_learner_has_no_head_entry():
    gt = {7: (np.array([[0, 0, 10, 10]], float), np.array([0]))}
    Xe = np.array([[1.0, 0.0]] * 4)
    fused = np.array([[0, 1, 0]] * 4, float)
    r = dg.head_vs_zeroshot(make_ctx(Xe, None, fused, variant="zeroshot"), gt)
    assert "head_only" not in r and r["fused"]["acc"] == 1.0


def test_report_formats_every_section_and_survives_missing_values():
    c = dg.calibration(np.array([1, 2]), np.array([1, 2]), np.array([.95, .6]))
    q = {"acc": 0.9, "fg_precision": 0.8, "bg_precision": float("nan")}
    t = {"task": 0, "classes": [0], "head_vs_zeroshot": {"n_boxes": 5, "zeroshot": {"acc": .5}, "head_only": {"acc": .6},
                                                         "fused": {"acc": .7}},
         "pseudo_calibration": c, "pseudo_confusions": [{"true": "cat", "pred": "dog", "n": 3, "share_of_true": .5}],
         "label_quality_zeroshot": q, "label_quality_refined": q, "kept": 2, "n": 2, "k": 4}
    t0 = {"task": 0, "classes": [0], "head_vs_zeroshot": {"n_boxes": 5, "zeroshot": {"acc": .5}, "fused": {"acc": .5}}}
    txt = dg.format_report({"E5_ours": {"tasks": [t]}, "E0_zeroshot": {"tasks": [t0]}})
    for s in ("## C.", "## A.", "### B.", "| cat | dog | 3 | 0.500 |", "bg precision n/a", "| E0_zeroshot | 1 | 5 | 0.500 | n/a"):
        assert s in txt


def test_cli_diagnose_end_to_end_and_probe_does_not_change_results(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *base])
    rp.main(["run", *base, "--only", "E5_ours", "E0_zeroshot"])
    rp.main(["diagnose", *base])
    d = json.load(open(os.path.join(wd, "results", "diagnose.json")))
    assert set(d) == {"E5_ours", "E5_no_refine", "E0_zeroshot"} and all(len(v["tasks"]) == 3 for v in d.values())
    ab = json.load(open(os.path.join(wd, "results", "ablation.json")))
    assert d["E5_ours"]["final_AP50"] == pytest.approx(ab["E5_ours"]["final_AP50"])        # same run as `run`
    assert d["E0_zeroshot"]["final_AP50"] == pytest.approx(ab["E0_zeroshot"]["final_AP50"])
    assert "head_only" not in d["E0_zeroshot"]["tasks"][0]["head_vs_zeroshot"]
    assert os.path.exists(os.path.join(wd, "results", "diagnose.md"))
    with pytest.raises(SystemExit, match="no tune images"):
        rp.main(["diagnose", *base, "--eval_on", "tune"])

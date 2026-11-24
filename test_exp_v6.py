"""exp v6: base-class-only tuning objective (no new-class labels used to pick settings) and the ImageNet-pretrained
detector backbone option."""
import dataclasses
import sys
import types

import numpy as np
import pytest

import incremental as inc
import tune as tn
import yolo_tools as yt
from inference import InferConfig
from test_pipeline import TASKS, bundle, same, world  # noqa: F401  (fixtures)


def test_base_objective_is_listed_and_unknown_still_refused(bundle):
    assert "final_AP50_base" in tn.OBJECTIVES
    with pytest.raises(ValueError):
        tn.run_search(bundle, TASKS, n_trials=2, objective="AP50_new")      # new-class labels: not a tuning objective


def test_infer_grid_base_column_is_the_mean_over_the_first_task_only(bundle, monkeypatch):
    per = {c: 0.1 * (c + 1) for c in range(9)}
    monkeypatch.setattr(tn, "coco_eval", lambda gt, dets, ids, order: {"AP50": 0.5, "AP": 0.3, "per_class_AP50": per})
    rows = tn.infer_grid(bundle, inc.RunConfig(tasks=TASKS), [InferConfig()])
    assert rows[0]["final_AP50_base"] == pytest.approx(np.mean([per[c] for c in TASKS[0]]))


def test_base_objective_selects_on_base_ap_and_ignores_new_class_ground_truth(bundle):
    t = tn.run_search(bundle, TASKS, n_trials=3, seed=0, objective="final_AP50_base", log=lambda *_: None)
    assert t["objective"] == "final_AP50_base"
    b = t["stage1"][t["best_train_trial"]]["final_AP50_base"]
    assert b == max(r["final_AP50_base"] for r in t["stage1"])
    assert t["tune_scores"]["E5_tuned"] == max(r["final_AP50_base"] for r in t["stage2_E5"])
    assert t["tune_scores"]["E0_tuned"] == max(r["final_AP50_base"] for r in t["stage2_E0"])
    assert "selection used base-class AP50 only" in tn.format_report(t)
    # the same search with every NON-base ground-truth box removed picks exactly the same settings
    import coco_utils as cu
    from evaluate import make_coco_gt
    base_ids = {cu.COCO_IDS[c] for c in TASKS[0]}
    ds = bundle.eval_gt.dataset
    stripped = {**ds, "annotations": [a for a in ds["annotations"] if a["category_id"] in base_ids]}
    assert len(stripped["annotations"]) < len(ds["annotations"])
    b2 = dataclasses.replace(bundle, eval_gt=make_coco_gt(stripped, bundle.eval_ids))
    t2 = tn.run_search(b2, TASKS, n_trials=3, seed=0, objective="final_AP50_base", log=lambda *_: None)
    assert t2["E5"] == t["E5"] and t2["E0"] == t["E0"]
    assert t2["tune_scores"] == pytest.approx(t["tune_scores"])


def test_imagenet_init_builds_from_yaml_and_loads_the_classifier_backbone(monkeypatch, tmp_path):
    calls = []

    class FakeYOLO:
        def __init__(self, src):
            calls.append(("new", src))

        def load(self, w):
            calls.append(("load", w)); return self

        def train(self, **kw):
            calls.append(("train", kw["data"], kw["epochs"]))
            return types.SimpleNamespace(save_dir=tmp_path)

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    yt.train_base("d.yaml", init="imagenet", size="s", epochs=3)
    assert calls[:2] == [("new", "yolov8s.yaml"), ("load", "yolov8s-cls.pt")] and calls[2][0] == "train"
    calls.clear()
    yt.train_base("d.yaml", init="scratch", size="s", epochs=3)
    assert calls[0] == ("new", "yolov8s.yaml") and all(c[0] != "load" for c in calls)
    with pytest.raises(ValueError):
        yt.train_base("d.yaml", init="resnet")


def test_cli_base_tuning_writes_its_own_file_and_feeds_variants_without_touching_tune_json(tmp_path):
    """The exact command order of the Kaggle check: tune --objective final_AP50_base --tag base, then variants with it."""
    import json
    import os
    import run_pipeline as rp
    from test_exp_round1 import _base
    wd = str(tmp_path / "wd"); base = _base(wd)
    for c in ("fetch", "make-yolo-data", "extract"):
        assert rp.main([c, *base]) == 0
    assert rp.main(["tune", *base, "--n_trials", "3"]) == 0
    tj = os.path.join(wd, "results", "tune.json"); before = open(tj, "rb").read()
    assert rp.main(["tune", *base, "--n_trials", "3", "--objective", "final_AP50_base", "--tag", "base"]) == 0
    tb = os.path.join(wd, "results", "tune_base.json")
    assert json.load(open(tb))["objective"] == "final_AP50_base" and open(tj, "rb").read() == before
    assert rp.main(["variants", *base, "--tuned", tb, "--on", "eval", "--n_folds", "2", "--tag", "basetune",
                    "--only", "rf_off"]) == 0
    r = json.load(open(os.path.join(wd, "results", "variants_eval_basetune.json")))
    assert {"E0_zeroshot_tuned", "E5_tuned", "rf_off"} <= set(r) and "AP" in r["E5_tuned"]["final_folds"]
    assert open(tj, "rb").read() == before

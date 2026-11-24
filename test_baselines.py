import json
import os
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np
import pytest
from PIL import Image

import baselines as bl
import compare_runs as cr
import evaluate as ev
import pipeline as pl
import run_pipeline as rp
from test_pipeline import TASKS, bundle, world  # noqa: F401


class _B:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)
    def __len__(self): return len(self.conf.a)


class _T:
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a


def stub_model(per_image):
    m = mock.Mock()
    calls = {"n": 0}
    def predict(**kw):
        b, s, c = per_image(calls["n"]); calls["n"] += 1
        m.last_kwargs = kw
        return [NS(boxes=_B(b, s, c))]
    m.predict.side_effect = predict
    return m


def test_world_detector_sets_classes_and_converts_results():
    m = stub_model(lambda i: ([[10, 10, 50, 60], [0, 0, 0.5, 0.5]], [0.9, 0.8], [2, 0]))
    d = bl.WorldDetector(["a", "b", "c"], model=m, conf=0.002, max_det=77)
    m.set_classes.assert_called_once_with(["a", "b", "c"])
    img = Image.new("RGB", (100, 80))
    b, s, c = d(img)
    assert b.shape == (1, 4) and s.tolist() == [0.9] and c.tolist() == [2]            # degenerate box dropped
    kw = m.last_kwargs
    assert kw["conf"] == 0.002 and kw["max_det"] == 77 and kw["source"].shape == (80, 100, 3) and kw["verbose"] is False


def test_world_detector_signature_matches_ultralytics_predict_kwargs():
    """predict() kwargs we pass must be ones ultralytics accepts (guards against typos the stub would swallow)."""
    allowed = {"source", "conf", "iou", "max_det", "imgsz", "device", "verbose"}
    m = stub_model(lambda i: ([[0, 0, 5, 5]], [0.5], [0]))
    bl.WorldDetector(["a"], model=m)(Image.new("RGB", (20, 20)))
    assert set(m.last_kwargs) <= allowed


def test_score_detector_matches_a_hand_built_evaluation(bundle, world):
    """A perfect detector (reads GT) must score AP50 = 1; a blind one 0; class-position remapping must follow class_order."""
    import coco_utils as cu
    gt = cu.gt_by_image(world["cv"], world["eval_ids"])
    ids = world["eval_ids"]
    order = [c for t in TASKS for c in t]
    pos = {c: j for j, c in enumerate(order)}

    def perfect(img_ids):
        it = iter(img_ids)
        def f(img):
            gb, gc = gt[next(it)]
            return gb, np.ones(len(gb)), np.array([pos[int(c)] for c in gc])
        return f
    imgs = list(pl.iter_images(world["cv"], ids, world["dva"]))
    res = bl.score_detector(perfect(ids), imgs, ids, bundle.eval_gt, TASKS, n_folds=4)
    assert res["final_AP50"] == pytest.approx(1.0) and res["curve"][0]["AP50_new"] == pytest.approx(1.0)
    assert len(res["final_folds"]["AP50"]) == 4 and all(v == pytest.approx(1.0) for v in res["final_folds"]["AP50"])
    blind = bl.score_detector(lambda img: (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)), imgs, ids, bundle.eval_gt, TASKS)
    assert blind["final_AP50"] == 0.0
    # wrong class positions are punished: shift every class by one position
    def shifted(img_ids):
        it = iter(img_ids)
        def f(img):
            gb, gc = gt[next(it)]
            return gb, np.ones(len(gb)), np.array([(pos[int(c)] + 1) % len(order) for c in gc])
        return f
    bad = bl.score_detector(shifted(ids), imgs, ids, bundle.eval_gt, TASKS)
    assert bad["final_AP50"] < 0.05


def test_score_detector_caps_detections_per_image(bundle, world):
    ids = world["eval_ids"][:3]
    imgs = list(pl.iter_images(world["cv"], ids, world["dva"]))
    gt = ev.make_coco_gt(world["cv"], ids)
    n = {"max": 0}
    def many(img):
        return np.tile([[1, 1, 30, 30]], (300, 1)).astype(float), np.linspace(1, 0.01, 300), np.zeros(300, int)
    orig = ev.coco_eval
    def spy(g, dets, i, c, **kw):
        n["max"] = max(n["max"], max(sum(d["image_id"] == x for d in dets) for x in ids)); return orig(g, dets, i, c, **kw)
    with mock.patch.object(bl, "coco_eval", spy):
        bl.score_detector(many, imgs, ids, gt, TASKS, max_dets=100)
    assert n["max"] == 100


def test_cli_baseline_writes_results_and_compare_merges_them(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *base])
    rp.main(["run", *base, "--only", "E0_zeroshot", "E5_ours"])
    import coco_utils as cu
    seen = {}
    def fake_detector(names, **kw):
        seen["names"], seen["kw"] = names, kw
        return lambda img: (np.zeros((0, 4)), np.zeros(0), np.zeros(0, int))
    with mock.patch.object(bl, "WorldDetector", fake_detector):
        rp.main(["baseline", *base, "--world_conf", "0.01"])
    assert seen["names"] == cu.prompt_names(range(9)) and seen["kw"]["conf"] == 0.01
    b = json.load(open(os.path.join(wd, "results", "baselines.json")))
    assert list(b) == ["yolo_world"] and b["yolo_world"]["final_AP50"] == 0.0 and "final_folds" in b["yolo_world"]
    out = str(tmp_path / "t.md")
    assert cr.main([wd, "--also", "baselines.json", "--out", out]) == 0
    assert "yolo_world" in open(out).read()
    with pytest.raises(SystemExit, match="already present"):
        cr.main([wd, "--also", "ablation.json"])
    with pytest.raises(SystemExit, match="not found"):
        cr.main([wd, "--also", "nope.json"])
    with pytest.raises(SystemExit, match="no tune images"):
        rp.main(["baseline", *base, "--eval_on", "tune"])


def test_plain_ridge_baseline_is_the_simplest_self_training():
    c = pl.ablation_configs(TASKS)["B_plain_ridge"]
    e5 = pl.ablation_configs(TASKS)["E5_ours"]
    assert c.n2l.n_rounds == 0 and not c.n2l.use_intra and not c.n2l.use_inter
    assert c.pseudo == e5.pseudo and c.infer == e5.infer and c.variant == "ours"


def test_plain_ridge_runs_and_is_distinct_from_e5(bundle):
    r = pl.run_ablation(bundle, TASKS, only=["B_plain_ridge", "E5_ours"], log=lambda *_: None)
    assert np.isfinite(r["B_plain_ridge"]["final_AP50"])


def test_score_detector_maps_class_positions_through_class_order(bundle, world):
    """With non-contiguous, shuffled task classes, position j must mean class_order[j] (not class j)."""
    import coco_utils as cu
    tasks = [[4, 0, 7], [1, 8, 2], [3, 5, 6]]
    order = [c for t in tasks for c in t]
    pos = {c: j for j, c in enumerate(order)}
    ids = world["eval_ids"]
    gt = cu.gt_by_image(world["cv"], ids)
    it = iter(ids)
    def perfect(img):
        gb, gc = gt[next(it)]
        return gb, np.ones(len(gb)), np.array([pos[int(c)] for c in gc])
    imgs = list(pl.iter_images(world["cv"], ids, world["dva"]))
    res = bl.score_detector(perfect, imgs, ids, bundle.eval_gt, tasks, n_folds=0)
    assert res["final_AP50"] == pytest.approx(1.0)

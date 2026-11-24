"""test_v2_infra.py -- held-out tune split, extract --splits, run --eval_on, image folds, probe hook."""
import json
import os

import numpy as np
import pytest

import incremental as inc
import pipeline as pl
import run_pipeline as rp
from evaluate import coco_eval
from fake_models import FakeModels
from test_pipeline import TASKS, bundle, same, world  # noqa: F401  (fixtures)


def cli(cmd, *args):
    return rp.main([cmd, *args])


def base_args(wd, n_tune=0):
    return ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50", "--n_tune", str(n_tune)]


def read(wd, name):
    return json.load(open(os.path.join(wd, name)))


# ------------------------------------------------------------------ tune split
def test_tune_ids_disjoint_deterministic_and_do_not_move_eval_ids(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    cli("fetch", *base_args(a, 0)); cli("fetch", *base_args(b, 25))
    sa, sb = read(a, "splits.json"), read(b, "splits.json")
    assert sa["tune_ids"] == [] and len(sb["tune_ids"]) == 25
    assert not set(sb["tune_ids"]) & set(sb["eval_ids"])                   # disjoint from the final eval images
    # (here the fake world has n_eval + n_tune val images, so the pool itself differs; on a fixed pool the eval ids
    # must not move -- tested directly in the next test)
    cli("fetch", *base_args(str(tmp_path / "c"), 25))
    assert read(str(tmp_path / "c"), "splits.json")["tune_ids"] == sb["tune_ids"]      # deterministic


def test_eval_ids_identical_with_and_without_tune_on_a_fixed_pool(tmp_path):
    """Real datasets have a fixed val pool: asking for tune images must not change which images are evaluated."""
    import coco_utils as cu
    from fake_models import make_image_world
    ct, cv, dtr, dva = make_image_world(str(tmp_path / "w"), TASKS, 10, 80, seed=1)
    json.dump(ct, open(tmp_path / "t.json", "w")); json.dump(cv, open(tmp_path / "v.json", "w"))
    out = []
    for n_tune in (0, 20):
        wd = str(tmp_path / f"wd{n_tune}")
        cli("fetch", "--workdir", wd, "--ann_train", str(tmp_path / "t.json"), "--ann_val", str(tmp_path / "v.json"),
            "--img_dir_train", dtr, "--img_dir_val", dva, "--tasks", "0,1,2|3,4,5|6,7,8", "--n_train_per_task", "10",
            "--n_eval", "30", "--n_tune", str(n_tune), "--coco_root", str(tmp_path / "none"))
        out.append(read(wd, "splits.json"))
    assert out[0]["eval_ids"] == out[1]["eval_ids"] and len(out[1]["eval_ids"]) == 30
    assert len(out[1]["tune_ids"]) == 20 and not set(out[1]["tune_ids"]) & set(out[1]["eval_ids"])
    assert out[0]["tune_ids"] == []


def test_n_tune_larger_than_the_pool_is_capped(tmp_path):
    from fake_models import make_image_world
    ct, cv, dtr, dva = make_image_world(str(tmp_path / "w"), TASKS, 10, 40, seed=1)
    json.dump(ct, open(tmp_path / "t.json", "w")); json.dump(cv, open(tmp_path / "v.json", "w"))
    wd = str(tmp_path / "wd")
    cli("fetch", "--workdir", wd, "--ann_train", str(tmp_path / "t.json"), "--ann_val", str(tmp_path / "v.json"),
        "--img_dir_train", dtr, "--img_dir_val", dva, "--tasks", "0,1,2|3,4,5|6,7,8", "--n_train_per_task", "10",
        "--n_eval", "30", "--n_tune", "500", "--coco_root", str(tmp_path / "none"))
    sp = read(wd, "splits.json")
    assert len(sp["eval_ids"]) == 30 and len(sp["tune_ids"]) == 10


@pytest.fixture()
def wd_tune(tmp_path):
    wd = str(tmp_path / "wd")
    args = base_args(wd, 20)
    for c in ("fetch", "gate-clip", "make-yolo-data", "extract", "gate-recall"):
        cli(c, *args)
    return wd, args


def test_extract_builds_tune_cache_and_records_settings(wd_tune):
    wd, _ = wd_tune
    from feature_cache import FeatureCache
    sp = read(wd, "splits.json")
    assert set(map(int, FeatureCache(os.path.join(wd, "cache", "tune")).image_ids)) == set(sp["tune_ids"])
    clip = read(wd, "clip.json")
    assert clip["max_boxes"] == 50 and clip["pad"] == 0.1 and "conf" in clip and "max_det" in clip


def test_partial_extract_touches_only_the_requested_cache(wd_tune):
    wd, args = wd_tune
    mt = lambda n: os.path.getmtime(os.path.join(wd, "cache", n, "feats.npy"))
    before = {n: mt(n) for n in ("train", "eval", "tune")}
    import time; time.sleep(0.05)
    cli("extract", *args, "--splits", "tune")
    after = {n: mt(n) for n in ("train", "eval", "tune")}
    assert after["train"] == before["train"] and after["eval"] == before["eval"] and after["tune"] > before["tune"]


def test_partial_extract_refuses_different_settings(wd_tune):
    wd, args = wd_tune
    bad = [x for x in args]
    bad[bad.index("--max_boxes") + 1] = "30"
    with pytest.raises(SystemExit, match="settings differ"):
        cli("extract", *bad, "--splits", "tune")


def test_extract_without_tune_ids_refuses_tune(tmp_path):
    wd = str(tmp_path / "wd"); args = base_args(wd, 0)
    cli("fetch", *args)
    with pytest.raises(SystemExit, match="no tune images"):
        cli("extract", *args, "--splits", "tune")


def test_run_eval_on_tune_writes_separate_files_and_uses_tune_images(wd_tune):
    wd, args = wd_tune
    cli("run", *args, "--only", "E0_zeroshot", "E5_ours")
    final = read(wd, "results/ablation.json")
    cli("run", *args, "--only", "E0_zeroshot", "--eval_on", "tune")
    assert read(wd, "results/ablation.json") == final                      # final numbers untouched
    tune = read(wd, "results/ablation_tune.json")
    assert list(tune) == ["E0_zeroshot"]
    sp = read(wd, "splits.json")
    pth = read(wd, "paths.json")
    by_hand = pl.run_ablation(
        pl.build_bundle(FakeModels(9), os.path.join(wd, "cache", "train"), sp["train_ids"],
                        json.load(open(pth["ann_train"])), os.path.join(wd, "cache", "tune"), sp["tune_ids"],
                        json.load(open(pth["ann_val"]))), sp["tasks"], only=["E0_zeroshot"], log=lambda *_: None)
    assert same(json.loads(json.dumps(by_hand, default=pl._json_default)), tune)


def test_run_eval_on_tune_without_tune_ids_fails_clearly(tmp_path):
    wd = str(tmp_path / "wd"); args = base_args(wd, 0)
    for c in ("fetch", "make-yolo-data", "extract"):
        cli(c, *args)
    with pytest.raises(SystemExit, match="no tune images"):
        cli("run", *args, "--eval_on", "tune")


# ------------------------------------------------------------------ folds
def test_image_folds_disjoint_cover_balanced_deterministic():
    ids = list(range(100, 203))
    f = inc.image_folds(ids, 5, 7)
    assert len(f) == 5 and sorted(i for x in f for i in x) == ids
    assert max(map(len, f)) - min(map(len, f)) <= 1
    assert f == inc.image_folds(ids[::-1], 5, 7)                         # independent of input order
    assert f != inc.image_folds(ids, 5, 8)                                # seed matters
    assert inc.image_folds(ids, 0) == [] and inc.image_folds(ids, 1) == [] and inc.image_folds(ids[:9], 5) == []


def test_fold_scores_equal_direct_evaluation_on_each_fold(bundle):
    cfg = inc.RunConfig(tasks=TASKS, variant="zeroshot", n_folds=4)
    seen = {}
    res = inc.run_incremental(bundle, cfg, log=lambda *_: None, probe=lambda c: seen.update(last=c))
    folds = inc.image_folds(bundle.eval_ids, 4, cfg.fold_seed)
    ff = res["final_folds"]
    assert len(ff["AP50"]) == 4 and ff["n_images"] == [len(x) for x in folds]
    # recompute one fold independently from the same fused scores
    from evaluate import dets_to_coco
    from inference import detect
    c = seen["last"]; Xe, e_img, Be, Se = c["eval"]; dets = []
    for k, iid in enumerate(bundle.eval_ids):
        m = e_img == k
        b, s, cl = detect(Be[m], Se[m], c["fused"][m], c["class_order"], cfg.infer)
        dets += dets_to_coco(iid, b, s, cl)
    r = coco_eval(bundle.eval_gt, dets, folds[2], c["class_order"])
    assert ff["AP50"][2] == pytest.approx(r["AP50"]) and ff["AP"][2] == pytest.approx(r["AP"])
    new = [r["per_class_AP50"][k] for k in TASKS[-1]]
    assert ff["AP50_new"][2] == pytest.approx(float(np.nanmean(new)))


def test_folds_off_means_no_fold_key(bundle):
    res = inc.run_incremental(bundle, inc.RunConfig(tasks=TASKS, variant="zeroshot", n_folds=0), log=lambda *_: None)
    assert "final_folds" not in res


# ------------------------------------------------------------------ probe hook
def test_probe_is_read_only_and_sees_the_right_things(bundle):
    cfg = inc.RunConfig(tasks=TASKS, variant="ours")
    plain = inc.run_incremental(bundle, cfg, log=lambda *_: None)
    ctxs = []
    probed = inc.run_incremental(bundle, cfg, log=lambda *_: None, probe=ctxs.append)
    assert same(json.loads(json.dumps(plain, default=pl._json_default)), json.loads(json.dumps(probed, default=pl._json_default)))
    assert [c["task"] for c in ctxs] == [0, 1, 2]
    c = ctxs[-1]
    assert c["class_order"] == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert c["fused"].shape[1] == 1 + 9 and c["train"]["logits"].shape[1] == c["fused"].shape[1]
    assert set(c["train"]) >= {"X", "imgidx", "boxes", "yolo", "true", "col_of", "keep", "logits", "result"}
    zs = inc.run_incremental(bundle, inc.RunConfig(tasks=TASKS, variant="zeroshot"), log=lambda *_: None,
                             probe=ctxs.append)
    assert ctxs[-1]["train"] is None


# ------------------------------------------------------------------ sharper fold tests (the fake world saturates AP50 at 1,
# so these use hand-built detections whose quality differs between folds / between old and new classes)
def _gt_dets(world, keep):
    """Perfect detections from the ground truth for the (image, class) pairs `keep(img_id, class) -> bool`."""
    import coco_utils as cu
    from evaluate import dets_to_coco
    gt = cu.gt_by_image(world["cv"], world["eval_ids"])
    dets = []
    for iid, (gb, gc) in gt.items():
        sel = [k for k, c in enumerate(gc) if keep(iid, int(c))]
        if sel:
            dets += dets_to_coco(iid, np.asarray(gb)[sel], np.ones(len(sel)), np.asarray(gc)[sel])
    return dets


def test_fold_scores_are_computed_on_their_own_fold(world):
    from evaluate import make_coco_gt
    ids = world["eval_ids"]
    folds = [ids[:20], ids[20:]]
    bad = set(folds[1])
    dets = _gt_dets(world, lambda iid, c: iid not in bad)                  # perfect on fold 0, nothing on fold 1
    ff = inc.fold_scores(make_coco_gt(world["cv"], ids), dets, folds, list(range(9)), [6, 7, 8])
    assert ff["AP50"][0] == pytest.approx(1.0) and ff["AP50"][1] < 0.05
    assert ff["AP"][0] > 0.9 and ff["AP"][1] < 0.05 and ff["n_images"] == [20, len(ids) - 20]


def test_fold_scores_new_class_ap_averages_only_the_new_classes(world):
    from evaluate import make_coco_gt
    ids = world["eval_ids"]
    new = [6, 7, 8]
    dets = _gt_dets(world, lambda iid, c: c not in new)                    # old classes perfect, new classes never found
    ff = inc.fold_scores(make_coco_gt(world["cv"], ids), dets, [ids], list(range(9)), new)
    assert ff["AP50_new"][0] == pytest.approx(0.0)                         # an all-class average would be about 2/3
    assert 0.5 < ff["AP50"][0] < 0.8
    none = inc.fold_scores(make_coco_gt(world["cv"], ids), dets, [ids], list(range(9)), [])
    assert np.isnan(none["AP50_new"][0])

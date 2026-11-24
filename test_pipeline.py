"""test_pipeline.py -- Step 7: the glue (fake models, image world, pipeline stages, CLI).

Everything runs offline: FakeModels stand in for CLIP/YOLO, make_image_world writes real PNGs +
COCO-format json. The CLI sub-commands are driven through main([...]) exactly as on Colab.
"""
import copy
import dataclasses
import inspect
import json
import os
import sys
import types
from unittest import mock

import numpy as np
import pytest
from PIL import Image

import coco_utils as cu
import incremental as inc
import pipeline as pl
import prompts as pr
import run_pipeline as rp
from boxes import iou_matrix
from fake_models import FakeModels, bin_rgb, make_image_world, pixel_bins
from feature_cache import FeatureCache
from inference import InferConfig

TASKS = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def same(a, b):
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


# ------------------------------------------------------------------ the world
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("world"))
    ct, cv, dtr, dva = make_image_world(root, TASKS, n_per_task=25, n_eval=40, seed=3)
    models = FakeModels(9)
    train_ids = [cu.select_task_images(ct, t, 25, 0) for t in TASKS]
    eval_ids = sorted(im["id"] for im in cv["images"])
    tr_union = sorted({i for ids in train_ids for i in ids})
    tr_cache = pl.stage_extract(models, ct, tr_union, dtr, os.path.join(root, "c_tr"))
    ev_cache = pl.stage_extract(models, cv, eval_ids, dva, os.path.join(root, "c_ev"))
    return dict(root=root, ct=ct, cv=cv, dtr=dtr, dva=dva, models=models, train_ids=train_ids,
                eval_ids=eval_ids, tr_union=tr_union, tr_cache=tr_cache, ev_cache=ev_cache)


@pytest.fixture(scope="module")
def bundle(world):
    w = world
    return pl.build_bundle(w["models"], w["tr_cache"], w["train_ids"], w["ct"], w["ev_cache"],
                           w["eval_ids"], w["cv"])


# ================================================================== A. fake world
def test_world_pixels_match_annotations(world):
    """Rendered colour at the centre of every annotation == that class's colour (renderer/annotation aligned)."""
    by_id = {im["id"]: im for im in world["ct"]["images"]}
    fm = world["models"]
    n = 0
    for a in world["ct"]["annotations"]:
        arr = np.asarray(Image.open(os.path.join(world["dtr"], by_id[a["image_id"]]["file_name"])).convert("RGB"))
        x, y, w, h = a["bbox"]
        cx, cy = int(x + w / 2), int(y + h / 2)
        assert tuple(arr[cy, cx]) == fm.class_rgb(cu.COCO_IDS.index(a["category_id"]))
        assert arr[int(y), int(x)].tolist() == list(fm.class_rgb(cu.COCO_IDS.index(a["category_id"])))
        n += 1
    assert n > 40


def test_world_boxes_inside_canvas_and_disjoint(world):
    for coco in (world["ct"], world["cv"]):
        gt = cu.gt_by_image(coco, [im["id"] for im in coco["images"]])
        for b, c in gt.values():
            assert (b >= 0).all() and (b <= 160).all()
            if len(b) > 1:
                m = iou_matrix(b, b); np.fill_diagonal(m, 0)
                assert m.max() == 0.0


def test_world_structure(world):
    ct, cv = world["ct"], world["cv"]
    assert len(ct["categories"]) == 80
    cu.verify_categories(ct["categories"]); cu.verify_categories(cv["categories"])
    cls_of = cu.image_class_index(ct)
    for t, ids in enumerate(world["train_ids"]):
        assert ids and all(cls_of[i] & set(TASKS[t]) for i in ids)       # every task image has a task class
    older_seen = [any(cls_of[i] & set(sum(TASKS[:t], [])) for i in ids)
                  for t, ids in enumerate(world["train_ids"]) if t > 0]
    assert all(older_seen)                                               # later tasks contain old objects (unlabelled)
    ev_cls = set().union(*cu.image_class_index(cv).values())
    assert ev_cls == set(range(9))
    assert not (set(world["tr_union"]) & set(world["eval_ids"]))         # train / eval image ids disjoint


def test_world_deterministic(tmp_path):
    a = make_image_world(str(tmp_path / "a"), TASKS, 5, 5, seed=11)
    b = make_image_world(str(tmp_path / "b"), TASKS, 5, 5, seed=11)
    c = make_image_world(str(tmp_path / "c"), TASKS, 5, 5, seed=12)
    core = lambda w: ([(i["id"], i["file_name"]) for i in w["images"]], w["annotations"])
    assert same(core(a[0]), core(b[0])) and same(core(a[1]), core(b[1]))
    assert not same(core(a[0]), core(c[0]))
    pa = np.asarray(Image.open(os.path.join(a[2], a[0]["images"][0]["file_name"])))
    pb = np.asarray(Image.open(os.path.join(b[2], b[0]["images"][0]["file_name"])))
    assert np.array_equal(pa, pb)                                   # pixels too, not only json


def test_fake_text_encoder_semantics():
    fm = FakeModels(9)
    for c, name in enumerate(fm.names):
        for t in pr.DET_TEMPLATES:
            v = fm.encode_text([t.format(name)])[0]
            assert v.argmax() == c + 1 and (v > 0).sum() == 1
    for t in pr.BG_PROMPTS:                                      # background prompts must NOT name a class
        assert fm.encode_text([t])[0].argmax() == 0
    with pytest.raises(ValueError):
        FakeModels(27)


def test_fake_image_encoder_histogram():
    fm = FakeModels(9)
    for c in range(9):
        crop = np.zeros((32, 32, 3), np.uint8); crop[:] = fm.class_rgb(c)
        assert fm.encode_images_u8(crop[None])[0].argmax() == c + 1
    assert pixel_bins(np.array([[bin_rgb(13)]], np.uint8))[0, 0] == 13


def test_fake_proposer_finds_objects_without_seeing_gt(world):
    """Proposals come from pixels only; every GT object has an (almost exact) component box; 3 random extras."""
    by_id = {im["id"]: im for im in world["cv"]["images"]}
    gt = cu.gt_by_image(world["cv"], world["eval_ids"])
    for i in world["eval_ids"][:15]:
        img = Image.open(os.path.join(world["dva"], by_id[i]["file_name"])).convert("RGB")
        b, s = world["models"].propose(img)
        assert len(b) == len(s) and (s >= 0).all()
        g = gt[i][0]
        assert (iou_matrix(g, b).max(1) > 0.97).all()
        assert (s < 0.5).sum() == 3
        b2, _ = world["models"].propose(img)
        assert np.array_equal(b, b2)                              # deterministic


# ================================================================== B. stages
def test_gate_clip_high_and_it_can_fail(world):
    r = pl.stage_gate_clip(world["models"], world["cv"], world["dva"], world["eval_ids"], TASKS)
    assert r["all"] > 0.97 and all(r[f"task{t}_within"] > 0.97 for t in range(3))

    class Shifted(FakeModels):                       # text tower mislabels every class -> the gate must notice
        def encode_text(self, texts):
            v = super().encode_text(texts)
            return np.concatenate([v[:, :1], np.roll(v[:, 1:10], 1, axis=1), v[:, 10:]], axis=1)

    bad = pl.stage_gate_clip(Shifted(9), world["cv"], world["dva"], world["eval_ids"], TASKS)
    assert bad["all"] < 0.2


def test_extract_structure_and_alignment(world):
    cache = FeatureCache(world["tr_cache"])
    assert list(cache.image_ids) == world["tr_union"]
    gt = cu.gt_by_image(world["ct"], world["tr_union"])
    hits = 0
    for i in world["tr_union"]:
        im = cache.image(i)
        assert im["feats"].shape[1] == 27 and im["feats"].dtype == np.float16
        assert im["boxes"].shape == (len(im["scores"]), 4) and len(im["boxes"]) <= 100
        gb, gc = gt[i]
        if len(gb):
            iou = iou_matrix(gb, im["boxes"])
            for g in range(len(gb)):
                j = iou[g].argmax()
                if iou[g, j] > 0.97:                           # exact box -> its feature must be that class's colour
                    assert im["feats"][j].astype(float).argmax() == gc[g] + 1
                    hits += 1
    assert hits > 60


def test_extract_deterministic_and_matches_second_run(world, tmp_path):
    again = pl.stage_extract(world["models"], world["ct"], world["tr_union"], world["dtr"], str(tmp_path / "x"))
    a, b = FeatureCache(world["tr_cache"]), FeatureCache(again)
    assert np.array_equal(np.asarray(a.feats), np.asarray(b.feats))
    assert np.array_equal(np.asarray(a.boxes), np.asarray(b.boxes))


def test_gate_recall_matches_independent_computation(world):
    r = pl.stage_gate_recall(world["ev_cache"], world["cv"], world["eval_ids"], TASKS)
    cache = FeatureCache(world["ev_cache"])
    gt = cu.gt_by_image(world["cv"], world["eval_ids"])
    for t, cls in enumerate(TASKS):
        tot = hit5 = hit7 = 0
        for i in world["eval_ids"]:
            gb, gc = gt[i]
            m = np.isin(gc, cls)
            if not m.any():
                continue
            best = iou_matrix(gb[m], cache.image(i)["boxes"]).max(1)
            tot += len(best); hit5 += int((best >= 0.5).sum()); hit7 += int((best >= 0.7).sum())
        assert r[f"task{t}"]["n_gt"] == tot
        assert r[f"task{t}"]["recall@0.5"] == pytest.approx(hit5 / tot)
        assert r[f"task{t}"]["recall@0.7"] == pytest.approx(hit7 / tot)
        assert r[f"task{t}"]["recall@0.5"] > 0.99


def test_gate_recall_detects_a_blind_proposer(world, tmp_path):
    class Blind(FakeModels):
        def propose(self, img):
            return np.array([[0.0, 0.0, 5.0, 5.0]]), np.array([0.5])      # never overlaps an object

    d = pl.stage_extract(Blind(9), world["cv"], world["eval_ids"], world["dva"], str(tmp_path / "blind"))
    r = pl.stage_gate_recall(d, world["cv"], world["eval_ids"], TASKS)
    assert all(r[f"task{t}"]["recall@0.5"] == 0.0 for t in range(3))


def test_text_assets_label_uses_first_template_only(world):
    seen = []
    orig = pr.class_embeddings

    def spy(names, enc, templates=pr.DET_TEMPLATES):
        seen.append(list(templates)); return orig(names, enc, templates)

    with mock.patch.object(pr, "class_embeddings", spy):
        T = pl.text_assets(world["models"])
    assert seen == [pr.DET_TEMPLATES[:1], list(pr.DET_TEMPLATES)]       # label: first only; eval: full ensemble
    assert T["text_label"].shape == (80, 27) and T["text_eval"].shape == (80, 27)
    assert T["bg_label"].shape == (1, 27) and T["bg_eval"].shape == (1, 27)
    assert np.allclose(np.linalg.norm(T["text_eval"], axis=1), 1.0)


# ================================================================== C. the experiment
def test_bundle_wiring(bundle, world):
    assert bundle.train_ids == world["train_ids"] and bundle.eval_ids == world["eval_ids"]
    assert set(bundle.train_gt) == set(world["tr_union"])
    assert bundle.eval_gt.getImgIds() and sorted(bundle.eval_gt.getImgIds()) == world["eval_ids"]
    assert bundle.train_cache.meta["feat_dim"] == 27


def test_core_runs_reach_expected_levels(bundle):
    res = pl.run_ablation(bundle, TASKS, only=["E0_zeroshot", "E5_ours", "E6_gt_ceiling"], log=lambda *_: None)
    assert res["E0_zeroshot"]["final_AP50"] > 0.95
    assert res["E5_ours"]["final_AP50"] > 0.95
    assert res["E6_gt_ceiling"]["final_AP50"] > 0.95
    assert len(res["E5_ours"]["curve"]) == 3


def test_all_ablations_run_and_are_finite(bundle):
    res = pl.run_ablation(bundle, TASKS, log=lambda *_: None)
    assert list(res) == list(pl.ablation_configs(TASKS))
    for n, r in res.items():
        assert np.isfinite([r["A_bar_AP"], r["final_AP"], r["final_AP50"]]).all(), n
        assert 0.0 <= r["final_AP50"] <= 1.0, n
    # Background matters here -- but with the default yolo_gamma=0.5 the fake world's low-score junk boxes are
    # already suppressed by the proposal score, so compare both runs with gamma 0 (the ablation's own effect).
    cfgs = pl.ablation_configs(TASKS)
    g0 = lambda c: dataclasses.replace(c, infer=InferConfig(yolo_gamma=0.0))
    with_bg = inc.run_incremental(bundle, g0(cfgs["E5_ours"]), lambda *_: None)
    no_bg = inc.run_incremental(bundle, g0(cfgs["E5_no_bg"]), lambda *_: None)
    assert no_bg["final_AP50"] < with_bg["final_AP50"] - 0.02


def test_unknown_ablation_name_rejected(bundle):
    with pytest.raises(KeyError):
        pl.run_ablation(bundle, TASKS, only=["E5_ours", "nope"])


def test_ablation_configs_differ_from_E5_in_one_thing():
    cfgs = pl.ablation_configs(TASKS)
    base = cfgs["E5_ours"]
    assert cfgs["E5_no_refine"].n2l.n_rounds == 0 and base.n2l.n_rounds == 3
    assert not cfgs["E5_no_weights"].n2l.use_intra and not cfgs["E5_no_weights"].n2l.use_inter
    assert not cfgs["E5_inter_only"].n2l.use_intra and cfgs["E5_inter_only"].n2l.use_inter
    assert not cfgs["E5_no_bg"].pseudo.use_bg and base.pseudo.use_bg
    assert cfgs["E5_expand2000"].expand == 2000 and base.expand == 0
    assert cfgs["E5_ridge_only"].infer.fusion_weight == 1.0 and base.infer.fusion_weight == 0.8
    assert cfgs["E5_fusion_0.5"].infer.fusion_weight == 0.5
    assert cfgs["E0_zeroshot"].variant == "zeroshot" and cfgs["E6_gt_ceiling"].variant == "gt_ceiling"
    for name, var, g in [("E0_zeroshot_g0", "zeroshot", 0.0), ("E0_zeroshot_g1", "zeroshot", 1.0),
                         ("E5_yolo_gamma0", "ours", 0.0), ("E5_yolo_gamma1", "ours", 1.0),
                         ("E6_gt_ceiling_g0", "gt_ceiling", 0.0), ("E6_gt_ceiling_g1", "gt_ceiling", 1.0)]:
        assert cfgs[name].variant == var and cfgs[name].infer.yolo_gamma == g
        assert cfgs[name].infer.fusion_weight == base.infer.fusion_weight and cfgs[name].expand == base.expand
    # gamma 0.5 is the default for every headline run (it helped in every COCO run); only the *_g0/_g1 rows change it
    for name in ("E0_zeroshot", "E5_ours", "E6_gt_ceiling"):
        assert cfgs[name].infer.yolo_gamma == 0.5, name
    assert not any(n.endswith("g0.5") or n.endswith("gamma0.5") for n in cfgs)
    assert cfgs["E5_k_energy80_nofloor"].n2l.pca_rho == 0.8 and cfgs["E5_k_energy80_nofloor"].n2l.pca_k_floor == "none"
    assert cfgs["E5_k_abs_official"].n2l.pca_rule == "abs_official"
    for n, c in cfgs.items():
        assert c.tasks == TASKS, n


def test_no_ground_truth_leaks_into_E5(world):
    """Garbage TRAIN annotations: E5 must be bit-identical (curve AND non-GT diagnostics); the GT ceiling must react.
    The ceiling is checked with ridge-only fusion, because the default 0.2*zero-shot term would rescue a head
    trained on garbage (that is a property of the fusion, not a leak)."""
    w = world
    bad = copy.deepcopy(w["ct"])
    for a in bad["annotations"]:
        k = cu.COCO_IDS.index(a["category_id"])
        a["category_id"] = cu.COCO_IDS[(k + 4) % 9]
        a["bbox"] = [a["bbox"][0] + 37, a["bbox"][1] + 37, a["bbox"][2], a["bbox"][3]]
    good = pl.build_bundle(w["models"], w["tr_cache"], w["train_ids"], w["ct"], w["ev_cache"], w["eval_ids"], w["cv"])
    evil = pl.build_bundle(w["models"], w["tr_cache"], w["train_ids"], bad, w["ev_cache"], w["eval_ids"], w["cv"])
    q = lambda *_: None
    for fw in (0.8, 1.0):
        cfg = inc.RunConfig(tasks=TASKS, variant="ours", infer=InferConfig(fusion_weight=fw))
        rg, rb = inc.run_incremental(good, cfg, q), inc.run_incremental(evil, cfg, q)
        assert same(rg["curve"], rb["curve"])
        assert [(d["n"], d["kept"], d["k"]) for d in rg["diag"]] == [(d["n"], d["kept"], d["k"]) for d in rb["diag"]]
    ceil = lambda b, fw: inc.run_incremental(
        b, inc.RunConfig(tasks=TASKS, variant="gt_ceiling", infer=InferConfig(fusion_weight=fw)), q)
    assert ceil(good, 1.0)["final_AP50"] > 0.9
    assert ceil(evil, 1.0)["final_AP50"] < 0.3                      # the ceiling really consumes the annotations
    assert ceil(good, 0.8)["final_AP50"] - ceil(evil, 0.8)["final_AP50"] > 0.03


def test_eval_annotations_drive_the_metric(world):
    w = world
    bad = copy.deepcopy(w["cv"])
    for a in bad["annotations"]:
        a["bbox"] = [a["bbox"][0] + 60, a["bbox"][1] + 60, a["bbox"][2], a["bbox"][3]]
    b = pl.build_bundle(w["models"], w["tr_cache"], w["train_ids"], w["ct"], w["ev_cache"], w["eval_ids"], bad)
    r = pl.run_ablation(b, TASKS, only=["E0_zeroshot"], log=lambda *_: None)
    assert r["E0_zeroshot"]["final_AP50"] < 0.5


def test_format_table_and_save_results(tmp_path):
    res = {"A": {"A_bar_AP": np.float32(0.5), "final_AP": 0.25, "final_AP50": np.float64(0.75),
                 "base_AP50_drop": -0.0125, "curve": [{"task": np.int64(0), "x": float("nan"), "arr": np.arange(2)}]}}
    t = pl.format_table(res)
    assert t.splitlines()[2] == "| A | 0.500 | 0.250 | 0.750 | -0.013 |"
    p = str(tmp_path / "r" / "out.json")
    pl.save_results(res, p)
    back = json.load(open(p))
    assert back["A"]["final_AP50"] == 0.75 and back["A"]["curve"][0]["arr"] == [0, 1]
    assert back["A"]["curve"][0]["task"] == 0 and np.isnan(back["A"]["curve"][0]["x"])


# ================================================================== D. the CLI
def run_cli(*argv):
    rc = rp.main(list(argv))
    assert rc == 0
    return rc


def test_parser_all_subcommands_and_defaults():
    p = rp.build_parser()
    for cmd in ["fetch", "gate-clip", "make-yolo-data", "train-yolo", "extract", "gate-recall", "run",
                "repro-cifar", "selftest"]:
        a = p.parse_args([cmd])
        assert a.func and a.workdir is None and a.tasks is None and a.fake is False
        assert a.dataset == "coco" and rp.scheme_of(a) == "40+40"          # default scheme is per dataset
        assert a.clip_weights == ("laion400m_e32" if cmd == "repro-cifar" else "openai")
    a = p.parse_args(["train-yolo", "--yolo_init", "coco", "--yolo_epochs", "5"])
    assert a.yolo_init == "coco" and a.yolo_epochs == 5
    assert p.parse_args(["train-yolo", "--yolo_init", "imagenet"]).yolo_init == "imagenet"     # exp v6
    with pytest.raises(SystemExit):
        p.parse_args(["train-yolo", "--yolo_init", "resnet"])
    with pytest.raises(SystemExit):
        p.parse_args([])
    assert p.parse_args(["run", "--only", "E5_ours", "E6_gt_ceiling"]).only == ["E5_ours", "E6_gt_ceiling"]


def test_custom_task_spec():
    assert rp.parse_custom_tasks("0,1|2,3,4") == [[0, 1], [2, 3, 4]]
    for bad in ["0,1|1,2", "0,1|", "0,80", "|"]:
        with pytest.raises(ValueError):
            rp.parse_custom_tasks(bad)


def test_cli_chain_fake(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]
    for cmd in ["fetch", "gate-clip", "make-yolo-data", "extract", "gate-recall"]:
        run_cli(cmd, *base)
    run_cli("run", *base, "--only", "E0_zeroshot", "E5_ours", "E6_gt_ceiling")
    sp = json.load(open(os.path.join(wd, "splits.json")))
    assert sp["tasks"] == TASKS
    assert len(sp["eval_ids"]) == 30 and all(len(x) == 20 for x in sp["train_ids"])
    assert not set(sp["eval_ids"]) & {i for ids in sp["train_ids"] for i in ids}
    assert set(sp["yolo_train"]) <= set(sp["train_ids"][0]) | set(range(10 ** 6))
    for f in ["paths.json", "results/gate_clip.json", "results/gate_recall.json", "results/ablation.json",
              "results/ablation.md", "yolo_data/data.yaml", "cache/train/image_ids.npy", "cache/eval/image_ids.npy"]:
        assert os.path.exists(os.path.join(wd, f)), f
    res = json.load(open(os.path.join(wd, "results", "ablation.json")))
    assert list(res) == ["E0_zeroshot", "E5_ours", "E6_gt_ceiling"]
    assert res["E6_gt_ceiling"]["final_AP50"] > 0.95
    md = open(os.path.join(wd, "results", "ablation.md")).read()
    assert "E5_ours" in md and md.count("\n") >= 4
    # `run` == the same experiment assembled by hand from splits.json + the two caches (no wiring slips)
    pth = json.load(open(os.path.join(wd, "paths.json")))
    B = pl.build_bundle(FakeModels(9), os.path.join(wd, "cache", "train"), sp["train_ids"],
                        json.load(open(pth["ann_train"])), os.path.join(wd, "cache", "eval"), sp["eval_ids"],
                        json.load(open(pth["ann_val"])))
    by_hand = pl.run_ablation(B, sp["tasks"], only=["E0_zeroshot", "E5_ours", "E6_gt_ceiling"], log=lambda *_: None)
    assert same(json.loads(json.dumps(by_hand, default=pl._json_default)), res)
    # the train cache only holds the selected images (not eval images, not unselected ones)
    c = FeatureCache(os.path.join(wd, "cache", "train"))
    assert set(map(int, c.image_ids)) == {i for ids in sp["train_ids"] for i in ids}
    assert json.load(open(os.path.join(wd, "clip.json")))["dim"] == 27


def test_yolo_dataset_contains_only_base_classes_and_correct_boxes(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "10",
            "--n_yolo_train", "15", "--n_yolo_val", "8"]
    run_cli("fetch", *base); run_cli("make-yolo-data", *base)
    paths = json.load(open(os.path.join(wd, "paths.json")))
    sp = json.load(open(os.path.join(wd, "splits.json")))
    ct = json.load(open(paths["ann_train"]))
    by_id = {im["id"]: im for im in ct["images"]}
    root = os.path.join(wd, "yolo_data")
    assert open(os.path.join(root, "data.yaml")).read().count("person") >= 1
    n_obj = 0
    for iid in sp["yolo_train"]:
        fn = by_id[iid]["file_name"]
        assert os.path.exists(os.path.join(root, "images", "train", fn))           # symlink resolves
        lines = [l.split() for l in open(os.path.join(root, "labels", "train", fn.rsplit(".", 1)[0] + ".txt"))
                 if l.strip()]
        want = [a for a in ct["annotations"] if a["image_id"] == iid and cu.COCO_IDS.index(a["category_id"]) in TASKS[0]]
        assert len(lines) == len(want)
        W, H = by_id[iid]["width"], by_id[iid]["height"]
        for l, a in zip(sorted(lines, key=lambda r: float(r[1])), sorted(want, key=lambda a: a["bbox"][0] + a["bbox"][2] / 2)):
            assert int(l[0]) == cu.COCO_IDS.index(a["category_id"]) and int(l[0]) in TASKS[0]
            x, y, w, h = a["bbox"]
            np.testing.assert_allclose([float(v) for v in l[1:]], [(x + w / 2) / W, (y + h / 2) / H, w / W, h / H], atol=1e-5)
            n_obj += 1
    assert n_obj > 5


def test_selftest_leaves_no_files_in_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert rp.main(["selftest"]) == 0
    assert not os.path.exists(tmp_path / "work") and list(tmp_path.iterdir()) == []


def test_fetch_real_path_with_local_mini_coco(tmp_path):
    """Non-fake `fetch`: annotations found under coco_root, images 'downloaded' (file:// copy) into images/<split>2017."""
    src = str(tmp_path / "src")
    ct, cv, dtr, dva = make_image_world(src, TASKS, 12, 16, seed=5)
    root = str(tmp_path / "coco"); os.makedirs(os.path.join(root, "annotations"))
    json.dump(ct, open(os.path.join(root, "annotations", "instances_train2017.json"), "w"))
    json.dump(cv, open(os.path.join(root, "annotations", "instances_val2017.json"), "w"))
    for coco, d in ((ct, dtr), (cv, dva)):                        # coco_url must be file:// for the offline test
        for im in coco["images"]:
            im["coco_url"] = "file://" + os.path.join(d, im["file_name"])
    json.dump(ct, open(os.path.join(root, "annotations", "instances_train2017.json"), "w"))
    json.dump(cv, open(os.path.join(root, "annotations", "instances_val2017.json"), "w"))
    wd = str(tmp_path / "wd")
    run_cli("fetch", "--workdir", wd, "--coco_root", root, "--tasks", "0,1,2|3,4,5|6,7,8",
            "--n_train_per_task", "6", "--n_eval", "5", "--n_yolo_train", "4", "--n_yolo_val", "3")
    sp = json.load(open(os.path.join(wd, "splits.json")))
    pth = json.load(open(os.path.join(wd, "paths.json")))
    assert pth["img_dir_train"] == os.path.join(root, "images", "train2017")
    need_tr = {i for ids in sp["train_ids"] for i in ids} | set(sp["yolo_train"])
    got_tr = set(os.listdir(pth["img_dir_train"]))
    by_id = {im["id"]: im["file_name"] for im in ct["images"]}
    assert {by_id[i] for i in need_tr} == got_tr                  # exactly the needed images, nothing else
    assert len(os.listdir(pth["img_dir_val"])) == len(set(sp["eval_ids"]) | set(sp["yolo_val"]))
    assert [len(x) for x in sp["train_ids"]] == [6, 6, 6]
    # downstream stages work on the downloaded layout
    base = ["--workdir", wd, "--fake", "--tasks", "0,1,2|3,4,5|6,7,8", "--max_boxes", "50"]
    run_cli("extract", *base); run_cli("gate-recall", *base)
    run_cli("run", *base, "--only", "E5_ours")


# ------------------------------------------------ real-model wiring (stubs with the REAL signatures)
def test_train_yolo_command_uses_real_signature(tmp_path):
    import yolo_tools
    sig = inspect.signature(yolo_tools.train_base)
    calls = []
    wd = str(tmp_path / "wd")

    def stub(*args, **kw):
        bound = sig.bind(*args, **kw)              # raises TypeError if the CLI passes something train_base can't take
        calls.append(dict(bound.arguments))
        best = os.path.join(bound.arguments["project"], "base40", "weights", "best.pt")
        os.makedirs(os.path.dirname(best)); open(best, "w").write("w")
        return best

    with mock.patch.object(yolo_tools, "train_base", stub):
        run_cli("train-yolo", "--workdir", wd, "--yolo_init", "scratch", "--yolo_size", "n", "--yolo_epochs", "3",
                "--device", "0")
    c = calls[0]
    assert c["init"] == "scratch" and c["epochs"] == 3 and c["size"] == "n" and c["device"] == "0"
    assert c["data_yaml"] == os.path.join(wd, "yolo_data", "data.yaml")
    assert os.path.isabs(c["project"])
    rec = json.load(open(os.path.join(wd, "yolo_best.json")))
    assert rec["init"] == "scratch" and os.path.isabs(rec["weights"]) and os.path.exists(rec["weights"])


def test_train_yolo_finds_weights_ultralytics_put_elsewhere_and_fails_loudly_otherwise(tmp_path):
    import yolo_tools
    wd = str(tmp_path / "wd")

    def reroot(*args, **kw):                       # reports a path that does not exist; real file is under project/
        real = os.path.join(kw["project"], "deep", "weights", "best.pt")
        os.makedirs(os.path.dirname(real)); open(real, "w").write("w")
        return "/nonexistent/best.pt"

    with mock.patch.object(yolo_tools, "train_base", reroot):
        run_cli("train-yolo", "--workdir", wd)
    assert json.load(open(os.path.join(wd, "yolo_best.json")))["weights"].endswith(os.path.join("deep", "weights", "best.pt"))

    wd2 = str(tmp_path / "wd2")
    with mock.patch.object(yolo_tools, "train_base", lambda *a, **k: "/nonexistent/best.pt"):
        with pytest.raises(SystemExit, match="no best.pt"):
            rp.main(["train-yolo", "--workdir", wd2])
    assert not os.path.exists(os.path.join(wd2, "yolo_best.json"))


def test_extract_picks_up_trained_weights_for_real_models(tmp_path):
    """Non-fake extract builds RealModels with the weights recorded by train-yolo."""
    import real_models
    wd = str(tmp_path / "wd")
    run_cli("fetch", "--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "5", "--n_eval", "5")
    json.dump({"weights": "/w/best.pt", "init": "scratch"}, open(os.path.join(wd, "yolo_best.json"), "w"))
    seen = {}

    class Stub(FakeModels):
        def __init__(self, clip_pretrained, yolo_weights, device, conf, max_det, clip_model):
            super().__init__(9); seen.update(clip=clip_pretrained, yolo=yolo_weights, conf=conf, max_det=max_det, model=clip_model)

    with mock.patch.object(real_models, "RealModels", Stub):
        run_cli("extract", "--workdir", wd, "--clip_weights", "laion400m_e32", "--fake_classes", "9",
                "--conf", "0.02", "--max_det", "77")
    assert seen == {"clip": "laion400m_e32", "yolo": "/w/best.pt", "conf": 0.02, "max_det": 77, "model": "ViT-B-16"}


def test_real_models_constructor_matches_dependency_signatures():
    import clip_encoder, real_models, yolo_tools
    clip_cls = mock.create_autospec(clip_encoder.ClipEncoder)
    clip_cls.return_value.dim = 512
    yolo_cls = mock.create_autospec(yolo_tools.YoloProposer)
    with mock.patch.object(real_models, "ClipEncoder", clip_cls), mock.patch.object(real_models, "YoloProposer", yolo_cls):
        m = real_models.RealModels("laion400m_e32", "w.pt", device="cpu", conf=0.05, max_det=50)
        assert m.dim == 512
        clip_cls.assert_called_once_with(pretrained="laion400m_e32", model_name="ViT-B-16", device="cpu", fp16=True)
        yolo_cls.assert_called_once_with("w.pt", conf=0.05, iou=0.7, max_det=50, device="cpu")
        m.encode_text(["a"]); m.encode_images_u8(np.zeros((1, 8, 8, 3), np.uint8)); m.propose("img")
        clip_cls.return_value.encode_text.assert_called_once_with(["a"])
        yolo_cls.return_value.assert_called_once_with("img")
        no_yolo = real_models.RealModels("openai", None)
        with pytest.raises(RuntimeError):
            no_yolo.propose("img")


def test_repro_cifar_driver_with_stubbed_clip_and_cifar(tmp_path, monkeypatch):
    """Drive `repro-cifar` end to end on a tiny fake CIFAR: checks class-order plumbing (names, targets, text rows)."""
    import cifar_repro as cr
    import clip_encoder
    C, per_tr, per_te, D = 100, 8, 2, 100
    names = [f"c{i:03d}" for i in range(C)]

    class FakeCifar:
        def __init__(self, root, train=True, download=True):
            n = per_tr if train else per_te
            self.targets = [c for c in range(C) for _ in range(n)]
            self.data = np.zeros((len(self.targets), 32, 32, 3), np.uint8)
            for k, c in enumerate(self.targets):
                self.data[k, :, :, 0] = c                          # constant image whose red value IS the class id
            self.classes = names

        def __len__(self):
            return len(self.targets)

    tv = types.ModuleType("torchvision"); tv.datasets = types.SimpleNamespace(CIFAR100=FakeCifar)
    monkeypatch.setitem(sys.modules, "torchvision", tv)

    class StubClip:
        def __init__(self, pretrained="openai", model_name="ViT-B-16", device=None, fp16=True):
            self.pretrained = pretrained

        def encode_text(self, texts):                  # one-hot of the class; the 2nd template adds a distractor
            out = np.zeros((len(texts), D))
            for i, t in enumerate(texts):
                c = int([w for w in t.replace(".", "").split() if w.startswith("c") and w[1:].isdigit()][0][1:])
                out[i, c] = 1
                if t.startswith("a bad photo"):
                    out[i, (c + 1) % D] = 0.5
            return out

        def encode_images_u8(self, crops):                         # 20 * one-hot of the red value
            assert crops.shape[1:] == (224, 224, 3) and crops.dtype == np.uint8     # official 32->224 squash reaches the encoder
            out = np.zeros((len(crops), D))
            out[np.arange(len(crops)), crops[:, 0, 0, 0].astype(int)] = 20.0
            return out

    monkeypatch.setattr(clip_encoder, "ClipEncoder", StubClip)
    monkeypatch.setattr(cr, "load_official_text_assets", lambda repo, key="cifar224": (names, ["a photo of a {}.", "a bad photo of a {}."]))
    captured = {}
    orig = cr.run_cifar_protocol

    def spy(*args, **kw):
        captured["label"], captured["eval"] = args[4], args[5]
        return orig(*args, **kw)

    monkeypatch.setattr(cr, "run_cifar_protocol", spy)
    wd = str(tmp_path / "wd")
    run_cli("repro-cifar", "--workdir", wd, "--cifar_root", str(tmp_path / "data"), "--n2l_repo", "x", "--expand", "256")
    res = json.load(open(os.path.join(wd, "results", "repro_cifar.json")))
    assert len(res["fused"]) == 10
    assert res["A_bar_zeroshot"] == 1.0                            # text rows are in the shuffled order, targets remapped alike
    assert res["A_bar_fused"] > 0.95
    assert min(res["label_acc"]) == 1.0
    # label text = FIRST template only (exact one-hot); eval text = ensemble (has the distractor); rows in shuffled order
    order = cr.official_class_order()
    assert np.allclose(captured["label"], np.eye(D)[order])
    assert not np.allclose(captured["eval"], captured["label"])
    assert (captured["eval"].argmax(1) == np.array(order)).all()


# ================================================================== E. extra sensitivity tests
def test_gate_clip_within_task_restricts_the_label_space(world):
    """Class 4's own text is diluted (large background component) while class 0's text also fires on class 4's
    colour, so class-4 crops are lost in the 80-way test but fine inside task 1 (classes 3,4,5).
    A gate that ignores `class_subset` cannot see the difference."""
    class Confusable(FakeModels):
        def encode_text(self, texts):
            v = super().encode_text(texts)
            for i in range(len(texts)):
                if v[i].argmax() == 1:                       # class 0 (person): also fires on class 4's colour bin
                    v[i, 5] = 8.0
                elif v[i].argmax() == 5:                     # class 4 (airplane): mostly 'black'
                    v[i, 0] = 20.0
            return v

    r = pl.stage_gate_clip(Confusable(9), world["cv"], world["dva"], world["eval_ids"], TASKS)
    assert r["task1_within"] > 0.97
    assert r["all"] < r["task1_within"] - 0.02


def test_iter_images_returns_rgb_for_grayscale_and_palette_files(tmp_path):
    Image.fromarray(np.full((20, 30), 90, np.uint8)).save(tmp_path / "g.png")                 # mode L
    Image.fromarray(np.full((20, 30), 3, np.uint8), mode="P").save(tmp_path / "p.png")        # mode P
    coco = {"images": [{"id": 1, "file_name": "g.png"}, {"id": 2, "file_name": "p.png"}]}
    out = dict(pl.iter_images(coco, [2, 1], str(tmp_path)))
    for im in out.values():
        assert im.mode == "RGB" and im.size == (30, 20) and np.asarray(im).shape == (20, 30, 3)
    assert list(dict(pl.iter_images(coco, [2, 1], str(tmp_path)))) == [2, 1]                  # order as requested


def test_extract_knobs_reach_the_extractor(world, tmp_path):
    ids = world["tr_union"][:12]
    few = pl.stage_extract(world["models"], world["ct"], ids, world["dtr"], str(tmp_path / "few"), max_boxes=2)
    assert all(len(FeatureCache(few).image(i)["boxes"]) <= 2 for i in ids)
    gt = cu.gt_by_image(world["ct"], ids)

    def exact_peak(pad):
        d = pl.stage_extract(world["models"], world["ct"], ids, world["dtr"], str(tmp_path / f"p{pad}"), pad_frac=pad)
        c, peaks = FeatureCache(d), []
        for i in ids:
            gb, gc = gt[i]
            if not len(gb):
                continue
            im = c.image(i)
            iou = iou_matrix(gb, im["boxes"])
            for g in range(len(gb)):
                j = iou[g].argmax()
                if iou[g, j] > 0.97:
                    peaks.append(float(im["feats"][j].astype(float)[gc[g] + 1]))
        return np.mean(peaks)

    assert exact_peak(0.0) > 9.9              # no padding: crop is pure object colour (histogram mass 1 * 10)
    assert exact_peak(0.25) < 0.6 * 10        # 25% padding each side: object is only 1/1.5^2 of the crop


def test_selftest_fails_when_the_pipeline_is_broken(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def roll(self, texts):
        v = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            hit = [c for c, p in enumerate(self._pat) if p.search(t)]
            v[i, ((hit[0] + 1) % 9 + 1) if hit else 0] = 7.0         # every class name points at the wrong colour
        return v

    monkeypatch.setattr(FakeModels, "encode_text", roll)
    with pytest.raises(SystemExit):
        rp.main(["selftest"])


def test_selftest_checks_each_criterion_can_fail_alone():
    ok_res = {"E0_zeroshot": {"final_AP50": 0.95}, "E5_ours": {"final_AP50": 0.95}, "E6_gt_ceiling": {"final_AP50": 0.95}}
    ok_rec = {f"task{t}": {"recall@0.5": 0.99} for t in range(3)}
    base = rp.selftest_checks(ok_res, 0.99, ok_rec)
    assert len(base) == 5 and all(base.values())
    for key, tweak in [("E0_zeroshot", 0.1), ("E5_ours", 0.1), ("E6_gt_ceiling", 0.1)]:
        r = copy.deepcopy(ok_res); r[key]["final_AP50"] = tweak
        bad = rp.selftest_checks(r, 0.99, ok_rec)
        assert sum(not v for v in bad.values()) == 1, key
    bad = rp.selftest_checks(ok_res, 0.5, ok_rec)
    assert sum(not v for v in bad.values()) == 1
    r2 = copy.deepcopy(ok_rec); r2["task2"]["recall@0.5"] = 0.2
    assert sum(not v for v in rp.selftest_checks(ok_res, 0.99, r2).values()) == 1


def _fake_workdir(tmp_path, extra=()):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "12", "--n_eval", "10", *extra]
    run_cli("fetch", *base); run_cli("extract", *base)
    return wd, base


def test_run_refuses_a_stale_cache_after_refetch(tmp_path):
    wd, base = _fake_workdir(tmp_path)
    run_cli("fetch", *base, "--n_train_per_task", "20")      # more images wanted than the cache holds
    with pytest.raises(SystemExit, match="lacks"):
        rp.main(["run", *base, "--only", "E0_zeroshot"])


def test_gate_recall_refuses_eval_ids_missing_from_cache(tmp_path):
    wd, base = _fake_workdir(tmp_path)
    f = os.path.join(wd, "splits.json")
    sp = json.load(open(f)); sp["eval_ids"].append(999999); json.dump(sp, open(f, "w"))
    with pytest.raises(SystemExit, match="lacks 1 of the"):
        rp.main(["gate-recall", *base])


def test_run_requires_extract_first(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "5", "--n_eval", "5"]
    run_cli("fetch", *base)
    with pytest.raises(SystemExit, match="extract"):
        rp.main(["run", *base])


def test_clip_weight_mismatch_between_extract_and_run_is_refused(tmp_path):
    wd, base = _fake_workdir(tmp_path)
    json.dump({"clip_weights": "openai", "dim": 27}, open(os.path.join(wd, "clip.json"), "w"))
    nonfake = [x for x in base if x != "--fake"]                      # simulate a real run's consistency check
    a = rp.build_parser().parse_args(["run", *nonfake, "--clip_weights", "laion400m_e32"])
    with pytest.raises(SystemExit, match="openai"):
        rp.check_clip_consistency(a)
    rp.check_clip_consistency(rp.build_parser().parse_args(["run", *nonfake, "--clip_weights", "openai"]))   # same -> ok


def test_build_bundle_rejects_dimension_mismatch(world):
    class Wide(FakeModels):
        dim = 40
        def encode_text(self, texts):
            return np.pad(super().encode_text(texts), ((0, 0), (0, 13)))

    with pytest.raises(ValueError, match="same CLIP"):
        pl.build_bundle(Wide(9), world["tr_cache"], world["train_ids"], world["ct"], world["ev_cache"],
                        world["eval_ids"], world["cv"])


# ================================================================== F. open_clip activation guard
class _StubOpenClip:
    """Mimics open_clip: warns when force_quick_gelu disagrees with what the pretrained tag was trained with."""
    def __init__(self, tag_qg, message=None):
        self.tag_qg, self.calls, self.message = tag_qg, [], message

    def create_model_and_transforms(self, model_name, pretrained=None, force_quick_gelu=None):
        import warnings
        self.calls.append(force_quick_gelu)
        qg = bool(force_quick_gelu)
        if qg != self.tag_qg:
            warnings.warn(self.message or f"QuickGELU mismatch between final model config (quick_gelu={qg}) and "
                          f"pretrained tag '{pretrained}' (quick_gelu={self.tag_qg}).", UserWarning)
        return ("model", None, None)


def test_create_clip_rebuilds_with_the_activation_the_weights_expect():
    import clip_encoder as ce
    oc = _StubOpenClip(tag_qg=True)                                       # like 'openai' on plain ViT-B-16
    assert ce.create_clip(oc, "ViT-B-16", "openai")[0] == "model"
    assert oc.calls == [None, True]                                       # first plain, then forced QuickGELU


def test_create_clip_single_call_when_no_mismatch():
    import clip_encoder as ce
    for tag_qg in (False,):
        oc = _StubOpenClip(tag_qg=tag_qg)
        ce.create_clip(oc, "ViT-B-16", "laion400m_e32")
        assert oc.calls == [None]


def test_create_clip_rebuilds_to_false_when_tag_says_plain_gelu():
    import clip_encoder as ce

    class Odd(_StubOpenClip):                                             # default build is QuickGELU, tag wants GELU
        def create_model_and_transforms(self, model_name, pretrained=None, force_quick_gelu=None):
            import warnings
            self.calls.append(force_quick_gelu)
            qg = True if force_quick_gelu is None else force_quick_gelu
            if qg != self.tag_qg:
                warnings.warn(f"QuickGELU mismatch between final model config (quick_gelu={qg}) and pretrained tag "
                              f"'{pretrained}' (quick_gelu={self.tag_qg}).", UserWarning)
            return ("model", None, None)

    oc = Odd(tag_qg=False)
    ce.create_clip(oc, "ViT-B-16-quickgelu", "x")
    assert oc.calls == [None, False]


def test_create_clip_raises_if_mismatch_cannot_be_resolved():
    import clip_encoder as ce
    with pytest.raises(RuntimeError, match="unparseable"):
        ce.create_clip(_StubOpenClip(True, message="QuickGELU mismatch: something odd"), "m", "p")

    class Stubborn(_StubOpenClip):
        def create_model_and_transforms(self, model_name, pretrained=None, force_quick_gelu=None):
            import warnings
            self.calls.append(force_quick_gelu)
            warnings.warn("QuickGELU mismatch between final model config (quick_gelu=False) and pretrained tag "
                          "'p' (quick_gelu=True).", UserWarning)
            return ("model", None, None)

    with pytest.raises(RuntimeError, match="persists"):
        ce.create_clip(Stubborn(True), "m", "p")


def test_create_clip_passes_other_warnings_through():
    import clip_encoder as ce
    import warnings

    class Noisy(_StubOpenClip):
        def create_model_and_transforms(self, *a, **k):
            warnings.warn("something unrelated", UserWarning)
            return ("model", None, None)

    with pytest.warns(UserWarning, match="unrelated"):
        ce.create_clip(Noisy(False), "m", "p")

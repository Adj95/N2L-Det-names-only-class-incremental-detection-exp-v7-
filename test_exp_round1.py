"""N2L-Det-exp round 1: rescoring knobs (off by default = frozen behaviour), the rescoring variants, and the box diagnostics."""
import dataclasses
import json
import os

import numpy as np
import pytest

import boxdiag as bd
import incremental as inc
import run_pipeline as rp
import tune as tn
import variants as vr
from boxes import batched_nms
from inference import InferConfig, detect
from test_pipeline import TASKS, bundle, world  # noqa: F401
from test_tune import _tune_dict


def _frozen_detect(boxes, yolo_scores, fused, class_order, cfg):
    """The frozen v6 detect(), copied verbatim (minus the new knobs): the reference the defaults must reproduce."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    fg = np.asarray(fused, dtype=np.float64)[:, 1:]
    n, K = fg.shape
    k = min(cfg.topk_classes, K)
    cols = np.argsort(-fg, axis=1, kind="stable")[:, :k]
    sc = np.take_along_axis(fg, cols, axis=1)
    if cfg.yolo_gamma > 0:
        sc = sc * (np.clip(np.asarray(yolo_scores, dtype=np.float64), 1e-6, 1.0) ** cfg.yolo_gamma)[:, None]
    bi = np.repeat(np.arange(n), k)
    cols, sc = cols.reshape(-1), sc.reshape(-1)
    ok = sc > cfg.min_score
    bi, cols, sc = bi[ok], cols[ok], sc[ok]
    labels = np.asarray(class_order, dtype=np.int64)[cols]
    keep = batched_nms(boxes[bi], sc, labels, cfg.nms_iou, max_keep=cfg.max_dets)
    return boxes[bi][keep], sc[keep], labels[keep]


def _rand_dets(n=60, K=6, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (n, 2)); wh = rng.uniform(5, 40, (n, 2))
    boxes = np.hstack([xy, xy + wh])
    fused = rng.dirichlet(np.ones(K + 1), n)
    return boxes, rng.uniform(0, 1, n), fused, [10, 11, 12, 20, 21, 22]


@pytest.mark.parametrize("gamma", [0.0, 0.25, 0.5])
def test_default_knobs_reproduce_the_frozen_detect_exactly(gamma):
    b, s, f, order = _rand_dets()
    cfg = InferConfig(yolo_gamma=gamma, topk_classes=3)
    got = detect(b, s, f, order, cfg, base_classes=[10, 11, 12])
    ref = _frozen_detect(b, s, f, order, cfg)
    for g, r in zip(got, ref):
        assert np.array_equal(g, r)


def test_gamma_new_equal_to_gamma_is_a_no_op_and_needs_base_classes():
    b, s, f, order = _rand_dets(seed=1)
    cfg = InferConfig(yolo_gamma=0.25, topk_classes=6)
    same = detect(b, s, f, order, dataclasses.replace(cfg, gamma_new=0.25), base_classes=[10, 11, 12])
    for g, r in zip(same, detect(b, s, f, order, cfg)):
        assert np.allclose(g, r)
    with pytest.raises(ValueError):
        detect(b, s, f, order, dataclasses.replace(cfg, gamma_new=0.0))


def test_gamma_new_changes_only_new_class_scores():
    b = np.array([[0, 0, 10, 10.0]]); s = np.array([0.04])
    f = np.array([[0.0, 0.5, 0.5]])                                   # one box, base class 10 and new class 20 tied
    cfg = InferConfig(yolo_gamma=0.5, topk_classes=2, gamma_new=0.0)
    _, sc, lab = detect(b, s, f, [10, 20], cfg, base_classes=[10])
    d = dict(zip(lab.tolist(), sc.tolist()))
    assert d[10] == pytest.approx(0.5 * 0.04 ** 0.5) and d[20] == pytest.approx(0.5)


def test_clip_obj_and_yolo_floor():
    b = np.array([[0, 0, 10, 10.0]]); s = np.array([0.001])
    f = np.array([[0.75, 0.25]])
    _, sc, _ = detect(b, s, f, [10], InferConfig(yolo_gamma=0.0, clip_obj=1.0))
    assert sc[0] == pytest.approx(0.25 * 0.25)
    _, sc, _ = detect(b, s, f, [10], InferConfig(yolo_gamma=1.0, yolo_floor=0.1))
    assert sc[0] == pytest.approx(0.25 * 0.1)
    _, sc, _ = detect(b, s, f, [10], InferConfig(yolo_gamma=1.0))
    assert sc.size == 0                                               # 0.25 * 0.001 < min_score: dropped without the floor


def test_old_tune_json_without_the_new_keys_still_loads_with_knobs_off():
    t = _tune_dict()
    for g in ("E5",):
        for k in ("gamma_new", "clip_obj", "yolo_floor"):
            t[g]["infer"].pop(k, None)
    for k in ("gamma_new", "clip_obj", "yolo_floor"):
        t["E0"]["infer"].pop(k, None)
    cf = tn.tuned_configs(t, TASKS)
    for n in ("E5_tuned", "E0_zeroshot_tuned"):
        assert (cf[n].infer.gamma_new, cf[n].infer.clip_obj, cf[n].infer.yolo_floor) == (-1.0, 0.0, 0.0)
    assert tn.changed_settings(t) == []


def test_rescoring_variants_change_only_inference_for_e5_and_e0():
    t = _tune_dict(infer__yolo_gamma=0.25, n2l__n_rounds=0)
    cfgs = vr.variant_configs(TASKS, t)
    for name in vr.RESCORE:
        for base, n in (("E5_tuned", name), ("E0_zeroshot_tuned", vr.E0_PREFIX + name)):
            a, b = cfgs[base], cfgs[n]
            assert dataclasses.replace(b, infer=a.infer) == a and b.infer != a.infer


def test_run_with_default_knobs_equals_the_frozen_scoring(bundle):
    """The new base_classes plumbing must not change any score when the knobs are off."""
    cfg = inc.RunConfig(tasks=TASKS, infer=InferConfig(yolo_gamma=0.25, topk_classes=3))
    box = {}

    def probe(ctx):
        if ctx["task"] == len(TASKS) - 1:
            box.update(ctx)

    r = inc.run_incremental(bundle, dataclasses.replace(cfg, n_folds=0), log=lambda *_: None, probe=probe)
    Xe, e_img, Be, Se = box["eval"]
    dets = []
    from evaluate import coco_eval, dets_to_coco
    for k, iid in enumerate(bundle.eval_ids):
        m = e_img == k
        dets += dets_to_coco(iid, *_frozen_detect(Be[m], Se[m], box["fused"][m], box["class_order"], cfg.infer))
    ref = coco_eval(bundle.eval_gt, dets, bundle.eval_ids, box["class_order"])
    assert r["final_AP50"] == ref["AP50"] and r["final_AP"] == ref["AP"]


# ------------------------------------------------------------------ coverage
def test_coverage_counts_topk_hits_scores_ranks_and_sizes():
    tasks = [[0], [1]]
    gt = {1: (np.array([[0, 0, 100, 100.0], [200, 200, 210, 210.0]]), np.array([0, 1]))}
    props = {1: (np.array([[0, 0, 100, 100.0],        # covers the base object, top score
                           [0, 0, 50, 50.0],          # background-ish
                           [201, 201, 210, 210.0]]),  # covers the small new object (IoU .81), lowest score
                 np.array([0.9, 0.5, 0.02]))}
    c = bd.coverage(props, gt, tasks, ks=(1, 2, 5))
    assert c["ks"] == ["1", "2", "all"]                                # K >= number of boxes collapses into 'all'
    base, new = c["groups"]["base"], c["groups"]["new"]
    assert base["recall@0.5"] == {"1": 1.0, "2": 1.0, "all": 1.0}
    assert new["recall@0.5"] == {"1": 0.0, "2": 0.0, "all": 1.0} and new["recall@0.7"]["all"] == 1.0
    assert new["best_cover_score"]["median"] == pytest.approx(0.02) and new["best_cover_rank"]["median"] == 2
    assert new["best_cover_score"]["frac_below_0.05"] == 1.0
    assert new["recall@0.5_by_size"]["small"] == {"n": 1, "recall": 1.0} and base["recall@0.5_by_size"]["large"]["n"] == 1


def test_coverage_with_no_proposals_and_unused_classes():
    tasks = [[0], [1]]
    gt = {1: (np.array([[0, 0, 10, 10.0], [0, 0, 20, 20.0]]), np.array([1, 5]))}     # class 5 is in no task: ignored
    c = bd.coverage({1: (np.zeros((0, 4)), np.zeros(0))}, gt, tasks)
    assert c["groups"]["new"]["n_gt"] == 1 and c["groups"]["new"]["recall@0.5"]["all"] == 0.0
    assert c["groups"]["base"] == {"n_gt": 0} and "best_cover_score" not in c["groups"]["new"]


def test_groups_of_multi_step():
    g = bd.groups_of([[0, 1], [2], [3, 4]])
    assert g == {"base": [0, 1], "new": [2, 3, 4], "task1": [2], "task2": [3, 4]}
    assert set(bd.groups_of([[0], [1]])) == {"base", "new"}


# ------------------------------------------------------------------ oracle ladder + naming
def test_oracle_ladder_is_monotone_and_naming_sums_to_one(bundle):
    t = _tune_dict(infer__yolo_gamma=0.25, n2l__n_rounds=0)
    cfg = tn.tuned_configs(t, TASKS)
    for name in ("E5_tuned", "E0_zeroshot_tuned"):
        o = bd.oracle_ladder(bundle, cfg[name], n_folds=0)
        ref = inc.run_incremental(bundle, dataclasses.replace(cfg[name], n_folds=0), log=lambda *_: None)
        assert o["ref"]["AP50"] == pytest.approx(ref["final_AP50"])     # 'ref' is exactly the normal run
        assert o["O2_no_bg"]["AP50"] >= o["O1_names"]["AP50"] - 1e-9
        assert o["O3_rank"]["AP50"] >= o["O2_no_bg"]["AP50"] - 1e-9
        for g, s in o["naming"].items():
            if s.get("n"):
                assert s["right"] + s["wrong_same_group"] + s["wrong_other_group"] == pytest.approx(1.0)


# ------------------------------------------------------------------ CLI
def _base(wd):
    return ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30", "--n_tune", "20",
            "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]


@pytest.fixture(scope="module")
def wd_tuned(tmp_path_factory):
    wd = str(tmp_path_factory.mktemp("exp") / "wd")
    base = _base(wd)
    for c in ("fetch", "make-yolo-data", "extract"):
        assert rp.main([c, *base]) == 0
    assert rp.main(["tune", *base, "--n_trials", "3"]) == 0
    return wd, base


def test_boxdiag_cli_on_tune_and_eval_never_final(wd_tuned, capsys):
    wd, base = wd_tuned
    tj = os.path.join(wd, "results", "tune.json")
    before = open(tj, "rb").read()
    for split in ("tune", "eval"):
        assert rp.main(["boxdiag", *base, "--eval_on", split, "--n_folds", "2"]) == 0
        r = json.load(open(os.path.join(wd, "results", f"boxdiag_{split}.json")))
        assert set(r["oracle"]) == {"E5_tuned", "E0_zeroshot_tuned"} and "groups" in r["coverage"]
        assert os.path.exists(os.path.join(wd, "results", f"boxdiag_{split}.md"))
    assert "Oracle ladder" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        rp.main(["boxdiag", *base, "--eval_on", "final"])
    assert open(tj, "rb").read() == before                            # read-only for the tuned settings


def test_propdiag_cli_fake(wd_tuned):
    wd, base = wd_tuned
    assert rp.main(["propdiag", *base, "--nms_ious", "0.7", "0.9", "--tag", "t"]) == 0
    r = json.load(open(os.path.join(wd, "results", "propdiag_t.json")))
    assert set(r) == {"yolo_nms0.7_tune", "yolo_nms0.7_eval", "yolo_nms0.9_tune", "yolo_nms0.9_eval"}
    assert r["yolo_nms0.7_tune"]["settings"]["split"] == "tune" and r["yolo_nms0.7_eval"]["coverage"]["n_images"] == 30
    with pytest.raises(SystemExit):
        rp.main(["propdiag", *base, "--splits", "final"])


def test_variants_cli_runs_rescoring_rows_for_e5_and_e0(wd_tuned):
    wd, base = wd_tuned
    only = ["gnew_0", "clipobj_1", vr.E0_PREFIX + "gnew_0", vr.E0_PREFIX + "clipobj_1"]
    assert rp.main(["variants", *base, "--only", *only, "--n_folds", "2", "--tag", "r1"]) == 0
    md = open(os.path.join(wd, "results", "variants_r1.md")).read()
    assert "applied to zero-shot E0" in md and "E0__gnew_0" in md


def test_marker_present():
    src = open(os.path.join(os.path.dirname(rp.__file__), "run_pipeline.py")).read()
    assert "N2L_EXP_MARKER_v7" in src


FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "n2l_det_frozen")


@pytest.mark.skipif(not os.path.isdir(FROZEN), reason="frozen v6 copy not next to this package (only on the dev machine)")
def test_frozen_package_gives_identical_selftest_results(tmp_path):
    """Whole fake pipeline with the frozen v6 code and with this code: identical ablation numbers."""
    import subprocess
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for name, pkg in (("frozen", FROZEN), ("exp", here)):
        wd = str(tmp_path / name)
        subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {pkg!r}); import run_pipeline as rp; "
                        f"rp.main(['selftest', '--workdir', {wd!r}])"], check=True, cwd=str(tmp_path), capture_output=True)
        out[name] = json.load(open(os.path.join(wd, "results", "ablation.json")))
    # exp v7 only ADDS fields (per-class AP50 per step, forgetting); every number the frozen code reports must be identical
    row = lambda c: {k: v for k, v in c.items() if k != "per_class_AP50"}
    strip = lambda r: {n: {k: ([row(c) for c in v] if k == "curve" else v) for k, v in x.items() if k not in ("diag", "forgetting")}
                       for n, x in r.items()}
    assert json.dumps(strip(out["frozen"]), sort_keys=True) == json.dumps(strip(out["exp"]), sort_keys=True)


# ------------------------------------------------------------------ the experimental Kaggle cells
def _exp_blocks():
    import re
    txt = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "KAGGLE_EXP.md")).read()
    return re.findall(r"```\n(.*?)```", txt, flags=re.S)


def test_exp_kaggle_cells_parse_never_touch_the_final_split_and_pick_only_the_exp_package():
    import re
    import shlex
    blocks = _exp_blocks()
    cells = [b for b in blocks if "run_pipeline.py" in b]
    assert len(cells) == 4                                                   # R1-A, R1-B, R1-C, R1-D
    for b in cells:
        assert "grep -l N2L_EXP_MARKER_v7" in b and "cmd_variants" not in b
        assert "after_final_split.tar" not in b and "--eval_on final" not in b and "--allow_repeat_final" not in b
        assert b.rstrip().endswith("ls -lh /kaggle/working/*.tar") and "rm -rf /kaggle/working/n2l_det_exp\n" in b
        for line in b.splitlines():
            line = line.strip()
            if not line.startswith("python run_pipeline.py"):
                continue
            argv = shlex.split(re.split(r"\s+\|\|?\s+", line)[0])
            a = rp.build_parser().parse_args(argv[2:])
            if a.cmd == "extract":
                assert a.splits and "final" not in a.splits and a.pad == 0.0 and a.max_det == 300 and a.conf == 0.001
            if a.cmd == "variants":
                assert a.on == "tune" and set(a.only) <= set(vr.variant_configs(TASKS, _tune_dict()))
            if a.cmd in ("boxdiag",):
                assert a.eval_on in ("tune", "eval")
            if a.cmd == "propdiag":
                assert set(a.splits) == {"tune", "eval"} and a.max_det == 300 and a.conf == 0.001
            if a.cmd == "fetch" and a.tasks == "10+2x5":
                assert a.n_final == 2000 and a.n_tune == 500 and a.n_eval == 2000          # same image pools as 15+5
    r1a = [b for b in cells if "BOXDIAG TUNE" in b][0]
    assert "work_voc_pad00_after_followup.tar" in r1a and "cache/final" in r1a and "tune.json unchanged" in r1a

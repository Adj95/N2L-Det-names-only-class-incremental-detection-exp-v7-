"""The untouched final split: disjoint, stable, single-use; and the YOLO validation images never overlap scored images."""
import json
import os

import pytest

import run_pipeline as rp


def _base(wd, extra=()):
    return ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30", "--n_tune", "20",
            "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50", *extra]


def _splits(wd):
    return json.load(open(os.path.join(wd, "splits.json")))


def test_draw_splits_disjoint_and_earlier_splits_do_not_depend_on_later_ones():
    pool = list(range(1000, 1400))
    e0, t0, f0 = rp.draw_splits(pool, 7, 100, 0, 0)
    e1, t1, f1 = rp.draw_splits(pool, 7, 100, 50, 0)
    e2, t2, f2 = rp.draw_splits(pool, 7, 100, 50, 80)
    assert e0 == e1 == e2 and t1 == t2 and t0 == [] and f0 == f1 == [] and len(f2) == 80      # existing caches stay valid
    E, T, F = set(e2), set(t2), set(f2)
    assert len(E) == 100 and len(T) == 50 and not (E & T) and not (E & F) and not (T & F) and (E | T | F) <= set(pool)
    assert rp.draw_splits(pool, 7, 100, 50, 80) == (e2, t2, f2)                                  # deterministic
    assert rp.draw_splits(pool, 8, 100, 50, 80)[2] != f2                                         # depends on the seed
    e, t, f = rp.draw_splits(list(range(10)), 0, 6, 6, 6)                                        # runs out of images: smaller, still disjoint
    assert len(e) == 6 and len(t) == 4 and f == [] and not (set(e) & set(t))


def test_fetch_stores_final_ids(tmp_path):
    b = str(tmp_path / "b")
    rp.main(["fetch", *_base(b, ["--n_final", "25"])])
    sb = _splits(b)
    assert len(sb["final_ids"]) == 25
    E, T, F = set(sb["eval_ids"]), set(sb["tune_ids"]), set(sb["final_ids"])
    assert not (E & T) and not (E & F) and not (T & F)


def test_yolo_val_images_are_never_scored_images(tmp_path):
    wd = str(tmp_path / "wd")
    rp.main(["fetch", *_base(wd, ["--n_final", "25"])])
    s = _splits(wd)
    assert s["yolo_val"] and not (set(s["yolo_val"]) & (set(s["eval_ids"]) | set(s["tune_ids"]) | set(s["final_ids"])))


def _prepare(wd, n_final=25):
    base = _base(wd, ["--n_final", str(n_final)])
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *base])
    rp.main(["tune", *base, "--n_trials", "3"])
    return base


def test_extract_builds_the_final_cache_by_default_and_run_final_is_single_use(tmp_path, capsys):
    wd = str(tmp_path / "wd")
    base = _prepare(wd)
    assert os.path.isdir(os.path.join(wd, "cache", "final"))
    tj = os.path.join(wd, "results", "tune.json")
    with pytest.raises(SystemExit, match="needs --tuned"):
        rp.main(["run", *base, "--eval_on", "final", "--only", "E0_zeroshot"])
    assert not os.path.exists(os.path.join(wd, "results", "final_log.json"))             # a refused call is not a use
    rp.main(["run", *base, "--eval_on", "final", "--tuned", tj, "--only", "E0_zeroshot_tuned", "E5_tuned"])
    assert os.path.exists(os.path.join(wd, "results", "ablation_final.json"))
    assert not os.path.exists(os.path.join(wd, "results", "ablation.json"))
    assert len(json.load(open(os.path.join(wd, "results", "final_log.json")))) == 1
    with pytest.raises(SystemExit, match="already used"):
        rp.main(["run", *base, "--eval_on", "final", "--tuned", tj, "--only", "E5_tuned"])
    rp.main(["run", *base, "--eval_on", "final", "--tuned", tj, "--only", "E5_tuned", "--allow_repeat_final"])
    log = json.load(open(os.path.join(wd, "results", "final_log.json")))
    assert [x["repeat"] for x in log] == [False, True]
    assert "REPEATED use of the final split" in open(os.path.join(wd, "results", "ablation_final.md")).read()


def test_final_split_scores_the_final_images_not_the_eval_images(tmp_path, monkeypatch):
    import pipeline as pl
    wd = str(tmp_path / "wd")
    base = _prepare(wd)
    seen = {}
    orig = pl.build_bundle
    monkeypatch.setattr(pl, "build_bundle", lambda m, tc, ti, ct, ec, eids, cv: seen.setdefault("ids", list(eids)) and orig(m, tc, ti, ct, ec, eids, cv))
    rp.main(["run", *base, "--eval_on", "final", "--tuned", os.path.join(wd, "results", "tune.json"), "--only", "E5_tuned"])
    assert seen["ids"] == _splits(wd)["final_ids"]


def test_final_split_missing_and_diagnose_refused(tmp_path):
    wd = str(tmp_path / "wd")
    base = _base(wd)
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *base])
    with pytest.raises(SystemExit, match="--n_final"):
        rp.main(["run", *base, "--eval_on", "final", "--tuned", "x"])
    with pytest.raises(SystemExit, match="--n_final"):
        rp.main(["extract", *base, "--splits", "final"])
    base2 = _base(wd, ["--n_final", "10"])
    rp.main(["fetch", *base2]); rp.main(["extract", *base2, "--splits", "final"])
    with pytest.raises(SystemExit, match="not allowed"):
        rp.main(["diagnose", *base2, "--eval_on", "final"])


def test_failed_final_run_is_not_recorded_as_a_use(tmp_path):
    wd = str(tmp_path / "wd")
    base = _prepare(wd)
    with pytest.raises(SystemExit, match="not found"):
        rp.main(["run", *base, "--eval_on", "final", "--tuned", str(tmp_path / "nope.json")])
    assert not os.path.exists(os.path.join(wd, "results", "final_log.json"))


def test_tiny_pool_falls_back_loudly_and_records_the_overlap(tmp_path, capsys):
    import test_voc
    argv, w = test_voc.fetch_args(tmp_path)                       # the toy VOC fixture: eval takes every val image
    rp.main(argv)
    s = json.load(open(w / "splits.json"))
    out = capsys.readouterr().out
    assert s["yolo_val_overlaps_scored"] is True and s["yolo_val"] and "WARNING: every validation image" in out
    assert set(s["yolo_val"]) <= set(s["eval_ids"])


def test_partial_pool_warns_but_stays_disjoint(tmp_path, capsys):
    import test_voc
    argv, w = test_voc.fetch_args(tmp_path)
    argv[argv.index("--n_eval") + 1] = "1"                        # leave one val image for the detector
    argv[argv.index("--n_yolo_val") + 1] = "5"
    rp.main(argv)
    s = json.load(open(w / "splits.json"))
    assert s["yolo_val_overlaps_scored"] is False and not (set(s["yolo_val"]) & set(s["eval_ids"]))
    assert "WARNING: only" in capsys.readouterr().out


def test_eval_and_tune_ids_are_exactly_what_the_original_fetch_drew():
    """Re-running `fetch` on an existing work dir (new package, same arguments) must not change eval_ids / tune_ids:
    the feature caches and tune.json already on disk refer to them. This is the ORIGINAL formula, copied verbatim."""
    import numpy as np
    pool = sorted(range(5000, 9952))
    for seed in (0, 1, 7):
        rng = np.random.default_rng(seed)
        eval_ids = sorted(rng.choice(pool, size=min(2000, len(pool)), replace=False).tolist())
        spare = sorted(set(pool) - set(eval_ids))
        tune_ids = sorted(np.random.default_rng(seed + 3).choice(spare, size=min(500, len(spare)), replace=False).tolist())
        e, t, f = rp.draw_splits(pool, seed, 2000, 500, 2000)
        assert e == eval_ids and t == tune_ids and len(f) == 2000


def test_refetch_keeps_the_detectors_real_validation_images_and_keeps_them_out_of_final(tmp_path, capsys):
    """The VOC detector was trained before this rule existed: its validation images overlap eval. A later fetch (adding the
    final split) must keep that record honest and must not hand those images to the final split."""
    wd = str(tmp_path / "wd")
    base = _base(wd, ["--n_final", "10"])                                    # the fake world's size depends on --n_final: keep it fixed
    rp.main(["fetch", *base])
    s = _splits(wd)
    old_val = sorted(s["eval_ids"][:6] + s["tune_ids"][:2] + s["final_ids"][:4])   # what an old fetch could have drawn
    s["yolo_val"] = old_val
    json.dump(s, open(os.path.join(wd, "splits.json"), "w"))
    capsys.readouterr()
    rp.main(["fetch", *base])
    out = capsys.readouterr().out
    t = _splits(wd)
    assert t["yolo_val"] == old_val and t["yolo_val_overlaps_scored"] is True
    assert "NOTE: 8 of the detector's 12 validation images" in out                 # only eval/tune overlaps are counted here
    assert not (set(t["final_ids"]) & set(old_val)) and set(s["final_ids"][:4]) <= set(old_val) and t["eval_ids"] == s["eval_ids"] and t["tune_ids"] == s["tune_ids"]
    # a work dir with a DIFFERENT detector-training set is a new detector: validation images are drawn afresh
    t["yolo_train"] = t["yolo_train"][:-1]
    json.dump(t, open(os.path.join(wd, "splits.json"), "w"))
    rp.main(["fetch", *base])
    u = _splits(wd)
    assert u["yolo_val"] != old_val and u["yolo_val_overlaps_scored"] is False
    assert not (set(u["yolo_val"]) & (set(u["eval_ids"]) | set(u["tune_ids"]) | set(u["final_ids"])))


def test_draw_splits_exclude_final_only_touches_the_final_split():
    pool = list(range(1000, 1400))
    e, t, f = rp.draw_splits(pool, 3, 100, 50, 80)
    e2, t2, f2 = rp.draw_splits(pool, 3, 100, 50, 80, exclude_final=f[:40] + [e[0]])
    assert (e, t) == (e2, t2) and not (set(f2) & set(f[:40])) and len(f2) == 80
    assert rp.draw_splits(pool, 3, 100, 50, 80, exclude_final=())[2] == f


def test_final_split_shortfall_is_reported(tmp_path, capsys):
    import test_voc
    argv, w = test_voc.fetch_args(tmp_path, "--n_final", "5")             # the toy VOC pool is exhausted by eval
    rp.main(argv)
    out = capsys.readouterr().out
    assert "WARNING: only 0 of the requested 5 final images are available" in out
    assert json.load(open(w / "splits.json"))["final_ids"] == []

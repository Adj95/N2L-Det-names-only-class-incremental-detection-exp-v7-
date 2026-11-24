"""exp v7: per-class AP50 at every step, the pre-registered forgetting measures, and variants on the final split."""
import json
import math
import os

import pytest

import incremental as inc


def _curve(rows):
    return [{"per_class_AP50": r} for r in rows]


def test_forgetting_by_hand_three_tasks():
    tasks = [[0, 1], [2], [3]]
    curve = _curve([{0: .8, 1: .6},
                    {0: .9, 1: .5, 2: .4},
                    {0: .7, 1: .5, 2: .1, 3: .3}])
    f = inc.forgetting(curve, tasks)
    assert f["old_class"] == pytest.approx(((.8 - .7) + (.6 - .5)) / 2)          # task-1 classes: step 1 minus last
    assert f["by_task"][0] == pytest.approx(((.9 - .7) + (.6 - .5)) / 2)          # best earlier minus last
    assert f["by_task"][1] == pytest.approx(.4 - .1)
    assert f["avg"] == pytest.approx(((.9 - .7) + (.6 - .5) + (.4 - .1)) / 3)     # every class learned before the last task
    assert len(f["by_task"]) == 2                                                  # the last task is never "forgotten"


def test_forgetting_skips_nan_classes_and_handles_string_keys_and_one_task():
    tasks = [[0, 1], [2]]
    curve = _curve([{"0": .5, "1": float("nan")}, {"0": .6, "1": float("nan"), "2": .2}])
    f = inc.forgetting(curve, tasks)
    assert f["old_class"] == pytest.approx(-.1) and f["avg"] == pytest.approx(-.1)   # improved -> negative
    g = inc.forgetting(_curve([{0: .5}]), [[0]])
    assert math.isnan(g["avg"]) and g["by_task"] == []


def test_runs_store_per_class_ap_and_forgetting(tmp_path):
    import run_pipeline as rp
    from test_exp_round1 import _base
    wd = str(tmp_path / "wd"); base = _base(wd) + ["--n_final", "20"]
    for c in ("fetch", "make-yolo-data"):
        assert rp.main([c, *base]) == 0
    assert rp.main(["extract", *base, "--splits", "train", "eval", "tune", "final"]) == 0
    assert rp.main(["tune", *base, "--n_trials", "2"]) == 0
    tj = os.path.join(wd, "results", "tune.json")
    assert rp.main(["variants", *base, "--tuned", tj, "--on", "eval", "--n_folds", "2", "--only", "mem_newonly"]) == 0
    r = json.load(open(os.path.join(wd, "results", "variants_eval.json")))
    sp = json.load(open(os.path.join(wd, "splits.json")))
    for n in ("E0_zeroshot_tuned", "E5_tuned", "mem_newonly"):
        assert len(r[n]["curve"]) == len(sp["tasks"])
        assert set(map(int, r[n]["curve"][-1]["per_class_AP50"])) == {c for t in sp["tasks"] for c in t}
        assert r[n]["forgetting"]["old_class"] == pytest.approx(r[n]["base_AP50_drop"], abs=1e-9)
    md = open(os.path.join(wd, "results", "variants_eval.md")).read()
    assert "## Forgetting" in md and "| mem_newonly |" in md

    # final split: rows must be named, one look only, recorded
    with pytest.raises(SystemExit):
        rp.main(["variants", *base, "--tuned", tj, "--on", "final", "--n_folds", "2"])
    assert not os.path.exists(os.path.join(wd, "results", "final_log.json"))
    assert rp.main(["variants", *base, "--tuned", tj, "--on", "final", "--n_folds", "2", "--only", "mem_newonly"]) == 0
    assert os.path.exists(os.path.join(wd, "results", "variants_final.json"))
    assert "FINAL SPLIT" in open(os.path.join(wd, "results", "variants_final.md")).read()
    with pytest.raises(SystemExit):
        rp.main(["variants", *base, "--tuned", tj, "--on", "final", "--n_folds", "2", "--only", "mem_newonly"])
    log = json.load(open(os.path.join(wd, "results", "final_log.json")))
    assert len(log) == 1 and log[0]["command"].startswith("variants")
    with open(tj, "rb") as f:
        assert json.load(f)                                                         # tune.json untouched and valid

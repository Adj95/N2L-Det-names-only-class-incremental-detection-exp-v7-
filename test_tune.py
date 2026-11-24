import dataclasses
import json
import os

import numpy as np
import pytest

import incremental as inc
import n2l_core as nc
import pipeline as pl
import pseudo as ps
import run_pipeline as rp
import tune as tn
from inference import InferConfig
from test_pipeline import TASKS, bundle, same, world  # noqa: F401


def test_default_params_equal_the_runconfig_defaults():
    d = inc.RunConfig(tasks=TASKS)
    cfg = tn.cfg_from(TASKS, tn.default_params())
    assert cfg.pseudo == d.pseudo and cfg.n2l == d.n2l and cfg.infer == d.infer


def test_candidates_start_with_default_then_no_refine_and_are_unique_and_deterministic():
    c = tn.candidate_params(12, seed=3)
    assert c[0] == tn.default_params() and c[1]["n2l"]["n_rounds"] == 0
    assert c[1]["pseudo"] == c[0]["pseudo"] and {k: v for k, v in c[1]["n2l"].items() if k != "n_rounds"} == \
        {k: v for k, v in c[0]["n2l"].items() if k != "n_rounds"}
    assert len(c) == 12 and len({json.dumps(x, sort_keys=True) for x in c}) == 12
    assert c == tn.candidate_params(12, seed=3) and c != tn.candidate_params(12, seed=4)
    assert len(tn.candidate_params(1, 0)) == 2                                   # the two fixed ones always exist


def test_sampled_values_come_from_the_declared_space():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = tn.sample_params(rng)
        for grp, space in tn.TRAIN_SPACE.items():
            assert all(p[grp][k] in v for k, v in space.items())
        tn.cfg_from(TASKS, p)                                                    # constructible


def test_infer_grid_equals_full_retrain_for_each_setting(bundle):
    """The grid trains once; each row must equal a separate full run with that inference config."""
    ics = [InferConfig(fusion_weight=0.2, yolo_gamma=0.25), InferConfig(fusion_weight=1.0, yolo_gamma=1.0, topk_classes=1)]
    base = inc.RunConfig(tasks=TASKS)
    rows = tn.infer_grid(bundle, base, ics)
    assert len(rows) == 2
    for ic, row in zip(ics, rows):
        full = inc.run_incremental(bundle, dataclasses.replace(base, infer=ic, n_folds=0), log=lambda *_: None)
        assert row["final_AP50"] == pytest.approx(full["final_AP50"]) and row["final_AP"] == pytest.approx(full["final_AP"])
        assert row["infer"] == dataclasses.asdict(ic)


def test_infer_grid_for_zeroshot_ignores_ridge_weight(bundle):
    cfg = inc.RunConfig(tasks=TASKS, variant="zeroshot")
    a, b = tn.infer_grid(bundle, cfg, [InferConfig(fusion_weight=0.2), InferConfig(fusion_weight=1.0)])
    assert a["final_AP50"] == b["final_AP50"]


def test_best_prefers_earlier_on_ties_and_survives_nan():
    rows = [{"x": 0.5}, {"x": 0.5}, {"x": float("nan")}]
    assert tn._best(rows, "x") == 0
    assert tn._best([{"x": float("nan")}, {"x": 0.1}], "x") == 1


def test_run_search_end_to_end_and_selection_is_consistent(bundle):
    lines = []
    t = tn.run_search(bundle, TASKS, n_trials=4, seed=1, log=lines.append)
    assert len(t["stage1"]) == 4 and t["n_tune_images"] == len(bundle.eval_ids)
    best = max(r["final_AP50"] for r in t["stage1"])
    assert t["stage1"][t["best_train_trial"]]["final_AP50"] == best
    assert t["stage1"][0]["params"] == tn.default_params()
    # the chosen E5 inference setting is the best row of its grid, and tuning can never score below the default
    assert t["tune_scores"]["E5_tuned"] == max(r["final_AP50"] for r in t["stage2_E5"])
    assert t["tune_scores"]["E0_tuned"] == max(r["final_AP50"] for r in t["stage2_E0"])
    assert t["tune_scores"]["E0_tuned"] >= t["tune_scores"]["E0_default"] - 1e-12     # default is inside the E0 grid
    assert len(t["stage2_E5"]) == len(tn.FUSION) * len(tn.GAMMA) + len(tn.TOPK)
    assert len(t["stage2_E0"]) == len(tn.GAMMA) * len(tn.TOPK)
    assert any("tune-split" in l for l in lines)
    with pytest.raises(ValueError):
        tn.run_search(bundle, TASKS, objective="bogus")
    rep = tn.format_report(t)
    assert "Stage 1" in rep and "E0 zero-shot" in rep


def test_tuned_configs_roundtrip_through_json_and_enter_the_ablation_table(bundle, tmp_path):
    t = {"E5": {**tn.default_params(), "infer": dataclasses.asdict(InferConfig(fusion_weight=0.5, yolo_gamma=1.0))},
         "E0": {"infer": dataclasses.asdict(InferConfig(yolo_gamma=0.25, topk_classes=5))}}
    t["E5"]["pseudo"]["tau_pos"] = 0.7
    f = tmp_path / "tune.json"; f.write_text(json.dumps(t))
    cfgs = pl.ablation_configs(TASKS, str(f))
    assert cfgs["E5_tuned"].pseudo.tau_pos == 0.7 and cfgs["E5_tuned"].infer.fusion_weight == 0.5
    assert cfgs["E0_zeroshot_tuned"].variant == "zeroshot" and cfgs["E0_zeroshot_tuned"].infer.topk_classes == 5
    assert "E5_tuned" not in pl.ablation_configs(TASKS)                          # only present when asked for
    with pytest.raises(KeyError):
        pl.run_ablation(bundle, TASKS, only=["E5_tuned"], log=lambda *_: None)
    res = pl.run_ablation(bundle, TASKS, only=["E5_tuned", "E0_zeroshot_tuned"], log=lambda *_: None, tuned=str(f))
    assert set(res) == {"E5_tuned", "E0_zeroshot_tuned"}


def test_cli_tune_then_run_with_tuned_uses_tune_images_for_search_only(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_tune", "20", "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *base])
    rp.main(["tune", *base, "--n_trials", "3"])
    t = json.load(open(os.path.join(wd, "results", "tune.json")))
    assert t["n_tune_images"] == 20 and os.path.exists(os.path.join(wd, "results", "tune.md"))
    rp.main(["run", *base, "--tuned", os.path.join(wd, "results", "tune.json"),
             "--only", "E0_zeroshot", "E0_zeroshot_tuned", "E5_ours", "E5_tuned"])
    res = json.load(open(os.path.join(wd, "results", "ablation.json")))
    assert set(res) == {"E0_zeroshot", "E0_zeroshot_tuned", "E5_ours", "E5_tuned"}
    assert all("final_folds" in r for r in res.values())                         # fold CIs come for free
    with pytest.raises(SystemExit, match="not found"):
        rp.main(["run", *base, "--tuned", str(tmp_path / "nope.json")])


def test_tune_refuses_without_tune_images(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *base])
    with pytest.raises(SystemExit, match="no tune images"):
        rp.main(["tune", *base])


# ------------------------------------------------------------------ sharper tuner tests (spies; the fake world saturates AP50)
def test_infer_grid_scores_only_the_final_task_over_all_classes(bundle, monkeypatch):
    seen = []
    orig = inc._eval_scores
    monkeypatch.setattr(inc, "_eval_scores", lambda Xe, order, *a, **k: (seen.append(list(order)), orig(Xe, order, *a, **k))[1])
    tn.infer_grid(bundle, inc.RunConfig(tasks=TASKS), [InferConfig(), InferConfig(yolo_gamma=1.0)])
    all_classes = [c for t in TASKS for c in t]
    # run_incremental itself scores each task once (3 calls); the grid adds exactly one call per setting, on all 9 classes
    assert [len(o) for o in seen].count(9) == 1 + 2 and len(seen) == 3 + 2
    assert seen[-1] == all_classes and seen[-2] == all_classes


def test_run_search_stage2_uses_the_right_models_and_the_winning_training_settings(bundle, monkeypatch):
    """Stage 1 is faked so that 'no refinement' wins; then check what every later stage was given."""
    def fake_run(b, cfg, log=print, probe=None):
        s = 0.3 if cfg.variant == "zeroshot" else (0.9 if cfg.n2l.n_rounds == 0 else 0.5)
        return {"final_AP50": s, "final_AP": s / 2, "A_bar_AP": s / 2}
    calls = []
    def fake_grid(b, cfg, infers):
        calls.append((cfg.variant, cfg.n2l.n_rounds, [dataclasses.asdict(i) for i in infers]))
        rows = []
        for i in infers:
            score = 0.6 + 0.1 * (i.fusion_weight == 0.5) + 0.1 * (i.yolo_gamma == 0.75) + 0.05 * (i.topk_classes == 5)
            rows.append({"infer": dataclasses.asdict(i), "final_AP50": score, "final_AP": score / 2})
        return rows
    monkeypatch.setattr(inc, "run_incremental", fake_run)
    monkeypatch.setattr(tn, "infer_grid", fake_grid)
    t = tn.run_search(bundle, TASKS, n_trials=3, seed=0, log=lambda *_: None)
    assert t["best_train_trial"] == 1 and t["E5"]["n2l"]["n_rounds"] == 0
    assert [(v, r) for v, r, _ in calls] == [("ours", 0), ("ours", 0), ("zeroshot", tn.default_params()["n2l"]["n_rounds"])]
    # the top-k stage starts from the winner of the fusion x gamma grid
    assert all(i["fusion_weight"] == 0.5 and i["yolo_gamma"] == 0.75 for i in calls[1][2])
    assert [i["topk_classes"] for i in calls[1][2]] == tn.TOPK
    assert t["E5"]["infer"]["topk_classes"] == 5
    # E0 default score comes from a real zero-shot run, not a placeholder
    assert t["tune_scores"]["E0_default"] == 0.3 and t["tune_scores"]["E5_default"] == 0.5


# ------------------------------------------------------------------ leave-one-out, tuned ceiling, training-subset seeds
def _tune_dict(**changes):
    t = {"E5": {**json.loads(json.dumps(tn.default_params())), "infer": dataclasses.asdict(InferConfig())},
         "E0": {"infer": dataclasses.asdict(InferConfig(yolo_gamma=0.25))}}
    for k, v in changes.items():
        g, key = k.split("__")
        t["E5"][g][key] = v
    return t


def test_changed_settings_lists_exactly_the_differences_from_default():
    assert tn.changed_settings(_tune_dict()) == []
    t = _tune_dict(n2l__n_rounds=0, pseudo__tau_bg=0.5, infer__yolo_gamma=0.25, infer__topk_classes=5)
    assert sorted(tn.changed_settings(t)) == sorted(["pseudo.tau_bg", "n2l.n_rounds", "infer.yolo_gamma", "infer.topk_classes"])
    # a value that is set but equals the default is not a change
    assert tn.changed_settings(_tune_dict(n2l__lam=tn.default_params()["n2l"]["lam"])) == []


def test_loo_configs_differ_from_the_tuned_one_in_exactly_the_named_setting():
    t = _tune_dict(n2l__n_rounds=0, pseudo__tau_bg=0.5, pseudo__use_loc=True, infer__yolo_gamma=0.25, infer__topk_classes=5)
    cfgs = tn.loo_configs(t, TASKS)
    assert set(cfgs) == {"E0_zeroshot_tuned", "E6_gt_ceiling_tuned", "E5_default", "E5_tuned", "E5_tuned_only_train", "E5_tuned_only_infer",
                         "E5_tuned_minus_pseudo.tau_bg", "E5_tuned_minus_pseudo.use_loc", "E5_tuned_minus_n2l.n_rounds",
                         "E5_tuned_minus_infer.yolo_gamma", "E5_tuned_minus_infer.topk_classes",
                          "E5_tuned_minus_group.pseudo_filters", "E5_tuned_minus_group.inference"}
    tuned, dflt = cfgs["E5_tuned"], cfgs["E5_default"]
    assert dflt == inc.RunConfig(tasks=TASKS) and tuned.n2l.n_rounds == 0 and tuned.pseudo.tau_bg == 0.5
    assert tuned.infer == InferConfig(yolo_gamma=0.25, topk_classes=5)
    # only-train: tuned training, default inference.  only-infer: default training, tuned inference
    assert cfgs["E5_tuned_only_train"] == dataclasses.replace(tuned, infer=InferConfig())
    assert cfgs["E5_tuned_only_infer"] == dataclasses.replace(dflt, infer=tuned.infer)
    # each minus-row reverts that one setting to the default and nothing else
    assert cfgs["E5_tuned_minus_n2l.n_rounds"] == dataclasses.replace(
        tuned, n2l=dataclasses.replace(tuned.n2l, n_rounds=dflt.n2l.n_rounds))
    assert cfgs["E5_tuned_minus_pseudo.tau_bg"] == dataclasses.replace(
        tuned, pseudo=dataclasses.replace(tuned.pseudo, tau_bg=dflt.pseudo.tau_bg))
    assert cfgs["E5_tuned_minus_pseudo.use_loc"].pseudo.use_loc is False and cfgs["E5_tuned_minus_pseudo.use_loc"].pseudo.tau_bg == 0.5
    assert cfgs["E5_tuned_minus_infer.yolo_gamma"].infer == InferConfig(yolo_gamma=InferConfig().yolo_gamma, topk_classes=5)
    assert cfgs["E5_tuned_minus_infer.topk_classes"].infer == InferConfig(yolo_gamma=0.25)
    assert cfgs["E0_zeroshot_tuned"].variant == "zeroshot" and cfgs["E0_zeroshot_tuned"].infer.yolo_gamma == 0.25
    # nothing tuned -> only the reference rows
    assert set(tn.loo_configs(_tune_dict(), TASKS)) == {"E0_zeroshot_tuned", "E6_gt_ceiling_tuned", "E5_default", "E5_tuned", "E5_tuned_only_train",
                                                       "E5_tuned_only_infer"}


def test_tuned_ceiling_uses_the_tuned_ridge_and_inference_settings():
    t = _tune_dict(n2l__lam=10.0, infer__yolo_gamma=1.0, infer__topk_classes=5)
    c = tn.tuned_configs(t, TASKS)["E6_gt_ceiling_tuned"]
    assert c.variant == "gt_ceiling" and c.n2l.lam == 10.0 and c.infer.yolo_gamma == 1.0 and c.infer.topk_classes == 5
    assert "E6_gt_ceiling_tuned" in pl.ablation_configs(TASKS, t) and "E6_gt_ceiling_tuned" not in pl.ablation_configs(TASKS)


def test_run_loo_runs_every_config_once_on_the_given_bundle(bundle, monkeypatch):
    t = _tune_dict(n2l__n_rounds=0, infer__yolo_gamma=0.25)
    seen = []
    real = inc.run_incremental
    monkeypatch.setattr(inc, "run_incremental", lambda b, cfg, log=print, probe=None: (seen.append((b, cfg)), real(b, cfg, log, probe))[1])
    res = tn.run_loo(bundle, t, TASKS, log=lambda *_: None)
    cfgs = tn.loo_configs(t, TASKS)
    assert set(res) == set(cfgs) and len(seen) == len(cfgs) and all(b is bundle for b, _ in seen)
    assert [c for _, c in seen] == list(cfgs.values())
    assert all("final_folds" in r for r in res.values())            # fold CIs come for free


def test_subsample_train_ids_is_a_deterministic_sorted_subset():
    ids = [list(range(100, 150)), list(range(200, 230)), [7]]
    a = tn.subsample_train_ids(ids, 0.5, seed=1)
    assert [len(x) for x in a] == [25, 15, 1]                                  # at least one image per task
    assert all(set(x) <= set(y) and x == sorted(x) and len(set(x)) == len(x) for x, y in zip(a, ids))
    assert a == tn.subsample_train_ids(ids, 0.5, seed=1) and a != tn.subsample_train_ids(ids, 0.5, seed=2)
    assert tn.subsample_train_ids(ids, 1.0, seed=3) == [sorted(x) for x in ids]
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            tn.subsample_train_ids(ids, bad, 0)


def test_run_seeds_retrains_on_different_subsets_and_runs_the_training_free_baseline_once(bundle, monkeypatch):
    t = _tune_dict(n2l__n_rounds=0)
    seen = []
    before = [list(x) for x in bundle.train_ids]
    real = inc.run_incremental
    monkeypatch.setattr(inc, "run_incremental", lambda b, cfg, log=print, probe=None: (seen.append((b, cfg)), real(b, cfg, log, probe))[1])
    res = tn.run_seeds(bundle, TASKS, t, n_seeds=3, train_frac=0.5, seed0=10, log=lambda *_: None)
    assert [r["seed"] for r in res["per_seed"]] == [10, 11, 12] and res["train_frac"] == 0.5
    zs = [(b, c) for b, c in seen if c.variant == "zeroshot"]
    assert len(zs) == 1                                                           # E0 does not train: run once, reused
    trained = [(b, c) for b, c in seen if c.variant != "zeroshot"]
    assert len(trained) == 3 * (len(tn.SEED_RUNS) - 1) and all(c.n_folds == 0 for _, c in seen)
    subs = {tuple(map(tuple, b.train_ids)) for b, _ in trained}
    assert len(subs) == 3                                                         # one distinct subset per seed
    for b, _ in trained:
        assert [len(x) for x in b.train_ids] == [round(0.5 * len(x)) for x in bundle.train_ids]
        assert all(set(x) <= set(y) for x, y in zip(b.train_ids, bundle.train_ids))
    assert all(set(tn.SEED_RUNS) <= set(r) for r in res["per_seed"])
    # E0 is identical across seeds (it never sees the training images); the original bundle is untouched
    assert len({r["E0_zeroshot_tuned"]["AP50"] for r in res["per_seed"]}) == 1
    assert bundle.train_ids == before
    with pytest.raises(ValueError):
        tn.run_seeds(bundle, TASKS, t, n_seeds=1)


def test_format_seeds_computes_mean_sd_and_paired_interval_over_seeds():
    import compare_runs as cr
    e5 = [0.70, 0.72, 0.68]; e0 = [0.65, 0.65, 0.65]
    per = [{"seed": i, **{n: {"AP50": 0.5, "AP": 0.3, "AP50_new": 0.2} for n in tn.SEED_RUNS},
            "E5_tuned": {"AP50": e5[i], "AP": 0.3, "AP50_new": 0.4}, "E0_zeroshot_tuned": {"AP50": e0[i], "AP": 0.3, "AP50_new": 0.3}}
           for i in range(3)]
    txt = tn.format_seeds({"n_seeds": 3, "train_frac": 0.8, "seed0": 0, "per_seed": per})
    d, ci, _ = cr.paired_ci(e5, e0)
    row = [l for l in txt.splitlines() if l.startswith("| E5_tuned |")][0]
    assert f"{np.mean(e5):.3f} | {np.std(e5, ddof=1):.3f} | {d:+.3f} | {ci:.3f} |" in row
    ref = [l for l in txt.splitlines() if l.startswith("| E0_zeroshot_tuned |")][0]
    assert "| 0.650 | 0.000 | - | - |" in ref
    assert "AP50_new mean" in txt and "80%" in txt


def test_cli_loo_and_seeds_end_to_end(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_tune", "20", "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *base])
    with pytest.raises(SystemExit, match="run `tune` first"):
        rp.main(["loo", *base])
    with pytest.raises(SystemExit, match="run `tune` first"):
        rp.main(["seeds", *base])
    rp.main(["tune", *base, "--n_trials", "3"])
    rp.main(["loo", *base])
    loo = json.load(open(os.path.join(wd, "results", "loo.json")))
    assert {"E0_zeroshot_tuned", "E6_gt_ceiling_tuned", "E5_default", "E5_tuned", "E5_tuned_only_train", "E5_tuned_only_infer"} <= set(loo)
    assert all("final_folds" in r for r in loo.values())
    md = open(os.path.join(wd, "results", "loo.md")).read()
    assert "difference vs E5_tuned" in md and "difference vs E0_zeroshot_tuned" in md and "AP50_new" in md
    rp.main(["seeds", *base, "--n_seeds", "2", "--train_frac", "0.5"])
    s = json.load(open(os.path.join(wd, "results", "seeds.json")))
    assert s["n_seeds"] == 2 and len(s["per_seed"]) == 2
    assert "sd over seeds" in open(os.path.join(wd, "results", "seeds.md")).read()
    # the eval images of the ordinary run are untouched by these commands
    assert not os.path.exists(os.path.join(wd, "results", "ablation.json"))


def test_format_seeds_has_no_significance_column_and_says_why():
    per = [{"seed": i, **{n: {"AP50": 0.5, "AP": 0.3, "AP50_new": 0.2} for n in tn.SEED_RUNS}} for i in range(3)]
    txt = tn.format_seeds({"n_seeds": 3, "train_frac": 0.8, "seed0": 0, "per_seed": per})
    assert "significant |" not in txt and "NOT whether the gain is significant" in txt


def test_loo_and_seeds_use_the_given_tune_file_the_chosen_fraction_and_check_the_eval_cache(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_tune", "20", "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *base])
    rp.main(["tune", *base, "--n_trials", "3"])
    # --tuned: a hand-made file with a setting nobody tuned must be the one that is used
    t = json.load(open(os.path.join(wd, "results", "tune.json")))
    t["E5"]["pseudo"]["tau_pos"] = 0.9; t["E5"]["infer"]["fusion_weight"] = 0.5
    other = tmp_path / "other_tune.json"; other.write_text(json.dumps(t))
    rp.main(["loo", *base, "--tuned", str(other)])
    loo = json.load(open(os.path.join(wd, "results", "loo.json")))
    assert "E5_tuned_minus_pseudo.tau_pos" in loo and "E5_tuned_minus_infer.fusion_weight" in loo
    # --train_frac reaches the training subsets
    sp = json.load(open(os.path.join(wd, "splits.json")))
    rp.main(["seeds", *base, "--n_seeds", "2", "--train_frac", "0.5"])
    s = json.load(open(os.path.join(wd, "results", "seeds.json")))
    assert s["train_frac"] == 0.5
    assert s["per_seed"][0]["n_train"] == [round(0.5 * len(x)) for x in sp["train_ids"]]
    # an eval image missing from the cache is reported, not silently ignored
    sp["eval_ids"] = sp["eval_ids"] + [987654321]
    json.dump(sp, open(os.path.join(wd, "splits.json"), "w"))
    for cmd in ("loo", "seeds"):
        with pytest.raises(SystemExit, match="lacks"):
            rp.main([cmd, *base])


# ------------------------------------------------------------------ round 3: groups, sweep, per-class, tags, templates
def _t_for(tmp, **over):
    t = {"E5": {"pseudo": dict(default_pseudo()), "n2l": dict(default_n2l()), "infer": dict(default_infer())},
         "E0": {"infer": dict(default_infer())}, "tune_scores": {}, "stage1": [], "n_tune_images": 1, "objective": "final_AP50", "seed": 0}
    return t


def default_pseudo():
    return tn.default_params()["pseudo"]


def default_n2l():
    return tn.default_params()["n2l"]


def default_infer():
    import dataclasses
    from inference import InferConfig
    return dataclasses.asdict(InferConfig())


def test_loo_group_rows_revert_all_members_together_and_only_when_two_or_more_changed():
    t = _t_for(None)
    t["E5"]["n2l"]["n_rounds"] = 0; t["E5"]["n2l"]["use_intra"] = False; t["E5"]["n2l"]["use_inter"] = False
    t["E5"]["pseudo"]["tau_bg"] = 0.5                                   # ONE change in pseudo_filters -> no group row
    tasks = [[0, 1, 2], [3, 4]]
    cf = tn.loo_configs(t, tasks)
    assert "E5_tuned_minus_group.n2l_machinery" in cf
    assert "E5_tuned_minus_group.pseudo_filters" not in cf and "E5_tuned_minus_group.inference" not in cf
    g = cf["E5_tuned_minus_group.n2l_machinery"]; d = tn.default_params()
    assert g.n2l.n_rounds == d["n2l"]["n_rounds"] and g.n2l.use_intra == d["n2l"]["use_intra"] and g.n2l.use_inter == d["n2l"]["use_inter"]
    assert g.pseudo.tau_bg == 0.5                                       # settings outside the group stay tuned
    single = cf["E5_tuned_minus_n2l.n_rounds"]
    assert single.n2l.use_intra is False                                # a single reversion leaves the others tuned


def test_loo_group_row_for_inference_reverts_the_inference_settings():
    t = _t_for(None)
    t["E5"]["infer"].update(yolo_gamma=0.25, topk_classes=5, fusion_weight=0.3)
    cf = tn.loo_configs(t, [[0, 1], [2]])
    g = cf["E5_tuned_minus_group.inference"]; di = default_infer()
    assert g.infer.yolo_gamma == di["yolo_gamma"] and g.infer.topk_classes == di["topk_classes"] and g.infer.fusion_weight == di["fusion_weight"]


def test_run_loo_passes_n_folds_to_every_run(monkeypatch):
    seen = []
    monkeypatch.setattr(tn.inc, "run_incremental", lambda b, cfg, log=None, probe=None: seen.append(cfg.n_folds) or {"final_AP50": 0.0, "final_AP": 0.0})
    tn.run_loo(None, _t_for(None), [[0, 1], [2]], n_folds=10)
    assert seen and set(seen) == {10}


def test_final_eval_scores_only_the_last_task_model_and_returns_per_class(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_tune", "20", "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *base])
    rp.main(["tune", *base, "--n_trials", "3"])
    for cmd in ("sweep", "perclass"):
        rp.main([cmd, *base])
    sw = json.load(open(os.path.join(wd, "results", "sweep.json")))
    assert len(sw["E0"]) == len(tn.SWEEP_GAMMAS) and len(sw["E5"]) == len(tn.SWEEP_GAMMAS) * len(tn.SWEEP_WEIGHTS)
    assert {"AP50", "AP50_new", "gamma", "weight"} <= set(sw["E5"][0])
    assert "best head - E0" in open(os.path.join(wd, "results", "sweep.md")).read()
    pc = json.load(open(os.path.join(wd, "results", "perclass.json")))
    assert len(pc["classes"]) == 9 and {c["task"] for c in pc["classes"]} == set(range(len(pc["tasks"])))
    assert all(n in pc["classes"][0] for n in tn.PERCLASS_RUNS)
    txt = open(os.path.join(wd, "results", "perclass.md")).read()
    assert "mean new" in txt and "mean base" in txt and "new classes" in txt


def test_format_perclass_numbers_and_ordering():
    rows = [{"class": 0, "name": "a", "task": 0, "E0_zeroshot_tuned": 0.5, "E5_tuned": 0.5, "E6_gt_ceiling_tuned": 0.5},
            {"class": 1, "name": "b", "task": 1, "E0_zeroshot_tuned": 0.2, "E5_tuned": 0.3, "E6_gt_ceiling_tuned": 0.4},
            {"class": 2, "name": "c", "task": 1, "E0_zeroshot_tuned": 0.6, "E5_tuned": 0.5, "E6_gt_ceiling_tuned": 0.6}]
    txt = tn.format_perclass({"tasks": [[0], [1, 2]], "classes": rows})
    lines = txt.splitlines()
    assert lines.index(next(l for l in lines if l.startswith("| b |"))) < lines.index(next(l for l in lines if l.startswith("| a |")))  # new first
    b = next(l for l in lines if l.startswith("| b |"))
    assert "+0.100" in b and "50%" in b                                  # 0.1 of the 0.2 headroom
    c = next(l for l in lines if l.startswith("| c |"))
    assert "-0.100" in c and c.rstrip().endswith("| - |")                # no headroom -> no share
    mn = next(l for l in lines if l.startswith("| **mean new**"))
    assert "| 0.400 | 0.400 | 0.500 | +0.000 |" in mn or "| 0.400 | 0.400 | 0.500 | -0.000 |" in mn
    assert "1 of 2 new classes" in txt


def test_format_sweep_columns_and_best_minus_e0():
    res = {"gammas": [0.0, 0.5], "weights": [0.0, 1.0],
           "E0": [{"gamma": 0.0, "AP50": 0.5, "AP50_new": 0.1}, {"gamma": 0.5, "AP50": 0.6, "AP50_new": 0.2}],
           "E5": [{"gamma": 0.0, "weight": 0.0, "AP50": 0.4, "AP50_new": 0.1}, {"gamma": 0.0, "weight": 1.0, "AP50": 0.55, "AP50_new": 0.3},
                  {"gamma": 0.5, "weight": 0.0, "AP50": 0.6, "AP50_new": 0.2}, {"gamma": 0.5, "weight": 1.0, "AP50": 0.58, "AP50_new": 0.25}]}
    txt = tn.format_sweep(res)
    assert "| 0 | 0.500 | 0.400 | 0.550 | +0.050 |" in txt and "| 0.5 | 0.600 | 0.600 | 0.580 | +0.000 |" in txt
    assert "| 0 | 0.100 | 0.100 | 0.300 | +0.200 |" in txt


def test_tag_keeps_earlier_results_and_label_templates_reach_the_text(tmp_path):
    import pipeline as pl
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_tune", "20", "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *base])
    rp.main(["tune", *base, "--n_trials", "3"])
    rp.main(["tune", *base, "--n_trials", "3", "--tune_seed", "1", "--tag", "s1"])
    r = os.path.join(wd, "results")
    assert os.path.exists(os.path.join(r, "tune.json")) and os.path.exists(os.path.join(r, "tune_s1.json"))
    assert json.load(open(os.path.join(r, "tune_s1.json")))["seed"] == 1 and json.load(open(os.path.join(r, "tune.json")))["seed"] == 0
    rp.main(["loo", *base, "--tag", "f10", "--n_folds", "10"])
    loo = json.load(open(os.path.join(r, "loo_f10.json")))
    assert not os.path.exists(os.path.join(r, "loo.json"))
    f = next(iter(loo.values()))["final_folds"]
    assert len(f["AP50"] if isinstance(f, dict) else f) == 10
    # label templates
    class M:
        def encode_text(self, texts):
            return np.ones((len(texts), 4))
    seen = []
    import prompts as pr
    orig = pr.class_embeddings
    pr.class_embeddings = lambda names, enc, templates: seen.append(len(templates)) or orig(names, enc, templates)
    try:
        for n, want in ((1, 1), (3, 3), (0, len(pr.DET_TEMPLATES))):
            pl.set_label_templates(n); seen.clear(); pl.text_assets(M())
            assert seen[0] == want, (n, seen)
    finally:
        pr.class_embeddings = orig; pl.set_label_templates(1)
    with pytest.raises(ValueError):
        pl.set_label_templates(11)
    assert pl.LABEL_TEMPLATES == 1
    rp.main(["loo", *base, "--label_templates", "0", "--tag", "ltall"])
    assert pl.LABEL_TEMPLATES == 0
    pl.set_label_templates(1)


def test_infer_grid_reports_AP50_new_as_the_mean_over_the_last_task_classes_only(bundle, monkeypatch):
    per = {c: 0.1 * (c + 1) for c in range(9)}                  # class 0..8 -> .1 .. .9
    monkeypatch.setattr(tn, "coco_eval", lambda gt, dets, ids, order: {"AP50": 0.5, "AP": 0.3, "per_class_AP50": per})
    rows = tn.infer_grid(bundle, inc.RunConfig(tasks=TASKS), [InferConfig()])
    assert rows[0]["AP50_new"] == pytest.approx(np.mean([per[c] for c in TASKS[-1]]))
    assert rows[0]["AP50_new"] != pytest.approx(np.mean(list(per.values())))


def test_run_sweep_scores_every_gamma_weight_pair_with_the_tuned_topk_and_returns_matching_rows(monkeypatch):
    calls = []

    def fake_grid(bundle, cfg, infers):
        calls.append((cfg.variant, [(i.yolo_gamma, i.fusion_weight, i.topk_classes) for i in infers]))
        return [{"final_AP50": i.yolo_gamma * 10 + i.fusion_weight, "final_AP": 0, "AP50_new": i.fusion_weight} for i in infers]

    monkeypatch.setattr(tn, "infer_grid", fake_grid)
    t = _t_for(None); t["E5"]["infer"].update(topk_classes=7); t["E0"]["infer"].update(topk_classes=2)
    res = tn.run_sweep(None, [[0, 1], [2]], t, gammas=(0.0, 0.5), weights=(0.0, 1.0))
    e0, e5 = calls
    assert e0[0] == "zeroshot" and e0[1] == [(0.0, t["E0"]["infer"]["fusion_weight"], 2), (0.5, t["E0"]["infer"]["fusion_weight"], 2)]
    assert e5[1] == [(0.0, 0.0, 7), (0.0, 1.0, 7), (0.5, 0.0, 7), (0.5, 1.0, 7)]
    assert [(r["gamma"], r["weight"], r["AP50"], r["AP50_new"]) for r in res["E5"]] == \
        [(0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 1.0, 1.0), (0.5, 0.0, 5.0, 0.0), (0.5, 1.0, 6.0, 1.0)]
    assert [(r["gamma"], r["AP50"]) for r in res["E0"]] == [(0.0, 0.0 + t["E0"]["infer"]["fusion_weight"]), (0.5, 5.0 + t["E0"]["infer"]["fusion_weight"])]

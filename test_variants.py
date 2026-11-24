import dataclasses
import json
import os

import numpy as np
import pytest
from scipy.special import softmax

import heads
import incremental as inc
import n2l_core as nc
import pipeline as pl
import pseudo as ps
import run_pipeline as rp
import tune as tn
import variants as vr
from inference import InferConfig, fuse
from test_pipeline import TASKS, bundle, same, world  # noqa: F401
from test_tune import _tune_dict


# ------------------------------------------------------------------ pseudo-label filters
def _inputs(n=40, K=4, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(n, 16)); T = rng.normal(size=(K, 16)); bg = rng.normal(size=(1, 16))
    return F, rng.uniform(0, 1, n), T, bg


def test_min_yolo_drops_exactly_the_low_score_boxes():
    F, s, T, bg = _inputs()
    s[0] = 0.4                                                          # exactly at the threshold: kept (>=)
    keep0, _, _ = ps.build_task_inputs(F, s, T, bg, ps.PseudoConfig())
    keep, logits, w = ps.build_task_inputs(F, s, T, bg, ps.PseudoConfig(min_yolo=0.4))
    assert keep0.all() and np.array_equal(keep, s >= 0.4) and len(logits) == keep.sum()
    keep1, _, _ = ps.build_task_inputs(F, s, T, bg, ps.PseudoConfig(min_yolo=0.4, tau_pos=0.5))     # filters combine (AND)
    ref, _, _ = ps.build_task_inputs(F, s, T, bg, ps.PseudoConfig(tau_pos=0.5))
    assert np.array_equal(keep1, ref & (s >= 0.4))


def test_class_topfrac_keeps_the_most_confident_per_class_and_at_least_one():
    F, s, T, bg = _inputs(n=200)
    cfg = ps.PseudoConfig(class_topfrac=0.5)
    keep, logits, _ = ps.build_task_inputs(F, s, T, bg, cfg)
    full = ps.build_label_logits(F, T, bg, cfg)
    p = softmax(full, axis=1); top = p.argmax(1); ptop = p.max(1)
    assert keep.sum() < len(F)
    for g in np.unique(top):
        idx = np.flatnonzero(top == g)
        kept = idx[keep[idx]]
        assert len(kept) == max(1, int(np.ceil(0.5 * len(idx))))
        assert ptop[kept].min() >= ptop[np.setdiff1d(idx, kept)].max() if len(kept) < len(idx) else True
    one, _, _ = ps.build_task_inputs(F[:3], s[:3], T, bg, ps.PseudoConfig(class_topfrac=0.01))
    assert one.sum() >= 1
    assert ps.build_task_inputs(F, s, T, bg, ps.PseudoConfig(class_topfrac=1.0))[0].all()


def test_filters_leave_weights_aligned_with_the_kept_rows():
    F, s, T, bg = _inputs()
    cfg = ps.PseudoConfig(min_yolo=0.3, use_loc=True)
    keep, logits, w = ps.build_task_inputs(F, s, T, bg, cfg)
    assert len(w) == len(logits) == keep.sum()
    assert np.allclose(w, (cfg.loc_floor + (1 - cfg.loc_floor) * s)[keep])


# ------------------------------------------------------------------ soft labels
def _task(n=60, K=4, d=12, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)); L = rng.normal(size=(n, K + 1)) * 3
    return X, L


def test_soft_zero_is_exactly_the_old_hard_label_behaviour():
    X, L = _task()
    a = nc.N2LLearner(12, nc.N2LConfig(n_rounds=0, pca_rule="energy", pca_rho=0.95))
    b = nc.N2LLearner(12, nc.N2LConfig(n_rounds=0, pca_rule="energy", pca_rho=0.95, soft=0.0))
    a.add_task(X, L, 5, rng=np.random.default_rng(0)); b.add_task(X, L, 5, rng=np.random.default_rng(0))
    assert np.array_equal(a.W, b.W)


def test_soft_targets_are_a_mix_of_onehot_and_zero_shot_softmax_and_rows_sum_to_one():
    X, L = _task()
    seen = {}
    lr = nc.N2LLearner(12, nc.N2LConfig(n_rounds=0, pca_rule="energy", pca_rho=0.95, soft=0.4))
    orig = lr.acc.update
    lr.acc.update = lambda F, t, w=None: (seen.setdefault("t", []).append(np.array(t)), orig(F, t, w))[1]
    res = lr.add_task(X, L, 5, rng=np.random.default_rng(0))
    Y = np.vstack(seen["t"])
    assert Y.shape == (60, 5) and np.allclose(Y.sum(1), 1.0)
    onehot = np.eye(5)[res.labels]
    assert np.allclose(Y, 0.6 * onehot + 0.4 * softmax(L, axis=1))
    hard = nc.N2LLearner(12, nc.N2LConfig(n_rounds=0, pca_rule="energy", pca_rho=0.95))
    hard.add_task(X, L, 5, rng=np.random.default_rng(0))
    assert not np.allclose(lr.W, hard.W)                                  # soft targets really change the head


def test_soft_targets_right_align_with_the_head_columns_on_a_later_task():
    X, L = _task(K=3)                                   # 4 logit columns, head will have 4 + 2 columns
    lr = nc.N2LLearner(12, nc.N2LConfig(n_rounds=0, pca_rule="energy", pca_rho=0.95, soft=1.0))
    lr.acc.add_classes(2)                               # pretend an earlier task owns the first 2 columns
    seen = {}
    orig = lr.acc.update
    lr.acc.update = lambda F, t, w=None: (seen.setdefault("t", []).append(np.array(t)), orig(F, t, w))[1]
    lr.add_task(X, L, 4, rng=np.random.default_rng(0))
    Y = np.vstack(seen["t"])
    assert Y.shape[1] == 6 and np.allclose(Y[:, :2], 0.0) and np.allclose(Y[:, 2:], softmax(L, axis=1))


def test_soft_outside_zero_one_is_rejected():
    X, L = _task()
    with pytest.raises(ValueError, match="soft"):
        nc.N2LLearner(12, nc.N2LConfig(n_rounds=0, soft=1.5)).add_task(X, L, 5)


# ------------------------------------------------------------------ fusion variants
def test_ridge_norm_clips_and_renormalises_and_falls_back_on_zero_rows():
    zs = np.array([[2.0, 0.0, 1.0], [0.0, 0.0, 0.0]]); r = np.array([[-1.0, 1.0, 3.0], [-1.0, -2.0, -3.0]])
    p = softmax(zs, axis=1)
    out = fuse(zs, r, InferConfig(fusion_weight=1.0, ridge_norm=True))
    assert np.allclose(out[0], [0.0, 0.25, 0.75]) and np.allclose(out[1], p[1])
    plain = fuse(zs, r, InferConfig(fusion_weight=1.0))
    assert np.allclose(plain, r)                                           # off by default
    assert np.allclose(fuse(zs, r, InferConfig(fusion_weight=0.5, ridge_norm=True)), 0.5 * p + 0.5 * np.array([[0, .25, .75], p[1]]))


def test_geo_fusion_is_a_normalised_weighted_geometric_mean():
    zs = np.array([[2.0, 0.0, 1.0]]); r = np.array([[0.2, 0.5, 0.3]]); p = softmax(zs, axis=1)
    out = fuse(zs, r, InferConfig(fusion_weight=0.25, geo_fusion=True))
    g = p ** 0.75 * r ** 0.25
    assert np.allclose(out, g / g.sum()) and np.isclose(out.sum(), 1.0)
    assert np.allclose(fuse(zs, r, InferConfig(fusion_weight=0.0, geo_fusion=True)), p)
    assert np.allclose(fuse(zs, r, InferConfig(fusion_weight=1.0, geo_fusion=True)), r / r.sum())
    assert np.all(np.isfinite(fuse(zs, np.array([[-1.0, 0.0, 2.0]]), InferConfig(geo_fusion=True))))   # negative/zero ridge scores are clipped


# ------------------------------------------------------------------ prototype head
def test_proto_scores_are_a_softmax_over_cosines_to_the_class_means():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 8)); labels = rng.integers(0, 3, 30)
    L = np.full((30, 3), -5.0); L[np.arange(30), labels] = 5.0
    h = heads.ProtoLearner(8, scale=10.0)
    res = h.add_task(X, L, 3)
    assert np.array_equal(res.labels, labels)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    P = np.stack([Xn[labels == c].sum(0) for c in range(3)]); P /= np.linalg.norm(P, axis=1, keepdims=True)
    Q = rng.normal(size=(5, 8)); Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    assert np.allclose(h.scores(Q), softmax(10.0 * Qn @ P.T, axis=1))


def test_proto_is_recursive_weighted_and_masks_unlabelled_columns():
    rng = np.random.default_rng(1)
    X1, X2 = rng.normal(size=(20, 6)), rng.normal(size=(25, 6))
    def labelled(n, K, lab):
        L = np.full((n, K), -4.0); L[np.arange(n), lab] = 4.0
        return L
    l1, l2 = rng.integers(0, 2, 20), rng.integers(0, 4, 25)
    a = heads.ProtoLearner(6); a.add_task(X1, labelled(20, 2, l1), 2); a.add_task(X2, labelled(25, 4, l2), 2)
    b = heads.ProtoLearner(6)
    b.add_task(np.vstack([X1, X2[:0]]), labelled(20, 2, l1), 2)
    b.add_task(X2, labelled(25, 4, l2), 2)
    assert np.allclose(a.S, b.S) and a.num_classes == 4
    # an old column keeps accumulating on the later task (labels range over all seen classes)
    n1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True); n2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    for c in range(4):
        exp = n1[l1 == c].sum(0) + n2[l2 == c].sum(0)
        assert np.allclose(a.S[c], exp) and np.isclose(a.cnt[c], (l1 == c).sum() + (l2 == c).sum())
    assert (l2 < 2).any()                                                 # the check above does cover an old column getting later data
    # weights scale a sample's contribution
    h = heads.ProtoLearner(6); L = labelled(20, 2, l1)
    h.add_task(X1, L, 2, base_weights=np.full(20, 2.0))
    g = heads.ProtoLearner(6); g.add_task(X1, L, 2)
    assert np.allclose(h.S, 2 * g.S) and np.allclose(h.cnt, 2 * g.cnt)
    # a column nobody was labelled with can never win
    z = heads.ProtoLearner(6); z.add_task(X1, labelled(20, 2, np.zeros(20, int)), 2)
    s = z.scores(rng.normal(size=(7, 6)))
    assert np.allclose(s[:, 1], 0.0) and np.allclose(s.sum(1), 1.0)


def test_proto_input_validation():
    h = heads.ProtoLearner(4)
    with pytest.raises(ValueError, match="disagree"):
        h.add_task(np.zeros((3, 4)), np.zeros((2, 2)), 2)
    with pytest.raises(ValueError, match="more columns"):
        h.add_task(np.zeros((3, 4)), np.zeros((3, 5)), 2)
    with pytest.raises(ValueError, match="base_weights"):
        h.add_task(np.ones((3, 4)), np.zeros((3, 2)), 2, base_weights=np.array([1, -1, 1.0]))
    with pytest.raises(RuntimeError):
        heads.ProtoLearner(4).scores(np.zeros((1, 4)))
    with pytest.raises(NotImplementedError):
        h.add_labeled()


def test_make_learner_picks_the_head_and_rejects_bad_combinations():
    c = inc.RunConfig(tasks=TASKS)
    assert isinstance(heads.make_learner(c, 8), nc.N2LLearner)
    assert isinstance(heads.make_learner(dataclasses.replace(c, head="proto"), 8), heads.ProtoLearner)
    with pytest.raises(ValueError, match="unknown head"):
        heads.make_learner(dataclasses.replace(c, head="mlp"), 8)
    with pytest.raises(ValueError, match="expand"):
        heads.make_learner(dataclasses.replace(c, head="proto", expand=10), 8)
    with pytest.raises(ValueError, match="only defined"):
        heads.make_learner(dataclasses.replace(c, head="proto", variant="gt_ceiling"), 8)


def test_proto_head_runs_end_to_end_and_differs_from_ridge(bundle):
    r = inc.run_incremental(bundle, inc.RunConfig(tasks=TASKS, head="proto"), log=lambda *_: None)
    e = inc.run_incremental(bundle, inc.RunConfig(tasks=TASKS), log=lambda *_: None)
    assert len(r["curve"]) == len(TASKS) and 0.0 <= r["final_AP50"] <= 1.0 and r["diag"][0]["k"] == 0
    assert r["curve"] != e["curve"] or r["diag"] != e["diag"]


# ------------------------------------------------------------------ the variants
def test_every_variant_differs_from_the_tuned_e5_in_exactly_its_named_setting():
    t = _tune_dict(n2l__n_rounds=0, pseudo__tau_bg=0.5, pseudo__use_loc=True)
    t["E5"]["infer"].update(yolo_gamma=0.25, topk_classes=5)
    cfgs = vr.variant_configs(TASKS, t)
    base = cfgs["E5_tuned"]
    assert set(cfgs) == set(vr.REFERENCES) | set(vr.VARIANTS) | {vr.E0_PREFIX + n for n in vr.RESCORE}
    want = {"min_yolo_0.1": ("pseudo", "min_yolo", 0.1), "topfrac_0.7": ("pseudo", "class_topfrac", 0.7), "soft_0.5": ("n2l", "soft", 0.5),
            "ridge_norm": ("infer", "ridge_norm", True), "geo_fusion": ("infer", "geo_fusion", True)}
    for name, c in cfgs.items():
        if name in vr.REFERENCES or name.startswith(vr.E0_PREFIX):        # E0__ rows: test_exp_round1
            continue
        if name == "rf_off" and base.n2l.n_rounds == 0:                   # already off in this tune dict: identical by design
            continue
        diffs = {}
        for f in dataclasses.fields(c):
            a, b = getattr(c, f.name), getattr(base, f.name)
            if dataclasses.is_dataclass(a):
                diffs.update({f"{f.name}.{g.name}": getattr(a, g.name) for g in dataclasses.fields(a) if getattr(a, g.name) != getattr(b, g.name)})
            elif a != b:
                diffs[f.name] = a
        assert diffs and len(diffs) <= 2, (name, diffs)
        if name in want:
            g, k, v = want[name]
            assert diffs == {f"{g}.{k}": v}
    assert {k: v for k, v in dataclasses.asdict(cfgs["head_proto"]).items() if k == "head"} == {"head": "proto"}
    assert cfgs["geo_fusion_ridge_norm"].infer.geo_fusion and cfgs["geo_fusion_ridge_norm"].infer.ridge_norm
    # tuned settings survive in every variant
    for name in vr.VARIANTS:
        assert cfgs[name].infer.yolo_gamma == 0.25 and cfgs[name].infer.topk_classes == 5 and cfgs[name].n2l.n_rounds == 0
        assert cfgs[name].pseudo.tau_bg == 0.5 and cfgs[name].pseudo.use_loc is True


def test_run_variants_runs_references_plus_requested_only_and_passes_n_folds(monkeypatch):
    seen = []
    monkeypatch.setattr(vr.inc, "run_incremental",
                        lambda b, cfg, log=None, probe=None: seen.append((cfg.head, cfg.n_folds)) or {"final_AP50": 0.5, "final_AP": 0.3})
    out = vr.run_variants(None, _tune_dict(), TASKS, only=["head_proto", "soft_0.5"], n_folds=7, log=lambda *_: None)
    assert list(out) == list(vr.REFERENCES) + ["head_proto", "soft_0.5"] and {n for _, n in seen} == {7}
    assert [h for h, _ in seen] == ["ridge"] * 3 + ["proto", "ridge"]
    with pytest.raises(ValueError, match="unknown variants"):
        vr.run_variants(None, _tune_dict(), TASKS, only=["nope"])


def test_variant_names_are_unique_documented_and_groups_named():
    assert all(len(v) == 3 and v[0] and v[1] and callable(v[2]) for v in vr.VARIANTS.values())
    assert len(vr.VARIANTS) == len(set(vr.VARIANTS)) >= 12


# ------------------------------------------------------------------ the CLI, on the tune split only
def _prep(wd):
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_tune", "20", "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *base])
    rp.main(["tune", *base, "--n_trials", "3"])
    return base


def test_cli_variants_scores_the_tune_images_matches_tune_json_and_writes_a_report(tmp_path, monkeypatch):
    wd = str(tmp_path / "wd")
    base = _prep(wd)
    seen = []
    orig = pl.build_bundle
    monkeypatch.setattr(pl, "build_bundle", lambda m, tc, ti, ct, ec, eids, cv: seen.append(list(eids)) or orig(m, tc, ti, ct, ec, eids, cv))
    rp.main(["variants", *base, "--only", "soft_0.5", "head_proto", "ridge_norm", "--n_folds", "5", "--tag", "t1"])
    sp = json.load(open(os.path.join(wd, "splits.json")))
    assert seen == [sp["tune_ids"]]                                           # only the tune images, never the eval ones
    res = json.load(open(os.path.join(wd, "results", "variants_t1.json")))
    assert set(res) == {"E0_zeroshot_tuned", "E5_tuned", "E6_gt_ceiling_tuned", "soft_0.5", "head_proto", "ridge_norm"}
    t = json.load(open(os.path.join(wd, "results", "tune.json")))
    assert res["E5_tuned"]["final_AP50"] == pytest.approx(t["tune_scores"]["E5_tuned"])    # same number the tuner reported
    assert res["E0_zeroshot_tuned"]["final_AP50"] == pytest.approx(t["tune_scores"]["E0_tuned"])
    assert len(res["soft_0.5"]["final_folds"]["AP50"]) == 5
    md = open(os.path.join(wd, "results", "variants_t1.md")).read()
    assert "difference vs E5_tuned" in md and "| soft_0.5 |" in md and "best of N" in md and "CANDIDATE" in md
    assert not os.path.exists(os.path.join(wd, "results", "variants.json"))


def test_cli_variants_on_eval_scores_only_the_eval_images_and_writes_separate_files(tmp_path, monkeypatch):
    wd = str(tmp_path / "wd")
    base = _prep(wd)
    rp.main(["variants", *base, "--only", "soft_1", "--n_folds", "5", "--tag", "t1"])             # tune run first: must stay untouched
    tune_json = open(os.path.join(wd, "results", "variants_t1.json")).read()
    seen = []
    orig = pl.build_bundle
    monkeypatch.setattr(pl, "build_bundle", lambda m, tc, ti, ct, ec, eids, cv: seen.append((ec, list(eids))) or orig(m, tc, ti, ct, ec, eids, cv))
    rp.main(["variants", *base, "--on", "eval", "--only", "soft_1", "topfrac_0.5", "--n_folds", "5", "--tag", "t1"])
    sp = json.load(open(os.path.join(wd, "splits.json")))
    assert len(seen) == 1 and seen[0][1] == sp["eval_ids"] and seen[0][0].endswith("eval")
    assert set(sp["eval_ids"]).isdisjoint(sp["tune_ids"])
    assert open(os.path.join(wd, "results", "variants_t1.json")).read() == tune_json               # tune results not overwritten
    res = json.load(open(os.path.join(wd, "results", "variants_eval_t1.json")))
    assert set(res) == {"E0_zeroshot_tuned", "E5_tuned", "E6_gt_ceiling_tuned", "soft_1", "topfrac_0.5"}
    assert len(res["soft_1"]["final_folds"]["AP50"]) == 5
    md = open(os.path.join(wd, "results", "variants_eval_t1.md")).read()
    assert "scored on the EVAL images" in md and "CONFIRMATION run: 2 variant(s)" in md
    assert "best of N" not in md and "TUNE images" not in md


def test_variants_report_wording_depends_on_the_split(tmp_path):
    import compare_runs as cr
    folds = {n: {"final_AP50": 0.55, "final_folds": {"AP50": [0.5, 0.6, 0.55], "AP50_new": [0.2, 0.3, 0.25]}} for n in vr.REFERENCES + ("soft_1",)}
    pth = str(tmp_path / "v.json"); json.dump(folds, open(pth, "w"))
    tune = vr.format_variants(folds, pth, 3)
    ev = vr.format_variants(folds, pth, 3, on="eval")
    assert "TUNE images" in tune and "best of N" in tune and "CANDIDATE" in tune and "EVAL" not in tune
    assert "EVAL images" in ev and "CONFIRMATION" in ev and "best of N" not in ev


def test_cli_variants_needs_tune_json_and_tune_images(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *base])
    with pytest.raises(SystemExit, match="run `tune` first"):
        rp.main(["variants", *base])
    os.makedirs(os.path.join(wd, "results"), exist_ok=True)
    open(os.path.join(wd, "results", "tune.json"), "w").write("{}")
    with pytest.raises(SystemExit, match="--n_tune"):
        rp.main(["variants", *base])

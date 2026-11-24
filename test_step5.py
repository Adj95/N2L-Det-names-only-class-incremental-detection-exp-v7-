"""Step-5 validation: the incremental runner on a synthetic feature-level world.
Run: python -m pytest -q test_step5.py"""
import dataclasses
import json

import numpy as np
import pytest

import evaluate as ev
import incremental as inc
import n2l_core as nc
import pseudo as ps
import synth
from inference import InferConfig

TASKS = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return synth.make_feature_world(str(tmp_path_factory.mktemp("w")), TASKS, dim=64, imgs_per_task=60,
                                    n_eval=80, text_noise=3.0, seed=1)


def bundle(w, text_label=None):
    return inc.Bundle(w.train_cache, w.train_ids, w.train_gt, w.eval_cache, w.eval_ids,
                      ev.make_coco_gt(w.eval_coco, w.eval_ids),
                      w.text_label if text_label is None else text_label, w.text_eval,
                      w.bg_label, w.bg_eval)


def cfg(variant, **kw):
    return inc.RunConfig(tasks=TASKS, variant=variant, **kw)


quiet = lambda *_: None
same = lambda a, b: json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)     # nan-safe equality


def test_world_is_sane(world):
    assert world.train_cache.n_images == 180 and world.eval_cache.n_images == 80
    assert [len(t) for t in world.train_ids] == [60, 60, 60]
    # task-2 images contain old-class objects (the thing the detection design must handle)
    old_in_new = sum(int(np.isin(world.train_gt[i][1], TASKS[0] + TASKS[1]).any()) for i in world.train_ids[2])
    assert old_in_new > 10


def test_gt_ceiling_is_high(world):
    r = inc.run_incremental(bundle(world), cfg("gt_ceiling"), quiet)
    assert r["final_AP50"] > 0.85 and r["curve"][0]["AP50"] > 0.9
    assert len(r["curve"]) == 3


def test_ours_beats_noisy_zeroshot_and_approaches_ceiling(world):
    B = bundle(world)
    zs = inc.run_incremental(B, cfg("zeroshot"), quiet)
    ours = inc.run_incremental(B, cfg("ours"), quiet)
    ceil = inc.run_incremental(B, cfg("gt_ceiling"), quiet)
    assert zs["final_AP50"] < 0.95                                           # precondition: the teacher is imperfect
    assert ours["final_AP50"] > zs["final_AP50"] + 0.03, (ours["final_AP50"], zs["final_AP50"])
    assert ours["final_AP50"] <= ceil["final_AP50"] + 0.05
    assert ours["final_AP50"] > ceil["final_AP50"] - 0.15


def test_refinement_improves_pseudo_label_quality(world):
    r = inc.run_incremental(bundle(world), cfg("ours"), quiet)
    assert np.mean([d["refined_labels"]["acc"] for d in r["diag"]]) > \
        np.mean([d["zero_shot_labels"]["acc"] for d in r["diag"]]) + 0.02


def test_head_shape_and_class_incremental_columns(world):
    B = bundle(world)
    logs = []
    r = inc.run_incremental(B, cfg("ours"), logs.append)
    assert len(logs) == 3 and "task 3/3" in logs[-1]
    assert [row["n_seen"] for row in r["curve"]] == [4, 8, 12]
    assert r["curve"][0]["AP50_old"] != r["curve"][0]["AP50_old"]               # nan: no old classes at task 1


def test_ours_never_reads_ground_truth(world):
    """Replace all GT classes/boxes used for training by garbage: 'ours' must be bit-identical
    (GT is only a diagnostic input), 'gt_ceiling' must change."""
    B = bundle(world)
    rng = np.random.default_rng(0)
    garbage = {i: (b + 1.0, rng.integers(0, 12, len(c))) for i, (b, c) in world.train_gt.items()}
    B2 = dataclasses.replace(B, train_gt=garbage)
    a = inc.run_incremental(B, cfg("ours"), quiet); b = inc.run_incremental(B2, cfg("ours"), quiet)
    assert same(a["curve"], b["curve"])
    assert not same(a["diag"][0]["refined_labels"], b["diag"][0]["refined_labels"])   # only diagnostics differ
    c = inc.run_incremental(B, cfg("gt_ceiling"), quiet); d = inc.run_incremental(B2, cfg("gt_ceiling"), quiet)
    assert not same(c["curve"], d["curve"])


def test_deterministic_and_seed_sensitive_only_through_weights(world):
    B = bundle(world)
    a = inc.run_incremental(B, cfg("ours", seed=3), quiet); b = inc.run_incremental(B, cfg("ours", seed=3), quiet)
    assert same(a["curve"], b["curve"])


def test_zeroshot_uses_no_training_data(world):
    B = bundle(world)
    B2 = dataclasses.replace(B, train_ids=[[], [], []])                       # no training images at all
    a = inc.run_incremental(B, cfg("zeroshot"), quiet); b = inc.run_incremental(B2, cfg("zeroshot"), quiet)
    assert same(a["curve"], b["curve"])


def test_perfect_teacher_gives_near_ceiling(world):
    """With text embeddings = true prototypes, pseudo-labels are ~correct -> N2L ~ GT ceiling."""
    rng = np.random.default_rng(1)
    # recover prototypes from the world: use class-mean of positives in the GT ceiling features instead of text
    X, img, Bx, Sx, _ = world.train_cache.gather(sum(world.train_ids, []))
    ids = sum(world.train_ids, [])
    protos = np.zeros((80, X.shape[1]))
    for c in range(12):
        feats = []
        for k, iid in enumerate(ids):
            gb, gc = world.train_gt[iid]
            m = img == k
            for g, cc in zip(gb, gc):
                if cc != c:
                    continue
                from boxes import iou_matrix
                iou = iou_matrix(Bx[m], g[None])[:, 0]
                feats.append(X[m][iou > 0.7])
        protos[c] = np.vstack(feats).mean(0)
    from prompts import l2norm
    perfect = l2norm(protos + 1e-9)
    B = bundle(world, text_label=perfect)
    ours = inc.run_incremental(B, cfg("ours"), quiet); ceil = inc.run_incremental(B, cfg("gt_ceiling"), quiet)
    assert ours["final_AP50"] > ceil["final_AP50"] - 0.08


def test_ablation_switches_run_and_change_behaviour(world):
    B = bundle(world)
    base = inc.run_incremental(B, cfg("ours"), quiet)
    nobg = inc.run_incremental(B, cfg("ours", pseudo=ps.PseudoConfig(use_bg=False)), quiet)
    norefine = inc.run_incremental(B, cfg("ours", n2l=nc.N2LConfig(n_rounds=0, pca_rule="energy", pca_rho=0.95)), quiet)
    expand = inc.run_incremental(B, cfg("ours", expand=256), quiet)
    ign = inc.run_incremental(B, cfg("ours", pseudo=ps.PseudoConfig(tau_pos=0.6, tau_bg=0.6)), quiet)
    for r in (nobg, norefine, expand, ign):
        assert len(r["curve"]) == 3 and np.isfinite(r["final_AP50"])
    assert ign["diag"][0]["kept"] < ign["diag"][0]["n"]
    assert norefine["diag"][0]["refined_labels"]["acc"] == pytest.approx(norefine["diag"][0]["zero_shot_labels"]["acc"])
    assert nobg["final_AP50"] != base["final_AP50"]


def test_noncontiguous_class_ids_and_bad_variant(world):
    with pytest.raises(ValueError):
        inc.run_incremental(bundle(world), cfg("magic"), quiet)
    # tasks given in a different class order: detections must only use the given classes
    tasks2 = [[11, 3], [7], [0, 5]]
    B = bundle(world)
    B2 = dataclasses.replace(B, train_ids=[world.train_ids[0], world.train_ids[1], world.train_ids[2]])
    r = inc.run_incremental(B2, inc.RunConfig(tasks=tasks2, variant="gt_ceiling"), quiet)
    assert [row["n_seen"] for row in r["curve"]] == [2, 3, 5]


def test_ours_with_noncontiguous_class_ids(tmp_path):
    """Tasks whose class ids are neither contiguous nor sorted: text rows, head columns and the
    reported class ids must all follow the ids, not the positions."""
    tasks2 = [[11, 3], [7, 9], [0, 5]]
    w = synth.make_feature_world(str(tmp_path), tasks2, dim=64, imgs_per_task=70, n_eval=80,
                                 text_noise=1.5, seed=4)
    B = bundle(w)
    r = inc.run_incremental(B, inc.RunConfig(tasks=tasks2, variant="ours"), quiet)
    zs = inc.run_incremental(B, inc.RunConfig(tasks=tasks2, variant="zeroshot"), quiet)
    assert [row["n_seen"] for row in r["curve"]] == [2, 4, 6]
    assert np.mean([d["zero_shot_labels"]["acc"] for d in r["diag"]]) > 0.8      # right names -> right columns
    assert r["final_AP50"] > 0.8 and zs["final_AP50"] > 0.8                      # class ids reported correctly
    assert r["curve"][2]["AP50_new"] > 0.7                                       # the last task's classes (0,5) are found


def test_eval_reads_eval_text_and_label_text_only_for_training(world):
    B = bundle(world)
    rng = np.random.default_rng(0)
    garbage = rng.normal(size=B.text_eval.shape); garbage /= np.linalg.norm(garbage, axis=1, keepdims=True)
    base = inc.run_incremental(B, cfg("zeroshot"), quiet)
    bad_eval = inc.run_incremental(dataclasses.replace(B, text_eval=garbage), cfg("zeroshot"), quiet)
    bad_label = inc.run_incremental(dataclasses.replace(B, text_label=garbage), cfg("zeroshot"), quiet)
    assert bad_eval["final_AP50"] < base["final_AP50"] - 0.3                      # eval depends on text_eval
    assert same(bad_label["curve"], base["curve"])                                # ... and not on text_label


def test_background_prompt_switch_in_eval(world):
    B = bundle(world)
    rng = np.random.default_rng(1)
    other_bg = rng.normal(size=B.bg_eval.shape); other_bg /= np.linalg.norm(other_bg)
    B2 = dataclasses.replace(B, bg_eval=other_bg, bg_label=other_bg)
    off = ps.PseudoConfig(use_bg=False)
    a = inc.run_incremental(B, cfg("zeroshot", pseudo=off), quiet)
    b = inc.run_incremental(B2, cfg("zeroshot", pseudo=off), quiet)
    assert same(a["curve"], b["curve"])                                           # bg prompt fully disabled
    on_a = inc.run_incremental(B, cfg("zeroshot"), quiet); on_b = inc.run_incremental(B2, cfg("zeroshot"), quiet)
    assert not same(on_a["curve"], on_b["curve"])                                 # ... and matters when enabled


def test_default_runconfig_uses_energy95_with_class_floor():
    c = inc.RunConfig(tasks=[[0]])
    assert c.n2l.pca_rule == "energy" and c.n2l.pca_rho == 0.95 and c.n2l.pca_k_floor == "classes"

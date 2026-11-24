#!/usr/bin/env python
"""run_pipeline.py -- the N2L-Det command line. One sub-command per validated build step.

    selftest      pre-flight: whole pipeline on a synthetic world with fake models (CPU, ~1 min)
    fetch         choose + download the COCO images (or convert VOC with --dataset voc); write splits.json
    gate-clip     STEP 2 gate  : zero-shot CLIP accuracy on ground-truth-box crops
    repro-cifar   STEP 2 check : reproduce N2L CIFAR-100 B0Inc10 with our core
    make-yolo-data / train-yolo   STEP 3 : build the 40-class dataset, train the base detector
    extract       STEP 3/4 : YOLO proposals + CLIP crop features -> cache
    gate-recall   STEP 3 gate  : box recall of the frozen proposer on each task's classes
    run           STEP 5-7 : E0 / E5 / ablations / E6 on cached features, mAP table

All sub-commands read/write under --workdir; paths found by `fetch` are remembered in paths.json.
`--fake` swaps in FakeModels + a synthetic world so every sub-command can be exercised offline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import coco_fetch as cf
import coco_utils as cu
import custom_data as cd
import pipeline as pl
import voc_utils as vu

N2L_EXP_MARKER = "N2L_EXP_MARKER_v7"     # the experimental code base (round 1+). Kaggle cells for it grep this; the frozen v6 lacks it.


# ------------------------------------------------------------------ helpers
def _p(workdir, *a):
    return os.path.join(workdir, *a)


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def require_cache(cache_dir, ids, what):
    """Fail early and clearly when the feature cache does not hold every image splits.json refers to."""
    if not os.path.exists(os.path.join(cache_dir, "image_ids.npy")):
        raise SystemExit(f"{what}: no feature cache at {cache_dir} -- run `extract` first")
    from feature_cache import FeatureCache
    missing = sorted(set(ids) - set(map(int, FeatureCache(cache_dir).image_ids)))
    if missing:
        raise SystemExit(f"{what}: cache {cache_dir} lacks {len(missing)} of the {len(set(ids))} images listed in "
                         f"splits.json (e.g. {missing[:3]}). Was `fetch` re-run after `extract`? Run `extract` again.")


def check_clip_consistency(a):
    """Text embeddings (made at `run`) must come from the same CLIP weights as the cached image features."""
    f = _p(a.workdir, "clip.json")
    if a.fake or not os.path.exists(f):
        return
    j = _read_json(f)
    rec = j["clip_weights"]
    if rec != a.clip_weights:
        raise SystemExit(f"features were extracted with CLIP weights '{rec}' but this command uses "
                         f"'{a.clip_weights}'. Pass --clip_weights {rec} (or re-run `extract`).")
    rec_m, now_m = j.get("clip_model", "ViT-B-16"), getattr(a, "clip_model", "ViT-B-16")      # older clip.json: the only model then
    if rec_m != now_m:
        raise SystemExit(f"features were extracted with CLIP model '{rec_m}' but this command uses '{now_m}'. "
                         f"Pass --clip_model {rec_m} (or re-run `extract`).")


def _out(a, stem, ext):
    """results/<stem>[_<tag>].<ext> : --tag keeps a second run from overwriting the first."""
    tag = getattr(a, "tag", "") or ""
    return _p(a.workdir, "results", f"{stem}_{tag}.{ext}" if tag else f"{stem}.{ext}")


def get_models(a):
    pl.set_label_templates(getattr(a, "label_templates", 1))
    if a.fake:
        from fake_models import FakeModels
        return FakeModels(n_classes=a.fake_classes)
    from real_models import RealModels
    kw = {}
    if getattr(a, "proposer", "yolo") != "yolo":              # only pass the new knobs when they are used
        kw = dict(proposer=a.proposer, world_weights=a.world_weights,
                  world_prompts=[w.strip() for w in a.world_prompts.split(",") if w.strip()])
    return RealModels(clip_pretrained=a.clip_weights, clip_model=getattr(a, "clip_model", "ViT-B-16"),
                      yolo_weights=a.yolo_weights, device=a.device,
                      conf=a.conf, max_det=a.max_det, **kw)


def parse_custom_tasks(spec: str):
    """'0,1,2|3,4,5|6,7,8' -> [[0,1,2],[3,4,5],[6,7,8]] (class indices 0..79, disjoint, non-empty)."""
    tasks = [[int(x) for x in part.split(",") if x.strip() != ""] for part in spec.split("|")]
    flat = [c for t in tasks for c in t]
    if any(len(t) == 0 for t in tasks) or len(set(flat)) != len(flat) or min(flat) < 0 or max(flat) > 79:
        raise ValueError(f"bad custom task spec {spec!r}: need disjoint non-empty groups of class indices 0..79")
    return tasks


def scheme_of(a):
    """The --tasks string, defaulting per dataset (COCO 40+40, VOC 15+5; custom datasets must say)."""
    ds = getattr(a, "dataset", "coco")
    if a.tasks:
        return a.tasks
    if ds == "custom":
        raise SystemExit("--dataset custom needs --tasks, e.g. '10+5' or '5+5x3' (sizes must add up to the number of classes)")
    return "15+5" if ds == "voc" else "40+40"


def _canonical_tasks(a, scheme):
    ds = getattr(a, "dataset", "coco")
    if ds == "voc" and not a.fake:
        return vu.voc_tasks(scheme)                 # VOC order, returned as COCO class indices
    if "|" in scheme:
        return parse_custom_tasks(scheme)
    if a.fake:
        k = a.fake_classes // 3
        return [list(range(i * k, (i + 1) * k)) for i in range(3)]
    return cu.make_tasks(scheme, cu.num_classes() if ds == "custom" else 80)


def tasks_of(a):
    """'40+40' | 'a,b|c,d' | 'random:<scheme>[:seed]' (same sizes, classes shuffled with that seed; default seed 0)."""
    scheme = scheme_of(a)
    if scheme.startswith("random:"):
        parts = scheme.split(":")
        if len(parts) not in (2, 3) or (len(parts) == 3 and not parts[2].lstrip("-").isdigit()):
            raise ValueError(f"bad random scheme {scheme!r}; use e.g. 'random:15+5:3'")
        return cu.random_tasks(_canonical_tasks(a, parts[1]), int(parts[2]) if len(parts) == 3 else 0)
    return _canonical_tasks(a, scheme)


def load_paths(a):
    pth = _read_json(_p(a.workdir, "paths.json"))
    if pth.get("classes"):                                   # a non-COCO dataset: re-install its class space
        c = pth["classes"]
        cu.set_class_space(c["names"], c["ids"], c.get("prompt_names"))
    return pth, cf.load_coco(pth["ann_train"]), cf.load_coco(pth["ann_val"])


# ------------------------------------------------------------------ commands
def split_ids(sp, which, what):
    """Image ids of the 'eval' | 'tune' | 'final' split, or a clear error naming the command that creates it."""
    ids = sp.get({"eval": "eval_ids", "tune": "tune_ids", "final": "final_ids"}[which], [])
    if not ids:
        flag = {"tune": "--n_tune", "final": "--n_final"}.get(which, "--n_eval")
        raise SystemExit(f"{what} --eval_on {which}: no {which} images; run `fetch ... {flag} N` then `extract --splits {which}`")
    return ids


def guard_final(a, what):
    """The final split is for ONE look at frozen settings. A second use needs --allow_repeat_final and is recorded."""
    f = _p(a.workdir, "results", "final_log.json")
    log = _read_json(f) if os.path.exists(f) else []
    if log and not getattr(a, "allow_repeat_final", False):
        raise SystemExit(f"{what} --eval_on final: the final split was already used ({log[-1]['command']}); looking at it again "
                         "after changing anything makes it a second tuning set. If you really mean it, pass --allow_repeat_final "
                         "(the repeat is recorded in results/final_log.json and printed in the table).")
    log.append({"command": what, "repeat": bool(log)})
    _write_json(f, log)
    return bool(log[:-1])


def draw_splits(all_val, seed, n_eval, n_tune, n_final=0, exclude_final=()):
    """Disjoint eval / tune / final image ids from the validation pool. Each later split is drawn from what is left, with its own
    generator, so eval_ids are identical with or without --n_tune and --n_final, and tune_ids with or without --n_final
    (existing feature caches and tune.json stay valid)."""
    rng = np.random.default_rng(seed)
    eval_ids = sorted(rng.choice(all_val, size=min(n_eval, len(all_val)), replace=False).tolist())
    spare = sorted(set(all_val) - set(eval_ids))
    tune_ids = sorted(np.random.default_rng(seed + 3).choice(spare, size=min(n_tune, len(spare)), replace=False).tolist()) \
        if n_tune and spare else []
    spare2 = sorted(set(spare) - set(tune_ids) - set(exclude_final))      # exclude_final: images the detector's checkpoint was chosen on
    final_ids = sorted(np.random.default_rng(seed + 5).choice(spare2, size=min(n_final, len(spare2)), replace=False).tolist()) \
        if n_final and spare2 else []
    return eval_ids, tune_ids, final_ids


def cmd_fetch(a):
    os.makedirs(a.workdir, exist_ok=True)
    if a.fake:
        from fake_models import make_image_world
        tasks = tasks_of(a)
        ct, cv, dtr, dva = make_image_world(_p(a.workdir, "fake_coco"), tasks, a.n_train_per_task,
                                            a.n_eval + a.n_tune + getattr(a, "n_final", 0) + (a.n_yolo_val if a.n_yolo_train else 0),
                                            seed=a.seed)
        ann_train, ann_val = _p(a.workdir, "fake_coco", "train.json"), _p(a.workdir, "fake_coco", "val.json")
        _write_json(ann_train, ct); _write_json(ann_val, cv)
    elif a.dataset == "voc":
        tasks = tasks_of(a)
        info = vu.prepare_voc(a.voc_root, a.voc_out or _p(a.workdir, "voc_coco"), a.voc_protocol,
                              download=not a.no_download)
        ann_train, ann_val, dtr, dva = info["ann_train"], info["ann_val"], info["img_dir_train"], info["img_dir_val"]
        ct, cv = cf.load_coco(ann_train), cf.load_coco(ann_val)
        cu.verify_categories(ct["categories"])
        a.voc_protocol_used = info["protocol"]
    elif a.dataset == "custom":
        need = {"--ann_train": a.ann_train, "--ann_val": a.ann_val, "--img_dir_train": a.img_dir_train,
                "--img_dir_val": a.img_dir_val}
        if not all(need.values()):
            raise SystemExit("--dataset custom needs " + ", ".join(k for k, v in need.items() if not v))
        ann_train, ann_val, dtr, dva = a.ann_train, a.ann_val, a.img_dir_train, a.img_dir_val
        ct, cv = cf.load_coco(ann_train), cf.load_coco(ann_val)
        classes = cd.read_classes(ct, cv, cd.load_name_map(a.name_map))
        cu.set_class_space(classes["names"], classes["ids"], classes["prompt_names"])
        cd.check_images(ct, dtr, "train"); cd.check_images(cv, dva, "val")
        tasks = tasks_of(a)
    else:
        tasks = tasks_of(a)
        if a.ann_train and a.ann_val:
            ann_train, ann_val = a.ann_train, a.ann_val
        else:
            ap = cf.ensure_annotations(a.coco_root)
            ann_train, ann_val = ap["train"], ap["val"]
        ct, cv = cf.load_coco(ann_train), cf.load_coco(ann_val)
        cu.verify_categories(ct["categories"])
        dtr = a.img_dir_train or _p(a.coco_root, "images", "train2017")
        dva = a.img_dir_val or _p(a.coco_root, "images", "val2017")
    train_ids = [cu.select_task_images(ct, cls, a.n_train_per_task, a.seed) for cls in tasks]
    all_val = sorted(im["id"] for im in cv["images"])
    yolo_train = cu.select_task_images(ct, tasks[0], a.n_yolo_train, a.seed + 1) if a.n_yolo_train else []
    # Re-running fetch on a work dir whose detector is already trained must not pretend the detector used other validation images:
    # keep the validation images it was really trained with, and keep them out of the final split.
    prev = _read_json(_p(a.workdir, "splits.json")) if os.path.exists(_p(a.workdir, "splits.json")) else {}
    reuse = bool(prev.get("yolo_val")) and prev.get("yolo_train") == yolo_train and bool(yolo_train)
    eval_ids, tune_ids, final_ids = draw_splits(all_val, a.seed, a.n_eval, a.n_tune, getattr(a, "n_final", 0),
                                                exclude_final=prev["yolo_val"] if reuse else ())
    if getattr(a, "n_final", 0) and len(final_ids) < a.n_final:
        print(f"WARNING: only {len(final_ids)} of the requested {a.n_final} final images are available "
              "(the validation pool is exhausted by eval, tune and the detector's validation images)")
    yolo_val_overlap = False
    if reuse:
        yolo_val = list(prev["yolo_val"])
        n_ov = len(set(yolo_val) & (set(eval_ids) | set(tune_ids)))
        yolo_val_overlap = n_ov > 0
        if n_ov:
            print(f"NOTE: {n_ov} of the detector's {len(yolo_val)} validation images (used to pick its best checkpoint) are eval/tune "
                  "images. The final split excludes them. Disclose this, or retrain the detector with the new fetch.")
    else:
        # the detector's checkpoint is chosen on its validation images: they must not be images we later score
        yolo_val = cu.select_task_images(cv, tasks[0], a.n_yolo_val, a.seed + 2,
                                         exclude=set(eval_ids) | set(tune_ids) | set(final_ids)) if a.n_yolo_train else []
        if a.n_yolo_train and len(yolo_val) < a.n_yolo_val:
            if not yolo_val:                              # tiny validation pool: fall back, but say so (and record it)
                yolo_val = cu.select_task_images(cv, tasks[0], a.n_yolo_val, a.seed + 2)
                yolo_val_overlap = True
                print("WARNING: every validation image is an eval/tune/final image, so the detector's checkpoint selection used "
                      "images that are also scored. Use a larger validation pool or fewer --n_eval/--n_tune/--n_final.")
            else:
                print(f"WARNING: only {len(yolo_val)} of the requested {a.n_yolo_val} detector-validation images are left "
                      "(they must not overlap the eval/tune/final images)")
    if not a.fake and a.dataset == "coco":
        need_tr = sorted({i for ids in train_ids for i in ids} | set(yolo_train))
        need_va = sorted(set(eval_ids) | set(yolo_val) | set(tune_ids) | set(final_ids))
        print(f"downloading {len(need_tr)} train + {len(need_va)} val images ...")
        cf.fetch_images(ct, need_tr, dtr); cf.fetch_images(cv, need_va, dva)
    _write_json(_p(a.workdir, "paths.json"), {"ann_train": ann_train, "ann_val": ann_val,
                                              "img_dir_train": dtr, "img_dir_val": dva,
                                              **({"classes": classes} if a.dataset == "custom" and not a.fake else {})})
    _write_json(_p(a.workdir, "splits.json"), {"tasks": tasks, "train_ids": train_ids, "eval_ids": eval_ids,
                                               "tune_ids": tune_ids, "final_ids": final_ids, "yolo_train": yolo_train, "yolo_val": yolo_val,
                                               "yolo_val_overlaps_scored": yolo_val_overlap,
                                               "dataset": "fake" if a.fake else a.dataset,
                                               "scheme": scheme_of(a),
                                               "voc_protocol": getattr(a, "voc_protocol_used", None)})
    print("tasks:", [len(t) for t in tasks], "| images/task:", [len(x) for x in train_ids],
          "| eval images:", len(eval_ids), "| tune images:", len(tune_ids), "| final images:", len(final_ids))
    return 0


def cmd_gate_clip(a):
    pth, ct, cv = load_paths(a)
    sp = _read_json(_p(a.workdir, "splits.json"))
    models = get_models(a)
    res = pl.stage_gate_clip(models, cv, pth["img_dir_val"], sp["eval_ids"], sp["tasks"])
    print("ZERO-SHOT CLIP ON GT-BOX CROPS (E0 ceiling, paper-free number you must report)")
    for k, v in res.items():
        print(f"  {k:>14s}: {v:.4f}")
    _write_json(_p(a.workdir, "results", "gate_clip.json"), res)
    return 0


def cmd_make_yolo_data(a):
    pth, ct, cv = load_paths(a)
    sp = _read_json(_p(a.workdir, "splits.json"))
    base = sp["tasks"][0]
    root = os.path.abspath(_p(a.workdir, "yolo_data"))
    by_id_t = {im["id"]: im for im in ct["images"]}; by_id_v = {im["id"]: im for im in cv["images"]}
    for split, coco, ids, src, by_id in (("train", ct, sp["yolo_train"], pth["img_dir_train"], by_id_t),
                                         ("val", cv, sp["yolo_val"], pth["img_dir_val"], by_id_v)):
        imgd = os.path.join(root, "images", split); os.makedirs(imgd, exist_ok=True)
        for i in ids:
            dst = os.path.join(imgd, by_id[i]["file_name"])
            if not os.path.exists(dst):
                os.symlink(os.path.abspath(os.path.join(src, by_id[i]["file_name"])), dst)
        st = cu.write_yolo_labels(coco, ids, base, os.path.join(root, "labels", split),
                                  agnostic=getattr(a, "agnostic", False))
        print(f"  {split}: {st}")
    agn = getattr(a, "agnostic", False)
    cu.write_dataset_yaml(os.path.join(root, "data.yaml"), root, ["object"] if agn else cu.prompt_names(base))
    print("dataset yaml:", os.path.join(root, "data.yaml"),
          f"| {'ONE class-agnostic class' if agn else str(len(base)) + ' base classes'} | NOTE: new-class objects inside these images are unlabeled (= background)")
    return 0


def cmd_train_yolo(a):
    from yolo_tools import train_base
    project = os.path.abspath(_p(a.workdir, "runs"))          # absolute: ultralytics may re-root a relative path
    best = train_base(_p(a.workdir, "yolo_data", "data.yaml"), init=a.yolo_init, size=a.yolo_size,
                      epochs=a.yolo_epochs, imgsz=a.imgsz, batch=a.batch, project=project, device=a.device)
    if not os.path.exists(best):                               # fall back to the newest best.pt under our project dir
        import glob
        found = sorted(glob.glob(os.path.join(project, "**", "weights", "best.pt"), recursive=True), key=os.path.getmtime)
        if not found:
            raise SystemExit(f"training reported {best} but no best.pt exists under {project}; check the training log")
        best = found[-1]
    best = os.path.abspath(best)
    _write_json(_p(a.workdir, "yolo_best.json"), {"weights": best, "init": a.yolo_init})
    print("best weights:", best, "| init:", a.yolo_init,
          "(COCO-pretrained init LEAKS new classes -- report it)" if a.yolo_init == "coco" else
          "(ImageNet-pretrained backbone: no box leak; ImageNet labels overlap some class names -- report it)"
          if a.yolo_init == "imagenet" else "")
    return 0


def _yolo_weights(a):
    if a.yolo_weights or a.fake or getattr(a, "proposer", "yolo") == "yoloworld":
        return a.yolo_weights
    f = _p(a.workdir, "yolo_best.json")
    return _read_json(f)["weights"] if os.path.exists(f) else None


EXTRACT_KEYS = ("conf", "max_det", "max_boxes", "pad", "proposer", "world_prompts")


def cmd_extract(a):
    a.yolo_weights = _yolo_weights(a)
    pth, ct, cv = load_paths(a)
    sp = _read_json(_p(a.workdir, "splits.json"))
    want = getattr(a, "splits", None) or ["train", "eval"] + (["tune"] if sp.get("tune_ids") else []) + (["final"] if sp.get("final_ids") else [])
    for sname, flag in (("tune", "--n_tune"), ("final", "--n_final")):
        if sname in want and not sp.get(sname + "_ids"):
            raise SystemExit(f"extract: no {sname} images in splits.json -- run `fetch ... {flag} N` first (same other args)")
    settings = {k: getattr(a, k, d) for k, d in zip(EXTRACT_KEYS, (None, None, None, None, "yolo", "object"))}
    f = _p(a.workdir, "clip.json")
    if getattr(a, "splits", None) and os.path.exists(f):                  # a partial extract must match what the other caches were made with
        prev = _read_json(f)
        diff = {k: (prev[k], v) for k, v in settings.items() if k in prev and prev[k] != v}
        if diff:
            raise SystemExit(f"extract --splits {' '.join(a.splits)}: settings differ from the existing caches "
                             f"{ {k: {'cache': c, 'now': n} for k, (c, n) in diff.items()} }; pass the same values.")
        if any(k not in prev for k in EXTRACT_KEYS):
            print("WARNING: the existing caches did not record their extract settings; make sure these match: "
                  + ", ".join(f"{k}={v}" for k, v in settings.items()))
    models = get_models(a)
    t0, n = time.time(), 0
    for name in want:
        ids = sorted({i for x in sp["train_ids"] for i in x}) if name == "train" else sp[name + "_ids"]
        coco, img_dir = (ct, pth["img_dir_train"]) if name == "train" else (cv, pth["img_dir_val"])
        pl.stage_extract(models, coco, ids, img_dir, _p(a.workdir, "cache", name), a.pad, a.max_boxes)
        n += len(ids)
    prev = _read_json(f) if os.path.exists(f) else {}
    _write_json(f, {**prev, "clip_weights": "fake" if a.fake else a.clip_weights,
                    "clip_model": "fake" if a.fake else getattr(a, "clip_model", "ViT-B-16"), "dim": int(models.dim),
                    "yolo_weights": a.yolo_weights, **settings})
    print(f"extracted {n} images ({', '.join(want)}) in {time.time() - t0:.0f}s")
    return 0


def cmd_gate_recall(a):
    pth, ct, cv = load_paths(a)
    sp = _read_json(_p(a.workdir, "splits.json"))
    require_cache(_p(a.workdir, "cache", "eval"), sp["eval_ids"], "gate-recall")
    r = pl.stage_gate_recall(_p(a.workdir, "cache", "eval"), cv, sp["eval_ids"], sp["tasks"])
    print("GATE 1 -- recall of the frozen proposer (eval images). Low recall on later tasks caps everything.")
    for k, v in r.items():
        print(f"  {k}: {v}")
    _write_json(_p(a.workdir, "results", "gate_recall.json"), r)
    return 0


def cmd_run(a):
    pth, ct, cv = load_paths(a)
    sp = _read_json(_p(a.workdir, "splits.json"))
    check_clip_consistency(a)
    which = getattr(a, "eval_on", "eval")                    # 'eval' (reported numbers) | 'tune' (choosing settings) | 'final' (once, frozen)
    ids = split_ids(sp, which, "run")
    require_cache(_p(a.workdir, "cache", "train"), [i for x in sp["train_ids"] for i in x], "run")
    require_cache(_p(a.workdir, "cache", which), ids, "run")
    if which == "final" and not getattr(a, "tuned", None):
        raise SystemExit("run --eval_on final needs --tuned results/tune.json: the final split is for FROZEN settings only")
    tuned = getattr(a, "tuned", None)
    if tuned and not os.path.exists(tuned):
        raise SystemExit(f"--tuned {tuned} not found; run `tune` first")
    models = get_models(a)
    B = pl.build_bundle(models, _p(a.workdir, "cache", "train"), sp["train_ids"], ct,
                        _p(a.workdir, "cache", which), ids, cv)
    repeated = guard_final(a, "run") if which == "final" else False          # recorded only once everything it needs exists
    res = pl.run_ablation(B, sp["tasks"], only=a.only, tuned=tuned)
    print("\n" + pl.format_table(res))
    suffix = {"eval": "", "tune": "_tune", "final": "_final"}[which]
    pl.save_results(res, _p(a.workdir, "results", f"ablation{suffix}.json"))
    with open(_p(a.workdir, "results", f"ablation{suffix}.md"), "w") as f:
        f.write(("**REPEATED use of the final split: not a clean test**\n\n" if repeated else "") + pl.format_table(res) + "\n")
    return 0


def cmd_baseline(a):
    import baselines as bl
    pth, ct, cv = load_paths(a)
    sp = _read_json(_p(a.workdir, "splits.json"))
    which = getattr(a, "eval_on", "eval")
    ids = split_ids(sp, which, "baseline")
    if which == "final":
        guard_final(a, "baseline")
    from evaluate import make_coco_gt
    names = cu.prompt_names([c for t in sp["tasks"] for c in t])
    det = bl.WorldDetector(names, weights=a.world_weights, conf=a.world_conf, device=a.device)
    res = bl.score_detector(det, pl.iter_images(cv, ids, pth["img_dir_val"]), ids, make_coco_gt(cv, ids), sp["tasks"])
    f = _p(a.workdir, "results", "baselines.json")
    prev = _read_json(f) if os.path.exists(f) else {}
    prev["yolo_world" + {"eval": "", "tune": "_tune", "final": "_final"}[which]] = res
    pl.save_results(prev, f)
    r = res["curve"][-1]
    print(f"YOLO-World zero-shot on {len(ids)} images: AP={r['AP']:.3f} AP50={r['AP50']:.3f} "
          f"base={r['AP50_base']:.3f} new={r['AP50_new']:.3f}   (reference only: pre-trained on O365/GQA/Flickr30k)")
    return 0


def cmd_tune(a):
    import tune as tn
    pth, ct, cv = load_paths(a)
    sp = _read_json(_p(a.workdir, "splits.json"))
    check_clip_consistency(a)
    if not sp.get("tune_ids"):
        raise SystemExit("tune: no tune images; run `fetch ... --n_tune N` then `extract --splits tune`")
    require_cache(_p(a.workdir, "cache", "train"), [i for x in sp["train_ids"] for i in x], "tune")
    require_cache(_p(a.workdir, "cache", "tune"), sp["tune_ids"], "tune")
    models = get_models(a)
    B = pl.build_bundle(models, _p(a.workdir, "cache", "train"), sp["train_ids"], ct,
                        _p(a.workdir, "cache", "tune"), sp["tune_ids"], cv)
    t = tn.run_search(B, sp["tasks"], n_trials=a.n_trials, seed=a.tune_seed, objective=a.objective)
    rep = tn.format_report(t)
    print("\n" + rep)
    pl.save_results(t, _out(a, "tune", "json"))
    with open(_out(a, "tune", "md"), "w") as f:
        f.write(rep + "\n")
    return 0


def _tuned_bundle(a, what, split="eval"):
    """Shared set-up for the commands that re-use tune.json on the eval images (loo, seeds) or on the tune images (variants)."""
    pth, ct, cv = load_paths(a)
    sp = _read_json(_p(a.workdir, "splits.json"))
    check_clip_consistency(a)
    tj = getattr(a, "tuned", None) or _p(a.workdir, "results", "tune.json")
    if not os.path.exists(tj):
        raise SystemExit(f"{what}: {tj} not found; run `tune` first")
    require_cache(_p(a.workdir, "cache", "train"), [i for x in sp["train_ids"] for i in x], what)
    ids = split_ids(sp, split, what)
    require_cache(_p(a.workdir, "cache", split), ids, what)
    B = pl.build_bundle(get_models(a), _p(a.workdir, "cache", "train"), sp["train_ids"], ct,
                        _p(a.workdir, "cache", split), ids, cv)
    return B, sp, _read_json(tj)


def cmd_loo(a):
    import compare_runs as cr
    import tune as tn
    B, sp, t = _tuned_bundle(a, "loo")
    print("changed settings:", ", ".join(tn.changed_settings(t)) or "(none: tuned == default)")
    res = tn.run_loo(B, t, sp["tasks"], n_folds=a.n_folds)
    pl.save_results(res, _out(a, "loo", "json"))
    folds = cr.load_folds(_out(a, "loo", "json"))
    out = []
    for ref in ("E5_tuned", "E0_zeroshot_tuned"):
        for metric in ("AP50", "AP50_new"):
            out.append(f"### {metric}, difference vs {ref}\n\n" + cr.format_rows(cr.compare(folds, ref, metric), ref, metric))
    note = ("A `E5_tuned_minus_X` row clearly BELOW E5_tuned (negative diff, 'yes') means setting X carries part of the gain; "
            f"a row equal to E5_tuned means X is not needed. `group.*` rows revert several settings together. {a.n_folds} image folds: indicative only.")
    text = "\n\n".join(out) + "\n\n" + note
    print("\n" + text)
    with open(_out(a, "loo", "md"), "w") as f:
        f.write(text + "\n")
    return 0


def cmd_seeds(a):
    import tune as tn
    B, sp, t = _tuned_bundle(a, "seeds")
    res = tn.run_seeds(B, sp["tasks"], t, n_seeds=a.n_seeds, train_frac=a.train_frac, seed0=a.seed0)
    rep = tn.format_seeds(res)
    print("\n" + rep)
    pl.save_results(res, _out(a, "seeds", "json"))
    with open(_out(a, "seeds", "md"), "w") as f:
        f.write(rep + "\n")
    return 0


def cmd_variants(a):
    import variants as vr
    on = getattr(a, "on", "tune")                                         # tune images (default) or eval images (confirmation only)
    stem = {"tune": "variants", "eval": "variants_eval", "final": "variants_final"}[on]
    B, sp, t = _tuned_bundle(a, "variants", split=on)
    if on == "final":                                                     # exp v7: one look, rows fixed beforehand, recorded
        if not a.only:
            raise SystemExit("variants --on final: name the rows with --only (fixed before looking); no open-ended search on final")
        repeated = guard_final(a, "variants " + " ".join(a.only))
        print("FINAL SPLIT" + (" (REPEAT, recorded in results/final_log.json)" if repeated else " (first and only look)"))
    res = vr.run_variants(B, t, sp["tasks"], only=a.only, n_folds=a.n_folds)
    pl.save_results(res, _out(a, stem, "json"))
    rep = vr.format_variants(res, _out(a, stem, "json"), a.n_folds, on=on)
    print("\n" + rep)
    with open(_out(a, stem, "md"), "w") as f:
        f.write(rep + "\n")
    return 0


def cmd_ovd(a):
    """exp: open-vocabulary detector baselines (YOLO-World, OWLv2, Grounding DINO) on the eval / tune / final images,
    scored like our runs (same COCOeval, same image folds). Reference rows: they need no new-class labels but were
    pre-trained on box data that contains these classes. The final split has its OWN log (results/final_log_ovd.json):
    scoring a fixed outside model never touches our method's one look at the final split."""
    import ovd
    from evaluate import make_coco_gt
    pth, ct, cv = load_paths(a)
    sp = _read_json(_p(a.workdir, "splits.json"))
    which = a.eval_on
    ids = split_ids(sp, which, "ovd")
    if a.limit:
        if which == "final":
            raise SystemExit("ovd: --limit is a smoke test; not allowed on the final split")
        ids = ids[:a.limit]
    if which == "final":
        f = _p(a.workdir, "results", "final_log_ovd.json")
        log = _read_json(f) if os.path.exists(f) else []
        done = sorted({e["model"] for e in log} & set(a.models))
        if done and not a.allow_repeat_final:
            raise SystemExit(f"ovd --eval_on final: {done} already scored on the final split; pass --allow_repeat_final "
                             "to repeat (recorded)")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        _write_json(f, log + [{"model": m, "repeat": m in done} for m in a.models])
    if a.restore_voc and sp.get("dataset") == "voc":
        ovd.restore_voc_images(pth, a.voc_root)
    by_id = {im["id"]: im for im in cv["images"]}
    missing = [i for i in ids if not os.path.exists(os.path.join(pth["img_dir_val"], by_id[i]["file_name"]))]
    if missing:
        raise SystemExit(f"ovd: {len(missing)} of {len(ids)} images are missing under {pth['img_dir_val']} "
                         "(VOC: pass --restore_voc --voc_root <VOC files>; COCO: attach the same image folders)")
    tasks = sp["tasks"]
    class_order = [int(c) for t in tasks for c in t]
    names = cu.prompt_names(class_order)
    gt = make_coco_gt(cv, ids)
    stem = f"ovd_{which}" + ("_smoke" if a.limit else "")
    fjson = _out(a, stem, "json")
    res = _read_json(fjson) if os.path.exists(fjson) else {}
    ids_model = {"yoloworld": a.world_weights, "owlv2": a.owl_model, "gdino": a.gdino_model}
    print(f"ovd on {len(ids)} {which} images, {len(class_order)} classes: {names}")
    for m in a.models:
        t0 = time.time()
        det = ovd.make_detector(m, names, ids_model[m], a.device)
        dets = ovd.collect(det, pl.iter_images(cv, ids, pth["img_dir_val"]), class_order)
        r = ovd.score(dets, ids, gt, tasks, n_folds=a.n_folds, name=m)
        r.update(model_id=ids_model[m], seconds=round(time.time() - t0, 1), split=which)
        res[m] = r
        pl.save_results(res, fjson)
        c = r["curve"][-1]
        print(f"  {m:10s} ({ids_model[m]}): AP={c['AP']:.3f} AP50={c['AP50']:.3f} base={c['AP50_base']:.3f} "
              f"new={c['AP50_new']:.3f}  [{r['seconds']:.0f} s, {r['n_dets']} dets]")
        del det
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:                                   # noqa: BLE001 - CPU-only machines
            pass
    names_by_idx = {c: n for c, n in zip(class_order, names)}
    rep = (f"# Open-vocabulary detector baselines, {which} images ({len(ids)}), all {len(class_order)} class names\n\n"
           "REFERENCE rows: no new-class labels (like ours), but pre-trained on large box/grounding data that already "
           "contains these classes, so the 'new' classes are not new to them.\n\n" + ovd.format_rows(res, names_by_idx))
    if a.ref:
        rep += ovd.paired_rows(res, _read_json(a.ref), a.ref_name)
    print("\n" + rep)
    with open(_out(a, stem, "md"), "w") as f:
        f.write(rep)
    return 0


def cmd_clone(a):
    """A new work dir that REUSES the images, splits and base detector of --src but none of its caches or results, so a
    different extraction setting (pad, CLIP model, proposer settings) can be tested without touching the original."""
    import shutil
    src = a.src
    need = ["splits.json", "paths.json"]
    missing = [f for f in need if not os.path.exists(_p(src, f))]
    if missing:
        raise SystemExit(f"clone: {src} lacks {', '.join(missing)}; run `fetch` there first")
    if os.path.abspath(src) == os.path.abspath(a.workdir):
        raise SystemExit("clone: --src and --workdir are the same directory")
    if os.path.exists(_p(a.workdir, "splits.json")):
        raise SystemExit(f"clone: {a.workdir} already holds a run; choose a new --workdir")
    os.makedirs(a.workdir, exist_ok=True)
    for f in need + ["yolo_best.json"]:
        if os.path.exists(_p(src, f)):
            shutil.copy(_p(src, f), _p(a.workdir, f))
    yb = _p(a.workdir, "yolo_best.json")
    if os.path.exists(yb):                                      # keep pointing at the ORIGINAL weights (not copied)
        w = _read_json(yb)
        w["weights"] = os.path.abspath(w["weights"]) if os.path.isabs(w["weights"]) else os.path.abspath(_p(src, w["weights"]))
        _write_json(yb, {**w, "cloned_from": os.path.abspath(src)})
    print(f"cloned {src} -> {a.workdir} (splits, paths" + (", detector weights" if os.path.exists(yb) else "") + "); no caches or results copied")
    return 0


def cmd_sweep(a):
    import tune as tn
    B, sp, t = _tuned_bundle(a, "sweep")
    res = tn.run_sweep(B, sp["tasks"], t)
    rep = tn.format_sweep(res)
    print("\n" + rep)
    pl.save_results(res, _out(a, "sweep", "json"))
    with open(_out(a, "sweep", "md"), "w") as f:
        f.write(rep + "\n")
    return 0


def cmd_perclass(a):
    import tune as tn
    B, sp, t = _tuned_bundle(a, "perclass")
    res = tn.run_perclass(B, sp["tasks"], t)
    rep = tn.format_perclass(res)
    print("\n" + rep)
    pl.save_results(res, _out(a, "perclass", "json"))
    with open(_out(a, "perclass", "md"), "w") as f:
        f.write(rep + "\n")
    return 0


def cmd_diagnose(a):
    import diagnose as dg
    pth, ct, cv = load_paths(a)
    sp = _read_json(_p(a.workdir, "splits.json"))
    check_clip_consistency(a)
    which = getattr(a, "eval_on", "eval")
    ids = split_ids(sp, which, "diagnose")
    if which == "final":
        raise SystemExit("diagnose --eval_on final: not allowed (diagnosis is for developing the method; the final split is only for the frozen run)")
    require_cache(_p(a.workdir, "cache", "train"), [i for x in sp["train_ids"] for i in x], "diagnose")
    require_cache(_p(a.workdir, "cache", which), ids, "diagnose")
    models = get_models(a)
    B = pl.build_bundle(models, _p(a.workdir, "cache", "train"), sp["train_ids"], ct,
                        _p(a.workdir, "cache", which), ids, cv)
    d = dg.run_diagnosis(B, sp["tasks"])
    rep = dg.format_report(d)
    print("\n" + rep)
    pl.save_results(d, _p(a.workdir, "results", "diagnose.json"))
    with open(_p(a.workdir, "results", "diagnose.md"), "w") as f:
        f.write(rep + "\n")
    return 0


def cmd_boxdiag(a):
    """exp round 1: where do new classes lose AP? coverage of the cached proposals + naming + oracle ladder (tune/eval only)."""
    import boxdiag as bd
    import tune as tn
    which = getattr(a, "eval_on", "tune")
    if which == "final":
        raise SystemExit("boxdiag --eval_on final: not allowed (diagnosis is for developing the method)")
    B, sp, t = _tuned_bundle(a, "boxdiag", split=which)
    cf_ = tn.tuned_configs(t, sp["tasks"])
    cfgs = {n: cf_[n] for n in ("E5_tuned", "E0_zeroshot_tuned")}
    res = bd.run_boxdiag(B, sp["tasks"], cfgs, n_folds=a.n_folds)
    rep = bd.format_report(res, which)
    print("\n" + rep)
    pl.save_results(res, _out(a, f"boxdiag_{which}", "json"))
    with open(_out(a, f"boxdiag_{which}", "md"), "w") as f:
        f.write(rep + "\n")
    return 0


def _make_proposer(kind, a, nms_iou, cache):
    """One proposer per (kind, NMS IoU); the heavy model is loaded once per kind and reused (cache dict)."""
    if a.fake:
        from fake_models import FakeModels
        m = cache.setdefault("fake", FakeModels(n_classes=a.fake_classes))
        return m.propose
    from yolo_tools import YoloProposer, YoloWorldProposer
    if kind == "yolo":
        w = _yolo_weights(a)
        if not w:
            raise SystemExit("propdiag: no base-detector weights (yolo_best.json or --yolo_weights)")
        if "yolo" not in cache:
            from ultralytics import YOLO
            cache["yolo"] = YOLO(w)
        return YoloProposer(w, conf=a.conf, iou=nms_iou, max_det=a.max_det, device=a.device, model=cache["yolo"])
    if "yoloworld" not in cache:
        from ultralytics import YOLOWorld
        cache["yoloworld"] = YOLOWorld(a.world_weights)
    prompts = [w.strip() for w in a.world_prompts.split(",") if w.strip()]
    return YoloWorldProposer(a.world_weights, prompts=prompts, conf=a.conf, iou=nms_iou, max_det=a.max_det, device=a.device,
                             model=cache["yoloworld"])


def cmd_propdiag(a):
    """exp round 1: coverage of FRESH proposals (base YOLO and/or YOLO-World, several NMS IoUs, up to max_det boxes) on tune/eval
    images. Only boxes and scores are computed (no CLIP), so it is quick. Nothing is cached or used by the pipeline."""
    import boxdiag as bd
    pth, ct, cv = load_paths(a)
    sp = _read_json(_p(a.workdir, "splits.json"))
    splits = a.splits or ["tune", "eval"]
    if "final" in splits:
        raise SystemExit("propdiag: the final split is not allowed (diagnosis is for developing the method)")
    out, cache = {}, {}
    for kind in a.proposers:
        for nms in a.nms_ious:
            prop = _make_proposer(kind, a, nms, cache)
            for which in splits:
                ids = split_ids(sp, which, "propdiag")
                t0, props = time.time(), {}
                for iid, img in pl.iter_images(cv, ids, pth["img_dir_val"]):
                    b, s = prop(img)
                    props[int(iid)] = (np.asarray(b, float).reshape(-1, 4), np.asarray(s, float).reshape(-1))
                sec = (time.time() - t0) / max(len(ids), 1)
                name = f"{kind}{'[' + a.world_prompts + ']' if kind == 'yoloworld' else ''}_nms{nms:g}_{which}"
                out[name] = {"coverage": bd.coverage(props, cu.gt_by_image(cv, ids), sp["tasks"]),
                             "settings": {"proposer": kind, "conf": a.conf, "nms_iou": nms, "max_det": a.max_det, "split": which,
                                          **({"world_prompts": a.world_prompts, "world_weights": a.world_weights} if kind == "yoloworld" else {})},
                             "seconds_per_image": sec}
                print(f"  {name}: {sec:.3f} s/image, {out[name]['coverage']['boxes_per_image']:.1f} boxes/image", flush=True)
                pl.save_results(out, _out(a, "propdiag", "json"))          # after every block: a later crash keeps the earlier ones
    rep = bd.format_propdiag(out)
    print("\n" + rep)
    pl.save_results(out, _out(a, "propdiag", "json"))
    with open(_out(a, "propdiag", "md"), "w") as f:
        f.write(rep + "\n")
    return 0


def cifar_features(clip, cifar_root, train, batch=512):
    """CIFAR-100 -> (CLIP features, original labels, torchvision class names).
    Preprocessing as the official config: bicubic squash 32->224, then CLIP mean/std (inside the encoder)."""
    import torchvision                                    # GPU box only
    from PIL import Image
    ds = torchvision.datasets.CIFAR100(cifar_root, train=train, download=True)
    out = []
    for i in range(0, len(ds), batch):
        crops = np.stack([np.asarray(Image.fromarray(ds.data[j]).resize((224, 224), Image.BICUBIC))
                          for j in range(i, min(i + batch, len(ds)))])
        out.append(clip.encode_images_u8(crops))
    return np.concatenate(out), np.array(ds.targets), list(ds.classes)


def cmd_repro_cifar(a):
    import cifar_repro as cr
    import prompts as pr
    from clip_encoder import ClipEncoder
    clip = ClipEncoder(pretrained=a.clip_weights, device=a.device)
    names, templates = cr.load_official_text_assets(a.n2l_repo)
    Xtr, ytr0, tv_names = cifar_features(clip, a.cifar_root, True)
    Xte, yte0, _ = cifar_features(clip, a.cifar_root, False)
    norm = lambda n: n.lower().replace("_", " ").replace("-", " ").strip()
    bad = [(i, x, y) for i, (x, y) in enumerate(zip(names, tv_names)) if norm(x) != norm(y)]
    if bad:                      # same class index must mean the same class in labels.json and torchvision
        print(f"WARNING: {len(bad)} official label names differ from torchvision's (first 5): {bad[:5]}")
        print("         (harmless if only wording differs, e.g. 'aquarium fish' vs 'aquarium_fish'; "
              "fatal if the ORDER differs)")
    order = cr.official_class_order()
    ytr, yte = cr.remap_targets(ytr0, order), cr.remap_targets(yte0, order)
    names_o = [names[i] for i in order]
    label_emb = pr.class_embeddings(names_o, clip.encode_text, templates[:1])
    eval_emb = pr.class_embeddings(names_o, clip.encode_text, templates)
    res = cr.run_cifar_protocol(Xtr, ytr, Xte, yte, label_emb, eval_emb, expand=a.expand)
    print(f"\nREPRO  A-bar fused {100 * res['A_bar_fused']:.2f}  A_B fused {100 * res['A_B_fused']:.2f}"
          f"  | zero-shot A-bar {100 * res['A_bar_zeroshot']:.2f}   (your BTP numbers: 87.86 / 81.17 / 81.81)")
    _write_json(_p(a.workdir, "results", "repro_cifar.json"), res)
    return 0


def selftest_checks(res, clip_acc, rec, n_tasks=3):
    """The pass/fail criteria of `selftest` (the fake world is easy, so every bar is high)."""
    return {
        "GT-crop zero-shot accuracy > 0.95": clip_acc > 0.95,
        "proposer recall@0.5 > 0.9 on every task": all(rec[f"task{t}"]["recall@0.5"] > 0.9 for t in range(n_tasks)),
        "E6 ceiling AP50 > 0.9": res["E6_gt_ceiling"]["final_AP50"] > 0.9,
        "E5 AP50 > 0.85": res["E5_ours"]["final_AP50"] > 0.85,
        "E0 AP50 > 0.85": res["E0_zeroshot"]["final_AP50"] > 0.85,
    }


def cmd_selftest(a):
    """Whole pipeline on a synthetic image world with fake models. Fails loudly on any glue bug."""
    import tempfile
    a.fake, a.fake_classes = True, 9
    a.workdir = a.workdir or tempfile.mkdtemp(prefix="n2l_selftest_")
    a.n_train_per_task, a.n_eval, a.n_yolo_train, a.n_yolo_val = 30, 40, 20, 10
    a.max_boxes, a.pad = 50, 0.1
    a.only = ["E0_zeroshot", "E5_ours", "E5_no_refine", "E5_no_bg", "E6_gt_ceiling"]
    for fn in (cmd_fetch, cmd_gate_clip, cmd_make_yolo_data, cmd_extract, cmd_gate_recall, cmd_run):
        if fn(a) != 0:
            raise SystemExit(f"selftest step {fn.__name__} failed")
    res = _read_json(_p(a.workdir, "results", "ablation.json"))
    clip_acc = _read_json(_p(a.workdir, "results", "gate_clip.json"))["all"]
    rec = _read_json(_p(a.workdir, "results", "gate_recall.json"))
    checks = selftest_checks(res, clip_acc, rec)
    for k, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + k)
    if not all(checks.values()):
        raise SystemExit("SELFTEST FAILED")
    print("SELFTEST OK  (workdir:", a.workdir + ")")
    return 0


# ------------------------------------------------------------------ CLI
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--workdir", default=None, help="default: ./work (selftest: a temp dir)")
        sp.add_argument("--coco_root", default="coco")
        sp.add_argument("--ann_train"); sp.add_argument("--ann_val")
        sp.add_argument("--img_dir_train"); sp.add_argument("--img_dir_val")
        sp.add_argument("--dataset", choices=["coco", "voc", "custom"], default="coco",
                        help="custom = any COCO-format json with its own classes (needs --ann_*, --img_dir_*, --tasks)")
        sp.add_argument("--name_map", default=None,
                        help="custom: json {category name: text for CLIP prompts}, e.g. {\"aeroplane\": \"airplane\"}")
        sp.add_argument("--tasks", default=None,
                        help="coco: '40+40' (default) | '70+10' | '40+4x10'; voc: '15+5' (default) | '10+10' | '19+1' | "
                             "'10+2x5'; or custom 'a,b,c|d,e' (COCO class indices; for voc: VOC positions 0..19); "
                             "'random:<scheme>[:seed]' shuffles which classes land in which task")
        sp.add_argument("--voc_root", default="voc", help="folder holding (or to download) VOC2007/VOC2012")
        sp.add_argument("--voc_protocol", choices=["auto", "07+12", "07", "12"], default="auto")
        sp.add_argument("--voc_out", default=None, help="converted json + image links (default <workdir>/voc_coco)")
        sp.add_argument("--no_download", action="store_true", help="never download VOC; fail if it is not found")
        sp.add_argument("--clip_weights", default="openai", help="open_clip tag, e.g. openai | laion400m_e32")
        sp.add_argument("--clip_model", default="ViT-B-16", help="open_clip architecture, e.g. ViT-B-16 (default) | ViT-L-14 | ViT-B-32")
        sp.add_argument("--yolo_weights"); sp.add_argument("--device")
        sp.add_argument("--n_train_per_task", type=int, default=2000)
        sp.add_argument("--n_eval", type=int, default=500)
        sp.add_argument("--n_tune", type=int, default=0,
                        help="extra held-out val images (disjoint from the eval images) for choosing hyper-parameters")
        sp.add_argument("--n_final", type=int, default=0,
                        help="untouched final test images (disjoint from eval and tune); used once, with frozen settings: run --eval_on final")
        sp.add_argument("--allow_repeat_final", action="store_true", help="allow a second look at the final split (recorded)")
        sp.add_argument("--n_yolo_train", type=int, default=0, help="base-class images for YOLO training")
        sp.add_argument("--n_yolo_val", type=int, default=300)
        sp.add_argument("--pad", type=float, default=0.1); sp.add_argument("--max_boxes", type=int, default=100)
        sp.add_argument("--conf", type=float, default=0.01); sp.add_argument("--max_det", type=int, default=100)
        sp.add_argument("--proposer", choices=["yolo", "yoloworld"], default="yolo",
                        help="yolo = our frozen base-trained detector (default); yoloworld = open-vocabulary proposer "
                             "(protocol change: needs your guide's approval; see README)")
        sp.add_argument("--world_weights", default="yolov8s-worldv2.pt")
        sp.add_argument("--world_prompts", default="object", help="comma-separated prompts for the yoloworld proposer")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--label_templates", type=int, default=1,
                        help="prompt templates used for the PSEUDO-LABEL text: 1 = first only (official N2L, default), 0 = all 10")
        sp.add_argument("--tag", default="", help="suffix for result files of tune/loo/seeds/sweep/perclass (keeps earlier results)")
        sp.add_argument("--fake", action="store_true", help="fake models + synthetic world (offline)")
        sp.add_argument("--fake_classes", type=int, default=9)
        sp.add_argument("--only", nargs="*", help="ablation names to run (default: all)")

    for name, fn, h in [("fetch", cmd_fetch, "choose + download images"),
                        ("gate-clip", cmd_gate_clip, "zero-shot CLIP on GT crops"),
                        ("make-yolo-data", cmd_make_yolo_data, "build base-class YOLO dataset"),
                        ("train-yolo", cmd_train_yolo, "train base detector"),
                        ("extract", cmd_extract, "proposals + CLIP features -> cache"),
                        ("gate-recall", cmd_gate_recall, "proposer box recall"),
                        ("run", cmd_run, "incremental experiments + mAP"),
                        ("baseline", cmd_baseline, "YOLO-World zero-shot reference on the same images"),
                        ("tune", cmd_tune, "choose settings on held-out tune images (E5 and E0 separately)"),
                        ("diagnose", cmd_diagnose, "why the head helps or not: pseudo-label calibration, head vs CLIP"),
                        ("loo", cmd_loo, "leave-one-out: which tuned setting carries the gain (eval images)"),
                        ("seeds", cmd_seeds, "re-train on random subsets of the training images (eval images fixed)"),
                        ("clone", cmd_clone, "new work dir reusing splits + detector of --src (for extraction-setting variants)"),
                        ("variants", cmd_variants, "single-change method variants of the tuned E5, scored on the TUNE images"),
                        ("sweep", cmd_sweep, "YOLO^gamma x fusion weight grid, zero-shot vs head (descriptive)"),
                        ("perclass", cmd_perclass, "per-class AP50: E0 tuned / E5 tuned / E6 ceiling"),
                        ("boxdiag", cmd_boxdiag, "exp: coverage, naming and oracle ladder of the cached proposals (tune/eval)"),
                        ("ovd", cmd_ovd, "exp: open-vocabulary detector baselines (YOLO-World, OWLv2, Grounding DINO)"),
                        ("propdiag", cmd_propdiag, "exp: coverage of fresh proposals (base YOLO / YOLO-World, NMS IoU, max_det)"),
                        ("repro-cifar", cmd_repro_cifar, "reproduce N2L on CIFAR-100"),
                        ("selftest", cmd_selftest, "offline pre-flight")]:
        sp = sub.add_parser(name, help=h)
        common(sp)
        sp.set_defaults(func=fn)
        if name == "train-yolo":
            sp.add_argument("--yolo_init", choices=["scratch", "coco", "imagenet"], default="scratch")
            sp.add_argument("--yolo_size", default="s"); sp.add_argument("--yolo_epochs", type=int, default=30)
            sp.add_argument("--imgsz", type=int, default=640); sp.add_argument("--batch", type=int, default=32)
        if name == "extract":
            sp.add_argument("--splits", nargs="*", choices=["train", "eval", "tune", "final"],
                            help="which caches to (re)build; default: train eval (+tune, +final when fetch made those ids)")
        if name == "make-yolo-data":
            sp.add_argument("--agnostic", action="store_true",
                            help="train ONE class-agnostic 'object' class instead of the base classes (proposer variant)")
        if name == "clone":
            sp.add_argument("--src", required=True, help="the finished work dir to reuse splits and detector from")
        if name == "baseline":
            sp.add_argument("--world_conf", type=float, default=0.001)
            sp.add_argument("--eval_on", choices=["eval", "tune", "final"], default="eval")
        if name == "tune":
            sp.add_argument("--n_trials", type=int, default=30, help="stage-1 random-search trials (incl. 2 fixed ones)")
            sp.add_argument("--objective", choices=["final_AP50", "final_AP", "A_bar_AP", "final_AP50_base"], default="final_AP50")
            sp.add_argument("--tune_seed", type=int, default=0)
        if name == "run":
            sp.add_argument("--tuned", default=None, help="results/tune.json: adds E5_tuned, E0_zeroshot_tuned, E6_gt_ceiling_tuned")
        if name == "boxdiag":
            sp.add_argument("--tuned", default=None, help="default: <workdir>/results/tune.json")
            sp.add_argument("--eval_on", choices=["tune", "eval", "final"], default="tune")
            sp.add_argument("--n_folds", type=int, default=10)
        if name == "ovd":
            sp.add_argument("--models", nargs="+", choices=["yoloworld", "owlv2", "gdino"], default=["yoloworld", "owlv2", "gdino"])
            sp.add_argument("--eval_on", choices=["eval", "tune", "final"], default="eval")
            sp.add_argument("--n_folds", type=int, default=10)
            sp.add_argument("--owl_model", default="google/owlv2-base-patch16-ensemble")
            sp.add_argument("--gdino_model", default="IDEA-Research/grounding-dino-tiny")
            sp.add_argument("--limit", type=int, default=0, help="smoke test on the first N images (results tagged _smoke)")
            sp.add_argument("--restore_voc", action="store_true",
                            help="re-create the VOC image links a work-dir tar leaves out, from the VOC files under --voc_root")
            sp.add_argument("--ref", default=None, help="result json with fold scores of our runs (e.g. results/variants_eval_r4.json)")
            sp.add_argument("--ref_name", default="E5_tuned")
        if name == "propdiag":
            sp.add_argument("--splits", nargs="*", choices=["tune", "eval", "final"], default=None, help="default: tune eval")
            sp.add_argument("--proposers", nargs="+", choices=["yolo", "yoloworld"], default=["yolo"])
            sp.add_argument("--nms_ious", nargs="+", type=float, default=[0.7])
        if name in ("loo", "seeds", "sweep", "perclass", "variants"):
            sp.add_argument("--tuned", default=None, help="default: <workdir>/results/tune.json")
        if name in ("loo", "variants"):
            sp.add_argument("--n_folds", type=int, default=10 if name == "variants" else 5,
                            help="image folds for the intervals (10 = tighter)")
        if name == "variants":
            sp.add_argument("--on", choices=["tune", "eval", "final"], default="tune",
                            help="tune (default): choose candidates on the tune images; eval: CONFIRM chosen candidates (use --only) on the eval images")
        if name == "seeds":
            sp.add_argument("--n_seeds", type=int, default=5)
            sp.add_argument("--train_frac", type=float, default=0.8, help="fraction of each task's training images per seed")
            sp.add_argument("--seed0", type=int, default=0)
        if name in ("run", "diagnose"):
            sp.add_argument("--eval_on", choices=["eval", "tune", "final"], default="eval",
                            help="'tune' scores on the held-out tune images (ablation_tune.json); 'final' scores the untouched final split ONCE (run only)")
        if name == "repro-cifar":
            sp.add_argument("--n2l_repo", required=False, default="ICLR2026-N2L")
            sp.add_argument("--cifar_root", default="data"); sp.add_argument("--expand", type=int, default=5000)
            sp.set_defaults(clip_weights="laion400m_e32")        # what the official CIFAR config uses
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.workdir is None and a.cmd != "selftest":
        a.workdir = "work"
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())

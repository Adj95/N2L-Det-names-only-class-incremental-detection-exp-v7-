"""Non-COCO datasets (own class names / ids) and random class splits."""
import json
import os

import numpy as np
import pytest

import coco_utils as cu
import custom_data as cd
import run_pipeline as rp
from fake_models import make_image_world

NAMES = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india"]
IDS = [101, 102, 105, 106, 110, 111, 112, 120, 121]                 # deliberately not 1..9 and not contiguous
TASKS9 = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


@pytest.fixture(autouse=True)
def restore_class_space():
    yield
    cu.reset_class_space()


def relabel(coco):
    """Re-express a COCO-80 fake world as a 9-class dataset with its own ids and names."""
    old2new = {cu.COCO_IDS[i]: IDS[i] for i in range(9)}
    out = json.loads(json.dumps(coco))
    for a in out["annotations"]:
        a["category_id"] = old2new[a["category_id"]]
    out["categories"] = [{"id": i, "name": n} for i, n in zip(IDS, NAMES)]
    return out


@pytest.fixture()
def custom_world(tmp_path):
    ct, cv, dtr, dva = make_image_world(str(tmp_path / "w"), TASKS9, 25, 60, seed=2)
    ct9, cv9 = relabel(ct), relabel(cv)
    json.dump(ct9, open(tmp_path / "train.json", "w")); json.dump(cv9, open(tmp_path / "val.json", "w"))
    return dict(tmp=tmp_path, train=str(tmp_path / "train.json"), val=str(tmp_path / "val.json"), dtr=dtr, dva=dva,
                ct=ct9, cv=cv9)


def argv(w, wd, *extra):
    return ["--workdir", wd, "--dataset", "custom", "--ann_train", w["train"], "--ann_val", w["val"],
            "--img_dir_train", w["dtr"], "--img_dir_val", w["dva"], "--tasks", "3+2x3", "--n_train_per_task", "20",
            "--n_eval", "30", "--n_tune", "10", "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50", *extra]


# ------------------------------------------------------------------ class space
def test_set_and_reset_class_space_in_place():
    ids_obj, names_obj, map_obj = cu.COCO_IDS, cu.COCO_NAMES, cu.ID2IDX
    cu.set_class_space(NAMES, IDS)
    assert cu.COCO_IDS is ids_obj and cu.COCO_NAMES is names_obj and cu.ID2IDX is map_obj     # same objects, mutated
    assert cu.num_classes() == 9 and cu.ID2IDX[105] == 2 and cu.prompt_names([1]) == ["bravo"]
    import evaluate as ev                                         # modules that imported the lists by name see it too
    assert ev.COCO_NAMES == NAMES
    cu.reset_class_space()
    assert cu.num_classes() == 80 and cu.COCO_NAMES[0] == "person" and cu.prompt_names([cu.COCO_NAMES.index("tv")]) == ["television"]


def test_custom_names_are_not_rewritten_by_the_coco_prompt_fix():
    cu.set_class_space(["tv", "mouse"], [1, 2])
    assert cu.prompt_names() == ["tv", "mouse"]                   # COCO's 'tv'->'television' fix must not apply
    cu.set_class_space(["tv", "mouse"], [1, 2], ["television set", "computer mouse"])
    assert cu.prompt_names() == ["television set", "computer mouse"]


def test_set_class_space_validation():
    for bad in (([], []), (["a"], [1, 2]), (["a", "a"], [1, 2]), (["a", "b"], [1, 1])):
        with pytest.raises(ValueError):
            cu.set_class_space(*bad)
    with pytest.raises(ValueError):
        cu.set_class_space(["a"], [1], ["x", "y"])
    assert cu.num_classes() == 80                                 # failed calls change nothing


# ------------------------------------------------------------------ read_classes / checks
def test_read_classes_sorted_by_id_with_name_map(custom_world):
    c = cd.read_classes(custom_world["ct"], custom_world["cv"], {"alpha": "alpha thing"})
    assert c["names"] == NAMES and c["ids"] == IDS and c["prompt_names"][0] == "alpha thing" and c["prompt_names"][1] == "bravo"


def test_read_classes_errors(custom_world):
    ct, cv = custom_world["ct"], custom_world["cv"]
    cv2 = json.loads(json.dumps(cv)); cv2["categories"][0]["name"] = "other"
    with pytest.raises(ValueError, match="different categories"):
        cd.read_classes(ct, cv2)
    dup = json.loads(json.dumps(ct)); dup["categories"][1]["name"] = "alpha"
    dup2 = json.loads(json.dumps(dup))
    with pytest.raises(ValueError, match="duplicate category names"):
        cd.read_classes(dup, dup2)
    bad = json.loads(json.dumps(ct)); bad["annotations"][0]["category_id"] = 9999
    with pytest.raises(ValueError, match="not in 'categories'"):
        cd.read_classes(bad, cv)
    with pytest.raises(ValueError, match="no 'categories'"):
        cd.read_classes({"annotations": []}, cv)
    with pytest.raises(ValueError, match="not categories"):
        cd.read_classes(ct, cv, {"zzz": "q"})


def test_check_images_missing_dir_and_missing_size(custom_world, tmp_path):
    cd.check_images(custom_world["ct"], custom_world["dtr"], "train")
    with pytest.raises(ValueError, match="missing under"):
        cd.check_images(custom_world["ct"], str(tmp_path / "nowhere"), "train")
    no_size = {"images": [{"id": 1, "file_name": "a.png"}], "annotations": []}
    with pytest.raises(ValueError, match="width/height"):
        cd.check_images(no_size, str(tmp_path), "val")


def test_name_map_file_validation(tmp_path):
    f = tmp_path / "m.json"; f.write_text(json.dumps({"a": "b"}))
    assert cd.load_name_map(str(f)) == {"a": "b"} and cd.load_name_map(None) is None
    f.write_text(json.dumps(["a"]))
    with pytest.raises(ValueError):
        cd.load_name_map(str(f))


# ------------------------------------------------------------------ the whole chain on a non-COCO dataset
def test_full_chain_on_a_custom_dataset(custom_world, tmp_path):
    w, wd = custom_world, str(tmp_path / "wd")
    rp.main(["fetch", *argv(w, wd)])                                # real (non-fake) loader
    pth = json.load(open(os.path.join(wd, "paths.json")))
    assert pth["classes"]["names"] == NAMES and pth["classes"]["ids"] == IDS
    sp = json.load(open(os.path.join(wd, "splits.json")))
    assert sp["tasks"] == TASKS9 and sp["dataset"] == "custom" and len(sp["tune_ids"]) == 10
    cu.reset_class_space()                                          # a new process would start from COCO-80
    fake = ["--fake_classes", "9"]
    # the remaining commands use fake CLIP/YOLO on the same files; they must re-install the class space themselves
    for cmd in ("gate-clip", "make-yolo-data", "extract", "gate-recall"):
        assert rp.main([cmd, *argv(w, wd), "--fake", *fake]) == 0
        cu.reset_class_space()
    clip = json.load(open(os.path.join(wd, "results", "gate_clip.json")))
    assert clip["all"] > 0.95                                       # the fake text encoder found the custom names
    yaml = open(os.path.join(wd, "yolo_data", "data.yaml")).read()
    assert "0: alpha" in yaml and "2: charlie" in yaml and "3: delta" not in yaml          # base classes only, custom names
    rp.main(["run", *argv(w, wd), "--fake", *fake, "--only", "E0_zeroshot", "E5_ours", "E6_gt_ceiling"])
    res = json.load(open(os.path.join(wd, "results", "ablation.json")))
    assert res["E6_gt_ceiling"]["final_AP50"] > 0.95 and res["E5_ours"]["final_AP50"] > 0.8
    cu.reset_class_space()
    assert rp.main(["tune", *argv(w, wd), "--fake", *fake, "--n_trials", "2", "--objective", "final_AP50"]) == 0


def test_custom_requires_its_inputs_and_tasks(custom_world, tmp_path):
    w = custom_world
    with pytest.raises(SystemExit, match="--ann_train"):
        rp.main(["fetch", "--workdir", str(tmp_path / "a"), "--dataset", "custom", "--tasks", "3+6"])
    a = argv(w, str(tmp_path / "b")); i = a.index("--tasks"); del a[i:i + 2]
    with pytest.raises(SystemExit, match="needs --tasks"):
        rp.main(["fetch", *a])
    bad = argv(w, str(tmp_path / "c")); bad[bad.index("--tasks") + 1] = "4+4"
    with pytest.raises(ValueError, match="covers 8 classes, expected 9"):
        rp.main(["fetch", *bad])


# ------------------------------------------------------------------ random splits
def test_random_tasks_same_sizes_disjoint_cover_and_deterministic():
    base = [list(range(15)), list(range(15, 20))]
    r = cu.random_tasks(base, 3)
    assert [len(t) for t in r] == [15, 5] and sorted(c for t in r for c in t) == list(range(20))
    assert all(t == sorted(t) for t in r) and r == cu.random_tasks(base, 3) and r != cu.random_tasks(base, 4)
    assert r != base


def test_random_scheme_through_the_cli_for_voc_and_coco():
    class A:  # minimal namespace
        dataset, fake, fake_classes, tasks = "voc", False, 9, "random:15+5:2"
    import voc_utils as vu
    t = rp.tasks_of(A)
    assert [len(x) for x in t] == [15, 5] and sorted(c for x in t for c in x) == sorted(vu.VOC_TO_COCO_IDX)
    assert t == rp.tasks_of(A)
    A.tasks = "random:15+5:3"
    assert rp.tasks_of(A) != t
    A.dataset, A.tasks = "coco", "random:40+40:1"
    t = rp.tasks_of(A)
    assert [len(x) for x in t] == [40, 40] and sorted(c for x in t for c in x) == list(range(80))
    A.tasks = "random:40+40"
    assert rp.tasks_of(A) == cu.random_tasks(cu.make_tasks("40+40"), 0)           # default seed 0
    for bad in ("random:", "random:40+40:x", "random:40+40:1:2"):
        A.tasks = bad
        with pytest.raises(ValueError):
            rp.tasks_of(A)

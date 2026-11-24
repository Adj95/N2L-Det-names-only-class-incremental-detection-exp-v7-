"""Tests for the Pascal VOC loader (voc_utils), its wiring into run_pipeline, and merge_results.

Everything runs offline on a tiny synthetic VOC tree and local tarballs (file:// URLs).
"""
import json
import os
import tarfile

import numpy as np
import pytest
from PIL import Image

import coco_utils as cu
import evaluate as ev
import merge_results as mr
import pipeline as pl
import run_pipeline as rp
import voc_utils as vu

XML = """<annotation><folder>VOC</folder><filename>{stem}.jpg</filename>
<size><width>{w}</width><height>{h}</height><depth>3</depth></size>
{objs}</annotation>"""
OBJ = """<object><name>{name}</name><pose>Unspecified</pose><truncated>0</truncated>{diff}
<bndbox><xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax></bndbox></object>"""


def write_item(vdir, stem, w, h, objs, jpg=True):
    """objs: list of (name, difficult or None, (x1,y1,x2,y2))."""
    for sub in ("Annotations", "JPEGImages"):
        os.makedirs(os.path.join(vdir, sub), exist_ok=True)
    o = "".join(OBJ.format(name=n, diff="" if d is None else f"<difficult>{d}</difficult>",
                           x1=b[0], y1=b[1], x2=b[2], y2=b[3]) for n, d, b in objs)
    with open(os.path.join(vdir, "Annotations", stem + ".xml"), "w") as f:
        f.write(XML.format(stem=stem, w=w, h=h, objs=o))
    if jpg:
        Image.new("RGB", (w, h), (120, 60, 30)).save(os.path.join(vdir, "JPEGImages", stem + ".jpg"))


def write_list(vdir, split, stems):
    os.makedirs(os.path.join(vdir, "ImageSets", "Main"), exist_ok=True)
    with open(os.path.join(vdir, "ImageSets", "Main", split + ".txt"), "w") as f:
        f.write("\n".join(stems) + "\n")


def make_tree(root, years=(2007, 2012), nested="VOCdevkit"):
    """A small VOC tree. 2007: trainval 000001-000004, test 000005-000006.
    2012: trainval 2008_000001-3, train 2008_000001-2, val 2008_000003."""
    base = os.path.join(root, nested) if nested else root
    if 2007 in years:
        d = os.path.join(base, "VOC2007")
        write_item(d, "000001", 100, 80, [("person", 0, (11, 21, 50, 60)), ("sheep", 0, (1, 1, 100, 80))])
        write_item(d, "000002", 100, 80, [("aeroplane", 0, (5, 5, 60, 40)), ("person", 1, (2, 2, 30, 30))])
        write_item(d, "000003", 100, 80, [("sofa", 0, (10, 10, 90, 70))])
        write_item(d, "000004", 100, 80, [("cat", None, (3, 4, 50, 60))])
        write_item(d, "000005", 100, 80, [("train", 0, (1, 1, 100, 80)), ("person", 0, (20, 20, 40, 60))])
        write_item(d, "000006", 100, 80, [("tvmonitor", 0, (10, 10, 50, 50)), ("car", 1, (60, 10, 99, 40))])
        write_list(d, "trainval", ["000001", "000002", "000003", "000004"])
        write_list(d, "test", ["000005", "000006"])
    if 2012 in years:
        d = os.path.join(base, "VOC2012")
        write_item(d, "2008_000001", 120, 90, [("dog", 0, (10, 10, 60, 70))])
        write_item(d, "2008_000002", 120, 90, [("pottedplant", 0, (10, 10, 60, 70)), ("bird", 0, (70, 5, 110, 50))])
        write_item(d, "2008_000003", 120, 90, [("bottle", 0, (10, 10, 30, 70))])
        write_list(d, "trainval", ["2008_000001", "2008_000002", "2008_000003"])
        write_list(d, "train", ["2008_000001", "2008_000002"])
        write_list(d, "val", ["2008_000003"])
    return base


# ================================================================== names, ids, splits
def test_voc_to_coco_class_mapping():
    assert len(vu.VOC_NAMES) == 20 and len(set(vu.VOC_TO_COCO_IDX)) == 20
    nm = lambda v: cu.COCO_NAMES[vu.VOC_NAME2COCO_IDX[v]]
    assert [nm(v) for v in ("aeroplane", "motorbike", "diningtable", "pottedplant", "sofa", "tvmonitor", "person")] == \
           ["airplane", "motorcycle", "dining table", "potted plant", "couch", "tv", "person"]
    for v in ("bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dog", "horse",
              "sheep", "train"):
        assert nm(v) == v
    assert vu.VOC_NAMES == sorted(vu.VOC_NAMES)                       # standard alphabetical order


@pytest.mark.parametrize("scheme,sizes", [("15+5", [15, 5]), ("10+10", [10, 10]), ("19+1", [19, 1]),
                                          ("10+2x5", [10, 5, 5])])
def test_voc_tasks_sizes_and_order(scheme, sizes):
    t = vu.voc_tasks(scheme)
    assert [len(x) for x in t] == sizes
    assert [c for g in t for c in g] == vu.VOC_TO_COCO_IDX          # VOC order, disjoint, covers all 20


def test_voc_15_5_last_five_are_the_standard_new_classes():
    new = [cu.COCO_NAMES[c] for c in vu.voc_tasks("15+5")[1]]
    assert new == ["potted plant", "sheep", "couch", "train", "tv"]


def test_voc_tasks_custom_and_errors():
    t = vu.voc_tasks("14,0|1,2")                                     # VOC positions: person, aeroplane | bicycle, bird
    assert [[cu.COCO_NAMES[c] for c in g] for g in t] == [["person", "airplane"], ["bicycle", "bird"]]
    for bad in ("15+6", "0,1|1,2", "0,1|", "0,20", "-1|2", "40+40", "0|20", "0,1|20", "19|20"):
        with pytest.raises(ValueError):
            vu.voc_tasks(bad)


def test_image_id_unique_and_deterministic():
    ids = [vu.image_id_for(s) for s in ("000001", "000002", "2007_000001", "2008_000001", "2012_004331")]
    assert len(set(ids)) == 5 and ids == [vu.image_id_for(s) for s in ("000001", "000002", "2007_000001",
                                                                       "2008_000001", "2012_004331")]
    assert vu.image_id_for("2008_000123") == 2008000123 and vu.image_id_for("000123") == 10_000_000_123


def test_box_conversion_one_based_inclusive_and_clipping():
    assert vu.voc_box_to_coco([1, 1, 100, 80], 100, 80) == [0.0, 0.0, 100.0, 80.0]      # full image
    assert vu.voc_box_to_coco([11, 21, 50, 60], 100, 80) == [10.0, 20.0, 40.0, 40.0]
    assert vu.voc_box_to_coco([0, 0, 120, 100], 100, 80) == [0.0, 0.0, 100.0, 80.0]     # clipped to the image


def test_parse_xml_difficult_default_and_floats(tmp_path):
    p = tmp_path / "a.xml"
    p.write_text(XML.format(stem="a", w=100, h=80, objs=OBJ.format(name="cat", diff="", x1="3.0", y1=4, x2=50, y2=60)
                            + OBJ.format(name="dog", diff="<difficult>1</difficult>", x1=1, y1=1, x2=9, y2=9)))
    r = vu.parse_voc_xml(str(p))
    assert (r["width"], r["height"]) == (100, 80)
    assert [(o["name"], o["difficult"]) for o in r["objects"]] == [("cat", 0), ("dog", 1)]
    assert r["objects"][0]["box"] == [3.0, 4.0, 50.0, 60.0]


# ================================================================== locating + protocol
def test_find_dirs_official_and_kaggle_layouts(tmp_path):
    a = tmp_path / "a"; make_tree(str(a))
    assert [os.path.basename(d) for d in vu.find_voc_dirs(str(a))] == ["VOC2007", "VOC2012"]
    b = tmp_path / "kaggle" / "datasets" / "someone" / "pascal-voc-2012"          # re-upload with another layout
    d = b / "data" / "VOC2012_trainval"
    write_item(str(d), "2008_000001", 50, 50, [("dog", 0, (1, 1, 20, 20))]); write_list(str(d), "train", ["2008_000001"])
    found = vu.find_voc_dirs(str(tmp_path / "kaggle"))
    assert found == [str(d)] and vu.year_of(found[0]) == 2012
    assert vu.find_voc_dirs(str(tmp_path / "nothing")) == []


def test_year_of_needs_a_year(tmp_path):
    with pytest.raises(ValueError):
        vu.year_of("/x/y/pascal")
    assert vu.year_of("/x/VOC2007_test") == 2007 and vu.year_of("/x/VOC2012/extra") == 2012


def test_read_split_skips_missing_files(tmp_path, capsys):
    root = tmp_path / "r"; make_tree(str(root))
    d = str(root / "VOCdevkit" / "VOC2007")
    write_list(d, "trainval", ["000001", "000002", "000003", "000004", "999999"])        # 999999 has nothing
    os.remove(os.path.join(d, "JPEGImages", "000004.jpg"))                                # xml but no jpg
    got = vu.read_split([d], 2007, "trainval")
    assert [s for _, s in got] == ["000001", "000002", "000003"]
    assert "skipped 2" in capsys.readouterr().out
    assert vu.read_split([d], 2012, "trainval") == [] and vu.read_split([d], 2007, "nosuch") == []


def test_duplicate_folders_of_the_same_year_are_deduplicated(tmp_path, capsys):
    """Kaggle can expose one dataset twice (e.g. VOCtest_... and voctest_...): no crash, no double counting."""
    root = tmp_path / "voc"
    make_tree(str(root / "a" / "VOCtrainval_06-Nov-2007"))
    make_tree(str(root / "a" / "voctrainval_06-nov-2007"))          # an identical second copy
    dirs = vu.find_voc_dirs(str(root))
    assert len(dirs) >= 4
    single = vu.read_split(vu.find_voc_dirs(str(root / "a" / "VOCtrainval_06-Nov-2007")), 2007, "trainval")
    both = vu.read_split(dirs, 2007, "trainval")
    assert len(both) == len(single) > 0 and len({s for _, s in both}) == len(both)
    assert "repeated copies" in capsys.readouterr().out
    proto, tr, va = vu.choose_protocol(dirs, "07+12")
    out = vu.convert(tr, va, str(tmp_path / "o"))                    # would raise 'duplicate image id' before the fix
    assert out is not None


def test_choose_protocol_variants(tmp_path):
    full = tmp_path / "full"; make_tree(str(full))
    dirs = vu.find_voc_dirs(str(full))
    used, tr, va = vu.choose_protocol(dirs, "auto")
    assert used == "07+12" and len(tr) == 4 + 3 and len(va) == 2
    assert vu.choose_protocol(dirs, "07")[0] == "07" and len(vu.choose_protocol(dirs, "07")[1]) == 4
    used, tr, va = vu.choose_protocol(dirs, "12")
    assert used == "12" and [s for _, s in tr] == ["2008_000001", "2008_000002"] and [s for _, s in va] == ["2008_000003"]
    only07 = tmp_path / "o7"; make_tree(str(only07), years=(2007,))
    assert vu.choose_protocol(vu.find_voc_dirs(str(only07)), "auto")[0] == "07"
    with pytest.raises(ValueError, match="07\\+12.*VOC2012 trainval"):
        vu.choose_protocol(vu.find_voc_dirs(str(only07)), "07+12")
    only12 = tmp_path / "o12"; make_tree(str(only12), years=(2012,))
    assert vu.choose_protocol(vu.find_voc_dirs(str(only12)), "auto")[0] == "12"
    with pytest.raises(ValueError):
        vu.choose_protocol([], "auto")
    with pytest.raises(ValueError):
        vu.choose_protocol(dirs, "nonsense")


def test_test_split_is_the_2007_test_never_trainval(tmp_path):
    root = tmp_path / "r"; make_tree(str(root))
    out = vu.prepare_voc(str(root), str(tmp_path / "out"), "07+12", download=False)
    val = json.load(open(out["ann_val"]))
    assert sorted(im["file_name"] for im in val["images"]) == ["000005.jpg", "000006.jpg"]
    train = json.load(open(out["ann_train"]))
    assert not ({im["id"] for im in train["images"]} & {im["id"] for im in val["images"]})


# ================================================================== conversion
@pytest.fixture()
def converted(tmp_path):
    root = tmp_path / "voc"; make_tree(str(root))
    out = vu.prepare_voc(str(root), str(tmp_path / "out"), "07+12", download=False)
    return out, json.load(open(out["ann_train"])), json.load(open(out["ann_val"]))


def test_converted_json_structure(converted):
    out, tr, va = converted
    assert out["stats"]["train"] == {"images": 7, "objects": 10, "difficult": 1}
    assert out["stats"]["val"] == {"images": 2, "objects": 4, "difficult": 1}
    cu.verify_categories(tr["categories"])                                   # the 80 COCO categories, in order
    assert len({a["id"] for a in tr["annotations"]}) == len(tr["annotations"])
    assert len({im["id"] for im in tr["images"]}) == 7
    by = {im["file_name"]: im for im in tr["images"]}
    assert (by["000001.jpg"]["width"], by["000001.jpg"]["height"]) == (100, 80)
    assert (by["2008_000001.jpg"]["width"], by["2008_000001.jpg"]["height"]) == (120, 90)
    a1 = [a for a in tr["annotations"] if a["image_id"] == by["000001.jpg"]["id"]]
    person = next(a for a in a1 if a["category_id"] == cu.COCO_IDS[cu.COCO_NAMES.index("person")])
    assert person["bbox"] == [10.0, 20.0, 40.0, 40.0] and person["area"] == 1600.0 and person["iscrowd"] == 0
    sheep = next(a for a in a1 if a["category_id"] == cu.COCO_IDS[cu.COCO_NAMES.index("sheep")])
    assert sheep["bbox"] == [0.0, 0.0, 100.0, 80.0]


def test_difficult_becomes_iscrowd_and_is_ignored_everywhere(converted):
    out, tr, va = converted
    crowd = [a for a in tr["annotations"] if a["iscrowd"]]
    assert len(crowd) == 1 and crowd[0]["category_id"] == cu.COCO_IDS[0]            # the difficult person in 000002
    idx = cu.image_class_index(tr)
    iid = vu.image_id_for("000002")
    assert idx[iid] == {cu.COCO_NAMES.index("airplane")}                             # difficult person not counted
    gt = cu.gt_by_image(tr, [iid])[iid]
    assert gt[1].tolist() == [cu.COCO_NAMES.index("airplane")]


def test_images_linked_and_readable(converted):
    out, tr, va = converted
    for split, coco in (("train", tr), ("val", va)):
        for im in coco["images"]:
            with Image.open(os.path.join(out[f"img_dir_{split}"], im["file_name"])) as p:
                assert p.size == (im["width"], im["height"])


def test_coco_eval_on_converted_gt_perfect_detections_and_difficult_ignored(converted):
    out, tr, va = converted
    ids = [im["id"] for im in va["images"]]
    gt = ev.make_coco_gt(va, ids)
    cls = [cu.COCO_NAMES.index(n) for n in ("train", "person", "tv", "car")]
    dets = []
    for i, (b, c) in cu.gt_by_image(va, ids).items():
        dets += ev.dets_to_coco(i, b, np.ones(len(b)), c)
    r = ev.coco_eval(gt, dets, ids, cls)
    assert r["AP50"] == pytest.approx(1.0) and r["n_gt"] == 3                        # the difficult car is not counted
    car = [a for a in va["annotations"] if a["iscrowd"]][0]
    x, y, w, h = car["bbox"]
    extra = ev.dets_to_coco(car["image_id"], np.array([[x, y, x + w, y + h]]), [0.99], [cu.COCO_NAMES.index("car")])
    r2 = ev.coco_eval(gt, dets + extra, ids, cls)
    assert r2["AP50"] == pytest.approx(1.0)                                          # hit on a difficult object: no false positive


def test_yolo_labels_only_base_classes_and_no_difficult(converted, tmp_path):
    out, tr, va = converted
    base = vu.voc_tasks("15+5")[0]
    ids = [im["id"] for im in tr["images"]]
    st = cu.write_yolo_labels(tr, ids, base, str(tmp_path / "labels"))
    assert st["images"] == 7
    lab = lambda stem: open(tmp_path / "labels" / (stem + ".txt")).read().split("\n")
    row = [r for r in lab("000001") if r][0].split()
    assert row[0] == str(base.index(cu.COCO_NAMES.index("person")))                 # remapped to 0..14
    assert [r for r in lab("000001") if r] == [" ".join(row)]                        # sheep is a NEW class: not written
    assert [r for r in lab("000002") if r][0].split()[0] == str(base.index(cu.COCO_NAMES.index("airplane")))
    assert len([r for r in lab("000002") if r]) == 1                                 # difficult person skipped
    assert float(row[3]) == pytest.approx(40 / 100) and float(row[4]) == pytest.approx(40 / 80)


def test_boxes_outside_or_degenerate_are_dropped(tmp_path):
    d = str(tmp_path / "voc" / "VOC2007")
    write_item(d, "000001", 100, 80, [("cat", 0, (200, 10, 210, 20)),          # entirely right of the image
                                      ("dog", 0, (10, 10, 40, 40))])
    coco, st = vu._convert_split([(d, "000001")], str(tmp_path / "img"))
    assert st["objects"] == 1 and [a["category_id"] for a in coco["annotations"]] == \
        [cu.COCO_IDS[cu.COCO_NAMES.index("dog")]]
    assert all(a["bbox"][2] > 0 and a["bbox"][3] > 0 for a in coco["annotations"])


def test_duplicate_image_in_one_split_rejected(tmp_path):
    root = tmp_path / "voc"; make_tree(str(root))
    d = str(root / "VOCdevkit" / "VOC2007")
    with pytest.raises(ValueError, match="duplicate"):
        vu.convert([(d, "000001"), (d, "000001")], [(d, "000005")], str(tmp_path / "o"))


def test_unknown_class_rejected(tmp_path):
    root = tmp_path / "voc"; d = str(root / "VOC2007")
    write_item(d, "000001", 50, 50, [("unicorn", 0, (1, 1, 20, 20))])
    with pytest.raises(ValueError, match="unicorn"):
        vu.convert([(d, "000001")], [(d, "000001")], str(tmp_path / "o"))


def test_rerun_is_idempotent(tmp_path):
    root = tmp_path / "voc"; make_tree(str(root))
    a = vu.prepare_voc(str(root), str(tmp_path / "o"), "auto", download=False)
    b = vu.prepare_voc(str(root), str(tmp_path / "o"), "auto", download=False)
    assert open(a["ann_train"]).read() == open(b["ann_train"]).read()


# ================================================================== download / extract
def make_tars(tdir):
    """Local stand-ins for the three official tarballs, built from a synthetic tree."""
    src = tdir / "src"
    make_tree(str(src))
    base = src / "VOCdevkit"
    names = {"VOCtrainval_06-Nov-2007.tar": ["VOC2007/Annotations", "VOC2007/JPEGImages", "VOC2007/ImageSets/Main/trainval.txt"],
             "VOCtest_06-Nov-2007.tar": ["VOC2007/ImageSets/Main/test.txt"],
             "VOCtrainval_11-May-2012.tar": ["VOC2012"]}
    urls = {}
    for tar, members in names.items():
        p = tdir / tar
        with tarfile.open(p, "w") as t:
            for m in members:
                t.add(base / m, arcname="VOCdevkit/" + m)
        urls[tar] = [(tdir / "missing" / tar).as_uri(), p.as_uri()]                   # first mirror is dead
    return urls


def test_download_with_mirror_fallback_then_convert(tmp_path):
    urls = make_tars(tmp_path)
    root = tmp_path / "dl"
    out = None
    import voc_utils
    saved = voc_utils.MIRRORS
    voc_utils.MIRRORS = urls
    try:
        out = vu.prepare_voc(str(root), str(tmp_path / "out"), "auto", download=True)
    finally:
        voc_utils.MIRRORS = saved
    assert out["protocol"] == "07+12" and out["stats"]["train"]["images"] == 7 and out["stats"]["val"]["images"] == 2


def test_no_download_flag_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="auto protocol needs"):
        vu.prepare_voc(str(tmp_path / "empty"), str(tmp_path / "out"), "auto", download=False)


def test_all_mirrors_dead_gives_actionable_error(tmp_path):
    dead = {t: [(tmp_path / "x" / t).as_uri()] for t in vu.MIRRORS}
    with pytest.raises(IOError, match="(?s)could not download.*--voc_root"):
        vu.download_voc(str(tmp_path / "r"), "12", mirrors=dead)


def test_unsafe_tar_rejected(tmp_path):
    evil = tmp_path / "evil.tar"
    (tmp_path / "f.txt").write_text("x")
    with tarfile.open(evil, "w") as t:
        t.add(tmp_path / "f.txt", arcname="../escaped.txt")
    with pytest.raises(ValueError, match="unsafe"):
        vu._safe_extract(str(evil), str(tmp_path / "dest"))
    assert not (tmp_path / "escaped.txt").exists()


def test_read_only_voc_root_downloads_next_to_output(tmp_path, monkeypatch):
    urls = make_tars(tmp_path)
    ro = tmp_path / "input"; ro.mkdir()
    real = os.access
    monkeypatch.setattr(os, "access", lambda p, m: False if os.path.abspath(p) == str(ro) else real(p, m))
    monkeypatch.setattr(vu, "MIRRORS", urls)
    out = vu.prepare_voc(str(ro), str(tmp_path / "out"), "07+12", download=True)
    assert os.path.isdir(tmp_path / "out" / "download" / "VOCdevkit") and not os.listdir(ro)
    assert out["protocol"] == "07+12"


# ================================================================== the CLI wiring
def fetch_args(tmp_path, *extra):
    root = tmp_path / "voc"
    make_tree(str(root))
    w = tmp_path / "work"
    return ["fetch", "--dataset", "voc", "--voc_root", str(root), "--workdir", str(w), "--no_download",
            "--n_train_per_task", "3", "--n_eval", "4", "--n_yolo_train", "5", "--n_yolo_val", "2", *extra], w


def test_cli_fetch_voc_writes_splits_and_paths(tmp_path):
    argv, w = fetch_args(tmp_path)
    assert rp.main(argv) == 0
    sp = json.load(open(w / "splits.json"))
    assert sp["dataset"] == "voc" and sp["scheme"] == "15+5" and sp["voc_protocol"] == "07+12"
    assert sp["tasks"] == vu.voc_tasks("15+5")
    pth = json.load(open(w / "paths.json"))
    tr = json.load(open(pth["ann_train"])); va = json.load(open(pth["ann_val"]))
    idx = cu.image_class_index(tr)
    for t, ids in enumerate(sp["train_ids"]):
        assert ids and all(idx[i] & set(sp["tasks"][t]) for i in ids)               # each task's images hold its classes
    assert set(sp["eval_ids"]) <= {im["id"] for im in va["images"]}
    assert set(sp["yolo_train"]) <= {im["id"] for im in tr["images"]} and sp["yolo_train"]
    assert os.path.isdir(pth["img_dir_train"]) and os.path.isdir(pth["img_dir_val"])


def test_cli_fetch_voc_other_schemes_and_custom(tmp_path):
    argv, w = fetch_args(tmp_path, "--tasks", "10+10")
    rp.main(argv)
    assert json.load(open(w / "splits.json"))["scheme"] == "10+10"
    assert [len(t) for t in json.load(open(w / "splits.json"))["tasks"]] == [10, 10]
    argv, w2 = fetch_args(tmp_path, "--tasks", "14,0,1|2,3")                        # custom, VOC positions
    argv[argv.index(str(w))] = str(w2)
    rp.main(argv)
    assert [[cu.COCO_NAMES[c] for c in t] for t in json.load(open(w2 / "splits.json"))["tasks"]] == \
           [["person", "airplane", "bicycle"], ["bird", "boat"]]


def test_cli_voc_protocol_flag_is_honoured(tmp_path):
    argv, w = fetch_args(tmp_path, "--voc_protocol", "12", "--tasks", "10+10")
    argv[argv.index("--n_train_per_task") + 1] = "2"
    rp.main(argv)
    sp = json.load(open(w / "splits.json"))
    assert sp["voc_protocol"] == "12"
    pth = json.load(open(w / "paths.json"))
    tr = json.load(open(pth["ann_train"])); va = json.load(open(pth["ann_val"]))
    assert {im["voc_year"] for im in tr["images"]} == {2012} and {im["voc_year"] for im in va["images"]} == {2012}
    assert sorted(im["file_name"] for im in va["images"]) == ["2008_000003.jpg"]               # VOC2012 val


def test_cli_no_download_never_touches_the_network(tmp_path, monkeypatch):
    dead = {t: [(tmp_path / "x" / t).as_uri()] for t in vu.MIRRORS}
    monkeypatch.setattr(vu, "MIRRORS", dead)
    calls = []
    monkeypatch.setattr(vu, "download_voc", lambda *a, **k: calls.append(a))
    argv = ["fetch", "--dataset", "voc", "--voc_root", str(tmp_path / "empty"), "--workdir", str(tmp_path / "w"),
            "--no_download"]
    with pytest.raises(ValueError, match="auto protocol needs"):
        rp.main(argv)
    assert calls == []
    with pytest.raises(ValueError):                         # without the flag it WOULD try to download
        rp.main(argv[:-1])
    assert len(calls) == 1


def test_cli_coco_scheme_default_unchanged():
    p = rp.build_parser()
    a = p.parse_args(["fetch"])
    assert rp.scheme_of(a) == "40+40" and rp.tasks_of(a) == cu.make_tasks("40+40")
    v = p.parse_args(["fetch", "--dataset", "voc"])
    assert rp.scheme_of(v) == "15+5" and rp.tasks_of(v) == vu.voc_tasks("15+5")
    c = p.parse_args(["fetch", "--tasks", "70+10"])
    assert [len(t) for t in rp.tasks_of(c)] == [70, 10]


def test_cli_make_yolo_data_for_voc(tmp_path):
    argv, w = fetch_args(tmp_path)
    rp.main(argv)
    assert rp.main(["make-yolo-data", "--workdir", str(w)]) == 0
    y = open(w / "yolo_data" / "data.yaml").read()
    names = [ln.split(": ", 1)[1] for ln in y.splitlines() if ln.startswith("  ")]
    assert names == cu.prompt_names(vu.voc_tasks("15+5")[0]) and len(names) == 15 and "airplane" in names
    assert "sheep" not in names and "couch" not in names                            # new classes are not in the detector
    lbl = list((w / "yolo_data" / "labels" / "train").glob("*.txt"))
    assert lbl
    for f in lbl:
        for ln in open(f).read().split("\n"):
            if ln:
                assert 0 <= int(ln.split()[0]) < 15


# ================================================================== gate-clip candidate classes
def test_gate_clip_all_uses_only_the_runs_classes(monkeypatch):
    seen = {}

    def fake_eval(items, enc, T, pad_frac=0.1, class_subset=None):
        seen.setdefault("subsets", []).append(None if class_subset is None else list(class_subset))
        return {"acc": 0.5}

    class M:
        encode_text = staticmethod(lambda t: np.eye(len(t), 80)[: len(t)])
        encode_images_u8 = staticmethod(lambda x: np.zeros((len(x), 80)))

    monkeypatch.setattr(pl, "gt_crop_zeroshot_eval", fake_eval)
    monkeypatch.setattr(pl, "text_assets", lambda m: {"text_eval": np.eye(80)})
    monkeypatch.setattr(pl, "iter_images", lambda *a, **k: iter(()))
    coco = {"images": [], "annotations": [], "categories": []}
    voc = vu.voc_tasks("15+5")
    pl.stage_gate_clip(M, coco, "x", [], voc)
    assert seen["subsets"][0] == sorted(vu.VOC_TO_COCO_IDX)                       # 'all' = the 20 VOC classes, not 80
    assert seen["subsets"][1] == voc[0] and seen["subsets"][2] == voc[1]
    seen.clear()
    pl.stage_gate_clip(M, coco, "x", [], cu.make_tasks("40+40"))
    assert seen["subsets"][0] is None                                             # COCO 80: unchanged behaviour


# ================================================================== merge_results
def put(workdir, rel, obj):
    p = os.path.join(workdir, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f)


def mk_run(tmp_path, name, clip=0.47, rec=((0.6, 1000), (0.3, 400), (0.5, 100)), ap=(0.20, 0.18, 0.26), splits=None):
    w = str(tmp_path / name)
    os.makedirs(w, exist_ok=True)
    if clip is not None:
        put(w, "results/gate_clip.json", {"all": clip, "task0_within": 0.5})
    if rec is not None:
        r = {f"task{t}": {"n_gt": n, "recall@0.5": v, "recall@0.7": v / 2} for t, (v, n) in enumerate(rec)}
        r["boxes_per_image"] = 55.0
        put(w, "results/gate_recall.json", r)
    if ap is not None:
        put(w, "results/ablation.json", {c: {"final_AP50": v} for c, v in zip(("E0_zeroshot", "E5_ours", "E6_gt_ceiling"), ap)})
    put(w, "splits.json", splits or {"dataset": "coco", "scheme": "70+10"})
    return w


def test_new_class_recall_is_ngt_weighted():
    rec = {"task0": {"n_gt": 10, "recall@0.5": 0.9}, "task1": {"n_gt": 300, "recall@0.5": 0.3},
           "task2": {"n_gt": 100, "recall@0.5": 0.7}, "boxes_per_image": 5}
    base, new, n = mr.new_class_recall(rec)
    assert base == 0.9 and n == 400 and new == pytest.approx((0.3 * 300 + 0.7 * 100) / 400)
    assert mr.new_class_recall(None) == (None, None, 0)
    assert mr.new_class_recall({"task0": {"n_gt": 3, "recall@0.5": 0.5}}) == (0.5, None, 0)


def test_summary_numbers_and_verdicts(tmp_path):
    cols = ["E0_zeroshot", "E5_ours", "E6_gt_ceiling"]
    r = mr.summarize_run("A", mk_run(tmp_path, "A", rec=((0.6, 1000), (0.6, 400)), ap=(0.20, 0.18, 0.26)), cols, 0.5, 0.03)
    assert r["clip_all"] == 0.47 and r["recall_new"] == pytest.approx(0.6) and r["n_new"] == 400
    assert r["headroom"] == pytest.approx(0.06) and r["e5_minus_e0"] == pytest.approx(-0.02) and r["verdict"] == "PASS"
    assert r["dataset"] == "coco" and r["scheme"] == "70+10" and r["boxes_per_image"] == 55.0
    low_recall = mr.summarize_run("B", mk_run(tmp_path, "B", rec=((0.6, 1000), (0.32, 445)), ap=(0.2, 0.18, 0.26)), cols, 0.5, 0.03)
    assert low_recall["verdict"] == "FAIL"                                           # recall decides even with headroom
    low_head = mr.summarize_run("C", mk_run(tmp_path, "C", rec=((0.6, 1000), (0.7, 445)), ap=(0.2, 0.18, 0.21)), cols, 0.5, 0.03)
    assert low_head["verdict"] == "FAIL" and low_head["headroom"] == pytest.approx(0.01)
    edge = mr.summarize_run("D", mk_run(tmp_path, "D", rec=((0.6, 1000), (0.5, 445)), ap=(0.2, 0.18, 0.23)), cols, 0.5, 0.03)
    assert edge["verdict"] == "PASS"                                                 # thresholds are inclusive


def test_missing_files_give_na_not_errors(tmp_path):
    cols = ["E0_zeroshot", "E5_ours", "E6_gt_ceiling"]
    r = mr.summarize_run("A", mk_run(tmp_path, "A", ap=None), cols, 0.5, 0.03)
    assert r["headroom"] is None and r["verdict"] == "FAIL"                          # recall 0.3 fails by itself
    r = mr.summarize_run("B", mk_run(tmp_path, "B", ap=None, rec=((0.6, 10), (0.8, 50))), cols, 0.5, 0.03)
    assert r["verdict"] == mr.NA                                                     # recall ok, headroom unknown
    r = mr.summarize_run("C", mk_run(tmp_path, "C", clip=None, rec=None, ap=None), cols, 0.5, 0.03)
    assert r["verdict"] == mr.NA and r["clip_all"] is None and r["recall_new"] is None
    bare = str(tmp_path / "bare"); os.makedirs(bare)
    assert mr.summarize_run("E", bare, cols, 0.5, 0.03)["dataset"] == mr.NA
    txt = mr.format_table([r], cols)
    assert txt.count(mr.NA) >= 4


def test_nan_values_are_treated_as_missing(tmp_path):
    w = mk_run(tmp_path, "N", rec=((0.6, 10), (float("nan"), 50)))                      # nan WITH ground truth
    r = mr.summarize_run("N", w, ["E0_zeroshot", "E5_ours", "E6_gt_ceiling"], 0.5, 0.03)
    assert r["recall_new"] is None and r["verdict"] == mr.NA


def test_table_format_and_cli(tmp_path, capsys):
    a = mk_run(tmp_path, "coco4040", rec=((0.6, 1000), (0.31, 1100)), ap=(0.238, 0.202, 0.262))
    b = mk_run(tmp_path, "voc155", rec=((0.7, 3000), (0.62, 800)), ap=(0.60, 0.58, 0.66),
               splits={"dataset": "voc", "scheme": "15+5"})
    out = tmp_path / "t.md"
    assert mr.main([f"coco4040={a}", f"voc155={b}", "--out", str(out)]) == 0
    txt = capsys.readouterr().out
    lines = [ln for ln in txt.splitlines() if ln.startswith("| coco4040") or ln.startswith("| voc155")]
    assert len(lines) == 2
    cells = [c.strip() for c in lines[0].strip("|").split("|")]
    assert cells[:3] == ["coco4040", "coco", "70+10"] or cells[0] == "coco4040"
    assert "0.238" in lines[0] and "+0.024" in lines[0] and "-0.036" in lines[0] and lines[0].rstrip().endswith("FAIL |")
    assert lines[1].rstrip().endswith("PASS |") and "0.620 (800)" in lines[1]
    assert out.read_text().startswith("| run |") and "proposed thresholds" in out.read_text()
    with pytest.raises(SystemExit):
        mr.main(["badarg"])
    with pytest.raises(SystemExit):
        mr.main(["x=" + str(tmp_path / "does_not_exist")])


def test_custom_columns_are_used(tmp_path):
    w = str(tmp_path / "r"); os.makedirs(w)
    put(w, "results/ablation.json", {"E0_zeroshot_g0": {"final_AP50": 0.1}, "E5_ours": {"final_AP50": 0.2},
                                     "E6_gt_ceiling_g0": {"final_AP50": 0.4}})
    r = mr.summarize_run("r", w, ["E0_zeroshot_g0", "E5_ours", "E6_gt_ceiling_g0"], 0.5, 0.03)
    assert r["headroom"] == pytest.approx(0.3) and r["e5_minus_e0"] == pytest.approx(0.1)


# ================================================================== end to end on a VOC workdir (stub models)
PAL = lambda i: (30 + (37 * i) % 200, 30 + (91 * i) % 200, 30 + (53 * i) % 200)       # a distinct colour per COCO class
BG = (235, 235, 235)


class ColourModels:
    """Real-signature stand-in for RealModels: an object's class is its fill colour.
    dim 81 = 80 COCO classes + background. propose() = connected non-background regions (+ one junk box)."""
    dim = 81

    def __init__(self):
        self.names = cu.prompt_names()
        self.pal = np.array([PAL(i) for i in range(80)], dtype=np.float64)

    def encode_text(self, texts):
        import prompts as pr
        out = np.zeros((len(texts), 81))
        look = {t.format(n): i for i, n in enumerate(self.names) for t in pr.DET_TEMPLATES}
        for k, t in enumerate(texts):
            out[k, look.get(t, 80)] = 1.0
        return out

    def encode_images_u8(self, crops):
        out = np.zeros((len(crops), 81))
        for k, c in enumerate(np.asarray(crops, dtype=np.float64)):
            col = np.median(c.reshape(-1, 3), axis=0)
            d = np.linalg.norm(self.pal - col, axis=1)
            out[k, int(d.argmin()) if d.min() < 25 else 80] = 1.0
        return out

    def propose(self, img):
        from scipy import ndimage
        a = np.asarray(img.convert("RGB"), dtype=np.int64)
        lab, n = ndimage.label(np.abs(a - np.array(BG)).sum(axis=2) > 30)
        boxes, scores = [], []
        for sl in ndimage.find_objects(lab):
            boxes.append([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]); scores.append(0.9)
        boxes.append([0, 0, 6, 6]); scores.append(0.1)                                 # junk background box
        return np.array(boxes, dtype=np.float64), np.array(scores)


def make_colour_tree(root, base_names, new_names):
    d = os.path.join(root, "VOCdevkit", "VOC2007")
    rng = np.random.default_rng(0)
    for split, n in (("trainval", 14), ("test", 10)):
        stems = []
        for k in range(n):
            stem = f"{(1 if split == 'trainval' else 500) + k:06d}"
            names = [str(rng.choice(base_names)), str(rng.choice(new_names))]
            img = Image.new("RGB", (96, 96), BG)
            objs, boxes = [], [(4, 4, 44, 44), (52, 52, 92, 92)]
            for nm, (x1, y1, x2, y2) in zip(names, boxes):
                img.paste(PAL(vu.VOC_NAME2COCO_IDX[nm]), (x1, y1, x2, y2))
                objs.append((nm, 0, (x1 + 1, y1 + 1, x2, y2)))                         # VOC is 1-based inclusive
            for sub in ("Annotations", "JPEGImages"):
                os.makedirs(os.path.join(d, sub), exist_ok=True)
            img.save(os.path.join(d, "JPEGImages", stem + ".jpg"), quality=100, subsampling=0)
            with open(os.path.join(d, "Annotations", stem + ".xml"), "w") as f:
                f.write(XML.format(stem=stem, w=96, h=96, objs="".join(
                    OBJ.format(name=n_, diff=f"<difficult>{dd}</difficult>", x1=b[0], y1=b[1], x2=b[2], y2=b[3])
                    for n_, dd, b in objs)))
            stems.append(stem)
        write_list(d, split, stems)


def test_end_to_end_voc_workdir_through_every_command_and_merge(tmp_path, monkeypatch, capsys):
    root, w = str(tmp_path / "voc"), str(tmp_path / "work")
    base = [vu.VOC_NAMES[i] for i in (0, 1, 3, 4, 5, 6, 7, 9, 11, 14)]               # 15+5 base classes
    new = ["pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    make_colour_tree(root, base, new)
    models = ColourModels()
    monkeypatch.setattr(rp, "get_models", lambda a: models)
    common = ["--workdir", w]
    assert rp.main(["fetch", "--dataset", "voc", "--voc_root", root, "--no_download", "--n_train_per_task", "12",
                    "--n_eval", "10", "--n_yolo_train", "10", "--n_yolo_val", "3", *common]) == 0
    assert rp.main(["gate-clip", *common]) == 0
    clip = json.load(open(os.path.join(w, "results", "gate_clip.json")))
    assert clip["all"] > 0.95 and clip["task0_within"] > 0.95 and clip["task1_within"] > 0.95
    assert rp.main(["extract", *common, "--max_boxes", "10"]) == 0
    assert rp.main(["gate-recall", *common]) == 0
    rec = json.load(open(os.path.join(w, "results", "gate_recall.json")))
    assert rec["task0"]["recall@0.5"] > 0.95 and rec["task1"]["recall@0.5"] > 0.95 and rec["task1"]["n_gt"] > 0
    assert rp.main(["run", *common, "--only", "E0_zeroshot", "E5_ours", "E6_gt_ceiling"]) == 0
    res = json.load(open(os.path.join(w, "results", "ablation.json")))
    assert res["E0_zeroshot"]["final_AP50"] > 0.9 and res["E6_gt_ceiling"]["final_AP50"] > 0.9
    assert res["E5_ours"]["final_AP50"] > 0.8
    capsys.readouterr()
    assert mr.main([f"voc155={w}"]) == 0
    line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("| voc155")][0]
    assert line.startswith("| voc155 | voc | 15+5 |")
    row = mr.summarize_run("voc155", w, ["E0_zeroshot", "E5_ours", "E6_gt_ceiling"], 0.5, 0.03)
    assert row["dataset"] == "voc" and row["scheme"] == "15+5" and row["recall_new"] > 0.95
    assert row["verdict"] == "FAIL"                                                   # perfect world: headroom ~0 < 0.03

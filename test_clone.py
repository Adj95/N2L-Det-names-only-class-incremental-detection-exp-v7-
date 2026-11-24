import argparse
import json
import os

import pytest

import run_pipeline as rp


def _base(wd):
    return ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30", "--n_tune", "20",
            "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]


def test_clone_copies_only_splits_paths_and_detector_pointer_and_leaves_the_source_untouched(tmp_path):
    src, dst = str(tmp_path / "src"), str(tmp_path / "dst")
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *_base(src)])
    os.makedirs(os.path.join(src, "results"), exist_ok=True)
    json.dump({"weights": "/abs/best.pt", "init": "scratch"}, open(os.path.join(src, "yolo_best.json"), "w"))
    json.dump({"x": 1}, open(os.path.join(src, "results", "tune.json"), "w"))
    before = sorted(os.listdir(src))
    rp.main(["clone", "--workdir", dst, "--src", src])
    assert sorted(os.listdir(dst)) == ["paths.json", "splits.json", "yolo_best.json"]       # no cache, no results, no clip.json
    assert open(os.path.join(dst, "splits.json")).read() == open(os.path.join(src, "splits.json")).read()
    yb = json.load(open(os.path.join(dst, "yolo_best.json")))
    assert yb["weights"] == "/abs/best.pt" and yb["cloned_from"] == os.path.abspath(src)
    assert sorted(os.listdir(src)) == before                                               # the original is untouched


def test_clone_makes_a_relative_weights_path_absolute_against_the_source(tmp_path):
    src, dst = tmp_path / "src", str(tmp_path / "dst")
    src.mkdir()
    for f in ("splits.json", "paths.json"):
        (src / f).write_text("{}")
    (src / "yolo_best.json").write_text(json.dumps({"weights": "yolo_runs/base/weights/best.pt"}))
    rp.main(["clone", "--workdir", dst, "--src", str(src)])
    assert json.load(open(os.path.join(dst, "yolo_best.json")))["weights"] == str(src / "yolo_runs/base/weights/best.pt")


def test_clone_refuses_missing_source_same_dir_and_existing_target(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    with pytest.raises(SystemExit, match="lacks splits.json, paths.json"):
        rp.main(["clone", "--workdir", str(tmp_path / "d"), "--src", str(src)])
    for f in ("splits.json", "paths.json"):
        (src / f).write_text("{}")
    with pytest.raises(SystemExit, match="same directory"):
        rp.main(["clone", "--workdir", str(src), "--src", str(src)])
    d = tmp_path / "d"; d.mkdir(); (d / "splits.json").write_text("{}")
    with pytest.raises(SystemExit, match="already holds a run"):
        rp.main(["clone", "--workdir", str(d), "--src", str(src)])


def test_extract_in_a_clone_works_and_does_not_touch_the_source_cache(tmp_path):
    src, dst = str(tmp_path / "src"), str(tmp_path / "dst")
    for c in ("fetch", "make-yolo-data", "extract"):
        rp.main([c, *_base(src)])
    snap = {f: os.path.getmtime(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(src, "cache")) for f in fs}
    rp.main(["clone", "--workdir", dst, "--src", src])
    rp.main(["extract", *_base(dst), "--pad", "0.3"])
    assert os.path.isdir(os.path.join(dst, "cache", "train"))
    assert json.load(open(os.path.join(dst, "clip.json")))["pad"] == 0.3
    assert snap == {f: os.path.getmtime(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(src, "cache")) for f in fs}


def test_clip_model_is_recorded_and_checked(tmp_path):
    wd = tmp_path / "wd"; wd.mkdir()
    A = lambda **k: argparse.Namespace(workdir=str(wd), fake=False, clip_weights="openai", **k)
    (wd / "clip.json").write_text(json.dumps({"clip_weights": "openai", "clip_model": "ViT-L-14"}))
    rp.check_clip_consistency(A(clip_model="ViT-L-14"))
    with pytest.raises(SystemExit, match="CLIP model 'ViT-L-14'.*--clip_model ViT-L-14"):
        rp.check_clip_consistency(A(clip_model="ViT-B-16"))
    (wd / "clip.json").write_text(json.dumps({"clip_weights": "openai"}))                  # an older record: ViT-B-16
    rp.check_clip_consistency(A(clip_model="ViT-B-16"))
    with pytest.raises(SystemExit, match="CLIP model 'ViT-B-16'"):
        rp.check_clip_consistency(A(clip_model="ViT-L-14"))
    assert rp.build_parser().parse_args(["run", "--clip_model", "ViT-L-14"]).clip_model == "ViT-L-14"
    assert rp.build_parser().parse_args(["run"]).clip_model == "ViT-B-16"


def test_get_models_passes_the_clip_model_to_the_encoder(monkeypatch):
    seen = {}
    import real_models
    monkeypatch.setattr(real_models, "RealModels", lambda **kw: seen.update(kw))
    a = rp.build_parser().parse_args(["run", "--clip_model", "ViT-L-14", "--clip_weights", "openai"])
    rp.get_models(a)
    assert seen["clip_model"] == "ViT-L-14" and seen["clip_pretrained"] == "openai"


def test_real_models_hands_the_architecture_to_clip(monkeypatch):
    import real_models
    seen = {}

    class Spy:
        dim = 8
        def __init__(self, **kw):
            seen.update(kw)
    monkeypatch.setattr(real_models, "ClipEncoder", Spy)
    real_models.RealModels(clip_pretrained="openai", clip_model="ViT-L-14")
    assert seen["model_name"] == "ViT-L-14" and seen["pretrained"] == "openai"
    seen.clear(); real_models.RealModels()
    assert seen["model_name"] == "ViT-B-16"


def test_extract_records_the_clip_model_and_weights_it_used(tmp_path, monkeypatch):
    from fake_models import FakeModels
    wd = str(tmp_path / "wd")
    rp.main(["fetch", *_base(wd)])
    assert not os.path.exists(os.path.join(wd, "clip.json"))
    rp.main(["extract", *_base(wd), "--splits", "train"])
    assert json.load(open(os.path.join(wd, "clip.json")))["clip_model"] == "fake"
    monkeypatch.setattr(rp, "get_models", lambda a: FakeModels(n_classes=9))
    argv = [x for x in _base(wd) if x != "--fake"]
    rp.main(["extract", *argv, "--splits", "eval", "--clip_model", "ViT-L-14", "--clip_weights", "laion400m_e32"])
    j = json.load(open(os.path.join(wd, "clip.json")))
    assert j["clip_model"] == "ViT-L-14" and j["clip_weights"] == "laion400m_e32"

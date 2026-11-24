import json
import os
from types import SimpleNamespace as NS
from unittest import mock

import numpy as np
import pytest
from PIL import Image

import coco_utils as cu
import real_models
import run_pipeline as rp
import yolo_tools as yt
from test_pipeline import TASKS, world  # noqa: F401


class _T:
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a


class _B:
    def __init__(self, xyxy, conf, cls): self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)
    def __len__(self): return len(self.conf.a)


def stub_world_model():
    m = mock.Mock()
    m.predict.side_effect = lambda **kw: [NS(boxes=_B([[5, 5, 50, 40], [1, 1, 1.2, 1.2]], [0.7, 0.9], [0, 0]))]
    return m


# ------------------------------------------------------------------ class-agnostic labels
def test_agnostic_yolo_labels_use_one_class_and_still_only_base_objects():
    coco = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [{"image_id": 1, "category_id": cu.COCO_IDS[0], "bbox": [10, 10, 20, 20], "iscrowd": 0},
                            {"image_id": 1, "category_id": cu.COCO_IDS[2], "bbox": [50, 50, 20, 20], "iscrowd": 0},
                            {"image_id": 1, "category_id": cu.COCO_IDS[7], "bbox": [0, 0, 30, 30], "iscrowd": 0}]}
    import tempfile
    d = tempfile.mkdtemp()
    cu.write_yolo_labels(coco, [1], [0, 1, 2], d, agnostic=True)
    rows = [r.split()[0] for r in open(os.path.join(d, "a.txt")).read().split("\n") if r]
    assert rows == ["0", "0"]                                               # classes 0 and 2 -> 0; class 7 not a base class
    cu.write_yolo_labels(coco, [1], [0, 1, 2], d)
    assert [r.split()[0] for r in open(os.path.join(d, "a.txt")).read().split("\n") if r] == ["0", "2"]   # default unchanged


def test_make_yolo_data_agnostic_flag(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_yolo_train", "15", "--n_yolo_val", "8"]
    rp.main(["fetch", *base]); rp.main(["make-yolo-data", *base, "--agnostic"])
    yaml = open(os.path.join(wd, "yolo_data", "data.yaml")).read()
    assert "0: object" in yaml and "1:" not in yaml
    lab = [l.split()[0] for f in os.listdir(os.path.join(wd, "yolo_data", "labels", "train"))
           for l in open(os.path.join(wd, "yolo_data", "labels", "train", f)) if l.strip()]
    assert lab and set(lab) == {"0"}
    rp.main(["make-yolo-data", *base])
    assert "2: " in open(os.path.join(wd, "yolo_data", "data.yaml")).read()


# ------------------------------------------------------------------ open-vocabulary proposer
def test_yoloworld_proposer_prompts_agnostic_nms_and_drops_degenerate():
    m = stub_world_model()
    p = yt.YoloWorldProposer(prompts=("object", "thing"), model=m, conf=0.02, max_det=50)
    m.set_classes.assert_called_once_with(["object", "thing"])
    boxes, conf = p(Image.new("RGB", (100, 80)))
    assert boxes.shape == (1, 4) and conf.tolist() == [0.7]
    kw = m.predict.call_args.kwargs
    assert kw["agnostic_nms"] is True and kw["conf"] == 0.02 and kw["max_det"] == 50 and kw["source"].shape == (80, 100, 3)
    allowed = {"source", "conf", "iou", "max_det", "imgsz", "device", "verbose", "agnostic_nms"}
    assert set(kw) <= allowed


def test_real_models_proposer_switch_and_signatures():
    import clip_encoder
    clip_cls = mock.create_autospec(clip_encoder.ClipEncoder); clip_cls.return_value.dim = 8
    yolo_cls = mock.create_autospec(yt.YoloProposer)
    world_cls = mock.create_autospec(yt.YoloWorldProposer)
    with mock.patch.object(real_models, "ClipEncoder", clip_cls), mock.patch.object(real_models, "YoloProposer", yolo_cls), \
            mock.patch.object(real_models, "YoloWorldProposer", world_cls):
        m = real_models.RealModels("openai", None, proposer="yoloworld", world_weights="w.pt", world_prompts=["object"],
                                   conf=0.05, max_det=30, device="cpu")
        world_cls.assert_called_once_with("w.pt", prompts=["object"], conf=0.05, iou=0.7, max_det=30, device="cpu")
        yolo_cls.assert_not_called()
        m.propose("img"); world_cls.return_value.assert_called_once_with("img")
        with pytest.raises(ValueError, match="proposer must be"):
            real_models.RealModels("openai", None, proposer="bogus")


def test_get_models_only_passes_proposer_knobs_when_used():
    base = dict(fake=False, clip_weights="openai", yolo_weights="w.pt", device=None, conf=0.01, max_det=100)
    with mock.patch("real_models.RealModels") as R:
        rp.get_models(NS(**base))
        assert R.call_args.kwargs == {"clip_pretrained": "openai", "clip_model": "ViT-B-16", "yolo_weights": "w.pt", "device": None,
                                      "conf": 0.01, "max_det": 100}
        rp.get_models(NS(**base, proposer="yoloworld", world_weights="x.pt", world_prompts="object, thing"))
        assert R.call_args.kwargs["proposer"] == "yoloworld" and R.call_args.kwargs["world_prompts"] == ["object", "thing"]


def test_extract_with_yoloworld_needs_no_yolo_weights_and_records_the_proposer(tmp_path):
    wd = str(tmp_path / "wd")
    base = ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30",
            "--n_yolo_train", "15", "--n_yolo_val", "8", "--max_boxes", "50"]
    rp.main(["fetch", *base])
    a = rp.build_parser().parse_args(["extract", *base[:-2], "--proposer", "yoloworld", "--world_prompts", "object,thing"])
    assert a.proposer == "yoloworld" and a.world_prompts == "object,thing" and a.world_weights == "yolov8s-worldv2.pt"
    a.yolo_weights = None
    assert rp._yolo_weights(a) is None                                    # no yolo_best.json lookup / crash
    rp.main(["extract", *base])
    clip = json.load(open(os.path.join(wd, "clip.json")))
    assert clip["proposer"] == "yolo" and clip["world_prompts"] == "object"
    with pytest.raises(SystemExit, match="settings differ"):
        rp.main(["extract", *base, "--proposer", "yoloworld", "--splits", "eval"])


def test_parser_defaults_keep_the_original_protocol():
    a = rp.build_parser().parse_args(["extract"])
    assert a.proposer == "yolo" and a.world_prompts == "object"
    assert rp.build_parser().parse_args(["make-yolo-data"]).agnostic is False

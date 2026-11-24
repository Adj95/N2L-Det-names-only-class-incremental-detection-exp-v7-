"""Step-3 validation: YOLO wrappers (stubbed) + Gate-1 recall aggregation.
Run: python -m pytest -q test_step3.py"""
import numpy as np
import pytest
from PIL import Image

import yolo_tools as yt
from gates import recall_over_dataset


class _T:                                   # mimics a torch tensor: .cpu().numpy()
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def __len__(self): return len(self.a)


class _Boxes:
    def __init__(self, xyxy, conf, cls): self.xyxy, self.conf, self.cls = _T(xyxy), _T(conf), _T(cls)
    def __len__(self): return len(self.xyxy)


class _Result:
    def __init__(self, boxes): self.boxes = boxes


class _StubModel:
    def __init__(self, result): self.result, self.kwargs, self.src = result, None, None
    def predict(self, source, **kw):
        self.kwargs, self.src = kw, source
        return [self.result]


def test_results_to_arrays_clips_and_drops_degenerate():
    r = _Result(_Boxes([[-5, -5, 50, 40], [10, 10, 10.5, 40], [90, 60, 140, 99], [0, 0, 20, 20]],
                       [0.9, 0.8, 0.7, 0.6], [3, 1, 2, 0]))
    b, c, k = yt.results_to_arrays(r, 100, 80)
    np.testing.assert_allclose(b, [[0, 0, 50, 40], [90, 60, 100, 80], [0, 0, 20, 20]])   # 2nd dropped
    np.testing.assert_allclose(c, [0.9, 0.7, 0.6]) and k.tolist() == [3, 2, 0]
    assert yt.results_to_arrays(_Result(None), 10, 10)[0].shape == (0, 4)
    assert yt.results_to_arrays(_Result(_Boxes(np.zeros((0, 4)), [], [])), 10, 10)[1].shape == (0,)


def test_proposer_passes_class_agnostic_low_conf_settings_and_bgr():
    res = _Result(_Boxes([[1, 2, 30, 40]], [0.05], [7]))
    m = _StubModel(res)
    p = yt.YoloProposer("unused.pt", conf=0.02, iou=0.65, max_det=77, imgsz=512, model=m)
    img = Image.new("RGB", (60, 50)); img.putpixel((0, 0), (10, 20, 30))
    boxes, scores = p(img)
    assert m.kwargs["agnostic_nms"] is True and m.kwargs["conf"] == 0.02 and m.kwargs["iou"] == 0.65
    assert m.kwargs["max_det"] == 77 and m.kwargs["imgsz"] == 512 and m.kwargs["verbose"] is False
    assert tuple(m.src[0, 0]) == (30, 20, 10)                    # BGR order handed to ultralytics
    assert m.src.shape == (50, 60, 3)
    np.testing.assert_allclose(boxes, [[1, 2, 30, 40]]) and np.testing.assert_allclose(scores, [0.05])


def test_train_base_rejects_bad_init():
    with pytest.raises(ValueError):
        yt.train_base("x.yaml", init="resnet")


def test_recall_over_dataset_hand_case():
    gt = {1: (np.array([[0, 0, 10, 10], [50, 50, 60, 60.0]]), np.array([0, 45])),
          2: (np.array([[0, 0, 10, 10.0]]), np.array([45])),
          3: (np.zeros((0, 4)), np.zeros(0, int))}
    props = {1: np.array([[0, 0, 10, 10], [50, 50, 60, 70.0]]),      # base hit (IoU 1), new IoU .5
             2: np.zeros((0, 4))}                                     # image 3 has no entry at all
    r = recall_over_dataset(props, gt, {"base": range(0, 40), "new": range(40, 80)})
    assert r["base"] == {"n_gt": 1, "recall@0.5": 1.0, "recall@0.7": 1.0}
    assert r["new"]["n_gt"] == 2 and r["new"]["recall@0.5"] == 0.5 and r["new"]["recall@0.7"] == 0.0
    assert abs(r["boxes_per_image"] - 2 / 3) < 1e-12
    empty = recall_over_dataset({}, gt, {"none": [99]})
    assert empty["none"]["n_gt"] == 0 and np.isnan(empty["none"]["recall@0.5"])

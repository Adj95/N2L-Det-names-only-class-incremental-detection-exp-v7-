"""yolo_tools.py -- thin ultralytics wrappers: base-class training + class-agnostic proposals.

Kept deliberately small (little untested surface). The result->numpy conversion is unit-tested
with a stub that mimics the ultralytics result objects; actual training / inference only run on
a GPU box (Colab/Kaggle).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image

from boxes import clip_boxes


def results_to_arrays(result, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ultralytics Results -> (boxes xyxy float64 (n,4) clipped to the image, conf (n,), cls (n,)).
    Drops boxes that are degenerate after clipping."""
    b = result.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int64)
    xyxy = clip_boxes(b.xyxy.cpu().numpy(), img_w, img_h)
    conf = b.conf.cpu().numpy().astype(np.float64).reshape(-1)
    cls = b.cls.cpu().numpy().astype(np.int64).reshape(-1)
    ok = (xyxy[:, 2] - xyxy[:, 0] > 1) & (xyxy[:, 3] - xyxy[:, 1] > 1)
    return xyxy[ok], conf[ok], cls[ok]


class YoloProposer:
    """Frozen YOLOv8 used ONLY as a box proposer: very low confidence, class-agnostic NMS.
    __call__(PIL) -> (boxes xyxy, scores) in original-image pixels."""

    def __init__(self, weights: str, conf: float = 0.01, iou: float = 0.7, max_det: int = 100,
                 imgsz: int = 640, device=None, model=None):
        if model is None:
            from ultralytics import YOLO  # lazy: only on the GPU box
            model = YOLO(weights)
        self.model, self.conf, self.iou = model, conf, iou
        self.max_det, self.imgsz, self.device = max_det, imgsz, device

    def __call__(self, img: Image.Image):
        res = self.model.predict(source=np.asarray(img.convert("RGB"))[..., ::-1],   # BGR like cv2
                                 conf=self.conf, iou=self.iou, agnostic_nms=True,
                                 max_det=self.max_det, imgsz=self.imgsz, device=self.device,
                                 verbose=False)[0]
        boxes, conf, _ = results_to_arrays(res, img.width, img.height)
        return boxes, conf


class YoloWorldProposer:
    """Open-vocabulary proposer: YOLO-World prompted with generic words (default just 'object'), class-agnostic NMS.
    It has NOT been trained on our base classes, so new-class objects are not treated as background by it; the price is
    that it was pre-trained on large detection sets (Objects365/GQA/Flickr30k) -- a protocol change to report.
    __call__(PIL) -> (boxes xyxy, scores)."""

    def __init__(self, weights: str = "yolov8s-worldv2.pt", prompts=("object",), conf: float = 0.01, iou: float = 0.7,
                 max_det: int = 100, imgsz: int = 640, device=None, model=None):
        if model is None:
            from ultralytics import YOLOWorld  # lazy: only on the GPU box
            model = YOLOWorld(weights)
        model.set_classes(list(prompts))
        self.model, self.conf, self.iou = model, conf, iou
        self.max_det, self.imgsz, self.device = max_det, imgsz, device

    def __call__(self, img: Image.Image):
        res = self.model.predict(source=np.asarray(img.convert("RGB"))[..., ::-1], conf=self.conf, iou=self.iou,
                                 agnostic_nms=True, max_det=self.max_det, imgsz=self.imgsz, device=self.device,
                                 verbose=False)[0]
        boxes, conf, _ = results_to_arrays(res, img.width, img.height)
        return boxes, conf


def train_base(data_yaml: str, init: str = "scratch", size: str = "s", epochs: int = 30,
               imgsz: int = 640, batch: int = 32, project: str = "runs", name: str = "base40",
               device=None, **kw) -> str:
    """Train the base detector on the base classes only. Returns the best.pt path.

    init='scratch': random init from yolov8{size}.yaml  -- NO COCO knowledge leaks in (clean
                    protocol, weaker boxes).
    init='coco'   : start from COCO-pretrained yolov8{size}.pt -- LEAKS the 40 'new' classes
                    into the backbone/box regressor. Only use as an upper bound on box recall
                    and say so in the report.
    init='imagenet' (exp v6): detector built from yolov8{size}.yaml, backbone copied from the ImageNet-1k
                    classifier yolov8{size}-cls.pt (matching layers by name and shape); neck and box head start
                    random. No box supervision leaks in; ImageNet image-level labels overlap some class
                    names (e.g. tv, ram), which the report must state. Standard in incremental-detection papers.
    """
    if init not in ("scratch", "coco", "imagenet"):
        raise ValueError("init must be 'scratch', 'coco' or 'imagenet'")
    from ultralytics import YOLO  # lazy
    if init == "imagenet":
        model = YOLO(f"yolov8{size}.yaml").load(f"yolov8{size}-cls.pt")
    else:
        model = YOLO(f"yolov8{size}.yaml" if init == "scratch" else f"yolov8{size}.pt")
    r = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, project=project,
                    name=name, device=device, exist_ok=True, **kw)
    return str(r.save_dir / "weights" / "best.pt") if hasattr(r, "save_dir") else \
        f"{project}/{name}/weights/best.pt"

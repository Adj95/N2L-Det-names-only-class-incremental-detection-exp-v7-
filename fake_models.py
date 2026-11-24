"""fake_models.py -- stand-ins for CLIP and YOLO so the WHOLE pipeline can be run (and tested)
without weights, a GPU or COCO: `python run_pipeline.py selftest`.

World: black images with filled coloured rectangles; class c <-> colour bin c+1 (bin 0 = black).
  FakeModels.encode_images_u8 : 27-bin colour histogram of the crop   (stands in for CLIP image tower)
  FakeModels.encode_text      : one-hot of the colour bin of the class named in the prompt;
                                prompts that name no class (background prompts) -> bin 0
  FakeModels.propose          : bounding boxes of non-black connected components + 3 random boxes
                                (never sees ground truth)
"""
from __future__ import annotations

import os
import re
from typing import List, Sequence

import numpy as np
from PIL import Image
from scipy import ndimage

import coco_utils as cu
from boxes import iou_matrix

LEVELS = (0, 128, 255)


def bin_rgb(b: int):
    return LEVELS[b // 9], LEVELS[(b // 3) % 3], LEVELS[b % 3]


def pixel_bins(arr_u8: np.ndarray) -> np.ndarray:
    q = np.digitize(arr_u8, [64, 192])                      # 0,1,2 per channel
    return q[..., 0] * 9 + q[..., 1] * 3 + q[..., 2]


class FakeModels:
    dim = 27

    def __init__(self, n_classes: int = 12, seed: int = 0):
        if n_classes > 26:
            raise ValueError("at most 26 fake classes (27 colour bins minus black)")
        self.n_classes = n_classes
        self.names = cu.prompt_names(range(n_classes))
        self._pat = [re.compile(r"\b" + re.escape(n) + r"\b") for n in self.names]
        self.seed = seed

    def class_rgb(self, c: int):
        return bin_rgb(c + 1)

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            hit = [c for c, p in enumerate(self._pat) if p.search(t)]
            out[i, (hit[0] + 1) if hit else 0] = 1.0
        return out * 7.0

    def encode_images_u8(self, crops: np.ndarray) -> np.ndarray:
        crops = np.asarray(crops)
        out = np.zeros((len(crops), self.dim))
        for i, c in enumerate(crops):
            out[i] = np.bincount(pixel_bins(c).ravel(), minlength=self.dim) / c.shape[0] / c.shape[1]
        return out * 10.0

    def propose(self, img: Image.Image):
        arr = np.asarray(img.convert("RGB"))
        lab, n = ndimage.label(arr.max(axis=2) > 40)
        boxes = []
        for sl in ndimage.find_objects(lab):
            y, x = sl
            if (y.stop - y.start) * (x.stop - x.start) >= 100:
                boxes.append([x.start, y.start, x.stop, y.stop])
        scores = [0.9] * len(boxes)
        rng = np.random.default_rng(int(arr.sum()) % (2 ** 32))          # deterministic per image
        H, W = arr.shape[:2]
        for _ in range(3):
            w, h = rng.uniform(20, 80, size=2)
            x1, y1 = rng.uniform(0, W - w), rng.uniform(0, H - h)
            boxes.append([x1, y1, x1 + w, y1 + h]); scores.append(0.2)
        return np.array(boxes, dtype=np.float64).reshape(-1, 4), np.array(scores)


def make_image_world(root: str, tasks: Sequence[Sequence[int]], n_per_task: int = 40, n_eval: int = 60,
                     canvas: int = 160, p_old: float = 0.4, seed: int = 0):
    """Write PNG images + COCO-format json dicts. Returns (coco_train, coco_val, train_dir, val_dir)."""
    rng = np.random.default_rng(seed)
    tasks = [list(t) for t in tasks]
    C = max(max(t) for t in tasks) + 1
    fm = FakeModels(C)
    train_dir, val_dir = os.path.join(root, "train"), os.path.join(root, "val")
    os.makedirs(train_dir, exist_ok=True); os.makedirs(val_dir, exist_ok=True)

    def render(classes, folder, iid):
        arr = np.zeros((canvas, canvas, 3), np.uint8)
        placed = []
        for c in classes:
            for _ in range(30):
                w, h = rng.integers(28, 60, size=2)
                x, y = int(rng.integers(0, canvas - w)), int(rng.integers(0, canvas - h))
                b = np.array([x, y, x + w, y + h], float)
                if all(iou_matrix(b[None], np.array([p[:4]]))[0, 0] == 0 and
                       not (b[0] < p[2] + 4 and p[0] < b[2] + 4 and b[1] < p[3] + 4 and p[1] < b[3] + 4)
                       for p in placed):
                    arr[y:y + h, x:x + w] = fm.class_rgb(c)
                    placed.append((*b, c))
                    break
        name = f"{iid:012d}.png"
        Image.fromarray(arr).save(os.path.join(folder, name))
        return name, placed

    def build(split_specs, folder, start):
        images, anns, iid = [], [], start
        for classes in split_specs:
            iid += 1
            name, placed = render(classes, folder, iid)
            images.append({"id": iid, "file_name": name, "width": canvas, "height": canvas,
                           "coco_url": "file://" + os.path.join(folder, name)})
            for (x1, y1, x2, y2, c) in placed:
                anns.append({"image_id": iid, "category_id": cu.COCO_IDS[int(c)], "iscrowd": 0,
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]})
        return {"images": images, "annotations": anns,
                "categories": [{"id": i, "name": n} for i, n in zip(cu.COCO_IDS, cu.COCO_NAMES)]}

    train_specs = []
    for t, cls_t in enumerate(tasks):
        older = [c for tt in tasks[:t] for c in tt]
        for _ in range(n_per_task):
            cls = [int(rng.choice(cls_t)) for _ in range(int(rng.integers(1, 3)))]
            if older and rng.random() < p_old:
                cls.append(int(rng.choice(older)))
            train_specs.append(cls)
    allc = [c for t in tasks for c in t]
    eval_specs = [[int(rng.choice(allc)) for _ in range(int(rng.integers(1, 4)))] for _ in range(n_eval)]
    return (build(train_specs, train_dir, 0), build(eval_specs, val_dir, 100000), train_dir, val_dir)

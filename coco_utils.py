"""coco_utils.py -- COCO metadata, class-incremental splits, COCO->YOLO conversion.

Pure python/numpy (+ json); no torch. Everything here is unit-tested on synthetic COCO json.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Sequence

import numpy as np

COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
            27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77,
            78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
COCO_NAMES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
              "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
              "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
              "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
              "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
              "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
              "toothbrush"]
# CLIP-friendly names (only where the COCO label reads badly in a prompt)
PROMPT_NAME_FIX = {"tv": "television", "hair drier": "hair dryer", "skis": "pair of skis",
                   "couch": "couch", "mouse": "computer mouse", "remote": "remote control"}
ID2IDX = {cid: i for i, cid in enumerate(COCO_IDS)}
_DEFAULT = (list(COCO_IDS), list(COCO_NAMES))
_CUSTOM_PROMPTS = None                      # None = COCO behaviour (PROMPT_NAME_FIX); else a list aligned with COCO_NAMES


def set_class_space(names: Sequence[str], ids: Sequence[int], prompt_names_: Sequence[str] = None) -> None:
    """Replace the class space IN PLACE (modules that did `from coco_utils import COCO_IDS` see the change).
    Used for non-COCO datasets: class index j <-> (ids[j], names[j]). Call reset_class_space() to get COCO-80 back."""
    global _CUSTOM_PROMPTS
    names, ids = [str(n) for n in names], [int(i) for i in ids]
    if len(names) != len(ids) or not names:
        raise ValueError("names and ids must be non-empty and the same length")
    if len(set(ids)) != len(ids) or len(set(names)) != len(names):
        raise ValueError("class ids and class names must be unique")
    if prompt_names_ is not None and len(prompt_names_) != len(names):
        raise ValueError("prompt_names must have one entry per class")
    COCO_IDS[:], COCO_NAMES[:] = ids, names
    ID2IDX.clear(); ID2IDX.update({c: i for i, c in enumerate(ids)})
    _CUSTOM_PROMPTS = list(prompt_names_) if prompt_names_ is not None else list(names)


def reset_class_space() -> None:
    global _CUSTOM_PROMPTS
    COCO_IDS[:], COCO_NAMES[:] = _DEFAULT
    ID2IDX.clear(); ID2IDX.update({c: i for i, c in enumerate(COCO_IDS)})
    _CUSTOM_PROMPTS = None


def num_classes() -> int:
    return len(COCO_NAMES)


def prompt_names(indices: Sequence[int] = None) -> List[str]:
    idx = range(num_classes()) if indices is None else indices
    if _CUSTOM_PROMPTS is not None:
        return [_CUSTOM_PROMPTS[i] for i in idx]
    return [PROMPT_NAME_FIX.get(COCO_NAMES[i], COCO_NAMES[i]) for i in idx]


def verify_categories(categories: List[dict]) -> None:
    """Assert that the annotation file's categories are exactly the 80 we assume, in order."""
    cats = sorted(categories, key=lambda c: c["id"])
    ids = [c["id"] for c in cats]
    names = [c["name"] for c in cats]
    if ids != COCO_IDS or names != COCO_NAMES:
        raise ValueError("annotation categories differ from the built-in COCO-80 list")


# ------------------------------------------------------------------ splits
def make_tasks(scheme: str, num_classes: int = 80) -> List[List[int]]:
    """'40+40' -> [[0..39],[40..79]];  '40+4x10' -> base 40 then four tasks of 10.
    Class order is COCO category-id order (the usual IOD convention)."""
    m = re.fullmatch(r"(\d+)\+(?:(\d+)x)?(\d+)", scheme.strip())
    if not m:
        raise ValueError(f"bad scheme {scheme!r}; use e.g. '40+40' or '40+4x10'")
    base = int(m.group(1))
    reps = int(m.group(2)) if m.group(2) else 1
    inc = int(m.group(3))
    if base + reps * inc != num_classes:
        raise ValueError(f"{scheme!r} covers {base + reps * inc} classes, expected {num_classes}")
    tasks, lo = [list(range(base))], base
    for _ in range(reps):
        tasks.append(list(range(lo, lo + inc)))
        lo += inc
    return tasks


def random_tasks(tasks: Sequence[Sequence[int]], seed: int) -> List[List[int]]:
    """Same task sizes, classes randomly reassigned (deterministic in seed). Each task is returned sorted."""
    flat = [c for t in tasks for c in t]
    perm = np.random.default_rng(seed).permutation(flat).tolist()
    out, lo = [], 0
    for t in tasks:
        out.append(sorted(int(c) for c in perm[lo:lo + len(t)]))
        lo += len(t)
    return out


def image_class_index(coco: dict, skip_crowd: bool = True) -> Dict[int, set]:
    """image_id -> set of class indices (0..79) present."""
    out: Dict[int, set] = {im["id"]: set() for im in coco["images"]}
    for a in coco["annotations"]:
        if skip_crowd and a.get("iscrowd", 0):
            continue
        out[a["image_id"]].add(ID2IDX[a["category_id"]])
    return out


def select_task_images(coco: dict, classes: Sequence[int], max_images: int = None,
                       seed: int = 0, exclude: Sequence[int] = ()) -> List[int]:
    """Images containing >=1 (non-crowd) object of `classes`. Deterministic subsample.
    (Standard IOD protocol: an image may appear in several tasks.) `exclude`: image ids that must not be drawn."""
    cls, ex = set(classes), set(exclude)
    ids = sorted(i for i, s in image_class_index(coco).items() if s & cls and i not in ex)
    if max_images is not None and len(ids) > max_images:
        rng = np.random.default_rng(seed)
        ids = sorted(rng.choice(ids, size=max_images, replace=False).tolist())
    return ids


def gt_by_image(coco: dict, image_ids: Sequence[int], classes: Sequence[int] = None):
    """image_id -> (boxes xyxy (n,4), class idx (n,)). Non-crowd only. `classes` filters."""
    keep = None if classes is None else set(classes)
    want = set(image_ids)
    acc: Dict[int, list] = {i: [] for i in image_ids}
    for a in coco["annotations"]:
        if a["image_id"] not in want or a.get("iscrowd", 0):
            continue
        c = ID2IDX[a["category_id"]]
        if keep is not None and c not in keep:
            continue
        x, y, w, h = a["bbox"]
        if w <= 1 or h <= 1:
            continue
        acc[a["image_id"]].append((x, y, x + w, y + h, c))
    out = {}
    for i, rows in acc.items():
        arr = np.array(rows, dtype=np.float64).reshape(-1, 5)
        out[i] = (arr[:, :4], arr[:, 4].astype(np.int64))
    return out


# ------------------------------------------------------------------ COCO -> YOLO
def write_yolo_labels(coco: dict, image_ids: Sequence[int], classes: Sequence[int],
                      label_dir: str, agnostic: bool = False) -> Dict[str, int]:
    """One .txt per image, only objects of `classes` (remapped to 0..len-1; all to 0 when `agnostic`).
    Objects of other classes are simply NOT written (standard IOD: unlabeled = background)."""
    os.makedirs(label_dir, exist_ok=True)
    remap = {c: (0 if agnostic else j) for j, c in enumerate(classes)}
    images = {im["id"]: im for im in coco["images"]}
    rows: Dict[int, list] = {i: [] for i in image_ids}
    n_obj = n_skip = 0
    for a in coco["annotations"]:
        i = a["image_id"]
        if i not in rows:
            continue
        c = ID2IDX[a["category_id"]]
        if c not in remap:
            continue
        x, y, w, h = a["bbox"]
        im = images[i]
        W, H = im["width"], im["height"]
        x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if a.get("iscrowd", 0) or (x2 - x1) <= 1 or (y2 - y1) <= 1:
            n_skip += 1
            continue
        rows[i].append(f"{remap[c]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                       f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        n_obj += 1
    for i, lines in rows.items():
        stem = os.path.splitext(images[i]["file_name"])[0]
        with open(os.path.join(label_dir, stem + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    return {"images": len(rows), "objects": n_obj, "skipped": n_skip}


def write_dataset_yaml(path: str, root: str, class_names: Sequence[str]) -> None:
    lines = [f"path: {root}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(class_names)]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

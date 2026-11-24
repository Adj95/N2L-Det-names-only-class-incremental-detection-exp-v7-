"""custom_data.py -- use ANY COCO-format detection dataset (own class names / category ids), not just COCO-80 / VOC.

The pipeline indexes classes 0..K-1 through coco_utils' class space. `read_classes` derives that space from the
annotation files (categories sorted by id, identical in train and val) and `coco_utils.set_class_space` installs it;
the space is stored in paths.json so every later command (gate-clip, extract, run, ...) re-installs it.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence


def read_classes(coco_train: dict, coco_val: dict, name_map: Optional[Dict[str, str]] = None) -> dict:
    """{'names': [...], 'ids': [...], 'prompt_names': [...]} in category-id order; train and val must agree."""
    def cats(c, which):
        out = sorted(c.get("categories", []), key=lambda x: x["id"])
        if not out:
            raise ValueError(f"{which} annotation file has no 'categories'")
        return [(int(x["id"]), str(x["name"])) for x in out]
    a, b = cats(coco_train, "train"), cats(coco_val, "val")
    if a != b:
        raise ValueError("train and val annotation files list different categories (ids/names must be identical)")
    ids, names = [i for i, _ in a], [n for _, n in a]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate category names: {sorted(n for n in set(names) if names.count(n) > 1)}")
    known = set(ids)
    for which, c in (("train", coco_train), ("val", coco_val)):
        bad = {x["category_id"] for x in c["annotations"]} - known
        if bad:
            raise ValueError(f"{which} annotations use category ids {sorted(bad)[:5]} that are not in 'categories'")
    nm = name_map or {}
    unknown = set(nm) - set(names)
    if unknown:
        raise ValueError(f"name_map has names that are not categories: {sorted(unknown)[:5]}")
    return {"names": names, "ids": ids, "prompt_names": [nm.get(n, n) for n in names]}


def check_images(coco: dict, img_dir: str, what: str, n: int = 25) -> None:
    """Fail early if the files the json names are not under img_dir, or the json lacks width/height."""
    ims = coco["images"][:n]
    if not ims:
        raise ValueError(f"{what}: annotation file has no images")
    for im in ims:
        if "width" not in im or "height" not in im:
            raise ValueError(f"{what}: image record {im.get('id')} lacks width/height (needed for YOLO labels)")
    missing = [im["file_name"] for im in ims if not os.path.exists(os.path.join(img_dir, im["file_name"]))]
    if missing:
        raise ValueError(f"{what}: {len(missing)} of the first {len(ims)} images are missing under {img_dir} "
                         f"(e.g. {missing[:2]}); check --img_dir_* and the json's file_name paths")


def load_name_map(path: Optional[str]) -> Optional[Dict[str, str]]:
    if not path:
        return None
    with open(path) as f:
        m = json.load(f)
    if not isinstance(m, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in m.items()):
        raise ValueError(f"{path} must be a JSON object mapping category name -> text used in CLIP prompts")
    return m

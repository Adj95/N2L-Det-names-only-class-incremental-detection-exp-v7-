"""coco_fetch.py -- download COCO annotations and only the images a run needs.

Works with http(s):// and file:// URLs (file:// is how the tests exercise it offline).
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_file(url: str, dest: str, retries: int = 3, timeout: int = 60) -> str:
    """Atomic download (writes dest.part, then renames). Skips if dest exists and is non-empty."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    last = None
    for _ in range(max(1, retries)):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(dest + ".part", "wb") as f:
                shutil.copyfileobj(r, f)
            if os.path.getsize(dest + ".part") == 0:
                raise IOError("empty download")
            os.replace(dest + ".part", dest)
            return dest
        except Exception as e:                      # noqa: BLE001 - retry any network/IO error
            last = e
    if os.path.exists(dest + ".part"):
        os.remove(dest + ".part")
    raise IOError(f"failed to download {url}: {last}")


def ensure_annotations(root: str, splits=("train", "val"), url: str = ANN_URL) -> Dict[str, str]:
    """Return {split: path to instances_<split>2017.json}, downloading+extracting if missing."""
    ann_dir = os.path.join(root, "annotations")
    paths = {s: os.path.join(ann_dir, f"instances_{s}2017.json") for s in splits}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    zpath = download_file(url, os.path.join(root, "annotations_trainval2017.zip"))
    with zipfile.ZipFile(zpath) as z:
        for s, p in paths.items():
            member = f"annotations/instances_{s}2017.json"
            if not os.path.exists(p):
                os.makedirs(ann_dir, exist_ok=True)
                with z.open(member) as src, open(p, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    return paths


def load_coco(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fetch_images(coco: dict, image_ids: Iterable[int], img_dir: str, workers: int = 16,
                 url_key: str = "coco_url") -> List[str]:
    """Download the listed images (by id) into img_dir/<file_name>. Returns local paths in order.
    Raises with the list of failures if any image could not be fetched."""
    by_id = {im["id"]: im for im in coco["images"]}
    ids = list(image_ids)
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"image ids not in annotations: {missing[:5]}")
    os.makedirs(img_dir, exist_ok=True)

    def one(i):
        im = by_id[i]
        return download_file(im[url_key], os.path.join(img_dir, im["file_name"]))

    failed, out = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [(i, ex.submit(one, i)) for i in ids]
        for i, fu in futs:
            try:
                out.append(fu.result())
            except Exception as e:                  # noqa: BLE001
                failed.append((i, str(e)))
    if failed:
        raise IOError(f"{len(failed)} image downloads failed, e.g. {failed[:3]}")
    return out

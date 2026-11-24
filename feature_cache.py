"""feature_cache.py -- on-disk cache of per-image proposal boxes + CLIP crop features.

Directory format (memory-mappable, no pickle):
    meta.json, image_ids.npy (N,), offsets.npy (N+1,), boxes.npy (M,4) f32 xyxy ORIGINAL pixels,
    scores.npy (M,) f32 (proposal confidence), feats.npy (M,D) f16 RAW (un-normalised) CLIP
    features of the PADDED crop, plus optional extra per-box arrays (extra_<name>.npy).
Raw features are stored because the official N2L code runs PCA/ridge on raw features and uses
normalised ones only for the zero-shot logits.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Iterator, Optional

import numpy as np


class CacheWriter:
    def __init__(self, out_dir: str, feat_dim: int, extra_dtypes: Optional[Dict[str, str]] = None):
        self.out_dir, self.D = out_dir, int(feat_dim)
        self.extra_dtypes = dict(extra_dtypes or {})
        self.ids, self.counts = [], []
        self.boxes, self.scores, self.feats = [], [], []
        self.extra = {k: [] for k in self.extra_dtypes}

    def add_image(self, image_id: int, boxes, scores, feats, **extra) -> None:
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        feats = np.asarray(feats, dtype=np.float32).reshape(len(boxes), -1) if len(boxes) else \
            np.zeros((0, self.D), np.float32)
        n = len(boxes)
        if len(scores) != n or feats.shape != (n, self.D):
            raise ValueError(f"inconsistent shapes: boxes {boxes.shape}, scores {scores.shape}, "
                             f"feats {feats.shape}, D={self.D}")
        if not (np.isfinite(boxes).all() and np.isfinite(feats).all() and np.isfinite(scores).all()):
            raise ValueError("non-finite values")
        if n and (np.any(boxes[:, 2] <= boxes[:, 0]) or np.any(boxes[:, 3] <= boxes[:, 1])):
            raise ValueError("degenerate box (x2<=x1 or y2<=y1)")
        if set(extra) != set(self.extra_dtypes):
            raise ValueError(f"extra arrays {sorted(extra)} != declared {sorted(self.extra_dtypes)}")
        for k, v in extra.items():
            v = np.asarray(v).reshape(n, *np.asarray(v).shape[1:]) if n else np.asarray(v)[:0]
            if len(v) != n:
                raise ValueError(f"extra '{k}' has {len(v)} rows, expected {n}")
            self.extra[k].append(v.astype(self.extra_dtypes[k]))
        self.ids.append(int(image_id))
        self.counts.append(n)
        self.boxes.append(boxes); self.scores.append(scores); self.feats.append(feats)

    def save(self) -> str:
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate image ids")
        os.makedirs(self.out_dir, exist_ok=True)
        offsets = np.concatenate([[0], np.cumsum(self.counts)]).astype(np.int64)
        cat = lambda xs, shape, dt: np.concatenate(xs).astype(dt) if xs and offsets[-1] else \
            np.zeros(shape, dt)
        np.save(os.path.join(self.out_dir, "image_ids.npy"), np.array(self.ids, dtype=np.int64))
        np.save(os.path.join(self.out_dir, "offsets.npy"), offsets)
        np.save(os.path.join(self.out_dir, "boxes.npy"), cat(self.boxes, (0, 4), np.float32))
        np.save(os.path.join(self.out_dir, "scores.npy"), cat(self.scores, (0,), np.float32))
        np.save(os.path.join(self.out_dir, "feats.npy"), cat(self.feats, (0, self.D), np.float16))
        for k, xs in self.extra.items():
            tail = xs[0].shape[1:] if xs else ()
            np.save(os.path.join(self.out_dir, f"extra_{k}.npy"),
                    cat(xs, (0, *tail), self.extra_dtypes[k]))
        with open(os.path.join(self.out_dir, "meta.json"), "w") as f:
            json.dump({"feat_dim": self.D, "n_images": len(self.ids), "n_boxes": int(offsets[-1]),
                       "extra": self.extra_dtypes}, f)
        return self.out_dir


class FeatureCache:
    def __init__(self, cache_dir: str, mmap: bool = True):
        mm = "r" if mmap else None
        L = lambda n: np.load(os.path.join(cache_dir, n), mmap_mode=mm)
        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.image_ids = np.load(os.path.join(cache_dir, "image_ids.npy"))
        self.offsets = np.load(os.path.join(cache_dir, "offsets.npy"))
        self.boxes, self.scores, self.feats = L("boxes.npy"), L("scores.npy"), L("feats.npy")
        self.extra = {k: L(f"extra_{k}.npy") for k in self.meta["extra"]}
        if self.offsets[-1] != len(self.boxes) or len(self.offsets) != len(self.image_ids) + 1:
            raise ValueError("corrupt cache: offsets disagree with array sizes")
        self._row = {int(i): r for r, i in enumerate(self.image_ids)}

    @property
    def n_images(self) -> int:
        return len(self.image_ids)

    @property
    def n_boxes(self) -> int:
        return int(self.offsets[-1])

    def span(self, image_id: int):
        r = self._row[int(image_id)]
        return int(self.offsets[r]), int(self.offsets[r + 1])

    def image(self, image_id: int) -> dict:
        a, b = self.span(image_id)
        return {"image_id": int(image_id), "boxes": np.asarray(self.boxes[a:b]),
                "scores": np.asarray(self.scores[a:b]), "feats": np.asarray(self.feats[a:b]),
                **{k: np.asarray(v[a:b]) for k, v in self.extra.items()}}

    def gather(self, image_ids, dtype=np.float64):
        """Concatenate the given images -> (feats[M,D], image_idx[M] positions in image_ids,
        boxes[M,4], scores[M], row_ids[M] global row numbers)."""
        F, I, B, S, R = [], [], [], [], []
        for k, iid in enumerate(image_ids):
            a, b = self.span(iid)
            F.append(np.asarray(self.feats[a:b], dtype=dtype)); B.append(np.asarray(self.boxes[a:b]))
            S.append(np.asarray(self.scores[a:b])); I.append(np.full(b - a, k, np.int64))
            R.append(np.arange(a, b))
        if not F:
            D = self.meta["feat_dim"]
            return np.zeros((0, D)), np.zeros(0, np.int64), np.zeros((0, 4), np.float32), \
                np.zeros(0, np.float32), np.zeros(0, np.int64)
        return np.concatenate(F), np.concatenate(I), np.concatenate(B), np.concatenate(S), \
            np.concatenate(R)

    def __iter__(self) -> Iterator[dict]:
        for iid in self.image_ids:
            yield self.image(int(iid))

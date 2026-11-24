"""extract_features.py -- build a FeatureCache from images + a proposal source + an image encoder.

All three heavy parts are injected callables, so the loop itself is fully testable:
    image_source   : iterable of (image_id, PIL.Image)
    proposal_fn    : PIL.Image -> (boxes_xyxy (n,4), scores (n,))
    encode_u8_fn   : uint8 (n,224,224,3) -> raw features (n,D)
"""
from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np
from PIL import Image

from clip_encoder import crops_for_image
from feature_cache import CacheWriter


def extract_cache(image_source: Iterable[Tuple[int, Image.Image]], proposal_fn: Callable,
                  encode_u8_fn: Callable, out_dir: str, feat_dim: int, pad_frac: float = 0.1,
                  size: int = 224, max_boxes: int = 100, log_every: int = 200) -> str:
    w = CacheWriter(out_dir, feat_dim)
    for n_done, (iid, img) in enumerate(image_source, 1):
        boxes, scores = proposal_fn(img)
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(boxes) > max_boxes:                       # keep the most confident
            top = np.argsort(-scores, kind="stable")[:max_boxes]
            boxes, scores = boxes[top], scores[top]
        if len(boxes):
            feats = encode_u8_fn(crops_for_image(img, boxes, pad_frac, size))
        else:
            feats = np.zeros((0, feat_dim), np.float32)
        w.add_image(iid, boxes, scores, feats)
        if log_every and n_done % log_every == 0:
            print(f"  extracted {n_done} images, {sum(w.counts)} boxes", flush=True)
    return w.save()

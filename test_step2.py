"""Step-2 validation: geometry, COCO utils, prompts, crops, cache, extraction plumbing.
Run:  python -m pytest -q test_step2.py"""
import json
import os

import numpy as np
import pytest
from PIL import Image

import boxes as bx
import coco_utils as cu
import prompts as pr
from clip_encoder import CLIP_MEAN, CLIP_STD, crop_resize, crops_for_image, to_clip_input
from extract_features import extract_cache
from feature_cache import CacheWriter, FeatureCache
from gates import box_recall, gt_crop_zeroshot_eval


# ============================================================ boxes
def test_iou_hand_cases():
    a = np.array([[0, 0, 2, 2.0]])
    b = np.array([[0, 0, 2, 2.0], [1, 0, 3, 2], [5, 5, 6, 6], [1, 1, 1, 1]])
    np.testing.assert_allclose(bx.iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0, 0.0])
    assert bx.iou_matrix(np.zeros((0, 4)), b).shape == (0, 4)
    assert bx.iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)


def test_xywh_roundtrip():
    r = np.random.default_rng(0).uniform(1, 50, size=(20, 4))
    np.testing.assert_allclose(bx.xyxy_to_xywh(bx.xywh_to_xyxy(r)), r)


def test_pad_boxes():
    np.testing.assert_allclose(bx.pad_boxes([[10, 10, 20, 20]], 0.1, 100, 100), [[9, 9, 21, 21]])
    np.testing.assert_allclose(bx.pad_boxes([[0, 0, 10, 10]], 0.5, 100, 100), [[0, 0, 15, 15]])  # clipped
    # non-square: padding is proportional to each side separately (width 40 -> 4, height 10 -> 1)
    np.testing.assert_allclose(bx.pad_boxes([[20, 20, 60, 30]], 0.1, 100, 100), [[16, 19, 64, 31]])
    out = bx.pad_boxes([[50, 50, 50, 50]], 0.1, 100, 100, min_size=2)                              # zero box
    assert (out[0, 2] - out[0, 0]) >= 2 and (out[0, 3] - out[0, 1]) >= 2
    edge = bx.pad_boxes([[99.9, 99.9, 100, 100]], 0.1, 100, 100, min_size=2)                       # at corner
    assert edge.min() >= 0 and edge.max() <= 100 and (edge[0, 2] - edge[0, 0]) >= 2 - 1e-9


def _nms_reference(boxes, scores, thr):
    """Independent O(n^2) loop implementation."""
    n = len(boxes)
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    suppressed = [False] * n
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j != i and not suppressed[j]:
                if bx.iou_matrix(boxes[i:i + 1], boxes[j:j + 1])[0, 0] > thr:
                    suppressed[j] = True
    return keep


@pytest.mark.parametrize("seed", range(8))
def test_nms_matches_reference(seed):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 80, size=(60, 2)); wh = rng.uniform(5, 40, size=(60, 2))
    boxes = np.hstack([xy, xy + wh]); scores = rng.random(60)
    for thr in (0.3, 0.5, 0.7):
        assert bx.nms(boxes, scores, thr).tolist() == _nms_reference(boxes, scores, thr)


def test_nms_hand_and_max_keep():
    b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60.0]])
    s = np.array([0.9, 0.8, 0.7])
    assert bx.nms(b, s, 0.5).tolist() == [0, 2]
    assert bx.nms(b, s, 0.99).tolist() == [0, 1, 2]
    assert bx.nms(b, s, 0.5, max_keep=1).tolist() == [0]
    assert bx.nms(np.zeros((0, 4)), np.zeros(0), 0.5).size == 0


def test_batched_nms_only_suppresses_within_class():
    b = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10.0]])
    s = np.array([0.9, 0.8, 0.7]); lab = np.array([1, 1, 2])
    assert bx.batched_nms(b, s, lab, 0.5).tolist() == [0, 2]      # same-class dup removed, other class kept
    assert bx.nms(b, s, 0.5).tolist() == [0]                      # class-agnostic would drop it


# ============================================================ coco_utils
def test_coco_constants():
    assert len(cu.COCO_IDS) == len(cu.COCO_NAMES) == 80
    assert len(set(cu.COCO_IDS)) == 80 and max(cu.COCO_IDS) == 90 and cu.COCO_IDS == sorted(cu.COCO_IDS)
    known = {1: "person", 3: "car", 13: "stop sign", 44: "bottle", 62: "chair", 72: "tv",
             77: "cell phone", 89: "hair drier", 90: "toothbrush", 55: "orange", 84: "book"}
    for cid, name in known.items():
        assert cu.COCO_NAMES[cu.ID2IDX[cid]] == name
    assert cu.prompt_names([cu.ID2IDX[72], cu.ID2IDX[89]]) == ["television", "hair dryer"]


def test_verify_categories():
    cats = [{"id": i, "name": n} for i, n in zip(cu.COCO_IDS, cu.COCO_NAMES)]
    cu.verify_categories(cats[::-1])                  # order independent
    bad = [dict(c) for c in cats]; bad[5]["name"] = "lorry"
    with pytest.raises(ValueError):
        cu.verify_categories(bad)
    with pytest.raises(ValueError):
        cu.verify_categories(cats[:-1])


def test_make_tasks():
    assert cu.make_tasks("40+40") == [list(range(40)), list(range(40, 80))]
    t = cu.make_tasks("40+4x10")
    assert [len(x) for x in t] == [40, 10, 10, 10, 10] and sum(t, []) == list(range(80))
    assert [len(x) for x in cu.make_tasks("70+10")] == [70, 10]
    for bad in ("40+30", "abc", "40+3x10"):
        with pytest.raises(ValueError):
            cu.make_tasks(bad)


def _toy_coco():
    P, C, T = 1, 3, 90                                   # person(0) car(2) toothbrush(79)
    imgs = [{"id": i, "file_name": f"{i:012d}.jpg", "width": 200, "height": 100} for i in (1, 2, 3, 4)]
    ann = [
        {"image_id": 1, "category_id": P, "bbox": [20, 10, 40, 30], "iscrowd": 0},
        {"image_id": 1, "category_id": T, "bbox": [100, 50, 20, 20], "iscrowd": 0},
        {"image_id": 2, "category_id": C, "bbox": [0, 0, 200, 100], "iscrowd": 0},
        {"image_id": 2, "category_id": P, "bbox": [10, 10, 50, 50], "iscrowd": 1},   # crowd: ignored
        {"image_id": 3, "category_id": T, "bbox": [190, 90, 30, 30], "iscrowd": 0},  # overflows image
        {"image_id": 4, "category_id": P, "bbox": [5, 5, 0.5, 30], "iscrowd": 0},    # degenerate
    ]
    return {"images": imgs, "annotations": ann}


def test_select_task_images_and_gt():
    coco = _toy_coco()
    assert cu.select_task_images(coco, [0]) == [1, 4]            # crowd-only image 2 excluded for person
    assert cu.select_task_images(coco, [79]) == [1, 3]
    assert cu.select_task_images(coco, [0, 2, 79]) == [1, 2, 3, 4]
    sub = cu.select_task_images(coco, [0, 2, 79], max_images=2, seed=1)
    assert len(sub) == 2 and sub == cu.select_task_images(coco, [0, 2, 79], max_images=2, seed=1)
    gt = cu.gt_by_image(coco, [1, 2, 4])
    np.testing.assert_allclose(gt[1][0], [[20, 10, 60, 40], [100, 50, 120, 70]])
    assert gt[1][1].tolist() == [0, 79]
    assert gt[2][1].tolist() == [2]                              # crowd skipped
    assert len(gt[4][1]) == 0                                    # degenerate skipped
    assert cu.gt_by_image(coco, [1], classes=[79])[1][1].tolist() == [79]


def test_write_yolo_labels(tmp_path):
    coco = _toy_coco()
    d = str(tmp_path / "labels")
    stats = cu.write_yolo_labels(coco, [1, 2, 3, 4], classes=[0, 2], label_dir=d)
    lines = lambda i: open(os.path.join(d, f"{i:012d}.txt")).read().split("\n")[:-1]
    # image 1: person -> class 0; toothbrush (not a base class) must NOT appear
    assert lines(1) == ["0 0.200000 0.250000 0.200000 0.300000"]
    assert lines(2) == ["1 0.500000 0.500000 1.000000 1.000000"]   # car -> remapped to 1; crowd skipped
    assert open(os.path.join(d, f"{3:012d}.txt")).read() == ""       # only a non-base object -> empty file
    assert open(os.path.join(d, f"{4:012d}.txt")).read() == ""       # degenerate skipped
    assert stats["objects"] == 2 and stats["skipped"] >= 1
    # clipped overflow box in image 3 when its class IS included
    cu.write_yolo_labels(coco, [3], classes=[79], label_dir=d)
    x = [float(v) for v in lines(3)[0].split()]
    assert abs(x[1] - 0.975) < 1e-6 and abs(x[2] - 0.95) < 1e-6 and abs(x[3] - 0.05) < 1e-6


def test_dataset_yaml(tmp_path):
    p = str(tmp_path / "d.yaml")
    cu.write_dataset_yaml(p, "/data/x", ["a", "b"])
    txt = open(p).read()
    assert "path: /data/x" in txt and "0: a" in txt and "1: b" in txt and "train: images/train" in txt


# ============================================================ prompts
def test_class_embeddings_unit_norm_and_ensemble_definition():
    import hashlib
    # hash() is salted per-process; use a stable one so the manual recomputation below matches
    def enc(texts):
        return np.array([np.random.default_rng(int(hashlib.md5(t.encode()).hexdigest()[:8], 16))
                         .normal(size=16) * 7 for t in texts])
    names = ["cat", "dog"]
    E = pr.class_embeddings(names, enc, pr.DET_TEMPLATES)
    assert E.shape == (2, 16)
    np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-12)
    manual = np.mean([pr.l2norm(enc([t.format("cat")]))[0] for t in pr.DET_TEMPLATES], axis=0)
    np.testing.assert_allclose(E[0], pr.l2norm(manual), atol=1e-12)
    B = pr.background_embedding(enc)
    assert B.shape == (1, 16) and abs(np.linalg.norm(B) - 1) < 1e-12
    assert pr.class_embeddings([], enc).shape[0] == 0


def test_zero_shot_logits_scale_invariant_and_bounded():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(5, 8)); T = pr.l2norm(rng.normal(size=(3, 8)))
    a, b = pr.zero_shot_logits(f, T), pr.zero_shot_logits(f * 13.0, T)
    np.testing.assert_allclose(a, b, atol=1e-9)
    assert np.abs(a).max() <= 100 + 1e-9


# ============================================================ crops
def _img_with_rect(color, box, size=(120, 100)):
    arr = np.zeros((size[1], size[0], 3), np.uint8)
    x1, y1, x2, y2 = box
    arr[y1:y2, x1:x2] = color
    return Image.fromarray(arr)


def test_crop_resize_exact_region_and_shape():
    img = _img_with_rect((255, 0, 0), (20, 30, 60, 70))
    c = crop_resize(img, [[20, 30, 60, 70]], 32)
    assert c.shape == (1, 32, 32, 3) and c.dtype == np.uint8
    assert (c[0, :, :, 0] > 250).all() and (c[0, :, :, 1:] < 5).all()          # pure red everywhere
    padded = crops_for_image(img, [[20, 30, 60, 70]], 0.25, 32)
    assert padded[0, 0, 0].sum() < 5 and padded[0, 16, 16, 0] > 250             # context border is black


def test_crop_resize_edge_cases_never_crash_or_empty():
    img = _img_with_rect((0, 255, 0), (0, 0, 120, 100))
    weird = [[-10, -10, 5, 5], [118, 98, 150, 140], [50.2, 50.2, 50.3, 50.3], [119.9, 99.9, 120, 100],
             [30, 30, 30, 30]]
    c = crop_resize(img, weird, 16)
    assert c.shape == (5, 16, 16, 3)
    assert (c[:, :, :, 1] > 200).all()                                          # all inside the green image


def test_to_clip_input():
    x = np.full((1, 4, 4, 3), 255, np.uint8)
    y = to_clip_input(x)
    assert y.shape == (1, 3, 4, 4) and y.dtype == np.float32
    np.testing.assert_allclose(y[0, :, 0, 0], (1 - CLIP_MEAN) / CLIP_STD, rtol=1e-6)
    red = np.zeros((1, 2, 2, 3), np.uint8); red[..., 0] = 255                       # channel order matters
    r = to_clip_input(red)[0, :, 0, 0]
    np.testing.assert_allclose(r, [(1 - CLIP_MEAN[0]) / CLIP_STD[0], -CLIP_MEAN[1] / CLIP_STD[1],
                                   -CLIP_MEAN[2] / CLIP_STD[2]], rtol=1e-6)
    z = to_clip_input(np.zeros((1, 4, 4, 3), np.uint8))
    np.testing.assert_allclose(z[0, :, 0, 0], -CLIP_MEAN / CLIP_STD, rtol=1e-6)


# ============================================================ cache
def test_cache_roundtrip_and_float16_error(tmp_path):
    rng = np.random.default_rng(0)
    w = CacheWriter(str(tmp_path / "c"), 12, extra_dtypes={"cls": "int16"})
    truth = {}
    for iid, n in ((7, 3), (9, 0), (4, 5)):
        b = rng.uniform(0, 50, size=(n, 2)); b = np.hstack([b, b + rng.uniform(2, 30, size=(n, 2))])
        f = rng.normal(scale=10, size=(n, 12)); s = rng.random(n)
        truth[iid] = (b, s, f)
        w.add_image(iid, b, s, f, cls=np.arange(n))
    w.save()
    c = FeatureCache(str(tmp_path / "c"))
    assert c.n_images == 3 and c.n_boxes == 8 and c.offsets.tolist() == [0, 3, 3, 8]
    for iid, (b, s, f) in truth.items():
        im = c.image(iid)
        np.testing.assert_allclose(im["boxes"], b, atol=1e-4)
        np.testing.assert_allclose(im["scores"], s, atol=1e-6)
        if len(b):
            assert np.abs(im["feats"].astype(np.float64) - f).max() <= 1e-3 * np.abs(f).max() + 1e-3   # f16
        else:
            assert im["feats"].shape == (0, 12) and im["boxes"].shape == (0, 4)
        assert im["cls"].tolist() == list(range(len(b)))
    F, I, B, S, R = c.gather([4, 7])
    assert F.shape == (8, 12) and I.tolist() == [0] * 5 + [1] * 3 and R.tolist() == [3, 4, 5, 6, 7, 0, 1, 2]
    assert F.dtype == np.float64
    F0, *_ = c.gather([9])
    assert F0.shape == (0, 12)


def test_cache_validation(tmp_path):
    w = CacheWriter(str(tmp_path / "v"), 4)
    ok = ([[0, 0, 5, 5]], [0.5], np.ones((1, 4)))
    with pytest.raises(ValueError): w.add_image(1, [[5, 5, 5, 9]], [0.5], np.ones((1, 4)))      # degenerate
    with pytest.raises(ValueError): w.add_image(1, [[0, 0, 5, 5]], [0.5], np.full((1, 4), np.nan))
    with pytest.raises(ValueError): w.add_image(1, [[0, 0, 5, 5]], [0.5, 0.6], np.ones((1, 4)))
    with pytest.raises(ValueError): w.add_image(1, [[0, 0, 5, 5]], [0.5], np.ones((1, 5)))
    assert w.ids == []                                                  # failed adds leave no trace
    w.add_image(1, *ok); w.add_image(1, *ok)
    with pytest.raises(ValueError): w.save()                            # duplicate ids
    empty = CacheWriter(str(tmp_path / "e"), 4); empty.save()
    assert FeatureCache(str(tmp_path / "e")).n_boxes == 0


# ============================================================ extraction plumbing + gates
PALETTE = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (0, 255, 255), 5: (255, 0, 255)}


def make_world(n_images=12, seed=0):
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_images):
        arr = np.zeros((100, 140, 3), np.uint8)
        gtb, gtc = [], []
        for _ in range(rng.integers(1, 4)):
            c = int(rng.integers(0, 6)); w, h = rng.integers(15, 40, size=2)
            x, y = rng.integers(0, 140 - w), rng.integers(0, 100 - h)
            if any(not (x + w < b[0] or x > b[2] or y + h < b[1] or y > b[3]) for b in gtb):
                continue                                    # keep objects disjoint
            arr[y:y + h, x:x + w] = PALETTE[c]; gtb.append([x, y, x + w, y + h]); gtc.append(c)
        if not gtb:
            arr[10:30, 10:30] = PALETTE[0]; gtb, gtc = [[10, 10, 30, 30]], [0]
        items.append((i, Image.fromarray(arr), np.array(gtb, float), np.array(gtc)))
    return items


def fake_encode_u8(crops):
    """Mean RGB of the crop in dims 0..2 (background is black so hue is preserved)."""
    m = crops.reshape(len(crops), -1, 3).mean(axis=1) / 255.0
    return np.hstack([m * 10, np.zeros((len(m), 5))])


FAKE_TEXT = np.hstack([pr.l2norm(np.array([PALETTE[c] for c in range(6)], float)), np.zeros((6, 5))])


def test_gt_crop_zeroshot_gate_measures_correctly_and_can_fail():
    world = make_world()
    stream = lambda: ((im, b, c) for _, im, b, c in world)
    good = gt_crop_zeroshot_eval(stream(), fake_encode_u8, FAKE_TEXT)
    assert good["n"] >= 12 and good["acc"] == 1.0
    wrong = gt_crop_zeroshot_eval(stream(), fake_encode_u8, FAKE_TEXT[::-1].copy())
    assert wrong["acc"] < 0.5                                       # the metric is able to fail
    sub = gt_crop_zeroshot_eval(stream(), fake_encode_u8, FAKE_TEXT, class_subset=[0, 1])
    assert sub["acc"] == 1.0 and sub["n"] < good["n"]
    assert np.isnan(sub["per_class"][5]) and sub["per_class"][0] == 1.0


def test_extract_cache_alignment(tmp_path):
    world = make_world()
    gt = {i: (b, c) for i, _, b, c in world}
    calls = []

    def proposal_fn(img):                                           # identify image by pixel hash
        for i, im, b, c in world:
            if im is img:
                calls.append(i)
                return b, np.linspace(0.9, 0.5, len(b))
        raise AssertionError

    out = extract_cache(((i, im) for i, im, _, _ in world), proposal_fn, fake_encode_u8,
                        str(tmp_path / "c"), feat_dim=8, pad_frac=0.1, log_every=0)
    c = FeatureCache(out)
    assert c.n_images == len(world) and calls == [i for i, *_ in world]
    for i, im, b, cl in world:
        rec = c.image(i)
        np.testing.assert_allclose(rec["boxes"], b, atol=1e-4)                 # ORIGINAL (unpadded) boxes stored
        want = fake_encode_u8(crops_for_image(im, b, 0.1))                     # independent recomputation
        np.testing.assert_allclose(rec["feats"].astype(float), want, atol=1e-2)
        pred = pr.zero_shot_logits(rec["feats"], FAKE_TEXT).argmax(1)
        assert pred.tolist() == cl.tolist()                                    # feature j belongs to box j


def test_extract_cache_max_boxes_and_empty(tmp_path):
    world = make_world(4)
    def prop(img):
        n = 7
        b = np.array([[i, i, i + 20, i + 20] for i in range(n)], float)
        return b, np.arange(n, dtype=float)
    out = extract_cache(((i, im) for i, im, _, _ in world), prop, fake_encode_u8,
                        str(tmp_path / "m"), 8, max_boxes=3, log_every=0)
    c = FeatureCache(out)
    assert all(len(c.image(i)["boxes"]) == 3 for i, *_ in world)
    assert c.image(0)["scores"].tolist() == [6.0, 5.0, 4.0]                    # kept the top-scoring
    out2 = extract_cache(((i, im) for i, im, _, _ in world),
                         lambda img: (np.zeros((0, 4)), np.zeros(0)), fake_encode_u8,
                         str(tmp_path / "z"), 8, log_every=0)
    assert FeatureCache(out2).n_boxes == 0


def test_box_recall_hand_case():
    gtb = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 110, 110.0]])
    gtc = np.array([1, 1, 2])
    props = np.array([[0, 0, 10, 10], [52, 50, 62, 60.0]])           # IoU 1.0 and 8/12=0.667
    r = box_recall(gtb, gtc, props, classes=[1])
    assert r == {"n_gt": 2, "hit@0.5": 2, "hit@0.7": 1}
    assert box_recall(gtb, gtc, props, classes=[2])["hit@0.5"] == 0
    edge = box_recall(np.array([[0, 0, 10, 10.0]]), np.array([1]), np.array([[0, 0, 10, 5.0]]), [1])
    assert edge["hit@0.5"] == 1                                       # IoU exactly 0.5 counts (>=)
    assert box_recall(gtb, gtc, np.zeros((0, 4)), classes=[1])["hit@0.5"] == 0

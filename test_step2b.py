"""Tests for coco_fetch.py and cifar_repro.py (no network, no torch).
Run: python -m pytest -q test_step2b.py"""
import json
import os
import zipfile

import numpy as np
import pytest

import cifar_repro as cr
import coco_fetch as cf
import n2l_core as nc
from prompts import l2norm


# ============================================================ coco_fetch
def _uri(p):
    return "file://" + str(p)


def test_download_file_atomic_skip_and_failures(tmp_path):
    src = tmp_path / "a.bin"; src.write_bytes(b"hello")
    dst = str(tmp_path / "out" / "a.bin")
    assert cf.download_file(_uri(src), dst) == dst and open(dst, "rb").read() == b"hello"
    src.write_bytes(b"CHANGED")
    cf.download_file(_uri(src), dst)                                   # exists -> skipped, not refetched
    assert open(dst, "rb").read() == b"hello"
    with pytest.raises(IOError):
        cf.download_file(_uri(tmp_path / "nope.bin"), str(tmp_path / "x.bin"), retries=2)
    assert not os.path.exists(str(tmp_path / "x.bin")) and not os.path.exists(str(tmp_path / "x.bin.part"))
    empty = tmp_path / "empty.bin"; empty.write_bytes(b"")
    with pytest.raises(IOError):
        cf.download_file(_uri(empty), str(tmp_path / "e.bin"), retries=1)
    assert not os.path.exists(str(tmp_path / "e.bin"))                 # no zero-byte file left behind


def test_ensure_annotations_extracts_from_zip(tmp_path):
    z = tmp_path / "ann.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("annotations/instances_train2017.json", json.dumps({"images": [1]}))
        zf.writestr("annotations/instances_val2017.json", json.dumps({"images": [2]}))
    root = str(tmp_path / "coco")
    p = cf.ensure_annotations(root, url=_uri(z))
    assert cf.load_coco(p["train"]) == {"images": [1]} and cf.load_coco(p["val"]) == {"images": [2]}
    os.remove(os.path.join(root, "annotations_trainval2017.zip"))
    assert cf.ensure_annotations(root, url="file:///does/not/exist") == p   # already extracted: no network


def test_fetch_images(tmp_path):
    srcs = []
    for i in (5, 6, 7):
        f = tmp_path / f"src{i}.jpg"; f.write_bytes(bytes([i]) * 10); srcs.append(f)
    coco = {"images": [{"id": i, "file_name": f"{i:012d}.jpg", "coco_url": _uri(s)}
                       for i, s in zip((5, 6, 7), srcs)]}
    paths = cf.fetch_images(coco, [7, 5], str(tmp_path / "imgs"), workers=4)
    assert [os.path.basename(p) for p in paths] == ["000000000007.jpg", "000000000005.jpg"]
    assert open(paths[0], "rb").read() == bytes([7]) * 10
    with pytest.raises(KeyError):
        cf.fetch_images(coco, [99], str(tmp_path / "imgs"))
    coco["images"][1]["coco_url"] = _uri(tmp_path / "gone.jpg")
    with pytest.raises(IOError):
        cf.fetch_images(coco, [5, 6], str(tmp_path / "imgs2"))


# ============================================================ cifar_repro
def test_class_order_is_the_official_global_seed_permutation():
    order = cr.official_class_order()
    assert sorted(order) == list(range(100)) and order == cr.official_class_order()
    np.random.seed(1993)
    assert order == np.random.permutation(100).tolist()


def test_remap_targets():
    order = [3, 0, 2, 1]                       # original class 3 becomes new class 0, etc.
    np.testing.assert_array_equal(cr.remap_targets(np.array([0, 1, 2, 3, 3]), order), [1, 3, 2, 0, 0])


_N2L_REPO = os.environ.get("N2L_REPO") or ("/tmp/claude-0/-home-claude/82a21fb4-e1ff-5833-ae2b-ab7f9d488a2e/"
                                           "scratchpad/ref/ICLR2026-N2L")


@pytest.mark.skipif(not os.path.isdir(_N2L_REPO), reason="official repo not cloned (set N2L_REPO=/path/to/repo)")
def test_official_text_assets_load():
    repo = _N2L_REPO
    names, templates = cr.load_official_text_assets(repo)
    assert len(names) == 100 and len(set(names)) == 100
    assert len(templates) == 18 and templates[0] == "a photo of a {}."


def _synthetic_clip_world(C=20, D=64, n_train=100, n_test=40, text_noise=0.6, sep=1.0, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=sep, size=(C, D))
    mk = lambda n: (np.vstack([means[c] + rng.normal(size=(n, D)) for c in range(C)]) * 5.0,
                    np.repeat(np.arange(C), n))
    (Xtr, ytr), (Xte, yte) = mk(n_train), mk(n_test)
    text = l2norm(means + rng.normal(scale=text_noise, size=means.shape))     # imperfect 'CLIP text'
    return Xtr, ytr, Xte, yte, text, means


def _fast_cfg(seed=0):
    return nc.N2LConfig(lam=0.1, pca_rule="energy", pca_rho=0.5, seed=seed)


def test_protocol_end_to_end_on_synthetic_features():
    Xtr, ytr, Xte, yte, text, _ = _synthetic_clip_world()
    logs = []
    r = cr.run_cifar_protocol(Xtr, ytr, Xte, yte, text, text, init_cls=5, increment=5, expand=256,
                              cfg=_fast_cfg(), log=logs.append)
    assert len(r["fused"]) == 4 and len(logs) == 4 and all(1 <= k <= 64 for k in r["k"])
    # independent recomputation of the final-task zero-shot accuracy
    from scipy.special import softmax
    zs = softmax(100 * l2norm(Xte) @ text.T, axis=1)
    assert abs(r["zeroshot"][-1] - np.mean(zs.argmax(1) == yte)) < 1e-12
    # first task evaluates on the 5 classes only
    m = yte < 5
    zs5 = softmax(100 * l2norm(Xte[m]) @ text[:5].T, axis=1)
    assert abs(r["zeroshot"][0] - np.mean(zs5.argmax(1) == yte[m])) < 1e-12
    assert r["A_bar_fused"] >= r["A_bar_zeroshot"] - 0.02          # N2L must not be worse than its teacher
    assert r["A_B_fused"] == r["fused"][-1]


def test_protocol_never_uses_ground_truth_labels_for_training():
    """Shuffle y inside each task block: the rows selected per task are unchanged, so if
    training touches GT labels anywhere the results would change."""
    Xtr, ytr, Xte, yte, text, _ = _synthetic_clip_world(seed=1)
    ytr2 = ytr.copy()
    rng = np.random.default_rng(0)
    for lo in range(0, 20, 5):
        idx = np.flatnonzero((ytr >= lo) & (ytr < lo + 5))
        ytr2[idx] = rng.permutation(ytr[idx])
    a = cr.run_cifar_protocol(Xtr, ytr, Xte, yte, text, text, 5, 5, 128, _fast_cfg(), log=lambda *_: None)
    b = cr.run_cifar_protocol(Xtr, ytr2, Xte, yte, text, text, 5, 5, 128, _fast_cfg(), log=lambda *_: None)
    assert a["fused"] == b["fused"] and a["ridge"] == b["ridge"]
    assert a["label_acc"] != b["label_acc"]                       # only the diagnostic changed


def test_protocol_bad_teacher_gives_bad_student():
    """If the text embeddings are wrong (teacher systematically mislabels), the student cannot
    repair it -- this guards against accidentally leaking truth into training."""
    Xtr, ytr, Xte, yte, text, _ = _synthetic_clip_world(seed=2, text_noise=0.0)
    good = cr.run_cifar_protocol(Xtr, ytr, Xte, yte, text, text, 5, 5, 128, _fast_cfg(), log=lambda *_: None)
    perm = text.copy()
    for lo in range(0, 20, 5):                                    # rotate names inside each task block
        perm[lo:lo + 5] = np.roll(text[lo:lo + 5], 1, axis=0)
    bad = cr.run_cifar_protocol(Xtr, ytr, Xte, yte, perm, text, 5, 5, 128, _fast_cfg(), log=lambda *_: None)
    assert np.mean(good["label_acc"]) > 0.9 and np.mean(bad["label_acc"]) < 0.1
    assert bad["A_bar_ridge"] < 0.3 < good["A_bar_ridge"]


def test_protocol_uneven_last_task():
    Xtr, ytr, Xte, yte, text, _ = _synthetic_clip_world(C=17, seed=3)
    r = cr.run_cifar_protocol(Xtr, ytr, Xte, yte, text, text, 5, 5, 64, _fast_cfg(), log=lambda *_: None)
    assert len(r["fused"]) == 4                                   # 5,5,5,2 classes


def test_fusion_weight_endpoints():
    """w=1 -> pure ridge head, w=0 -> pure zero-shot (pins the meaning of the fusion weight)."""
    Xtr, ytr, Xte, yte, text, _ = _synthetic_clip_world(seed=4, text_noise=2.5, sep=0.5)
    kw = dict(init_cls=5, increment=5, expand=128, log=lambda *_: None)
    one = cr.run_cifar_protocol(Xtr, ytr, Xte, yte, text, text, cfg=_fast_cfg(), fusion_weight=1.0, **kw)
    zero = cr.run_cifar_protocol(Xtr, ytr, Xte, yte, text, text, cfg=_fast_cfg(), fusion_weight=0.0, **kw)
    assert one["ridge"] != one["zeroshot"]          # precondition: the two heads really differ here
    assert one["fused"] == one["ridge"]
    assert zero["fused"] == zero["zeroshot"]

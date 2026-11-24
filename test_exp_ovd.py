"""N2L-Det-exp: open-vocabulary detector baselines (ovd.py + `run_pipeline.py ovd`) and the VOC image-link restore."""
import json
import os

import numpy as np
import pytest
from PIL import Image

import coco_utils as cu
import ovd
import run_pipeline as rp
from evaluate import make_coco_gt


# ------------------------------------------------------------------ decoding
def test_cxcywh_to_xyxy():
    np.testing.assert_allclose(ovd.cxcywh_to_xyxy([[0.5, 0.5, 0.2, 0.4]]), [[0.4, 0.3, 0.6, 0.7]])


def test_decode_owl_padded_square_scaling_classes_and_order():
    # image 80 wide x 48 high -> OWLv2 pads to an 80 x 80 square: boxes scale by 80 on BOTH axes
    logits = np.array([[5.0, -5.0, -5.0], [-5.0, -5.0, 3.0], [-9.0, -9.0, -9.0]])
    boxes = np.array([[0.5, 0.25, 0.5, 0.25], [0.25, 0.25, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]])
    b, s, c = ovd.decode_owl(logits, boxes, 80, 48, max_det=10)
    np.testing.assert_allclose(b[0], [20, 10, 60, 30])
    np.testing.assert_allclose(b[1], [16, 16, 24, 24])
    assert list(c) == [0, 2, 0] and np.all(np.diff(s) <= 0)
    b1, _, _ = ovd.decode_owl(logits, boxes, 80, 48, max_det=1)
    assert len(b1) == 1
    bn, _, _ = ovd.decode_owl(logits[:1], boxes[:1], 80, 48, padded_square=False)   # OWL-ViT v1: (w, h)
    np.testing.assert_allclose(bn[0], [20, 6, 60, 18])


def test_decode_owl_clips_and_drops_empty_boxes():
    logits = np.array([[1.0], [2.0]])
    boxes = np.array([[0.95, 0.5, 0.3, 0.2], [1.5, 0.5, 0.1, 0.1]])               # second lies outside the image
    b, s, c = ovd.decode_owl(logits, boxes, 100, 100)
    assert len(b) == 1 and b[0, 2] == 100


def test_gdino_prompt_spans_and_token_sets():
    text, spans = ovd.build_gdino_prompt(["Sheep", "potted  plant", "tv"])
    assert text == "sheep . potted plant . tv ."
    assert [text[s:e] for s, e in spans] == ["sheep", "potted plant", "tv"]
    # offsets as a BERT fast tokenizer gives them: [CLS] sheep . potted plant . tv . [SEP]
    offs = [(0, 0), (0, 5), (6, 7), (8, 14), (15, 20), (21, 22), (23, 25), (26, 27), (0, 0)]
    assert ovd.token_sets(offs, spans) == [[1], [3, 4], [6]]
    with pytest.raises(ValueError):
        ovd.token_sets(offs[:2], spans)


def test_decode_gdino_phrase_max_and_scaling():
    T = 12
    logits = np.full((3, T), -np.inf)
    logits[:, :9] = -6.0
    logits[0, 4] = 4.0                     # second token of phrase 1 ("potted plant")
    logits[1, 1] = 2.0                     # phrase 0
    logits[2, 6] = 1.0                     # phrase 2
    boxes = np.array([[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]])
    b, s, c = ovd.decode_gdino(logits, boxes, [[1], [3, 4], [6]], 200, 100)
    assert list(c) == [1, 0, 2]
    np.testing.assert_allclose(b[0], [50, 25, 150, 75])                            # (w, h) scaling, no padding
    np.testing.assert_allclose(s[0], 1 / (1 + np.exp(-4.0)))
    with pytest.raises(ValueError):
        ovd.decode_gdino(logits, boxes, [[1], [30]], 200, 100)


def test_chunk_names_fits_the_token_budget():
    names = ["a", "bb", "ccc", "dddd", "e"]
    count = lambda t: 2 + len(t.split())                                         # CLS + SEP + one token per word / dot
    chunks = ovd.chunk_names(names, count, max_tokens=7)
    assert sum(chunks, []) == list(range(5))
    assert all(count(ovd.build_gdino_prompt([names[j] for j in ch])[0]) <= 7 for ch in chunks)
    assert len(ovd.chunk_names(names, count, 256)) == 1
    with pytest.raises(ValueError):
        ovd.chunk_names(["x y z w"], count, max_tokens=4)


# ------------------------------------------------------------------ real transformers code paths on tiny random models
transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")


def _clip_tokenizer(tmp_path, words):
    vocab = {"<|startoftext|>": 0, "<|endoftext|>": 1}
    for w in words:
        vocab[w + "</w>"] = len(vocab)
    for ch in "abcdefghijklmnopqrstuvwxyz":
        vocab.setdefault(ch, len(vocab)); vocab.setdefault(ch + "</w>", len(vocab))
    (tmp_path / "vocab.json").write_text(json.dumps(vocab)); (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    return transformers.CLIPTokenizer(str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt")), len(vocab)


@pytest.fixture(scope="module")
def tiny_owl(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("owl")
    tok, nv = _clip_tokenizer(tmp, ["a", "photo", "of", "sheep", "dog", "train"])
    cfg = transformers.Owlv2Config(
        text_config=dict(vocab_size=nv + 5, hidden_size=32, intermediate_size=37, num_hidden_layers=2, num_attention_heads=4,
                         max_position_embeddings=16),
        vision_config=dict(hidden_size=32, intermediate_size=37, num_hidden_layers=2, num_attention_heads=4, image_size=64,
                           patch_size=16), projection_dim=32)
    torch.manual_seed(0)
    model = transformers.Owlv2ForObjectDetection(cfg).eval()
    ip = transformers.Owlv2ImageProcessor(size={"height": 64, "width": 64})
    return model, transformers.Owlv2Processor(image_processor=ip, tokenizer=tok)


def test_owl_detector_runs_on_the_real_processor_and_model(tiny_owl):
    model, proc = tiny_owl
    det = ovd.OwlDetector(["sheep", "dog", "train"], model=model, processor=proc, device="cpu", max_det=1000)
    img = Image.fromarray((np.random.default_rng(0).random((48, 80, 3)) * 255).astype("uint8"))
    b, s, c = det(img)
    assert len(b) <= 16 and (len(b) == 0 or (c.max() < 3 and b[:, 2].max() <= 80 and b[:, 3].max() <= 48))
    assert det.padded and det.max_len == 16


def test_owl_processor_pads_bottom_right_as_decode_assumes(tiny_owl):
    """decode_owl scales by max(w, h) because OWLv2 pads the image at the bottom/right: check the processor in use."""
    _, proc = tiny_owl
    for (h, w) in ((32, 64), (64, 32)):
        img = Image.fromarray(np.full((h, w, 3), 255, dtype="uint8"))          # white image, grey padding
        px = proc.image_processor(images=img, return_tensors="pt")["pixel_values"][0, 0].numpy()   # 64 x 64
        img_rows, img_cols = round(64 * h / max(h, w)), round(64 * w / max(h, w))
        assert np.allclose(px[: img_rows - 2, : img_cols - 2], px[0, 0])          # image region: white
        assert not np.isclose(px[-1, -1], px[0, 0])                                  # bottom-right corner: padding
        if h < w:
            assert np.allclose(px[img_rows + 2:, :], px[-1, -1])                     # padding rows at the BOTTOM
        else:
            assert np.allclose(px[:, img_cols + 2:], px[-1, -1])                     # padding columns at the RIGHT


@pytest.mark.parametrize("h,w", [(48, 80), (64, 64), (90, 50)])
def test_decode_owl_matches_the_official_post_processor(tiny_owl, h, w):
    _, proc = tiny_owl
    from transformers.models.owlv2.modeling_owlv2 import Owlv2ObjectDetectionOutput
    rng = np.random.default_rng(h + w)
    logits = rng.normal(0, 2, (1, 30, 3)).astype("float32")
    cxcy = rng.uniform(0.1, 0.9, (1, 30, 2)); wh = rng.uniform(0.05, 0.3, (1, 30, 2))
    boxes = np.concatenate([cxcy, wh], axis=2).astype("float32")
    out = Owlv2ObjectDetectionOutput(logits=torch.tensor(logits), pred_boxes=torch.tensor(boxes))
    S = max(h, w)                                                            # the padded square, as the OWLv2 docs prescribe
    ref = proc.image_processor.post_process_object_detection(out, threshold=-1.0, target_sizes=[(S, S)])[0]
    rb, rs, rl = ref["boxes"].numpy().astype(float), ref["scores"].numpy(), ref["labels"].numpy()
    rb[:, [0, 2]] = rb[:, [0, 2]].clip(0, w); rb[:, [1, 3]] = rb[:, [1, 3]].clip(0, h)
    ok = (rb[:, 2] > rb[:, 0] + 1e-6) & (rb[:, 3] > rb[:, 1] + 1e-6)
    rb, rs, rl = rb[ok], rs[ok], rl[ok]
    order = np.argsort(-rs, kind="stable")
    b, s, c = ovd.decode_owl(logits[0], boxes[0], w, h, max_det=1000)
    assert len(b) == len(rb) > 10
    np.testing.assert_allclose(s, rs[order], rtol=1e-5)
    np.testing.assert_allclose(b, rb[order], rtol=1e-4, atol=1e-3)
    assert list(c) == list(rl[order])


@pytest.fixture(scope="module")
def tiny_gdino(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("gd")
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "sheep", "potted", "plant", "dining", "table", "tv"]
    (tmp / "vocab.txt").write_text("\n".join(words) + "\n")
    tok = transformers.BertTokenizerFast(str(tmp / "vocab.txt"))
    ip = transformers.GroundingDinoImageProcessor(size={"shortest_edge": 64, "longest_edge": 96})
    proc = transformers.GroundingDinoProcessor(image_processor=ip, tokenizer=tok)
    bb = transformers.SwinConfig(image_size=64, patch_size=4, embed_dim=8, depths=[1, 1, 1, 1], num_heads=[1, 1, 1, 1],
                                 window_size=2, out_features=["stage2", "stage3", "stage4"])
    tc = transformers.BertConfig(vocab_size=len(words), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                 intermediate_size=16)
    cfg = transformers.GroundingDinoConfig(backbone_config=bb, text_config=tc, d_model=32, encoder_layers=1, decoder_layers=2,
                                           encoder_ffn_dim=32, decoder_ffn_dim=32, num_queries=20, encoder_attention_heads=2,
                                           decoder_attention_heads=2, num_feature_levels=4, encoder_n_points=2,
                                           decoder_n_points=2)
    torch.manual_seed(0)
    return transformers.GroundingDinoForObjectDetection(cfg).eval(), proc


def test_gdino_detector_runs_and_maps_tokens_to_classes(tiny_gdino):
    model, proc = tiny_gdino
    names = ["sheep", "potted plant", "dining table", "tv"]
    det = ovd.GDinoDetector(names, model=model, processor=proc, device="cpu", max_det=50)
    assert len(det.prompts) == 1
    text, ids, tsets, ch = det.prompts[0]
    assert text == "sheep . potted plant . dining table . tv ."
    assert [len(t) for t in tsets] == [1, 2, 2, 1]
    img = Image.fromarray((np.random.default_rng(1).random((48, 80, 3)) * 255).astype("uint8"))
    b, s, c = det(img)                                         # the real processor + model: shapes and token check pass
    assert len(b) == 20 and c.max() < 4 and b[:, 2].max() <= 80 and b[:, 3].max() <= 48
    # force query 0 onto the 2nd token of "dining table": its class must be 2
    real = det._forward
    def forced(im, t):
        lg, bx, got = real(im, t)
        lg = lg.copy(); lg[0, tsets[2][1]] = 50.0
        return lg, bx, got
    det._forward = forced
    b, s, c = det(img)
    assert c[0] == 2 and s[0] > 0.99


def test_gdino_chunks_when_the_prompt_is_too_long(tiny_gdino):
    model, proc = tiny_gdino
    names = ["sheep", "potted plant", "dining table", "tv"]
    det = ovd.GDinoDetector(names, model=model, processor=proc, device="cpu", max_tokens=8)
    assert len(det.prompts) > 1 and sorted(sum([p[3] for p in det.prompts], [])) == [0, 1, 2, 3]
    img = Image.fromarray((np.random.default_rng(2).random((48, 80, 3)) * 255).astype("uint8"))
    b, s, c = det(img)
    assert len(b) and set(c.tolist()) <= {0, 1, 2, 3}


# ------------------------------------------------------------------ scoring
def _toy_coco():
    images = [{"id": i, "file_name": f"{i}.jpg", "width": 100, "height": 100} for i in range(1, 41)]
    anns = [{"image_id": i, "category_id": cu.COCO_IDS[(i % 4)], "bbox": [10, 10, 40, 40], "iscrowd": 0} for i in range(1, 41)]
    return {"images": images, "annotations": anns}


def test_score_perfect_and_paired_rows():
    coco = _toy_coco()
    ids = [im["id"] for im in coco["images"]]
    gt = make_coco_gt(coco, ids)
    tasks = [[0, 1], [2, 3]]
    perfect = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
               for a in coco["annotations"]]
    r = ovd.score(perfect, ids, gt, tasks, n_folds=4, name="p")
    c = r["curve"][-1]
    assert c["AP50"] == pytest.approx(1.0) and c["AP50_new"] == pytest.approx(1.0) and c["AP50_base"] == pytest.approx(1.0)
    assert len(r["final_folds"]["AP50"]) == 4 and r["n_dets"] == 40
    half = [d for d in perfect if cu.COCO_IDS.index(d["category_id"]) in (0, 1)]
    r2 = ovd.score(half, ids, gt, tasks, n_folds=4, name="h")
    assert r2["curve"][-1]["AP50_new"] == pytest.approx(0.0)
    md = ovd.paired_rows({"p": r, "h": r2}, {"E5_tuned": r2}, "E5_tuned")
    assert "| p |" in md and "| +0.500 |" in md
    assert "skipped" in ovd.paired_rows({"p": r}, {}, "E5_tuned")
    assert "| p |" in ovd.format_rows({"p": r}, {0: "a", 1: "b"})


# ------------------------------------------------------------------ CLI on the fake world
def _base(wd):
    return ["--workdir", wd, "--fake", "--fake_classes", "9", "--n_train_per_task", "20", "--n_eval", "30", "--n_tune", "20",
            "--n_final", "12", "--n_yolo_train", "15", "--n_yolo_val", "8"]


class _GTDetector:
    """Returns the ground-truth boxes of each image (an oracle) so the CLI scoring can be checked exactly."""
    def __init__(self, coco, names):
        self.by_size = {}
        self.anns = {}
        for a in coco["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)
        self.queue = []

    def __call__(self, img):
        a = self.anns.get(self.queue.pop(0), [])
        b = np.array([[x["bbox"][0], x["bbox"][1], x["bbox"][0] + x["bbox"][2], x["bbox"][1] + x["bbox"][3]] for x in a]).reshape(-1, 4)
        return b, np.full(len(a), 0.9), np.array([self.pos[cu.COCO_IDS.index(x["category_id"])] for x in a], dtype=np.int64)


def test_ovd_cli_scores_guards_final_and_pairs(tmp_path, monkeypatch, capsys):
    wd = str(tmp_path / "wd")
    assert rp.main(["fetch", *_base(wd)]) == 0
    pth, ct, cv = rp.load_paths(rp.build_parser().parse_args(["ovd", *_base(wd)]))
    sp = json.load(open(os.path.join(wd, "splits.json")))
    order = [c for t in sp["tasks"] for c in t]
    made = []

    def fake_make(model, names, model_id=None, device=None, max_det=100):
        d = _GTDetector(cv, names)
        d.pos = {c: j for j, c in enumerate(order)}
        made.append((model, list(names)))
        return d

    real_iter = rp.pl.iter_images
    def iter_and_queue(coco, ids, img_dir):
        for iid, img in real_iter(coco, ids, img_dir):
            for d in _live:
                d.queue.append(iid)
            yield iid, img
    _live = []
    def fake_make_live(*a, **k):
        d = fake_make(*a, **k); _live[:] = [d]; return d
    monkeypatch.setattr(ovd, "make_detector", fake_make_live)
    monkeypatch.setattr(rp.pl, "iter_images", iter_and_queue)

    assert rp.main(["ovd", *_base(wd), "--models", "owlv2", "gdino", "--n_folds", "3"]) == 0
    res = json.load(open(os.path.join(wd, "results", "ovd_eval.json")))
    assert set(res) == {"owlv2", "gdino"}
    for r in res.values():
        assert r["curve"][-1]["AP50"] == pytest.approx(1.0) and len(r["final_folds"]["AP50"]) == 3
    assert made[0][1] == cu.prompt_names(order)                       # every class name, in task order
    # paired against a reference file with the same folds
    ref = os.path.join(wd, "results", "ref.json")
    json.dump({"E5_tuned": res["gdino"]}, open(ref, "w"))
    assert rp.main(["ovd", *_base(wd), "--models", "owlv2", "--n_folds", "3", "--ref", ref]) == 0
    assert "vs E5_tuned" in capsys.readouterr().out
    # smoke runs never overwrite the main file
    assert rp.main(["ovd", *_base(wd), "--models", "owlv2", "--limit", "7", "--n_folds", "3"]) == 0
    assert os.path.exists(os.path.join(wd, "results", "ovd_eval_smoke.json"))
    # final: own log, once per model; --limit refused; the method's final log untouched
    with pytest.raises(SystemExit):
        rp.main(["ovd", *_base(wd), "--eval_on", "final", "--limit", "3"])
    assert rp.main(["ovd", *_base(wd), "--eval_on", "final", "--models", "owlv2", "--n_folds", "3"]) == 0
    with pytest.raises(SystemExit):
        rp.main(["ovd", *_base(wd), "--eval_on", "final", "--models", "owlv2"])
    assert rp.main(["ovd", *_base(wd), "--eval_on", "final", "--models", "gdino", "--n_folds", "3"]) == 0
    assert not os.path.exists(os.path.join(wd, "results", "final_log.json"))
    log = json.load(open(os.path.join(wd, "results", "final_log_ovd.json")))
    assert [e["model"] for e in log] == ["owlv2", "gdino"]


def test_ovd_cli_reports_missing_images(tmp_path):
    wd = str(tmp_path / "wd")
    assert rp.main(["fetch", *_base(wd)]) == 0
    pth = json.load(open(os.path.join(wd, "paths.json")))
    import shutil
    shutil.rmtree(pth["img_dir_val"])
    with pytest.raises(SystemExit, match="missing"):
        rp.main(["ovd", *_base(wd), "--models", "owlv2"])


# ------------------------------------------------------------------ VOC image links after a tar restore
def test_restore_voc_images_relinks_without_touching_json(tmp_path):
    import voc_utils as vu
    from test_voc import make_tree
    root = tmp_path / "vocsrc"
    make_tree(str(root))
    out = tmp_path / "work" / "voc_coco"
    info = vu.prepare_voc(str(root), str(out), "auto", download=False)
    paths = {"ann_train": str(out / "annotations" / "voc_train.json"), "ann_val": str(out / "annotations" / "voc_val.json"),
             "img_dir_train": str(out / "images" / "train"), "img_dir_val": str(out / "images" / "val")}
    before = {k: open(paths[k]).read() for k in ("ann_train", "ann_val")}
    import shutil
    shutil.rmtree(out / "images")                                           # what a tar --exclude leaves
    assert ovd.restore_voc_images(paths, str(root)) is True
    val = json.load(open(paths["ann_val"]))
    for im in val["images"]:
        assert Image.open(os.path.join(paths["img_dir_val"], im["file_name"])).size
    assert {k: open(paths[k]).read() for k in ("ann_train", "ann_val")} == before
    assert ovd.restore_voc_images(paths, str(root)) is False              # already there: nothing to do
    # a json listing other images -> refuse
    shutil.rmtree(str(out) + "_rebuild"); os.remove(out / "images")
    v = json.load(open(paths["ann_val"])); v["images"] = v["images"][:-1]; json.dump(v, open(paths["ann_val"], "w"))
    with pytest.raises(SystemExit, match="different images"):
        ovd.restore_voc_images(paths, str(root))

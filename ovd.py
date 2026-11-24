"""ovd.py -- open-vocabulary detector baselines scored with the SAME images, folds and COCOeval as our runs.

    yoloworld : ultralytics YOLO-World (yolov8s-worldv2 by default), class names as the vocabulary
    owlv2     : OWLv2 (google/owlv2-base-patch16-ensemble), "a photo of a {name}" queries
    gdino     : Grounding DINO (IDEA-Research/grounding-dino-tiny), "name . name . ... ." prompts (chunked to 256 tokens)

They need NO labels for the new classes, like our method, so they are the closest outside comparison. They are REFERENCE
rows, not competitors under our protocol: all three were pre-trained on large box-annotated / grounding data (Objects365,
GoldG, Visual Genome, ...) that already contains the VOC / COCO classes, so the "new" classes are not new to them. Every
detector is told ALL class names at once (the final state of the class sequence) and is scored like the last task.

The raw model outputs are decoded here (not with the transformers post-processors, whose arguments changed between
versions): OWLv2 boxes are relative to the image padded to a square (bottom/right), Grounding DINO boxes to the image.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import incremental as inc
from evaluate import coco_eval, dets_to_coco

MODELS = ("yoloworld", "owlv2", "gdino")
DEFAULT_IDS = {"yoloworld": "yolov8s-worldv2.pt", "owlv2": "google/owlv2-base-patch16-ensemble",
               "gdino": "IDEA-Research/grounding-dino-tiny"}
OWL_TEMPLATE = "a photo of a {}"


# ------------------------------------------------------------------ decoding (pure numpy, tested offline)
def _sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def cxcywh_to_xyxy(b) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    cx, cy, w, h = b.T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def _finish(xyxy, score, cls, width, height, max_det):
    """Clip to the image, drop empty boxes, keep the max_det highest scores (stable order)."""
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4).copy()
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height)
    score, cls = np.asarray(score, dtype=np.float64).reshape(-1), np.asarray(cls, dtype=np.int64).reshape(-1)
    ok = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1]) & np.isfinite(score)
    xyxy, score, cls = xyxy[ok], score[ok], cls[ok]
    order = np.argsort(-score, kind="stable")[:max_det]
    return xyxy[order], score[order], cls[order]


def decode_owl(logits, boxes, width: int, height: int, max_det: int = 100, padded_square: bool = True):
    """OWL-ViT / OWLv2 outputs of ONE image: logits (patches, queries), boxes (patches, 4) cxcywh in [0, 1].
    One detection per patch: its best query (as the official post-processor). OWLv2 pads the image to a square at the
    bottom/right, so its boxes are scaled by max(width, height) on both axes; OWL-ViT v1 boxes by (width, height)."""
    p = _sigmoid(logits)
    if p.ndim != 2:
        raise ValueError(f"logits must be (patches, queries), got {p.shape}")
    cls, score = p.argmax(axis=1), p.max(axis=1)
    xyxy = cxcywh_to_xyxy(boxes)
    if padded_square:
        s = float(max(width, height))
        xyxy = xyxy * s
    else:
        xyxy = xyxy * np.array([width, height, width, height], dtype=np.float64)
    return _finish(xyxy, score, cls, width, height, max_det)


def build_gdino_prompt(names: Sequence[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """'a . b c . d .' (lower case, as Grounding DINO expects) and the character span of every name."""
    parts, spans, pos = [], [], 0
    for n in names:
        n = " ".join(str(n).lower().split())
        if not n:
            raise ValueError("empty class name")
        spans.append((pos, pos + len(n)))
        parts.append(n)
        pos += len(n) + len(" . ")
    return " . ".join(parts) + " .", spans


def token_sets(offsets: Sequence[Tuple[int, int]], spans: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Token positions of every phrase: a token belongs to the phrase whose character span contains it.
    Special tokens have empty offsets (s == e) and belong to none. Every phrase must get at least one token."""
    out = []
    for (ps, pe) in spans:
        out.append([i for i, (s, e) in enumerate(offsets) if e > s and s >= ps and e <= pe])
    empty = [k for k, t in enumerate(out) if not t]
    if empty:
        raise ValueError(f"phrases {empty} got no tokens: prompt and offsets do not match")
    return out


def decode_gdino(logits, boxes, tsets: Sequence[Sequence[int]], width: int, height: int, max_det: int = 100):
    """Grounding DINO outputs of ONE image: logits (queries, text positions; -inf past the text), boxes (queries, 4)
    cxcywh in [0, 1] of the image. Class score = max over the phrase's tokens; one detection per query (its best class)."""
    p = _sigmoid(logits)
    if p.ndim != 2:
        raise ValueError(f"logits must be (queries, tokens), got {p.shape}")
    if max(max(t) for t in tsets) >= p.shape[1]:
        raise ValueError("a phrase token lies beyond the logits: prompt too long for the model")
    per = np.stack([p[:, list(t)].max(axis=1) for t in tsets], axis=1)          # (queries, classes)
    cls, score = per.argmax(axis=1), per.max(axis=1)
    xyxy = cxcywh_to_xyxy(boxes) * np.array([width, height, width, height], dtype=np.float64)
    return _finish(xyxy, score, cls, width, height, max_det)


def chunk_names(names: Sequence[str], n_tokens: Callable[[str], int], max_tokens: int = 256) -> List[List[int]]:
    """Greedy chunks of class positions whose prompt fits in max_tokens (special tokens included)."""
    chunks, cur = [], []
    for k in range(len(names)):
        trial = cur + [k]
        if n_tokens(build_gdino_prompt([names[j] for j in trial])[0]) <= max_tokens:
            cur = trial
        else:
            if not cur:
                raise ValueError(f"class name {names[k]!r} alone exceeds {max_tokens} tokens")
            chunks.append(cur)
            cur = [k]
    if cur:
        chunks.append(cur)
    return chunks


# ------------------------------------------------------------------ detectors: PIL image -> (xyxy, score, class position)
def _device(device):
    if device not in (None, "", "auto"):
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class OwlDetector:
    def __init__(self, names: Sequence[str], model_id: str = DEFAULT_IDS["owlv2"], device=None, max_det: int = 100,
                 template: str = OWL_TEMPLATE, model=None, processor=None):
        self.names, self.max_det = list(names), max_det
        self.queries = [template.format(n) for n in self.names]
        if model is None or processor is None:
            from transformers import AutoProcessor, Owlv2ForObjectDetection, OwlViTForObjectDetection
            processor = processor or AutoProcessor.from_pretrained(model_id)
            cls = Owlv2ForObjectDetection if "owlv2" in model_id.lower() else OwlViTForObjectDetection
            model = model or cls.from_pretrained(model_id)
        self.device = _device(device)
        self.model, self.processor = model.to(self.device).eval(), processor
        self.padded = getattr(model.config, "model_type", "owlv2") == "owlv2"
        self.max_len = int(getattr(getattr(model.config, "text_config", None), "max_position_embeddings", 16) or 16)

    def _forward(self, img):
        import torch
        inp = self.processor(text=[self.queries], images=img, return_tensors="pt", padding="max_length", truncation=True,
                             max_length=self.max_len)
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.no_grad():
            out = self.model(**inp)
        return out.logits[0].float().cpu().numpy(), out.pred_boxes[0].float().cpu().numpy()

    def __call__(self, img):
        logits, boxes = self._forward(img)
        return decode_owl(logits, boxes, img.width, img.height, self.max_det, self.padded)


class GDinoDetector:
    def __init__(self, names: Sequence[str], model_id: str = DEFAULT_IDS["gdino"], device=None, max_det: int = 100,
                 model=None, processor=None, max_tokens: Optional[int] = None):
        self.names, self.max_det = list(names), max_det
        if model is None or processor is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            processor = processor or AutoProcessor.from_pretrained(model_id)
            model = model or AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.device = _device(device)
        self.model, self.processor = model.to(self.device).eval(), processor
        self.tok = processor.tokenizer
        mt = max_tokens or int(getattr(model.config, "max_text_len", 256))
        self.chunks = chunk_names(self.names, lambda t: len(self.tok(t)["input_ids"]), mt)
        self.prompts = []
        for ch in self.chunks:
            text, spans = build_gdino_prompt([self.names[j] for j in ch])
            enc = self.tok(text, return_offsets_mapping=True)
            self.prompts.append((text, list(enc["input_ids"]), token_sets(enc["offset_mapping"], spans), ch))

    def _forward(self, img, text):
        import torch
        inp = self.processor(images=img, text=text, return_tensors="pt")
        ids = inp["input_ids"][0].tolist()
        inp = {k: v.to(self.device) for k, v in inp.items()}
        with torch.no_grad():
            out = self.model(**inp)
        return out.logits[0].float().cpu().numpy(), out.pred_boxes[0].float().cpu().numpy(), ids

    def __call__(self, img):
        allb, alls, allc = [], [], []
        for text, ids, tsets, ch in self.prompts:
            logits, boxes, got = self._forward(img, text)
            if list(got) != list(ids):
                raise RuntimeError("the processor tokenised the prompt differently from the tokenizer: token positions "
                                   "would be wrong. Prompt: " + text)
            b, s, c = decode_gdino(logits, boxes, tsets, img.width, img.height, self.max_det)
            allb.append(b); alls.append(s); allc.append(np.asarray(ch, dtype=np.int64)[c])
        return _finish(np.concatenate(allb), np.concatenate(alls), np.concatenate(allc), img.width, img.height,
                       self.max_det)


def make_detector(model: str, names: Sequence[str], model_id: Optional[str] = None, device=None, max_det: int = 100):
    if model == "yoloworld":
        from baselines import WorldDetector
        return WorldDetector(names, weights=model_id or DEFAULT_IDS[model], conf=0.001, max_det=max_det,
                             device=None if device in (None, "", "auto") else device)
    if model == "owlv2":
        return OwlDetector(names, model_id or DEFAULT_IDS[model], device, max_det)
    if model == "gdino":
        return GDinoDetector(names, model_id or DEFAULT_IDS[model], device, max_det)
    raise ValueError(f"model must be one of {MODELS}, got {model!r}")


# ------------------------------------------------------------------ running and scoring
def collect(detector: Callable, images: Iterable, class_order: Sequence[int], log: Callable = print,
            every: int = 250) -> List[dict]:
    """COCO result dicts of `detector` on (image_id, PIL) pairs; class position j -> class_order[j]."""
    co = np.asarray(class_order, dtype=np.int64)
    dets, n = [], 0
    for iid, img in images:
        b, s, c = detector(img)
        if len(b):
            dets += dets_to_coco(iid, np.asarray(b), np.asarray(s), co[np.asarray(c, dtype=np.int64)])
        n += 1
        if every and n % every == 0:
            log(f"    {n} images")
    return dets


def score(dets: List[dict], eval_ids: Sequence[int], gt, tasks, n_folds: int = 10, fold_seed: int = 123,
          name: str = "") -> Dict:
    """The final-state row (all classes), per-class AP50 and fold scores, in the shape compare_runs.py reads."""
    class_order = [int(c) for t in tasks for c in t]
    r = coco_eval(gt, dets, eval_ids, class_order)
    pc = r["per_class_AP50"]
    mean = lambda cs: float(np.nanmean([pc[k] for k in cs])) if len(cs) else float("nan")
    old = [c for t in tasks[:-1] for c in t]
    row = {"task": len(tasks) - 1, "n_seen": len(class_order), "AP": r["AP"], "AP50": r["AP50"], "AP75": r["AP75"],
           "AP50_base": mean(tasks[0]), "AP50_old": mean(old), "AP50_new": mean(tasks[-1])}
    out = {"variant": name, "curve": [row], "diag": [], "A_bar_AP": float("nan"), "final_AP": r["AP"],
           "final_AP50": r["AP50"], "base_AP50_drop": float("nan"),
           "per_class_AP50": {str(k): float(v) for k, v in pc.items()},
           "eval_ids_sha": ids_sha(eval_ids), "n_dets": len(dets)}
    folds = inc.image_folds(eval_ids, n_folds, fold_seed)
    if folds:
        out["final_folds"] = inc.fold_scores(gt, dets, folds, class_order, list(tasks[-1]))
    return out


def ids_sha(ids) -> str:
    return hashlib.sha256(json.dumps(sorted(int(i) for i in ids)).encode()).hexdigest()[:16]


def paired_rows(res: Dict[str, dict], ref_res: Dict[str, dict], ref_name: str, metrics=("AP50", "AP50_new")) -> str:
    """Markdown: every OVD row against `ref_name` (e.g. E5_tuned) from another result file, paired over the same folds."""
    import compare_runs as cr
    if ref_name not in ref_res or "final_folds" not in ref_res[ref_name]:
        return f"(no fold scores for {ref_name} in the reference file: paired comparison skipped)\n"
    rf = ref_res[ref_name]["final_folds"]
    lines = []
    for m in metrics:
        lines += [f"\n### {m}: OVD reference rows vs {ref_name} (paired over the same {len(rf[m])} image folds)\n",
                  "| run | mean | diff vs ref | 95% CI (+-) | significant |", "|---|---|---|---|---|"]
        mr, _, _ = cr.mean_sd(rf[m])
        lines.append(f"| {ref_name} | {mr:.3f} | - | - | - |")
        for n, r in res.items():
            f = r.get("final_folds")
            if not f or f["n_images"] != rf["n_images"]:
                lines.append(f"| {n} | - | folds differ from the reference: not paired | | |")
                continue
            mu, _, _ = cr.mean_sd(f[m])
            d, h, _ = cr.paired_ci(f[m], rf[m])
            sig = "yes" if (h == h and abs(d) > h) else "no"
            lines.append(f"| {n} | {mu:.3f} | {d:+.3f} | {h:.3f} | {sig} |")
    return "\n".join(lines) + "\n"


def format_rows(res: Dict[str, dict], class_names: Dict[int, str] = None) -> str:
    lines = ["| model | AP | AP50 | AP50 base | AP50 old | AP50 new | fold AP50 mean | fold AP50_new mean |",
             "|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        c = r["curve"][-1]
        f = r.get("final_folds") or {}
        fm = lambda k: f"{np.nanmean(f[k]):.3f}" if f.get(k) else "-"
        lines.append(f"| {n} | {c['AP']:.3f} | {c['AP50']:.3f} | {c['AP50_base']:.3f} | {c['AP50_old']:.3f} | "
                     f"{c['AP50_new']:.3f} | {fm('AP50')} | {fm('AP50_new')} |")
    if class_names:
        lines += ["", "Per-class AP50:", "", "| model | " + " | ".join(class_names.values()) + " |",
                  "|---|" + "---|" * len(class_names)]
        for n, r in res.items():
            pc = r.get("per_class_AP50", {})
            lines.append(f"| {n} | " + " | ".join(f"{pc.get(str(k), float('nan')):.3f}" for k in class_names) + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ VOC work dirs restored from a tar (image links excluded)
def restore_voc_images(paths: dict, voc_root: str, log: Callable = print) -> bool:
    """Work-dir tars exclude the VOC image links. Re-create them WITHOUT touching splits.json or the annotation files:
    the conversion is rebuilt in <out>_rebuild and its images are linked in only if its json lists exactly the same
    images (id, file name) as the work dir's json. Returns True if links were (re)created."""
    import os
    import shutil
    import voc_utils as vu
    img_val = paths["img_dir_val"]
    if os.path.isdir(img_val) and os.listdir(img_val):
        return False
    out = os.path.dirname(os.path.dirname(os.path.abspath(img_val)))          # <out>/images/val
    tmp = out.rstrip("/") + "_rebuild"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    vu.prepare_voc(voc_root, tmp, "auto", download=False)
    for split, key in (("train", "ann_train"), ("val", "ann_val")):
        with open(os.path.join(tmp, "annotations", f"voc_{split}.json")) as f:
            new = [(im["id"], im["file_name"]) for im in json.load(f)["images"]]
        with open(paths[key]) as f:
            old = [(im["id"], im["file_name"]) for im in json.load(f)["images"]]
        if new != old:
            raise SystemExit(f"restore: the rebuilt VOC {split} json lists different images than {paths[key]} "
                             f"({len(new)} vs {len(old)}): attach the same VOC dataset the work dir was made with")
    images = os.path.join(out, "images")
    if os.path.islink(images) or (os.path.isdir(images) and not any(os.scandir(images))):
        os.remove(images) if os.path.islink(images) else shutil.rmtree(images)
    elif os.path.exists(images):                                             # partial folders: replace the empty split
        for split in ("train", "val"):
            d = os.path.join(images, split)
            if os.path.isdir(d) and not os.listdir(d):
                os.rmdir(d)
            if not os.path.exists(d):
                os.symlink(os.path.join(tmp, "images", split), d)
        log(f"restore: VOC image links re-created under {images}")
        return True
    os.makedirs(out, exist_ok=True)
    os.symlink(os.path.join(tmp, "images"), images)
    log(f"restore: VOC image links re-created ({images} -> {tmp}/images); annotation files and splits untouched")
    return True

"""voc_utils.py -- Pascal VOC -> COCO-format conversion, so the rest of the pipeline is unchanged.

All 20 VOC classes are also COCO classes. We therefore write VOC annotations as COCO json that
keeps the COCO category ids; class indices stay in the COCO 0..79 index space and every other
module (splits, YOLO labels, prompts, CLIP text, COCOeval) works as it is. A VOC split such as
'15+5' is just a list of COCO class indices (the first 15 / last 5 VOC classes in the standard
alphabetical VOC order).

Conventions (standard for VOC):
  * box coordinates are 1-based inclusive pixels -> COCO [x-1, y-1, w, h] with w = xmax-xmin+1;
  * objects flagged `difficult` become iscrowd=1: never used for training or crops, and ignored
    (neither hit nor false positive) by COCOeval -- the same effect as the VOC protocol;
  * protocol 07+12: train = VOC2007 trainval + VOC2012 trainval, eval pool = VOC2007 test.

Pure python (xml, json, tarfile, os) + coco_fetch.download_file. No torch.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
from typing import Dict, List, Sequence, Tuple

import coco_fetch as cf
import coco_utils as cu

# standard alphabetical VOC order (the order incremental-detection papers split by)
VOC_NAMES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
             "train", "tvmonitor"]
VOC_TO_COCO_NAME = {"aeroplane": "airplane", "motorbike": "motorcycle", "diningtable": "dining table",
                    "pottedplant": "potted plant", "sofa": "couch", "tvmonitor": "tv"}
# VOC position (0..19) -> COCO class index (0..79)
VOC_TO_COCO_IDX = [cu.COCO_NAMES.index(VOC_TO_COCO_NAME.get(n, n)) for n in VOC_NAMES]
VOC_NAME2COCO_IDX = dict(zip(VOC_NAMES, VOC_TO_COCO_IDX))

MIRRORS = {
    "VOCtrainval_06-Nov-2007.tar": ["http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
                                    "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar"],
    "VOCtest_06-Nov-2007.tar": ["http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
                                "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar"],
    "VOCtrainval_11-May-2012.tar": ["http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
                                    "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"],
}
PROTOCOL_TARS = {"07+12": ["VOCtrainval_06-Nov-2007.tar", "VOCtest_06-Nov-2007.tar", "VOCtrainval_11-May-2012.tar"],
                 "07": ["VOCtrainval_06-Nov-2007.tar", "VOCtest_06-Nov-2007.tar"],
                 "12": ["VOCtrainval_11-May-2012.tar"]}


# ------------------------------------------------------------------ splits
def voc_tasks(scheme: str) -> List[List[int]]:
    """'15+5' | '10+10' | '19+1' | '10+2x5' (10 base, then two tasks of 5; VOC order) or custom '0,1,2|3,4'
    (VOC positions 0..19) -> lists of COCO class indices."""
    scheme = scheme.strip()
    if "|" in scheme:
        groups = [[int(x) for x in part.split(",") if x.strip() != ""] for part in scheme.split("|")]
        flat = [c for g in groups for c in g]
        if any(len(g) == 0 for g in groups) or len(set(flat)) != len(flat) or min(flat) < 0 or max(flat) > 19:
            raise ValueError(f"bad custom VOC task spec {scheme!r}: need disjoint non-empty groups of positions 0..19")
    else:
        groups = cu.make_tasks(scheme, num_classes=20)
    return [[VOC_TO_COCO_IDX[p] for p in g] for g in groups]


# ------------------------------------------------------------------ parsing
def image_id_for(stem: str) -> int:
    """Unique int id: '2008_000123' -> 2008000123; plain VOC2007 stems '000123' -> 10_000_000_123."""
    if "_" in stem:
        return int(stem.replace("_", ""))
    return 10_000_000_000 + int(stem)


def parse_voc_xml(path: str) -> Dict:
    root = ET.parse(path).getroot()
    size = root.find("size")
    w, h = int(float(size.findtext("width"))), int(float(size.findtext("height")))
    objs = []
    for o in root.findall("object"):
        bb = o.find("bndbox")
        objs.append({"name": o.findtext("name").strip(),
                     "difficult": int(float(o.findtext("difficult", "0") or 0)),
                     "box": [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]})
    return {"width": w, "height": h, "objects": objs}


def voc_box_to_coco(box: Sequence[float], width: int, height: int) -> List[float]:
    """VOC 1-based inclusive [xmin,ymin,xmax,ymax] -> COCO 0-based [x,y,w,h], clipped to the image."""
    x1, y1, x2, y2 = box
    x1, y1 = max(x1 - 1.0, 0.0), max(y1 - 1.0, 0.0)
    x2, y2 = min(x2, float(width)), min(y2, float(height))
    return [x1, y1, x2 - x1, y2 - y1]


# ------------------------------------------------------------------ locating the VOC tree
def find_voc_dirs(root: str, max_depth: int = 7) -> List[str]:
    """Every directory under root that looks like a VOC devkit year folder
    (has Annotations/, JPEGImages/ and ImageSets/Main/). Layout-agnostic: works for the official
    tarballs (VOCdevkit/VOC2007) and for Kaggle-style re-uploads."""
    found, root = [], os.path.abspath(root)
    if not os.path.isdir(root):
        return found
    base_depth = root.rstrip(os.sep).count(os.sep)
    for d, subdirs, _ in os.walk(root, followlinks=True):
        if d.count(os.sep) - base_depth >= max_depth:
            subdirs[:] = []
        if {"Annotations", "JPEGImages", "ImageSets"} <= set(subdirs) and \
                os.path.isdir(os.path.join(d, "ImageSets", "Main")):
            found.append(d)
    return sorted(found)


def year_of(voc_dir: str) -> int:
    """2007 or 2012, taken from the folder name, else from its parents."""
    parts = [p for p in voc_dir.replace("\\", "/").split("/") if p][::-1]
    for p in parts:
        m = re.search(r"(?<!\d)(2007|2012)(?!\d)", p)
        if m:
            return int(m.group(1))
    raise ValueError(f"cannot tell if {voc_dir} is VOC2007 or VOC2012 (put '2007' or '2012' in a folder name)")


def read_split(voc_dirs: Sequence[str], year: int, split: str) -> List[Tuple[str, str]]:
    """[(voc_dir, stem)] for ImageSets/Main/<split>.txt over all folders of that year; only stems
    whose xml and jpg both exist (a skipped count is printed)."""
    out, skipped, dups, seen = [], 0, 0, set()
    for d in voc_dirs:
        if year_of(d) != year:
            continue
        lst = os.path.join(d, "ImageSets", "Main", f"{split}.txt")
        if not os.path.exists(lst):
            continue
        with open(lst) as f:
            stems = [ln.split()[0] for ln in f if ln.strip()]
        for s in stems:
            if os.path.exists(os.path.join(d, "Annotations", s + ".xml")) and \
                    os.path.exists(os.path.join(d, "JPEGImages", s + ".jpg")):
                if s in seen:                       # same image reachable twice (e.g. a dataset mounted twice, case-variant copies)
                    dups += 1
                    continue
                seen.add(s)
                out.append((d, s))
            else:
                skipped += 1
    if skipped:
        print(f"  VOC{year} {split}: skipped {skipped} listed images with a missing xml/jpg")
    if dups:
        print(f"  VOC{year} {split}: ignored {dups} repeated copies of the same image (duplicate folders)")
    return out


def choose_protocol(voc_dirs: Sequence[str], protocol: str = "auto"):
    """-> (protocol used, train entries, val entries)."""
    sets = {(y, s): read_split(voc_dirs, y, s) for y, s in
            ((2007, "trainval"), (2007, "test"), (2012, "trainval"), (2012, "train"), (2012, "val"))}
    have = lambda y, s: len(sets[(y, s)]) > 0
    if protocol == "auto":
        if have(2007, "trainval") and have(2007, "test"):
            protocol = "07+12" if have(2012, "trainval") else "07"
        elif have(2012, "train") and have(2012, "val"):
            protocol = "12"
        else:
            raise ValueError("auto protocol needs VOC2007 trainval+test, or VOC2012 train+val; found: " +
                             ", ".join(f"{y}/{s}={len(v)}" for (y, s), v in sets.items()))
    if protocol == "07+12":
        need = [(2007, "trainval"), (2007, "test"), (2012, "trainval")]
        train, val = sets[(2007, "trainval")] + sets[(2012, "trainval")], sets[(2007, "test")]
    elif protocol == "07":
        need = [(2007, "trainval"), (2007, "test")]
        train, val = sets[(2007, "trainval")], sets[(2007, "test")]
    elif protocol == "12":
        need = [(2012, "train"), (2012, "val")]
        train, val = sets[(2012, "train")], sets[(2012, "val")]
    else:
        raise ValueError(f"protocol must be auto | 07+12 | 07 | 12, got {protocol!r}")
    missing = [f"VOC{y} {s}" for y, s in need if not have(y, s)]
    if missing:
        raise ValueError(f"protocol {protocol} needs {missing}, which were not found under the VOC root")
    return protocol, train, val


# ------------------------------------------------------------------ conversion
def _link_or_copy(src: str, dst: str) -> None:
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def _convert_split(entries: Sequence[Tuple[str, str]], img_dir: str) -> Tuple[dict, Dict]:
    os.makedirs(img_dir, exist_ok=True)
    images, anns, seen = [], [], set()
    n_diff = n_obj = 0
    for d, stem in entries:
        iid = image_id_for(stem)
        fname = stem + ".jpg"
        if iid in seen:
            raise ValueError(f"duplicate image id for {stem}: the same image appears twice in this split")
        seen.add(iid)
        rec = parse_voc_xml(os.path.join(d, "Annotations", stem + ".xml"))
        _link_or_copy(os.path.join(d, "JPEGImages", fname), os.path.join(img_dir, fname))
        images.append({"id": iid, "file_name": fname, "width": rec["width"], "height": rec["height"],
                       "voc_year": year_of(d)})
        for o in rec["objects"]:
            if o["name"] not in VOC_NAME2COCO_IDX:
                raise ValueError(f"{stem}: unknown VOC class {o['name']!r}")
            bbox = voc_box_to_coco(o["box"], rec["width"], rec["height"])
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            anns.append({"id": len(anns) + 1, "image_id": iid,
                         "category_id": cu.COCO_IDS[VOC_NAME2COCO_IDX[o["name"]]],
                         "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": int(o["difficult"])})
            n_obj += 1
            n_diff += o["difficult"]
    coco = {"images": images, "annotations": anns,
            "categories": [{"id": i, "name": n} for i, n in zip(cu.COCO_IDS, cu.COCO_NAMES)]}
    return coco, {"images": len(images), "objects": n_obj, "difficult": n_diff}


def convert(train_entries, val_entries, out_dir: str) -> Dict:
    """Write out_dir/annotations/voc_{train,val}.json and out_dir/images/{train,val}/ (symlinks)."""
    ann_dir = os.path.join(out_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    paths, stats = {}, {}
    for split, entries in (("train", train_entries), ("val", val_entries)):
        img_dir = os.path.join(out_dir, "images", split)
        coco, stats[split] = _convert_split(entries, img_dir)
        p = os.path.join(ann_dir, f"voc_{split}.json")
        with open(p, "w") as f:
            json.dump(coco, f)
        paths[f"ann_{split}"], paths[f"img_dir_{split}"] = p, img_dir
    return {**paths, "stats": stats}


# ------------------------------------------------------------------ download / extract
def _safe_extract(tar_path: str, dest: str) -> None:
    dest = os.path.abspath(dest)
    with tarfile.open(tar_path) as t:
        for m in t.getmembers():
            target = os.path.abspath(os.path.join(dest, m.name))
            if not (target == dest or target.startswith(dest + os.sep)):
                raise ValueError(f"unsafe path in tar: {m.name}")
            if m.issym() or m.islnk():
                raise ValueError(f"links are not allowed in the VOC tar: {m.name}")
        if hasattr(tarfile, "data_filter"):                  # Python >= 3.12: also refuse special files etc.
            t.extractall(dest, filter="data")
        else:                                               # older Python: members were vetted above
            t.extractall(dest)


def download_voc(root: str, protocol: str, mirrors: Dict[str, List[str]] = None) -> List[str]:
    """Download + extract the tarballs a protocol needs into root. Tries each mirror in order."""
    mirrors = mirrors or MIRRORS
    os.makedirs(root, exist_ok=True)
    got = []
    for tar in PROTOCOL_TARS[protocol]:
        dest, last = os.path.join(root, tar), None
        for url in mirrors[tar]:
            try:
                cf.download_file(url, dest)
                last = None
                break
            except IOError as e:
                last = e
        if last is not None:
            raise IOError(f"could not download {tar} from any mirror: {last}\n"
                          f"Download it by hand into {root} (or add a Kaggle VOC dataset and pass --voc_root).")
        _safe_extract(dest, root)
        got.append(dest)
    return got


def prepare_voc(voc_root: str, out_dir: str, protocol: str = "auto", download: bool = True,
                download_dir: str = None) -> Dict:
    """Locate VOC under voc_root (or download what the protocol needs), convert to COCO-format json
    under out_dir. A read-only voc_root (e.g. a Kaggle input) is never written to: downloads go to
    out_dir/download instead, and both places are searched."""
    if download_dir is None:
        writable = (not os.path.exists(voc_root)) or os.access(voc_root, os.W_OK)
        download_dir = voc_root if writable else os.path.join(out_dir, "download")
    roots = [voc_root] + ([download_dir] if os.path.abspath(download_dir) != os.path.abspath(voc_root) else [])
    find = lambda: sorted({d for r in roots for d in find_voc_dirs(r)})
    dirs = find()
    try:
        used, train, val = choose_protocol(dirs, protocol)
    except ValueError as e:
        if not download:
            raise
        want = "07+12" if protocol == "auto" else protocol
        print(f"VOC data incomplete for protocol '{protocol}' ({e}); downloading '{want}' into {download_dir} ...")
        download_voc(download_dir, want)
        dirs = find()
        used, train, val = choose_protocol(dirs, protocol)
    out = convert(train, val, out_dir)
    out["protocol"] = used
    print(f"VOC protocol {used}: train {out['stats']['train']} | eval pool {out['stats']['val']}")
    return out

#!/usr/bin/env python
"""merge_results.py -- one screening table from several N2L-Det working folders.

    python merge_results.py coco40+40=/content/work  coco70+10=/content/work_70_10  voc15+5=/content/work_voc \\
        [--cols E0_zeroshot,E5_ours,E6_gt_ceiling] [--recall_min 0.5] [--headroom_min 0.03] [--out table.md]

Per run it reads (whatever exists; a missing file gives 'n/a', never an error):
    results/gate_clip.json    E0: zero-shot CLIP accuracy on ground-truth crops
    results/gate_recall.json  Gate 1: proposal recall@0.5, base task vs the later (new) tasks
    results/ablation.json     AP50 of the chosen columns (default E0 / E5 / E6) and the gaps between them
    splits.json               dataset, split scheme (labels only)

Pass rule (a PROPOSAL -- agree the thresholds with your guide before looking at the table):
    new-class recall@0.5 >= --recall_min   AND   E6 - E0 headroom (AP50) >= --headroom_min
A run without an ablation.json gets the recall half only (headroom 'n/a'): FAIL if recall fails, else 'n/a'.
Old runs made before gamma 0.5 became the default used gamma 0 for E0_zeroshot / E5_ours / E6_gt_ceiling;
for those pass the matching columns explicitly, e.g. --cols E0_zeroshot,E5_ours,E6_gt_ceiling (same names, gamma 0).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

NA = "n/a"


def _load(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _num(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)) else None


def new_class_recall(rec: Optional[dict], key: str = "recall@0.5"):
    """n_gt-weighted recall over every task after the first. Returns (base recall, new recall, n_new)."""
    if not rec:
        return None, None, 0
    tasks = sorted((k for k in rec if k.startswith("task") and k[4:].isdigit()), key=lambda k: int(k[4:]))
    if not tasks:
        return None, None, 0
    base = _num(rec[tasks[0]].get(key))
    hit = n = 0.0
    for k in tasks[1:]:
        r, g = _num(rec[k].get(key)), rec[k].get("n_gt", 0)
        if r is not None and g:
            hit += r * g
            n += g
    return base, (hit / n if n else None), int(n)


def summarize_run(name: str, workdir: str, cols: Sequence[str], recall_min: float, headroom_min: float) -> Dict:
    clip = _load(os.path.join(workdir, "results", "gate_clip.json")) or {}
    rec = _load(os.path.join(workdir, "results", "gate_recall.json"))
    abl = _load(os.path.join(workdir, "results", "ablation.json")) or {}
    spl = _load(os.path.join(workdir, "splits.json")) or {}
    base, new, n_new = new_class_recall(rec)
    ap50 = {c: _num(abl.get(c, {}).get("final_AP50")) for c in cols}
    e0, e5, e6 = (ap50[c] for c in cols[:3]) if len(cols) >= 3 else (None, None, None)
    headroom = (e6 - e0) if (e0 is not None and e6 is not None) else None
    gap = (e5 - e0) if (e0 is not None and e5 is not None) else None
    if new is None:
        verdict = NA
    elif new < recall_min:
        verdict = "FAIL"
    elif headroom is None:
        verdict = NA
    else:
        verdict = "PASS" if headroom >= headroom_min else "FAIL"
    return {"run": name, "dataset": spl.get("dataset", NA), "scheme": spl.get("scheme", NA),
            "clip_all": _num(clip.get("all")), "recall_base": base, "recall_new": new, "n_new": n_new,
            "boxes_per_image": _num((rec or {}).get("boxes_per_image")), "ap50": ap50,
            "headroom": headroom, "e5_minus_e0": gap, "verdict": verdict}


def _f(x, nd=3, sign=False):
    if x is None:
        return NA
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def format_table(rows: List[Dict], cols: Sequence[str]) -> str:
    head = (["run", "dataset", "split", "E0 crop acc", "recall@.5 base", "recall@.5 new (n)", "boxes/img"]
            + [f"AP50 {c}" for c in cols] + ["E6-E0", "E5-E0", "rule"])
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["run"], str(r["dataset"]), str(r["scheme"]), _f(r["clip_all"]), _f(r["recall_base"]),
                 f"{_f(r['recall_new'])} ({r['n_new']})", _f(r["boxes_per_image"], 1)]
        cells += [_f(r["ap50"][c]) for c in cols]
        cells += [_f(r["headroom"], sign=True), _f(r["e5_minus_e0"], sign=True), r["verdict"]]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def parse_runs(items: Sequence[str]) -> List[tuple]:
    runs = []
    for it in items:
        if "=" not in it:
            raise SystemExit(f"bad run {it!r}: use name=/path/to/workdir")
        name, path = it.split("=", 1)
        if not os.path.isdir(path):
            raise SystemExit(f"run {name!r}: folder not found: {path}")
        runs.append((name, path))
    return runs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="name=workdir (one per dataset/split)")
    p.add_argument("--cols", default="E0_zeroshot,E5_ours,E6_gt_ceiling",
                   help="ablation rows to show; the first three are read as E0, E5, E6 for the gaps")
    p.add_argument("--recall_min", type=float, default=0.5)
    p.add_argument("--headroom_min", type=float, default=0.03)
    p.add_argument("--out", default=None, help="also write the table to this file")
    a = p.parse_args(argv)
    cols = [c.strip() for c in a.cols.split(",") if c.strip()]
    rows = [summarize_run(n, d, cols, a.recall_min, a.headroom_min) for n, d in parse_runs(a.runs)]
    table = format_table(rows, cols)
    note = (f"\nrule = new-class recall@0.5 >= {a.recall_min} and E6-E0 AP50 >= {a.headroom_min} "
            f"(proposed thresholds). n/a = file or run missing.")
    print(table + note)
    if a.out:
        with open(a.out, "w") as f:
            f.write(table + note + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

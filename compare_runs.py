"""compare_runs.py -- mean +- sd over image folds, and paired differences with 95% confidence intervals.

    python compare_runs.py <workdir> [--file ablation.json] [--ref E0_zeroshot] [--metric AP50|AP|AP50_new]
                           [--runs E0_zeroshot E5_ours ...] [--also baselines.json] [--out table.md]

Every run in ablation.json is scored on the same K disjoint image folds (RunConfig.n_folds, default 5). For a pair
(run, ref) the per-fold differences give  mean_diff +- t_{K-1,0.975} * sd_diff / sqrt(K).  'sig' means the interval
excludes 0. K is small, so treat this as a sanity check on noise, not a strict test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

METRICS = ("AP50", "AP", "AP50_new")


def _t975(dof: int) -> float:
    from scipy.stats import t
    return float(t.ppf(0.975, dof))


def mean_sd(x: Sequence[float]):
    v = np.asarray([a for a in x if a == a], dtype=float)          # drop nan folds
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), len(v)


def paired_ci(a: Sequence[float], b: Sequence[float]):
    """(mean(a-b), half-width of the 95% CI, n pairs) over folds where both are defined."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError(f"fold counts differ: {a.shape} vs {b.shape}")
    ok = ~(np.isnan(a) | np.isnan(b))
    d = (a - b)[ok]
    n = int(ok.sum())
    if n < 2:
        return (float(d.mean()) if n else float("nan")), float("nan"), n
    return float(d.mean()), _t975(n - 1) * float(d.std(ddof=1)) / math.sqrt(n), n


def load_folds(path: str) -> Dict[str, dict]:
    with open(path) as f:
        res = json.load(f)
    return {n: r["final_folds"] for n, r in res.items() if "final_folds" in r}


def compare(folds: Dict[str, dict], ref: str, metric: str = "AP50", runs: Optional[Sequence[str]] = None) -> List[dict]:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    if ref not in folds:
        raise KeyError(f"reference run {ref!r} has no fold scores (have: {sorted(folds)})")
    rows = []
    for n in (runs or list(folds)):
        if n not in folds:
            raise KeyError(f"run {n!r} has no fold scores (have: {sorted(folds)})")
        m, sd, k = mean_sd(folds[n][metric])
        row = {"run": n, "mean": m, "sd": sd, "k": k, "diff": None, "ci": None, "sig": None}
        if n != ref:
            d, ci, _ = paired_ci(folds[n][metric], folds[ref][metric])
            row.update(diff=d, ci=ci, sig=bool(ci == ci and abs(d) > ci))
        rows.append(row)
    return rows


def format_rows(rows: List[dict], ref: str, metric: str) -> str:
    out = [f"| run | {metric} mean | sd over folds | diff vs {ref} | 95% CI (+-) | significant |", "|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda v, s=False: "-" if v is None or v != v else (f"{v:+.3f}" if s else f"{v:.3f}")
        out.append(f"| {r['run']} | {f(r['mean'])} | {f(r['sd'])} | {f(r['diff'], True)} | {f(r['ci'])} | "
                   f"{'-' if r['sig'] is None else ('yes' if r['sig'] else 'no')} |")
    return "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("workdir"); p.add_argument("--file", default="ablation.json")
    p.add_argument("--ref", default="E0_zeroshot"); p.add_argument("--metric", default="AP50", choices=METRICS)
    p.add_argument("--runs", nargs="*"); p.add_argument("--out")
    p.add_argument("--also", nargs="*", default=[], help="more files in results/ to merge in, e.g. baselines.json")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    path = os.path.join(a.workdir, "results", a.file)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run `run` first")
    folds = load_folds(path)
    for extra in a.also:
        ep = os.path.join(a.workdir, "results", extra)
        if not os.path.exists(ep):
            raise SystemExit(f"{ep} not found")
        more = load_folds(ep)
        clash = set(more) & set(folds)
        if clash:
            raise SystemExit(f"{extra}: run names already present: {sorted(clash)}")
        k = {len(v["AP50"]) for v in list(folds.values()) + list(more.values())}
        if len(k) > 1:
            raise SystemExit(f"{extra}: different number of folds ({sorted(k)}); re-run with the same n_folds")
        folds.update(more)
    if not folds:
        raise SystemExit(f"{path} has no fold scores (it was made before folds existed, or with n_folds=0); re-run `run`")
    table = format_rows(compare(folds, a.ref, a.metric, a.runs), a.ref, a.metric)
    k = len(next(iter(folds.values()))["AP50"])
    note = f"\n\n{k} disjoint image folds; CI = t(0.975, {k - 1}) * sd(diff) / sqrt({k}). Small K: indicative only."
    print(table + note)
    if a.out:
        with open(a.out, "w") as f:
            f.write(table + note + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

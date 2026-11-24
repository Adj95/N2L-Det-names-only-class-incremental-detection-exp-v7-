import json
import math
import os

import numpy as np
import pytest

import compare_runs as cr


def test_mean_sd_ignores_nan_folds():
    m, sd, k = cr.mean_sd([0.1, 0.2, float("nan"), 0.3])
    assert m == pytest.approx(0.2) and sd == pytest.approx(0.1) and k == 3
    assert cr.mean_sd([float("nan")])[2] == 0


def test_paired_ci_matches_hand_computation():
    a, b = [0.50, 0.52, 0.49, 0.55, 0.51], [0.48, 0.50, 0.50, 0.51, 0.49]
    d = np.array(a) - np.array(b)
    from scipy.stats import t
    want = t.ppf(0.975, 4) * d.std(ddof=1) / math.sqrt(5)
    m, ci, n = cr.paired_ci(a, b)
    assert m == pytest.approx(d.mean()) and ci == pytest.approx(want) and n == 5


def test_paired_ci_skips_nan_pairs_and_rejects_shape_mismatch():
    m, ci, n = cr.paired_ci([1, 2, float("nan"), 4], [0, 1, 1, 3])
    assert n == 3 and m == pytest.approx(1.0) and ci == pytest.approx(0.0)      # constant difference
    assert cr.paired_ci([1.0], [0.0])[1] != cr.paired_ci([1.0], [0.0])[1]       # one pair: no CI
    with pytest.raises(ValueError):
        cr.paired_ci([1, 2], [1, 2, 3])


def make_folds(a, b, c):
    f = lambda x: {"AP50": x, "AP": x, "AP50_new": x, "n_images": [10] * len(x)}
    return {"E0": f(a), "E5": f(b), "E6": f(c)}


def test_significance_flag_follows_the_interval():
    base = [0.60, 0.62, 0.58, 0.61, 0.59]
    clearly = [x + 0.05 + e for x, e in zip(base, [0.001, -0.001, 0.0, 0.002, -0.002])]
    noisy = [x + e for x, e in zip(base, [0.03, -0.04, 0.05, -0.03, 0.02])]
    rows = {r["run"]: r for r in cr.compare(make_folds(base, clearly, noisy), "E0")}
    assert rows["E5"]["sig"] is True and rows["E6"]["sig"] is False and rows["E0"]["sig"] is None
    assert rows["E5"]["diff"] == pytest.approx(0.05, abs=0.003)


def test_compare_errors_and_formatting():
    folds = make_folds([0.5] * 5, [0.6] * 5, [0.7] * 5)
    with pytest.raises(KeyError, match="reference"):
        cr.compare(folds, "nope")
    with pytest.raises(KeyError):
        cr.compare(folds, "E0", runs=["zzz"])
    with pytest.raises(ValueError):
        cr.compare(folds, "E0", metric="bogus")
    txt = cr.format_rows(cr.compare(folds, "E0", "AP50", ["E0", "E5"]), "E0", "AP50")
    assert txt.count("\n") == 3 and "| E5 | 0.600 | 0.000 | +0.100 |" in txt and "| E0 | 0.500" in txt


def test_cli_end_to_end_and_missing_file(tmp_path, capsys):
    wd = tmp_path / "wd"; (wd / "results").mkdir(parents=True)
    folds = make_folds([0.5, 0.6, 0.55, 0.52, 0.58], [0.55, 0.66, 0.6, 0.58, 0.62], [0.6] * 5)
    (wd / "results" / "ablation.json").write_text(json.dumps({k: {"final_folds": v} for k, v in folds.items()}
                                                              | {"legacy_run": {"final_AP50": 0.1}}))
    out = tmp_path / "t.md"
    assert cr.main([str(wd), "--ref", "E0", "--out", str(out)]) == 0
    txt = capsys.readouterr().out
    assert "E5" in txt and "legacy_run" not in txt and "5 disjoint image folds" in txt and out.exists()
    with pytest.raises(SystemExit, match="not found"):
        cr.main([str(tmp_path / "x")])
    (wd / "results" / "ablation.json").write_text(json.dumps({"E0": {"final_AP50": 0.1}}))
    with pytest.raises(SystemExit, match="no fold scores"):
        cr.main([str(wd)])

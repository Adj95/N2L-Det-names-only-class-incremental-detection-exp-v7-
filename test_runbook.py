"""test_runbook.py -- the README runbook and the Colab notebook must stay in sync with the real CLI.

Every `run_pipeline.py ...` line in either document is parsed by the real argparse parser, the two documents must
contain the same commands, the order must follow the data flow, and the claimed test count must be the real one.
"""
import json
import os
import re
import shlex
import subprocess
import sys

import pytest

import run_pipeline as rp

HERE = os.path.dirname(os.path.abspath(__file__))


def _readme_blocks():
    txt = open(os.path.join(HERE, "README.md")).read()
    return txt, re.findall(r"```\n(.*?)```", txt, flags=re.S)


def _notebook_cells():
    nb = json.load(open(os.path.join(HERE, "colab_driver.ipynb")))
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _cli_lines(block_text):
    return [l.strip().lstrip("!").strip() for l in block_text.splitlines() if "run_pipeline.py" in l]


def _all_cli_lines():
    _, blocks = _readme_blocks()
    readme = [l for b in blocks for l in _cli_lines(b)]
    nb = [l for c in _notebook_cells() for l in _cli_lines("".join(c["source"]))]
    return readme, nb


def test_every_cli_command_in_the_docs_parses_with_the_real_parser():
    readme, nb = _all_cli_lines()
    assert len(readme) >= 12
    for line in readme + nb:
        argv = shlex.split(line)
        assert argv[:2] == ["python", "run_pipeline.py"], line
        try:
            args = rp.build_parser().parse_args(argv[2:])
        except SystemExit:
            pytest.fail(f"documented command does not parse: {line}")
        assert args.func is not None


def test_readme_and_notebook_contain_identical_commands():
    readme, nb = _all_cli_lines()
    assert readme == nb


def test_every_subcommand_is_documented():
    readme, _ = _all_cli_lines()
    used = {shlex.split(l)[2] for l in readme}
    assert used == {"fetch", "gate-clip", "make-yolo-data", "train-yolo", "extract", "gate-recall", "run",
                    "repro-cifar", "selftest", "diagnose", "tune", "baseline", "loo", "seeds", "sweep", "perclass", "variants", "clone"}


def test_documented_order_follows_the_data_flow():
    readme, _ = _all_cli_lines()
    seq = [(shlex.split(l)[2], rp.build_parser().parse_args(shlex.split(l)[2:]).workdir) for l in readme]
    first = [c for c, w in seq if w == "/content/work" or c == "selftest"]
    order = ["selftest", "repro-cifar", "fetch", "gate-clip", "make-yolo-data", "train-yolo", "extract", "gate-recall", "run"]
    assert [c for c in order if c in first] == sorted(set(first), key=order.index)           # each appears in this order
    pos = {c: first.index(c) for c in set(first)}
    assert all(pos[a] < pos[b] for a, b in zip(order[2:], order[3:]))
    # second protocol: fetch -> extract -> gate-recall -> run in its own work dir
    second = [c for c, w in seq if w == "/content/work_4x10"]
    assert second == ["fetch", "extract", "gate-recall", "run"]
    assert any("40+4x10" in l for l in readme)
    # third protocol: Pascal VOC 15+5, the full chain in its own work dir
    third = [c for c, w in seq if w == "/content/work_voc"]
    assert third == ["fetch", "gate-clip", "make-yolo-data", "train-yolo", "extract", "gate-recall", "run",
                     # improvement chain: tune split -> diagnose -> tune -> final table -> external baseline
                     "fetch", "extract", "diagnose", "tune", "run", "loo", "seeds",
                     # round 3 (cache-only): loo f10, perclass, sweep, loo lt_all, tune s1, loo s1, tune s2, loo s2
                     "loo", "perclass", "sweep", "loo", "tune", "loo", "tune", "loo",
                     # variants on the tune images, their confirmation on eval, the final split (fetch + extract); the frozen final run now
                     # lives in the pad 0.0 work dir (checked in test_documented_final_split_variants_and_clone_steps)
                     "variants", "variants", "fetch", "extract", "baseline"]
    # the tune split must exist before anything that reads it, and the tuned run must come after `tune`
    assert third.index("diagnose") > third.index("extract", 7) and third.index("tune") > third.index("diagnose")
    assert third.index("run", 7) > third.index("tune") and third.index("loo") > third.index("run", 7)   # loo/seeds need tune.json
    r3 = third[third.index("seeds") + 1:third.index("variants")]
    assert r3.index("sweep") > r3.index("perclass") > r3.index("loo") and r3.count("tune") == 2 and r3.count("loo") == 4
    # variants live in their own work dirs
    assert [c for c, w in seq if w == "/content/work_voc_world"] == ["fetch", "extract", "gate-recall", "run"]
    assert [c for c, w in seq if w == "/content/work_voc_r1"] == ["fetch"]
    assert [c for c, w in seq if w == "/content/work_custom"] == ["fetch"]


def test_documented_paths_are_consistent():
    readme, _ = _all_cli_lines()
    parsed = [rp.build_parser().parse_args(shlex.split(l)[2:]) for l in readme if "selftest" not in l]
    assert {a.workdir for a in parsed} == {"/content/work", "/content/work_4x10", "/content/work_voc", "/content/work_voc_r1",
                                           "/content/work_voc_world", "/content/work_custom", "/content/work_voc_pad03", "/content/work_voc_pad00", "/content/work_voc_clipL14"}
    fetches = [a for a in parsed if a.cmd == "fetch"]
    assert {a.coco_root for a in fetches if a.dataset == "coco"} == {"/content/coco"}
    voc = [a for a in fetches if a.dataset == "voc" and a.workdir == "/content/work_voc"]
    assert len(voc) == 3 and all(a.voc_root == "/content/voc" and rp.scheme_of(a) == "15+5" for a in voc)
    assert all(a.voc_protocol == "auto" for a in voc)
    # the second VOC fetch only ADDS the tune split: every other setting must equal the first (else train/eval ids would change)
    first, again, with_final = voc
    assert (first.n_tune, again.n_tune, with_final.n_tune) == (0, 500, 500) and (first.n_final, again.n_final, with_final.n_final) == (0, 0, 2000)
    for k in ("n_train_per_task", "n_eval", "n_yolo_train", "n_yolo_val", "seed", "tasks", "voc_root", "voc_protocol"):
        assert getattr(first, k) == getattr(again, k) == getattr(with_final, k), k
    rnd = [a for a in fetches if a.workdir == "/content/work_voc_r1"][0]
    assert rnd.tasks == "random:15+5:1" and rnd.n_tune == 500
    cust = [a for a in fetches if a.dataset == "custom"][0]
    assert cust.tasks == "0,1,2|3,4" and cust.n_tune == 300
    cifar = [a for a in parsed if a.cmd == "repro-cifar"][0]
    assert cifar.n2l_repo == "/content/N2L" and cifar.clip_weights == "laion400m_e32"
    assert [a for a in parsed if a.cmd == "train-yolo"][0].yolo_init == "scratch"       # the clean protocol is the default path


def test_notebook_is_valid_and_has_a_gpu_marker():
    nb = json.load(open(os.path.join(HERE, "colab_driver.ipynb")))
    assert nb["nbformat"] == 4 and nb["metadata"]["accelerator"] == "GPU"
    for c in nb["cells"]:
        assert isinstance(c["source"], list) and "".join(c["source"]).strip()
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None


def test_claimed_test_count_matches_reality():
    txt, _ = _readme_blocks()
    m = re.search(r"(\d+) passed(?:, (\d+) skipped)?", txt)
    claimed = {int(m.group(1)) + int(m.group(2) or 0)}
    out = subprocess.run([sys.executable, "-B", "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                         cwd=HERE, capture_output=True, text=True, timeout=300).stdout
    real = int(re.search(r"(\d+) tests? collected", out).group(1))
    assert claimed == {real}, f"README claims {claimed}, suite has {real}"


# ---------------------------------------------------------------- Kaggle docs
def _kaggle_md_blocks():
    txt = open(os.path.join(HERE, "KAGGLE.md")).read()
    return txt, re.findall(r"```\n(.*?)```", txt, flags=re.S)


def _kaggle_cells():
    nb = json.load(open(os.path.join(HERE, "kaggle_driver.ipynb")))
    return nb, [c for c in nb["cells"] if c["cell_type"] == "code"]


def _kaggle_cli(text):
    return [re.split(r"\s+\|\|?\s+", l.strip())[0] for l in text.splitlines() if l.strip().startswith("python run_pipeline.py")]


def test_kaggle_commands_parse_and_notebook_matches_markdown():
    txt, blocks = _kaggle_md_blocks()
    md = [l for b in blocks for l in _kaggle_cli(b)]
    nb_, cells = _kaggle_cells()
    nb = [l for c in cells for l in _kaggle_cli("".join(c["source"]))]
    assert len(md) >= 9
    # the notebook repeats the restore cell and the md also shows a COCO 70+10 fetch, so compare as sets of the shared chain
    assert [l for l in md if "work_voc" in l] == [l for l in nb if "work_voc" in l]
    for line in md:
        argv = shlex.split(line)
        assert argv[:2] == ["python", "run_pipeline.py"], line
        try:
            args = rp.build_parser().parse_args(argv[2:])
        except SystemExit:
            pytest.fail(f"Kaggle command does not parse: {line}")
        assert args.func is not None


def test_kaggle_chain_order_paths_and_safety():
    _, blocks = _kaggle_md_blocks()
    chain = [b for b in blocks if "train-yolo" in b][0]
    assert chain.startswith("%%bash") and "set -e" in chain
    assert "CUDA_VISIBLE_DEVICES=0" in chain
    cmds = [shlex.split(l)[2] for l in _kaggle_cli(chain)]
    assert cmds == ["selftest", "fetch", "gate-clip", "make-yolo-data", "train-yolo", "extract", "gate-recall", "run"]
    parsed = [rp.build_parser().parse_args(shlex.split(l)[2:]) for l in _kaggle_cli(chain) if "selftest" not in l]
    assert {a.workdir for a in parsed} == {"/kaggle/working/work_voc"}
    assert parsed[0].dataset == "voc" and parsed[0].voc_root == "/kaggle/input"
    assert [a for a in parsed if a.cmd == "train-yolo"][0].yolo_init == "scratch"
    # every stage banner and a tar per stage, so a crash still leaves the earlier stages downloadable
    for b in ("=== SETUP ===", "=== FETCH ===", "=== GATE-CLIP ===", "=== YOLO ===", "=== EXTRACT + GATE-RECALL ===",
              "=== RUN ===", "=== ALL DONE ==="):
        assert b in chain
    for stage in ("fetch", "yolo", "extract", "run"):
        assert f"work_voc_after_{stage}.tar" in chain


def test_kaggle_run_uses_only_real_ablation_names():
    import pipeline as pl
    import dataclasses
    import tune as tn
    from inference import InferConfig
    d = {"E5": {**tn.default_params(), "infer": dataclasses.asdict(InferConfig())}, "E0": {"infer": dataclasses.asdict(InferConfig())}}
    _, blocks = _kaggle_md_blocks()
    runs = [l for b in blocks for l in _kaggle_cli(b) if " run " in l]
    assert len(runs) >= 2
    for line in runs:
        a = rp.build_parser().parse_args(shlex.split(line)[2:])
        assert set(a.only) <= set(pl.ablation_configs([[0], [1]], d)), a.only
        if not a.tuned:                                           # without --tuned the tuned names do not exist
            assert set(a.only) <= set(pl.ablation_configs([[0], [1]])), a.only


def test_kaggle_notebook_is_valid_and_restore_cell_matches_markdown():
    nb, cells = _kaggle_cells()
    assert nb["nbformat"] == 4
    for c in nb["cells"]:
        assert isinstance(c["source"], list) and "".join(c["source"]).strip()
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
    txt, _ = _kaggle_md_blocks()
    assert "work_voc_after_extract.tar" in "".join("".join(c["source"]) for c in cells)
    assert "Save & Run All (Commit)" in txt


# ---------------------------------------------------------------- improvement chain (steps 16-22)
def test_extract_tune_uses_the_same_settings_as_the_original_extract():
    readme, _ = _all_cli_lines()
    ex = [rp.build_parser().parse_args(shlex.split(l)[2:]) for l in readme if " extract " in l and "work_voc " in l + " "]
    ex = [a for a in ex if a.workdir == "/content/work_voc"]
    assert len(ex) == 3 and ex[1].splits == ["tune"] and ex[2].splits == ["final"] and ex[0].splits is None
    for k in ("conf", "max_det", "max_boxes", "pad", "proposer"):
        assert getattr(ex[0], k) == getattr(ex[1], k) == getattr(ex[2], k), k     # else the tune/final cache would be refused / inconsistent


def test_tuned_run_and_compare_commands_are_consistent():
    import pipeline as pl
    readme, _ = _all_cli_lines()
    runs = [rp.build_parser().parse_args(shlex.split(l)[2:]) for l in readme if " run " in l]
    dg = [rp.build_parser().parse_args(shlex.split(l)[2:]) for l in readme if " diagnose " in l]
    assert len(dg) == 1 and dg[0].eval_on == "tune"                   # diagnose on the tune images, never the eval images
    tuned = [a for a in runs if a.tuned and a.workdir == "/content/work_voc" and a.eval_on == "eval"]
    assert len(tuned) == 1 and tuned[0].tuned == "/content/work_voc/results/tune.json"
    assert {"E5_tuned", "E0_zeroshot_tuned", "B_plain_ridge"} <= set(tuned[0].only)
    import tune as tn
    from inference import InferConfig
    import dataclasses
    d = {"E5": {**tn.default_params(), "infer": dataclasses.asdict(InferConfig())}, "E0": {"infer": dataclasses.asdict(InferConfig())}}
    assert set(tuned[0].only) <= set(pl.ablation_configs([[0], [1]], d))
    txt, _ = _readme_blocks()
    import compare_runs as cr
    lines = re.findall(r"python compare_runs\.py [^\n]+", txt)
    assert len(lines) >= 3
    for line in lines:
        ns = cr.build_parser().parse_args(shlex.split(line)[2:])
        assert ns.ref == "E0_zeroshot_tuned" and ns.metric in cr.METRICS and ns.workdir == "/content/work_voc"
    assert {cr.build_parser().parse_args(shlex.split(l)[2:]).metric for l in lines} == {"AP50", "AP50_new"}


def test_human_decision_steps_are_flagged_in_the_readme():
    txt, _ = _readme_blocks()
    assert txt.count("[YOU DECIDE]") >= 2 and "Decide with your guide first" in txt
    assert "Do not put this in the main results table" in txt


def test_kaggle_improvement_cell_order_and_names():
    import pipeline as pl
    _, blocks = _kaggle_md_blocks()
    chain = [b for b in blocks if "=== TUNE SPLIT ===" in b][0]
    assert chain.startswith("%%bash") and "set -e" in chain
    cmds = [shlex.split(l)[2] for l in _kaggle_cli(chain)]
    assert cmds == ["selftest", "fetch", "extract", "diagnose", "tune", "run", "baseline"]
    assert chain.index("=== TUNE SPLIT ===") < chain.index("=== DIAGNOSE ===") < chain.index("=== TUNE ===") < chain.index("=== FINAL TABLE ===")
    parsed = {c: rp.build_parser().parse_args(shlex.split(l)[2:]) for c, l in
              ((shlex.split(l)[2], l) for l in _kaggle_cli(chain) if "selftest" not in l)}
    assert {a.workdir for a in parsed.values()} == {"/kaggle/working/work_voc"}
    assert parsed["fetch"].n_tune == 500 and parsed["extract"].splits == ["tune"] and parsed["diagnose"].eval_on == "tune"
    assert parsed["run"].tuned == "/kaggle/working/work_voc/results/tune.json"
    # settings that must match the original chain so the tune cache is accepted and eval/train ids do not move
    orig = [b for b in blocks if "=== GATE-CLIP ===" in b][0]
    o = {shlex.split(l)[2]: rp.build_parser().parse_args(shlex.split(l)[2:]) for l in _kaggle_cli(orig) if "selftest" not in l}
    for k in ("n_train_per_task", "n_eval", "n_yolo_train", "n_yolo_val", "seed", "tasks", "dataset", "voc_root"):
        assert getattr(o["fetch"], k) == getattr(parsed["fetch"], k), k
    for k in ("conf", "max_det", "max_boxes", "pad"):
        assert getattr(o["extract"], k) == getattr(parsed["extract"], k), k
    d = {"E5": {**__import__("tune").default_params(), "infer": {}}, "E0": {"infer": {}}}
    assert set(parsed["run"].only) <= set(pl.ablation_configs([[0], [1]], d))
    assert "work_voc_after_improve.tar" in chain
    # restores only when the run is not already in this session, and says what to attach when it is missing
    assert "already here" in chain and "attach the first version" in chain


def test_documented_loo_and_seeds_commands_and_kaggle_cell4():
    readme, _ = _all_cli_lines()
    loo = [rp.build_parser().parse_args(shlex.split(l)[2:]) for l in readme if " loo " in l and "--tag" not in l]
    seeds = [rp.build_parser().parse_args(shlex.split(l)[2:]) for l in readme if " seeds " in l]
    assert len(loo) == 1 and len(seeds) == 1 and loo[0].workdir == seeds[0].workdir == "/content/work_voc"
    assert seeds[0].n_seeds == 5 and seeds[0].train_frac == 0.8
    _, blocks = _kaggle_md_blocks()
    c4 = [b for b in blocks if "=== LOO ===" in b][0]
    assert c4.startswith("%%bash") and "set -e" in c4
    cmds = [shlex.split(l)[2] for l in _kaggle_cli(c4)]
    assert cmds == ["selftest", "loo", "seeds"]
    assert c4.index("=== RESTORE ===") < c4.index("=== LOO ===") < c4.index("=== SEEDS ===")
    assert "work_voc_after_improve.tar" in c4 and "work_voc_after_loo.tar" in c4 and "tune.json" in c4
    # must not pick an older package copy (e.g. inside the previous notebook's output): select by the loo command itself
    assert 'grep -q cmd_loo "$f"' in c4 and "tune.py" not in c4
    for l in _kaggle_cli(c4):
        if "selftest" not in l:
            assert rp.build_parser().parse_args(shlex.split(l)[2:]).workdir == "/kaggle/working/work_voc"


def test_documented_round3_commands_and_kaggle_cell5():
    readme, _ = _all_cli_lines()
    P = lambda l: rp.build_parser().parse_args(shlex.split(l)[2:])
    loos = [P(l) for l in readme if " loo " in l and "--tag" in l and P(l).workdir == "/content/work_voc"]   # round 3 (pad 0.1); pad 0.0 repeats it in 19i
    assert [(a.tag, a.n_folds, a.label_templates, bool(a.tuned)) for a in loos] == \
        [("f10", 10, 1, False), ("lt_all", 10, 0, False), ("s1", 10, 1, True), ("s2", 10, 1, True)]
    tunes = [P(l) for l in readme if " tune " in l and "--tag" in l and P(l).workdir == "/content/work_voc"]
    assert [(a.tag, a.tune_seed, a.n_trials) for a in tunes] == [("s1", 1, 20), ("s2", 2, 20)]
    for a in loos:                                       # a --tuned file must be the one the matching `tune --tag` writes
        if a.tuned:
            assert a.tuned.endswith(f"results/tune_{a.tag}.json")
    _, blocks = _kaggle_md_blocks()
    c5 = [b for b in blocks if "=== SWEEP ===" in b][0]
    assert c5.startswith("%%bash") and 'grep -q cmd_perclass "$f"' in c5 and "tune.py" not in c5
    banners = ["=== RESTORE ===", "=== LOO 10 FOLDS + GROUPS ===", "=== PER CLASS ===", "=== SWEEP ===", "=== LOO ALL LABEL TEMPLATES ===",
               "=== TUNE SEED 1 ===", "=== LOO WITH TUNE SEED 1 ===", "=== TUNE SEED 2 ===", "=== LOO WITH TUNE SEED 2 ===", "=== RESULT FILES ===", "=== ALL DONE ==="]
    assert [c5.index(b) for b in banners] == sorted(c5.index(b) for b in banners)
    assert "work_voc_after_improve.tar" in c5 and "work_voc_after_round3.tar" in c5
    cli = _kaggle_cli(c5)
    assert [shlex.split(l)[2] for l in cli] == ["selftest", "loo", "perclass", "sweep", "loo", "tune", "loo", "tune", "loo"]
    for l in cli:
        if "selftest" not in l:
            assert P(l).workdir == "/kaggle/working/work_voc"
    # every stage is allowed to fail without stopping the others, and says so
    stage_lines = [l for l in c5.splitlines() if l.startswith("python run_pipeline.py") and "selftest" not in l]
    assert len(stage_lines) == 8 and all("|| echo" in l and "FAILED" in l for l in stage_lines)
    assert "set -e" not in c5


def test_documented_final_split_variants_and_clone_steps():
    import pipeline as pl
    readme, _ = _all_cli_lines()
    P = lambda l: rp.build_parser().parse_args(shlex.split(l)[2:])
    fin = [P(l) for l in readme if " run " in l and "--eval_on final" in l]
    assert len(fin) == 1 and fin[0].workdir == "/content/work_voc_pad00" and fin[0].tuned == "/content/work_voc_pad00/results/tune.json"   # frozen run of the MAIN (pad 0.0) pipeline
    assert not fin[0].allow_repeat_final and set(fin[0].only) == {"E0_zeroshot_tuned", "E5_tuned", "E6_gt_ceiling_tuned"}
    fetches = [P(l) for l in readme if " fetch " in l and "--n_final" in l]
    assert len(fetches) == 1 and fetches[0].n_final == 2000 and fetches[0].n_tune == 500 and fetches[0].n_eval == 2000
    v = [P(l) for l in readme if " variants " in l]
    assert len(v) == 2 and all(x.workdir == "/content/work_voc" and x.n_folds == 10 and x.tag == "v1" for x in v)
    assert [x.on for x in v] == ["tune", "eval"] and v[0].only is None and v[1].only == ["soft_1", "topfrac_0.5"]   # confirmation names its candidates
    # clones: every command of a clone repeats its own extraction flags; tuned files are the clone's own
    base = [P(l) for l in readme if " extract " in l and P(l).workdir == "/content/work_voc"][0]
    for name, flag in (("pad03", ("pad", 0.3)), ("pad00", ("pad", 0.0)), ("clipL14", ("clip_model", "ViT-L-14"))):
        wd = f"/content/work_{'voc_' + name}"
        cl = [P(l) for l in readme if P(l).workdir == wd]
        assert [a.cmd for a in cl][:4] == ["clone", "extract", "tune", "run"], name
        assert cl[0].src == "/content/work_voc" and cl[3].tuned == f"{wd}/results/tune.json"
        assert set(cl[3].only) <= set(pl.ablation_configs([[0], [1]], {"E5": {**__import__("tune").default_params(), "infer": {}}, "E0": {"infer": {}}}))
        for a in cl[1:4]:
            if flag[0] == "clip_model":
                assert a.clip_model == "ViT-L-14", (name, a.cmd)
        ex = cl[1]
        assert getattr(ex, flag[0]) == flag[1]
        for k in ("conf", "max_det", "max_boxes", "proposer"):
            assert getattr(ex, k) == getattr(base, k), (name, k)             # only the named setting differs from the original extract
    assert [a for a in [P(l) for l in readme if P(l).workdir == "/content/work_voc_pad03"]][1].clip_model == "ViT-B-16"
    # the pad 0.0 pipeline: after its clone chain come the stability checks (19i), then, last of all, the single frozen final run (19h)
    p0 = [P(l) for l in readme if P(l).workdir == "/content/work_voc_pad00"]
    assert [a.cmd for a in p0] == ["clone", "extract", "tune", "run", "loo", "perclass", "tune", "loo", "tune", "loo", "run"]
    assert [a.tag for a in p0 if a.cmd in ("tune", "loo")][1:] == ["f10", "s1", "s1", "s2", "s2"] and [a.tune_seed for a in p0 if a.cmd == "tune"][1:] == [1, 2]
    assert p0[5 - 1].n_folds == 10 and p0[-1].eval_on == "final" and p0[-1].tuned.endswith("/results/tune.json")   # the FROZEN settings = the automatic tune.json


def test_kaggle_cell6_variants_and_final_split_cache():
    _, blocks = _kaggle_md_blocks()
    c6 = [b for b in blocks if "=== FINAL SPLIT ===" in b][0]
    P = lambda l: rp.build_parser().parse_args(shlex.split(l)[2:])
    assert c6.startswith("%%bash") and 'grep -q cmd_variants "$f"' in c6 and "tune.py" not in c6 and "set -e" not in c6
    assert "work_voc_after_improve.tar" in c6 and "work_voc_after_final_split.tar" in c6
    cmds = [shlex.split(l)[2] for l in _kaggle_cli(c6)]
    assert cmds == ["selftest", "variants", "fetch", "extract"]
    ban = ["=== RESTORE ===", "=== VARIANTS", "=== FINAL SPLIT ===", "=== RESULT FILES ===", "=== ALL DONE ==="]
    assert [c6.index(b) for b in ban] == sorted(c6.index(b) for b in ban)
    v, f, e = [P(l) for l in _kaggle_cli(c6) if "selftest" not in l]
    assert v.tag == "v1" and v.n_folds == 10 and v.workdir == "/kaggle/working/work_voc"
    assert f.n_final == 2000 and f.n_tune == 500 and f.n_eval == 2000                       # the same draw as the original fetch + the final split
    assert e.splits == ["final"] and (e.conf, e.max_det, e.max_boxes) == (0.001, 300, 60)   # the settings of Cell 2's extraction
    # the final extraction only runs if eval and tune ids are provably unchanged, and the ids are saved before the re-fetch
    assert c6.index("splits_before_final.json") < c6.index("run_pipeline.py fetch") < c6.index("ids changed") < c6.index("--splits final")
    assert "eval_ids" in c6 and "tune_ids" in c6 and "SKIPPED" in c6
    stages = [l for l in c6.splitlines() if l.startswith("python run_pipeline.py") and "selftest" not in l]
    assert all("|| echo" in l and "FAILED" in l for l in stages)


def test_kaggle_cell6b_confirms_only_the_named_variants_on_the_eval_images():
    _, blocks = _kaggle_md_blocks()
    c = [b for b in blocks if "=== CONFIRM ON EVAL" in b]
    assert len(c) == 1
    c = c[0]
    P = lambda l: rp.build_parser().parse_args(shlex.split(l)[2:])
    assert c.startswith("%%bash") and 'grep -q variants_eval "$f"' in c and "set -e" not in c
    cmds = [P(l) for l in _kaggle_cli(c) if "selftest" not in l]
    assert [a.cmd for a in cmds] == ["variants"]
    v = cmds[0]
    assert v.on == "eval" and v.only == ["soft_1", "topfrac_0.5"] and v.n_folds == 10 and v.tag == "v1"
    assert v.workdir == "/kaggle/working/work_voc"
    import variants as vr
    assert set(v.only) <= set(vr.VARIANTS)
    assert "work_voc_after_improve.tar" in c and "work_voc_after_confirm.tar" in c and "fetch" not in c
    assert "|| echo" in c and "stage CONFIRM FAILED" in c
    assert "variants_eval" in open(os.path.join(os.path.dirname(__file__), "run_pipeline.py")).read()


def _cell(marker):
    _, blocks = _kaggle_md_blocks()
    c = [b for b in blocks if marker in b]
    assert len(c) == 1, marker
    return c[0]


def test_kaggle_cell8_pad00_followup_is_cache_only_and_uses_both_tars():
    c = _cell("work_voc_pad00_after_followup.tar")
    P = lambda l: rp.build_parser().parse_args(shlex.split(l)[2:])
    w = "/kaggle/working/work_voc_pad00"
    assert c.startswith("%%bash") and 'grep -q cmd_variants "$f"' in c and "set -e" not in c
    assert "work_voc_after_improve.tar" in c and "work_voc_pad00_after_variant.tar" in c and "work_voc_pad00_after_followup.tar" in c
    cmds = [P(l) for l in _kaggle_cli(c) if "selftest" not in l]
    assert [a.cmd for a in cmds] == ["loo", "perclass", "tune", "loo", "tune", "loo"]            # cache only: no fetch, no extract
    assert all(a.workdir == w for a in cmds) and cmds[0].n_folds == 10 and cmds[0].tag == "f10"
    assert [a.tune_seed for a in cmds if a.cmd == "tune"] == [1, 2] and [a.tag for a in cmds if a.cmd == "tune"] == ["s1", "s2"]
    assert [a.tuned for a in cmds if a.cmd == "loo"][1:] == [f"{w}/results/tune_s1.json", f"{w}/results/tune_s2.json"]
    assert all(a.clip_model == "ViT-B-16" for a in cmds if hasattr(a, "clip_model"))
    assert "extract" not in c.replace("extract_features", "") and "fetch" not in c
    assert all("|| echo" in l and "FAILED" in l for l in c.splitlines() if l.startswith("python run_pipeline.py") and "selftest" not in l)


def test_kaggle_cell9_builds_the_final_cache_at_pad00_with_the_same_extract_settings():
    c = _cell("=== FINAL SPLIT AT PAD 0.0")
    P = lambda l: rp.build_parser().parse_args(shlex.split(l)[2:])
    S, w = "/kaggle/working/work_voc", "/kaggle/working/work_voc_pad00"
    assert c.startswith("%%bash") and 'grep -q cmd_variants "$f"' in c and "set -e" not in c
    assert "work_voc_after_improve.tar" in c and "work_voc_pad00_after_variant.tar" in c and "work_voc_pad00_after_final_split.tar" in c
    cmds = [P(l) for l in _kaggle_cli(c) if "selftest" not in l]
    assert [a.cmd for a in cmds] == ["fetch", "extract"]
    fetch, ex = cmds
    assert fetch.workdir == S and fetch.n_final == 2000 and fetch.n_tune == 500 and fetch.n_eval == 2000   # images/ids come from the ORIGINAL work dir
    assert ex.workdir == w and ex.splits == ["final"]
    # same extract settings as the pad00 clone (a partial extract must match its clip.json or it refuses)
    ref = [P(l) for l in _kaggle_cli(_cell("=== CLONE pad00 ==="))][3]
    assert ref.cmd == "extract" and ref.workdir == w
    for k in ("conf", "max_det", "max_boxes", "pad", "proposer"):
        assert getattr(ex, k) == getattr(ref, k), k
    # the eval/tune ids are verified unchanged BEFORE the pad00 splits are replaced and before anything is extracted
    i_chk, i_cp, i_ex = c.index("ids changed"), c.index(f"cp {S}/splits.json {w}/splits.json"), c.index("--splits final")
    assert i_chk < i_cp < i_ex and f"cp {w}/splits.json {w}/splits_before_final.json" in c and "SKIPPED" in c


def test_kaggle_cell7_extraction_variants_change_exactly_one_extract_setting():
    _, blocks = _kaggle_md_blocks()
    P = lambda l: rp.build_parser().parse_args(shlex.split(l)[2:])
    cells = {n: [b for b in blocks if f"=== CLONE {n} ===" in b] for n in ("pad03", "pad00", "boxes100", "clipL14")}
    assert all(len(v) == 1 for v in cells.values())
    want = {"pad03": ("pad", 0.3), "pad00": ("pad", 0.0), "boxes100": ("max_boxes", 100), "clipL14": ("clip_model", "ViT-L-14")}
    for n, (c,) in cells.items():
        assert c.startswith("%%bash") and 'grep -q cmd_variants "$f"' in c and "set -e" not in c
        w = f"/kaggle/working/work_voc_{n}"
        cmds = [P(l) for l in _kaggle_cli(c) if "selftest" not in l]
        assert [a.cmd for a in cmds] == ["fetch", "clone", "extract", "tune", "run"], n
        fetch, clone, ex, tune, run = cmds
        assert fetch.workdir == "/kaggle/working/work_voc" and fetch.n_tune == 500 and clone.src == "/kaggle/working/work_voc"
        assert clone.workdir == ex.workdir == tune.workdir == run.workdir == w
        assert run.tuned == f"{w}/results/tune.json" and set(run.only) == {"E0_zeroshot_tuned", "E5_tuned", "E6_gt_ceiling_tuned"}
        base = dict(conf=0.001, max_det=300, max_boxes=60, pad=0.0 if n == "boxes100" else 0.1, clip_model="ViT-B-16")   # 7c is judged against 7b (pad 0.0)
        for k, v0 in base.items():
            expect = want[n][1] if k == want[n][0] else v0
            assert getattr(ex, k) == expect, (n, k)
        for a in (tune, run):                                                # later commands repeat the CLIP model, or are refused
            assert a.clip_model == ex.clip_model, (n, a.cmd)
        assert "work_voc_after_improve.tar" in c and f"work_voc_{n}_after_variant.tar" in c
        assert c.count("compare_runs.py") == 2 and "--metric AP50_new" in c
        assert all("|| echo" in l and "FAILED" in l for l in c.splitlines() if l.startswith("python run_pipeline.py") and "selftest" not in l)

# N2L-Det-exp (EXPERIMENTAL copy of the frozen v6 pipeline)

This folder is a separate code base for round 1+ experiments. The frozen pipeline is `n2l_det_v6.zip` (Kaggle dataset `n2l-det-v6`);
its file checksums are in `FROZEN_MANIFEST_v6.sha256`. Kaggle cells for this copy are in `KAGGLE_EXP.md` (they look for the marker
`N2L_EXP_MARKER_v7` and can never pick the frozen code).

Exp v7 (19 Nov): every step's per-class AP50 is stored in each run's curve, and each run reports the pre-registered forgetting
measures (`forgetting`: old_class = task-1 classes after task 1 minus after the last task; avg = Chaudhry et al. over every class
learned before the last task; by_task). `variants --on final --only ROWS` scores fixed rows once on the final split (logged in
results/final_log.json; a second look needs --allow_repeat_final). Reported numbers are otherwise identical to v6.
Exp v6 (19 Nov): adds `tune --objective final_AP50_base` (settings chosen on base-class AP50 only, so no new-class labels are
used for tuning; write it with `--tag base` so the frozen tune.json is never touched) and `train-yolo --yolo_init imagenet`
(detector built from yolov8{size}.yaml with the backbone copied from the ImageNet-1k classifier yolov8{size}-cls.pt).
Defaults unchanged: every earlier command gives the same numbers.
Exp v5 (16 Nov): adds `ovd` (open-vocabulary detector baselines YOLO-World / OWLv2 / Grounding DINO, scored with the same
images, folds and COCOeval; reference rows; own final-split log `final_log_ovd.json`; `--restore_voc` re-creates VOC image links
after a tar restore). No method code changed.
Exp v4 (16 Nov): adds forgetting baselines (`N2LConfig.memory` = 'none' / 'none_newonly'; variants `mem_none`,
`mem_newonly`, `headonly`, `headonly_mem_newonly`), ablations only, default 'full' = unchanged.
Exp v3 (16 Nov): adds round-2 refinement options (`N2LConfig.refine_gate`, `anchor`, `refine_mode`; variants `rf_*`), all OFF
by default (tests compare against the frozen refinement). Exp v2 (16 Nov): includes the v5.4 per-class AP fix (`evaluate.py` looks each class up by category id; regression test in
`test_step6_eval.py`). Exp v1 reported AP50_new / base / per-class over the WRONG classes on VOC; overall AP50/AP were right.

Changes against v6 (all OFF by default; `test_exp_round1.py` checks that the defaults reproduce v6 exactly):
- `InferConfig.gamma_new`, `clip_obj`, `yolo_floor`: rescoring knobs; `variants` rows `gnew_*`, `clipobj_*`, `yfloor_*` and the same on E0 (`E0__*`).
- `boxdiag` (coverage of the cached proposals, naming of covered objects, oracle ladder) and `propdiag` (coverage of fresh proposals:
  base YOLO / YOLO-World, NMS IoU, up to 300 boxes). Both refuse the final split.

---

# N2L-Det: Naming-to-Learn for class-incremental object detection (YOLOv8 proposals)

Frozen YOLOv8 proposals -> padded crops -> frozen CLIP features -> pseudo-labels from class **names** (+ a background prompt) -> N2L refinement -> bi-level weights -> recursive closed-form ridge head -> blend with zero-shot CLIP -> class-wise NMS. No ground-truth labels are used for the new classes.

## Results so far (measured on Colab / Kaggle T4; base detector YOLOv8-s trained from scratch on base classes only)

| | COCO 40+40 | COCO 70+10 |
|---|---|---|
| CLIP zero-shot on ground-truth crops (all classes) | 0.487 | 0.470 |
| base detector mAP50 | 0.165 | 0.122 |
| proposal recall@0.5, base classes | 0.60 | 0.53 |
| proposal recall@0.5, **new** classes | **0.31** (1,100 boxes) | **0.32** (445 boxes) |
| boxes per image | 56 | 58 |

Reading: new-class recall is about 0.3 on COCO whichever split is used, so the frozen base-trained detector caps every method. With a leaky
COCO-initialised detector (upper bound only) on 40+40: E0 0.238, E5 0.202, E5 without refinement 0.226, E6 ground-truth head 0.250 AP50
(gamma 0), and base-class forgetting about 0. The 70+10 detector was undertrained (5k images, 30 epochs), so that comparison is confounded.
Pascal VOC (steps 10-15) is the benchmark chosen to test whether larger, clearer objects change this.


Free Colab GPU quota runs out; the same runs work as one committed Kaggle notebook: see `KAGGLE.md`. Steps 0-9 are the COCO runbook; steps 10-15 are the Pascal VOC benchmark and the comparison table; steps 16-22 are the improvement and testing chain (held-out tuning, diagnosis, confidence intervals, external baseline, other splits/datasets, proposer variant). Steps marked **[YOU DECIDE]** need a decision from you and your guide before they count for the paper.

## What to do after downloading (step by step)

Run these in order, each code block in its own Colab cell (`colab_driver.ipynb` has them ready). Each step ends with what to expect and when to stop. Do not skip a step: each one validates the one before it.

### Step 0 - Set up Colab (GPU) and unpack

One-time setup. Runtime -> Change runtime type -> GPU (T4 is enough). Upload `n2l_det.zip` to the Colab file panel.

```
!unzip -q -o /content/n2l_det.zip -d /content
%cd /content/n2l_det
!pip -q install numpy scipy pytest pillow pycocotools open_clip_torch ultralytics
!nvidia-smi -L
```

**Expect:** A line like `GPU 0: Tesla T4`. If it says no GPU, switch the runtime type first.

### Step 1 - Run all tests, then the offline self-test (no GPU needed, ~1 min)

Proves the code you downloaded is intact on this machine before any GPU time is spent.

```
!python -m pytest -q
!python run_pipeline.py selftest
```

**Expect:** pytest ends with `428 passed, 1 skipped` (on Kaggle `424 passed, 5 skipped`: four tests need the frozen v6 copy next to this one) (the skipped test needs the official repo; it runs in step 2); selftest prints five `PASS` lines and `SELFTEST OK`.

**Stop / fix if:** Any failure here: stop and send me the full output. Nothing later is meaningful.

### Step 2 - Reproduce your N2L CIFAR-100 result with the new core (validates core + CLIP wrapper)

Your BTP reports A-bar 87.86, A_B 81.17 and zero-shot CLIP A-bar 81.81 (B0Inc10). If the new code, run on the same frozen CLIP, reproduces them, the learner and the image/text preprocessing are right. Uses CLIP ViT-B/16 `laion400m_e32` as the official CIFAR config.

```
!git clone -q https://github.com/zhoujiahuan1991/ICLR2026-N2L /content/N2L
!N2L_REPO=/content/N2L python -m pytest -q test_step2b.py -k official_text_assets
!python run_pipeline.py repro-cifar --workdir /content/work --n2l_repo /content/N2L --cifar_root /content/data
```

**Expect:** The pytest line shows `1 passed` (the official label/template files load: 100 names, 18 templates). Then last line `REPRO  A-bar fused ~87.9  A_B fused ~81.2 | zero-shot A-bar ~81.8`. Within about 1 point of your numbers = pass.

**Stop / fix if:** More than ~1 point off, or a WARNING about class names/order: stop and send me the output. Do not continue to detection with an unvalidated core.

### Step 3 - Choose and download the COCO images (resumable)

Downloads the COCO annotations (~250 MB) and only the images this run needs: 2000 per task, 500 evaluation images, 5000 base-class images for the detector. Re-running skips files already present.

```
!python run_pipeline.py fetch --workdir /content/work --coco_root /content/coco --tasks 40+40 --n_train_per_task 2000 --n_eval 500 --n_yolo_train 5000 --n_yolo_val 300
```

**Expect:** `tasks: [40, 40] | images/task: [2000, 2000] | eval images: 500`.

**Stop / fix if:** A download error list: just re-run the same command.

### Step 4 - GATE E0 - zero-shot CLIP accuracy on ground-truth box crops

The number every later result is judged against: if CLIP cannot name a perfectly-boxed object, nothing built on its text can.

```
!python run_pipeline.py gate-clip --workdir /content/work
```

**Expect:** Prints `all` (80-way) and `task0_within`, `task1_within`. I have NOT measured this for COCO; I would expect a broad 50-70% for ViT-B/16 on padded crops.

**Stop / fix if:** Below ~30%, or near chance (1.25%): the crops, class names or CLIP weights are wrong; send me the output.

### Step 5 - Train the base detector on the 40 base classes only

Builds the 40-class YOLO dataset (new-class objects inside those images stay unlabeled = background) and trains YOLOv8-s from scratch, so no knowledge of the 40 new classes leaks in. ~1 h on a T4 (my guess, not measured).

```
!python run_pipeline.py make-yolo-data --workdir /content/work
!python run_pipeline.py train-yolo --workdir /content/work --yolo_init scratch --yolo_size s --yolo_epochs 30
```

**Expect:** `best weights: .../best.pt | init: scratch`.

**Stop / fix if:** Shortcut for a first look only: `--yolo_init coco` starts from COCO-pretrained weights. That LEAKS the new classes into the detector, so its numbers are an upper bound and must be labelled as such. Redo with `scratch` for anything you report.

### Step 6 - Extract proposals and CLIP features (the slow, GPU step)

Runs the frozen detector on every selected image, crops each box (10% padding, squash to 224), and stores CLIP features. Everything after this reads the cache, so experiments take minutes. Uses the weights from step 5 automatically.

```
!python run_pipeline.py extract --workdir /content/work
```

**Expect:** `extracted ~3980 train + 500 eval images in ...s` (the 2 tasks share a few images). Roughly 250k crops: tens of minutes on a T4 (my guess). Use `--max_boxes 50` to halve it.

**Stop / fix if:** Out-of-memory: re-run; if it persists, tell me.

### Step 7 - GATE 1 - can the frozen detector even box the NEW classes?

Recall of ground-truth boxes by the detector's proposals, per task. Task 0 = base classes it was trained on; task 1 = classes it never saw.

```
!python run_pipeline.py gate-recall --workdir /content/work
```

**Expect:** `recall@0.5` for task0 and task1. A clear gap (task1 much lower) is the structural cap of the frozen-detector design.

**Stop / fix if:** Task1 recall@0.5 below ~0.4: the frozen proposer is the bottleneck; tell me before running more.

### Step 8 - Run the experiments: zero-shot, ours, ground-truth ceiling, and ablations

E0 zero-shot CLIP; E5 ours (names only, no labels); E6 same head trained on ground-truth labels (upper bound); and ablations that each change exactly one component. Results land in `work/results/ablation.md`.

```
!python run_pipeline.py run --workdir /content/work
!cat /content/work/results/ablation.md
```

**Expect:** A table with A-bar AP, final AP, final AP50 and base-AP50 drop per run. Read it with the gates below. Headline rows use proposal-score weighting `yolo_gamma=0.5` (the default); `*_g0` rows switch it off. Runs made before this default changed used gamma 0, i.e. they correspond to the `*_g0` rows.

### Step 9 - Repeat for the 40+4x10 protocol (reuses the detector and downloaded images)

Same experiment with five tasks (40 base + 4 increments of 10).

```
!python run_pipeline.py fetch --workdir /content/work_4x10 --coco_root /content/coco --tasks 40+4x10 --n_train_per_task 1000 --n_eval 500
!cp /content/work/yolo_best.json /content/work_4x10/
!python run_pipeline.py extract --workdir /content/work_4x10
!python run_pipeline.py gate-recall --workdir /content/work_4x10
!python run_pipeline.py run --workdir /content/work_4x10
!cat /content/work_4x10/results/ablation.md
```

**Expect:** A second table, with 5 tasks per run.

### Step 10 - Pascal VOC 15+5: convert VOC and choose the images (second benchmark)

VOC has large, clear objects and only 20 classes, and 15+5 is the standard incremental split (new = potted plant, sheep, sofa, train, tv). `--voc_root` is searched for VOC2007/VOC2012 folders in any layout; if none is found the three official tarballs (~2.9 GB) are downloaded there. The converter writes COCO-style json (VOC `difficult` objects become ignore regions) so every later step is the same as for COCO. Train = VOC07+12 trainval, eval pool = VOC07 test (protocol `07+12`).

```
!python run_pipeline.py fetch --dataset voc --tasks 15+5 --workdir /content/work_voc --voc_root /content/voc --n_train_per_task 2000 --n_eval 2000 --n_yolo_train 12000 --n_yolo_val 500
```

**Expect:** A line `VOC protocol 07+12: train {'images': 16551, ...} | eval pool {'images': 4952, ...}` (exact counts depend on the files found), then `tasks: [15, 5] | images/task: [2000, 2000] | eval images: 2000`. If it prints protocol `12` or `07` the tarballs for the other year were not found: that is NOT the standard benchmark.

**Stop / fix if:** `protocol 12` or `07` when you wanted 07+12; a download error (the official host is slow or down at times): download the tar files by hand into the voc folder and re-run the same command.

### Step 11 - VOC: GATE E0 (CLIP on ground-truth crops, 20 classes)

For VOC the `all` number is accuracy among the 20 VOC classes, not among 80, because only those 20 compete in this run.

```
!python run_pipeline.py gate-clip --workdir /content/work_voc
```

**Expect:** `all`, `task0_within` (15-way), `task1_within` (5-way). I have NOT measured this on VOC; the run decides whether VOC is easy (high) or not.

**Stop / fix if:** Near chance (5%): class names or crops are wrong; send me the output.

### Step 12 - VOC: train the base detector on the 15 base classes (all base images, 40 epochs)

Uses all ~12k base-class images so the detector is not the weak link (the COCO 70+10 detector was undertrained: 5k images, mAP50 0.122). Roughly 1.5 h on a T4 (my estimate from the COCO run: 2.5 it/s).

```
!python run_pipeline.py make-yolo-data --workdir /content/work_voc
!python run_pipeline.py train-yolo --workdir /content/work_voc --yolo_init scratch --yolo_size s --yolo_epochs 40
```

**Expect:** `best weights: .../best.pt | init: scratch`; the validation mAP50 over the 15 base classes should be far above the COCO run's 0.12.

**Stop / fix if:** mAP50 still near zero after 10 epochs: tell me before continuing.

### Step 13 - VOC: extract proposals + CLIP features, then GATE 1

Same as steps 6-7 with a low confidence threshold so more boxes survive (the pass rule is judged on this recall).

```
!python run_pipeline.py extract --workdir /content/work_voc --conf 0.001 --max_det 300 --max_boxes 60
!python run_pipeline.py gate-recall --workdir /content/work_voc
```

**Expect:** `task0` and `task1` recall@0.5. The proposed pass rule for a dataset is new-class (task1) recall@0.5 >= 0.5 (COCO gave 0.31 and 0.32).

**Stop / fix if:** Task1 recall below 0.5: the frozen proposer is again the limit; send me the output before running experiments.

### Step 14 - VOC: experiments (screening subset)

E0 zero-shot, E5 ours, E5 without refinement, and the ground-truth ceiling E6, each with and without proposal-score weighting. Drop `--only ...` to run every ablation.

```
!python run_pipeline.py run --workdir /content/work_voc --only E0_zeroshot E0_zeroshot_g0 E5_ours E5_no_refine E5_yolo_gamma0 E6_gt_ceiling E6_gt_ceiling_g0
!cat /content/work_voc/results/ablation.md
```

**Expect:** The table. E6 minus E0 is the headroom any head can add; E5 minus E0 says whether N2L helped.

### Step 15 - Merge every run into one screening table (CPU, seconds)

One row per dataset/split: E0 crop accuracy, recall on base vs new classes, AP50 for E0 / E5 / E6, the headroom E6-E0, the gap E5-E0 and a PASS/FAIL against the proposed rule (new-class recall@0.5 >= 0.5 and E6-E0 >= +0.03 AP50). The thresholds are a proposal: agree them with your guide BEFORE reading the table. Missing runs/files show `n/a`. Runs made before the gamma default changed used gamma 0 for the same row names.

```
!python merge_results.py coco40+40=/content/work coco70+10=/content/work_70_10 voc15+5=/content/work_voc --out /content/screening.md
!cat /content/screening.md
```

**Expect:** A markdown table with one row per run you list.

### Step 16 - VOC: add a held-out TUNE split (images that are never evaluated on)

Choosing settings on the same images you report on inflates the result. `--n_tune 500` draws 500 extra VOC07-test images that are disjoint from the 2000 eval images; the eval images and the train images do not change (re-running `fetch` with one extra flag is safe). The tune images are then passed through the SAME frozen detector and CLIP (`extract --splits tune`; the settings must match step 13).

```
!python run_pipeline.py fetch --dataset voc --tasks 15+5 --workdir /content/work_voc --voc_root /content/voc --n_train_per_task 2000 --n_eval 2000 --n_yolo_train 12000 --n_yolo_val 500 --n_tune 500
!python run_pipeline.py extract --workdir /content/work_voc --splits tune --conf 0.001 --max_det 300 --max_boxes 60
```

**Expect:** `fetch` prints `tune images: 500` (disjoint from eval); `extract` reports ~500 images. If it says the settings differ from the other caches, copy the exact `--conf/--max_det/--max_boxes` you used in step 13.

**Stop / fix if:** `extract` says the cache settings differ: use the same flags as step 13.

### Step 17 - VOC: DIAGNOSE why the head does not beat zero-shot CLIP (read-only, minutes)

Splits the question into three measurable parts: (A) how accurate the pseudo-labels are as a function of CLIP's confidence and what dropping low-confidence ones would keep (a 'what-if' table); (B) the most frequent pseudo-label confusions; (C) classification accuracy on proposals that overlap a GT box for zero-shot CLIP vs the ridge head vs the fused scores. Ground truth is used only for the report. Run it on the tune images so the eval images stay untouched.

```
!python run_pipeline.py diagnose --workdir /content/work_voc --eval_on tune
```

**Expect:** `results/diagnose.md`. Look for: pseudo-label accuracy rising with confidence (then thresholds help); head accuracy below CLIP accuracy in part C (then refinement/head is the problem, not localisation).

### Step 18 - VOC: TUNE the settings on the tune images (E5 and E0 tuned separately)

Random search over pseudo-label thresholds, N2L rounds, ridge strength, PCA energy and the bi-level weights (trial 0 = our default, trial 1 = no refinement), then a grid over the fusion weight, YOLO^gamma and top-k, for E5 AND, on its own grid, for zero-shot E0, so the baseline is tuned as hard as our method. Ties go to the earlier (simpler) candidate. 20 trials is a few minutes on cached features.

```
!python run_pipeline.py tune --workdir /content/work_voc --n_trials 20 --objective final_AP50
```

**Expect:** `results/tune.md` and `tune.json`; the last line shows tune-split AP50 for E5 default -> tuned and E0 default -> tuned.

**Stop / fix if:** If the best stage-1 row is barely above trial 0, the settings do not matter; if trial 1 (no refinement) is near the top, refinement is not helping.

### Step 19 - VOC: final table on the EVAL images with the tuned settings, plus confidence intervals

Adds `E5_tuned`, `E0_zeroshot_tuned` and the plain-ridge baseline `B_plain_ridge` to the earlier rows (all scored on the eval images, once). `compare_runs.py` splits the eval images into 5 disjoint folds and gives mean +- sd and paired differences with 95% intervals; 'sig' means the interval excludes 0. With 5 folds treat it as a noise check, not a strict test.

```
!python run_pipeline.py run --workdir /content/work_voc --tuned /content/work_voc/results/tune.json --only E0_zeroshot E0_zeroshot_g0 E5_ours E5_no_refine E5_yolo_gamma0 E6_gt_ceiling E6_gt_ceiling_g0 E0_zeroshot_tuned E5_tuned B_plain_ridge
!python compare_runs.py /content/work_voc --ref E0_zeroshot_tuned --metric AP50 --out /content/work_voc/results/compare_AP50.md
!python compare_runs.py /content/work_voc --ref E0_zeroshot_tuned --metric AP50_new --out /content/work_voc/results/compare_AP50_new.md
```

**Expect:** Two tables: AP50 over all classes and AP50 over the 5 new classes, each row vs tuned zero-shot CLIP. This is the table for the paper.

### Step 19b - VOC: which tuned setting carries the gain? (leave-one-out, eval images)

Tuning changed several settings at once, and on VOC it switched N2L refinement and the bi-level weights OFF. This step re-runs the tuned E5 with ONE tuned setting put back to its default at a time (`E5_tuned_minus_<setting>`), plus `E5_tuned_only_train` (tuned training, default inference), `E5_tuned_only_infer` (default training, tuned inference) and the ground-truth ceiling with the tuned settings (`E6_gt_ceiling_tuned`). A row clearly BELOW `E5_tuned` means that setting carries part of the gain; a row equal to it means it is not needed. Only the eval images are used, once each, and nothing is re-tuned.

```
!python run_pipeline.py loo --workdir /content/work_voc
```

**Expect:** `results/loo.md`: four tables (AP50 and new-class AP50, each against `E5_tuned` and against `E0_zeroshot_tuned`) with 95% fold intervals.

**Stop / fix if:** `run `tune` first`: results/tune.json is missing.

### Step 19c - VOC: does the result depend on WHICH training images were used? (5 random subsets)

Re-trains E5 default, E5 tuned, the plain-ridge baseline and the tuned ground-truth ceiling on 5 different random 80% subsets of the cached training images and scores each on the SAME eval images. The training-free zero-shot E0 does not change between subsets. The settings are NOT re-tuned. This measures training-sample variation only: not the detector, the class split or the eval images (the folds cover those).

```
!python run_pipeline.py seeds --workdir /content/work_voc --n_seeds 5 --train_frac 0.8
```

**Expect:** `results/seeds.md`: mean, sd over seeds and a paired interval against `E0_zeroshot_tuned`, for AP50 and new-class AP50.

### Step 19d - VOC: round-3 checks, all cache-only (per-class, gamma x fusion sweep, 10-fold groups, tuning stability, full-template pseudo-labels)

Seven checks on the finished VOC run; none needs a new detector or new features. (1) `loo --n_folds 10 --tag f10`: the leave-one-out table with 10 image folds and GROUP rows (`group.n2l_machinery`, `group.pseudo_filters`, `group.inference`) that put several settings back together, because one reversion alone can hide a gain shared by two settings. (2) `perclass`: AP50 per class for E0 tuned, E5 tuned and the ceiling: does the gain come from a few classes? (3) `sweep`: YOLO^gamma x fusion weight, zero-shot vs head. It is scored on the eval images, so it is DESCRIPTIVE and must not be used to pick settings. (4) `tune --tune_seed 1/2 --tag s1/s2` then `loo --tuned ... --tag s1/s2`: is the tuned setting stable when the random search is repeated, and does the gain survive a different tuned setting? (5) `loo --label_templates 0 --tag lt_all`: pseudo-labels from all 10 prompt templates instead of the first only (the official N2L uses one). `--tag` keeps every earlier result file.

```
!python run_pipeline.py loo --workdir /content/work_voc --n_folds 10 --tag f10
!python run_pipeline.py perclass --workdir /content/work_voc
!python run_pipeline.py sweep --workdir /content/work_voc
!python run_pipeline.py loo --workdir /content/work_voc --n_folds 10 --label_templates 0 --tag lt_all
!python run_pipeline.py tune --workdir /content/work_voc --n_trials 20 --tune_seed 1 --tag s1
!python run_pipeline.py loo --workdir /content/work_voc --tuned /content/work_voc/results/tune_s1.json --n_folds 10 --tag s1
!python run_pipeline.py tune --workdir /content/work_voc --n_trials 20 --tune_seed 2 --tag s2
!python run_pipeline.py loo --workdir /content/work_voc --tuned /content/work_voc/results/tune_s2.json --n_folds 10 --tag s2
```

**Expect:** `loo_f10.md`, `perclass.md`, `sweep.md`, `loo_lt_all.md`, `tune_s1.md`, `loo_s1.md`, `tune_s2.md`, `loo_s2.md` in `results/`.

**Stop / fix if:** `run `tune` first`: results/tune.json is missing (not needed for the `tune` commands themselves).

### Step 19e - VOC: single-change method variants of the tuned E5, judged on the TUNE images (cache-only)

Each variant changes ONE thing in the tuned E5 and is scored on the held-out tune images only, never on eval or final: `min_yolo_*` (drop training boxes with a low detector score), `topfrac_*` (per pseudo-class keep the most confident boxes), `soft_*` (ridge targets mix one-hot with the zero-shot softmax), `ridge_norm` and `geo_fusion` (how the head and CLIP scores are fused), `head_proto` (a class-mean prototype head on the same pseudo-labels: the simplest self-training baseline). They do not depend on each other. The report prints paired fold intervals against `E5_tuned`, and the warning that the best of N differences is optimistic: a winner is only a CANDIDATE until it is confirmed on images it was not chosen on.

```
!python run_pipeline.py variants --workdir /content/work_voc --n_folds 10 --tag v1
!python run_pipeline.py variants --workdir /content/work_voc --on eval --only soft_1 topfrac_0.5 --n_folds 10 --tag v1
```

**Expect:** `results/variants_v1.md`: AP50 and new-class AP50 differences vs `E5_tuned` with intervals, plus a table of what each variant changes. The second command CONFIRMS candidates chosen on the tune images by scoring only them (`--only`) on the eval images (`results/variants_eval_v1.md`); write down which variants you will confirm BEFORE looking at it, and count one as confirmed only if its AP50 interval excludes 0 with a positive sign and AP50_new does not drop.

**Stop / fix if:** `run `tune` first`, or `no tune images`: steps 16 and 18 must be done.

### Step 19f - VOC: create the untouched FINAL split (images nobody has looked at)

Every method choice so far was judged on the tune images or on the eval images (which the loo / sweep / perclass diagnostics have already used). A paper number should come from images that influenced nothing. `--n_final 2000` draws 2000 more VOC07-test images, disjoint from eval and tune (the eval and tune ids do not change, so every existing cache and `tune.json` stay valid). The detector's own validation images are now also kept disjoint from eval, tune and final. Do NOT score the final split until the method and settings are frozen.

```
!python run_pipeline.py fetch --dataset voc --tasks 15+5 --workdir /content/work_voc --voc_root /content/voc --n_train_per_task 2000 --n_eval 2000 --n_yolo_train 12000 --n_yolo_val 500 --n_tune 500 --n_final 2000
!python run_pipeline.py extract --workdir /content/work_voc --splits final --conf 0.001 --max_det 300 --max_boxes 60
```

**Expect:** `fetch` prints `tune images: 500 | final images: 2000`; `extract` reports 2000 images. If it says the settings differ from the other caches, copy the flags of step 13.

**Stop / fix if:** `extract` says the cache settings differ: use the same flags as step 13.

### Step 19g - VOC: extraction-setting variants in CLONED work dirs (crop padding, CLIP model, boxes per image)

These change the cached features, so they need a re-extraction, but `clone` makes a NEW work dir that reuses the images, splits and base detector and copies no cache or result: the original run stays intact. In each clone E0 and E5 are tuned separately again (`tune`), then scored once on the eval images with the three reference rows. `--pad` is the context around each crop (the default stays 0.1 so old work dirs keep working), `--clip_model ViT-L-14` is a larger CLIP (every later command of that clone must repeat the flag), `--max_boxes` is how many proposals per image are kept. A bigger CLIP changes E0 as well, so judge the E5 - E0 gap, not E5 alone. RESULT (VOC, 13 Nov 2026): tighter crops are better for E0 and E5 alike (pad 0.3 / 0.1 / 0.0: E0 .621 / .658 / .671, E5 .631 / .676 / .687 on eval; the same order on the tune images), the E5 - E0 gap stays about +0.017, and ViT-L/14 does not help. So `work_voc_pad00` (pad 0.0) is the MAIN pipeline from here on; every extraction command of it passes `--pad 0.0` explicitly. Because the final split was cloned in already (step 19f runs first), the pad00 `extract` below also builds the final cache. Crop padding is a per-dataset setting: re-check it on any other dataset.

```
!python run_pipeline.py clone --workdir /content/work_voc_pad03 --src /content/work_voc
!python run_pipeline.py extract --workdir /content/work_voc_pad03 --conf 0.001 --max_det 300 --max_boxes 60 --pad 0.3
!python run_pipeline.py tune --workdir /content/work_voc_pad03 --n_trials 20
!python run_pipeline.py run --workdir /content/work_voc_pad03 --tuned /content/work_voc_pad03/results/tune.json --only E0_zeroshot_tuned E5_tuned E6_gt_ceiling_tuned
!python run_pipeline.py clone --workdir /content/work_voc_pad00 --src /content/work_voc
!python run_pipeline.py extract --workdir /content/work_voc_pad00 --conf 0.001 --max_det 300 --max_boxes 60 --pad 0.0
!python run_pipeline.py tune --workdir /content/work_voc_pad00 --n_trials 20
!python run_pipeline.py run --workdir /content/work_voc_pad00 --tuned /content/work_voc_pad00/results/tune.json --only E0_zeroshot_tuned E5_tuned E6_gt_ceiling_tuned
!python run_pipeline.py clone --workdir /content/work_voc_clipL14 --src /content/work_voc
!python run_pipeline.py extract --workdir /content/work_voc_clipL14 --conf 0.001 --max_det 300 --max_boxes 60 --clip_model ViT-L-14
!python run_pipeline.py tune --workdir /content/work_voc_clipL14 --n_trials 20 --clip_model ViT-L-14
!python run_pipeline.py run --workdir /content/work_voc_clipL14 --tuned /content/work_voc_clipL14/results/tune.json --clip_model ViT-L-14 --only E0_zeroshot_tuned E5_tuned E6_gt_ceiling_tuned
```

**Expect:** Per clone: `results/tune.md` and `results/ablation.md`. Compare the E5_tuned - E0_zeroshot_tuned gap with the one in `work_voc`.

**Stop / fix if:** `extract` in a clone fails on missing images: run step 16's `fetch` in the original work dir first (a clone points at its image folder).

### Step 19i - VOC: stability checks on the pad 0.0 pipeline (cache-only) before freezing it

The frozen settings of the pad 0.1 pipeline do not carry over: the pad 0.0 clone has its OWN `tune.json` (picked automatically from 20 trials, tune images only). Before freezing it, repeat the cheap checks there: `loo --n_folds 10` with the group rows (does the N2L machinery matter at this padding? at pad 0.1 putting it back cost -0.010, but at pad 0.0 the top tuning trials were within 0.002 of each other with it on or off), `perclass`, and two more tuning seeds with a leave-one-out for each (is the gain still there when the random search picks something else?). Freeze the `tune.json` this produces; do NOT hand-pick a trial.

```
!python run_pipeline.py loo --workdir /content/work_voc_pad00 --n_folds 10 --tag f10
!python run_pipeline.py perclass --workdir /content/work_voc_pad00
!python run_pipeline.py tune --workdir /content/work_voc_pad00 --n_trials 20 --tune_seed 1 --tag s1
!python run_pipeline.py loo --workdir /content/work_voc_pad00 --tuned /content/work_voc_pad00/results/tune_s1.json --n_folds 10 --tag s1
!python run_pipeline.py tune --workdir /content/work_voc_pad00 --n_trials 20 --tune_seed 2 --tag s2
!python run_pipeline.py loo --workdir /content/work_voc_pad00 --tuned /content/work_voc_pad00/results/tune_s2.json --n_folds 10 --tag s2
```

**Expect:** `loo_f10.md`, `perclass.md`, `tune_s1.md`, `loo_s1.md`, `tune_s2.md`, `loo_s2.md` in `/content/work_voc_pad00/results/`.

**Stop / fix if:** `run `tune` first`: step 19g's pad00 chain must be done.

### Step 19h - [LAST, ONCE] Frozen final numbers on the untouched final split

Only after every method choice and every dataset decision is made. This scores the final split ONCE with the frozen `tune.json` of the pad 0.0 pipeline (`work_voc_pad00`). The command refuses without `--tuned` and refuses a second use (a repeat needs `--allow_repeat_final` and is marked in the table), because looking at the final split again after changing anything turns it into another tuning set.

```
!python run_pipeline.py run --workdir /content/work_voc_pad00 --eval_on final --tuned /content/work_voc_pad00/results/tune.json --only E0_zeroshot_tuned E5_tuned E6_gt_ceiling_tuned
```

**Expect:** `results/ablation_final.md` and `results/final_log.json` (one entry).

**Stop / fix if:** `already used`: the final split was scored before; do not repeat it unless you accept it is no longer a clean test.

### Step 20 - VOC: external reference, YOLO-World zero-shot (needs Internet for the weights)

An open-vocabulary detector that is told the 20 class names and scored on the same eval images and the same metric. It was pre-trained on Objects365/GoldG/CC3M-style data, so it is a REFERENCE row, not a fair competitor. Add it to the comparison with `--also baselines.json`.

```
!pip -q install ultralytics
!python run_pipeline.py baseline --workdir /content/work_voc
!python compare_runs.py /content/work_voc --ref E0_zeroshot_tuned --metric AP50 --also baselines.json
```

**Expect:** `YOLO-World zero-shot on 2000 images: AP=... AP50=... base=... new=...` then the comparison table with a `yolo_world` row.

**Stop / fix if:** Download error for `yolov8s-worldv2.pt`: Internet must be on; otherwise skip this step.

### Step 21 - [YOU DECIDE] Other class splits and other datasets

A result on one VOC split can be an accident. (a) A random 15+5 split puts different classes in the new task: use `random:15+5:SEED`; every split needs its OWN base detector (about 100 min on a T4), so run them one at a time. (b) Any COCO-format dataset with its own classes: `--dataset custom` with train/val json and image folders, `--tasks` as class positions (alphabetical by category id), `--name_map` to rename classes for CLIP. Repeat steps 10-13 and 16-19 in a new `--workdir`.

```
!python run_pipeline.py fetch --dataset voc --tasks random:15+5:1 --workdir /content/work_voc_r1 --voc_root /content/voc --n_train_per_task 2000 --n_eval 2000 --n_tune 500 --n_yolo_train 12000 --n_yolo_val 500
# then, in /content/work_voc_r1, the same commands as steps 11-14 and 16-19 (make-yolo-data, train-yolo, extract, run ...)
# a custom COCO-format dataset (the paths and the class split below are placeholders for YOUR data):
!python run_pipeline.py fetch --dataset custom --ann_train /content/mydata/train.json --ann_val /content/mydata/val.json --img_dir_train /content/mydata/train_images --img_dir_val /content/mydata/val_images --tasks 0,1,2|3,4 --workdir /content/work_custom --n_train_per_task 1000 --n_eval 500 --n_tune 300 --n_yolo_train 5000
```

**Expect:** Same prints as step 10, with the class split shown in the `tasks:` line.

**Stop / fix if:** **Decide with your guide first** which extra datasets/splits count for the paper, and agree the screening rule before looking at results.

### Step 22 - [YOU DECIDE] Proposer variant: YOLO-World as the box proposer

Our proposer is frozen and trained on the base classes only, and it caps new-class recall (COCO ~0.3). An open-vocabulary proposer (prompt `object`) usually boxes unseen classes much better, but it was pre-trained on far more data, so using it CHANGES THE PROTOCOL. Report it only as a clearly labelled variant, and only after your guide agrees. No detector training is needed. (A class-agnostic variant of our own detector: `make-yolo-data --agnostic`, then train and extract as in steps 12-13.)

```
!python run_pipeline.py fetch --dataset voc --tasks 15+5 --workdir /content/work_voc_world --voc_root /content/voc --n_train_per_task 2000 --n_eval 2000 --n_yolo_train 12000 --n_yolo_val 500 --n_tune 500
!python run_pipeline.py extract --workdir /content/work_voc_world --proposer yoloworld --world_prompts object --conf 0.001 --max_det 300 --max_boxes 60
!python run_pipeline.py gate-recall --workdir /content/work_voc_world
!python run_pipeline.py run --workdir /content/work_voc_world --only E0_zeroshot E5_ours E5_no_refine E6_gt_ceiling
```

**Expect:** Compare `gate-recall` task1 recall with step 13: that is the headroom the proposer was hiding. Then compare the AP50 rows.

**Stop / fix if:** **Do not put this in the main results table** unless the protocol change is approved.

### How to read the results (decision rules, fixed before seeing any number)

| Check | Look at | If it fails |
|---|---|---|
| GATE E0 (step 4) | `all`, `task*_within` | CLIP itself is weak on these crops: every method is capped; fix crops/names first |
| GATE 1 (step 7) | task1 `recall@0.5` vs task0 | the frozen detector cannot box unseen classes: expect E5 and E6 both capped by it |
| GATE 2 (step 8) | `E6_gt_ceiling` vs `E0_zeroshot` | if E6 is not clearly above E0, the head adds nothing even with perfect labels: the features/proposals are the bottleneck, not N2L |
| GATE 3 (step 8) | `E5_ours` vs `E0_zeroshot`, then each `E5_*` ablation vs `E5_ours` | a component that does not help in the real table is removed from the method, not defended |
| Forgetting | `base AP50 drop` column | by construction (recursive ridge == joint training) it should be ~0 for the head; a non-zero value comes from the zero-shot fusion or new classes taking over detections |

### What to send back to me

Paste the printed output of steps 1, 2, 4 and 7, plus the contents of `work/results/ablation.md` (and the 4x10 one); for VOC the outputs of steps 10, 11, 12 (last lines), 13 and 14; `screening.md` from step 15; and for the improvement chain `diagnose.md`, `tune.md`, the two `compare_*.md` tables, `loo.md` and `seeds.md` (steps 17-19c). Those four gates tell me where the pipeline stands without a single guess.

### If something goes wrong

- `run`/`gate-recall` says the cache *lacks N images*: you re-ran `fetch` with different settings after `extract`; run `extract` again.
- `run` says features were extracted with CLIP weights X: pass the same `--clip_weights X`.
- Colab session died: re-run the same commands; downloads and the cache are reused (the work folder lives in `/content`, so copy `/content/work` to Drive if you want it to survive a runtime reset).

## What was and was not verified (read this)

**Verified here, offline (370 tests + mutation checks):** the Pascal VOC converter (box offsets, `difficult`, ids, protocols, tar safety) and the results-merge script on synthetic VOC trees and local tarballs, plus an end-to-end VOC workdir through every command; the ridge/N2L core (recursive == joint solution, weights, PCA rules, label refinement), box utilities, COCO class/task/YOLO-label handling, caches, crop extraction, pseudo-labelling, the COCO-style evaluator (cross-checked against an independent AP implementation), inference/NMS (cross-checked against a reference NMS), the incremental runner, and every CLI sub-command end-to-end on a synthetic image world with stand-in models. Each test file was mutation-checked: I broke the code on purpose and confirmed the tests fail. Ground truth is proven not to leak into the E5 run (garbage training annotations leave E5 bit-identical).

**NOT verified, because the sandbox cannot reach them:** the real VOC download URLs (the official host and a mirror; if both fail, download the tars by hand or attach a Kaggle VOC dataset), real VOC files, real CLIP weights, real YOLOv8 training/inference, the real COCO download, the CIFAR-100 reproduction numbers, any real mAP, GPU timings (all time estimates above are guesses). The code that touches them is covered only by tests that use stubs with the *real* function signatures. Expect to hit small environment problems on first contact; step 1-2 are designed to find them early and cheaply.

**Known design risks (these are research risks, not bugs):**
1. The frozen detector may not box unseen classes well (GATE 1) - and a base detector trained with new-class objects as unlabeled background actively learns to suppress them.
2. The background text prompts are hand-written heuristics; `E5_no_bg` in the ablation tells you whether they help.
3. The PCA rank rule is scale dependent; the default (95% energy, rank at least #classes-1) is my choice for detection, the official absolute-sigma rule is available as `E5_k_abs_official`. The real table decides.
4. Synthetic results say the plumbing is right, not what real mAP will be.
5. `--yolo_init coco` leaks; E1-E4 (fine-tuned-detector baselines from your professor's flowchart) are NOT implemented - they need a trainable detector, not cached features.

## Files

| file | role |
|---|---|
| `n2l_core.py` | ridge accumulators, PCA/k rules, bi-level weights, label refinement, `N2LLearner` |
| `boxes.py`, `coco_utils.py`, `prompts.py` | box maths, COCO-80 + task splits + YOLO labels, prompt templates and background prompts |
| `feature_cache.py`, `clip_encoder.py`, `extract_features.py`, `yolo_tools.py` | disk cache, CLIP wrapper, crop extraction, YOLO proposer/trainer |
| `pseudo.py`, `incremental.py`, `inference.py`, `evaluate.py` | pseudo-labels, the class-incremental runner, fusion + class-wise NMS, COCO AP |
| `gates.py`, `coco_fetch.py`, `cifar_repro.py` | gate metrics, resumable download, CIFAR-100 protocol |
| `voc_utils.py` | Pascal VOC -> COCO-format conversion (`fetch --dataset voc`), VOC splits (15+5, 10+10, ...) |
| `merge_results.py` | one screening table from several working folders |
| `diagnose.py`, `tune.py`, `compare_runs.py` | step 17 (why the head helps or not), step 18 (tuning on held-out images), step 19 (fold CIs, paired differences), steps 19b/19c (leave-one-out, training-subset seeds) |
| `baselines.py`, `custom_data.py` | YOLO-World reference (step 20), any COCO-format dataset (step 21) |
| `KAGGLE.md`, `kaggle_driver.ipynb` | the same runs as one committed Kaggle notebook |
| `fake_models.py`, `synth.py`, `pipeline.py`, `real_models.py`, `run_pipeline.py` | offline stand-ins, stage glue, real-model wrappers, the CLI |
| `test_*.py` | the tests (`python -m pytest -q`); `test_voc.py` covers VOC and the merge script |
| `colab_driver.ipynb` | this runbook as a notebook |

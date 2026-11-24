# N2L-Det-exp on Kaggle: round 1 (boxes) + the 10+5+5 detector

This is the EXPERIMENTAL code base (folder `n2l_det_exp`, marker `N2L_EXP_MARKER_v7` in `run_pipeline.py`). v2 = v1 + the v5.4
per-class AP fix; v1 cells/outputs report AP50_new and per-class AP over the wrong VOC classes (overall AP50 was right). The frozen pipeline
(`n2l-det-v6`, pad 0.0, `work_voc_pad00/results/tune.json`) is never changed: these cells only READ its notebook outputs, and every
Kaggle notebook works on its own copy of them.

## Rules for every cell here
- Upload `n2l_det_exp.zip` as a NEW Kaggle dataset named **n2l-det-exp** (not a new version of `n2l-det-v6`).
- Attach **n2l-det-exp**. A cell only uses a package that contains the marker, so the frozen v6 code (also present inside older notebook
  outputs) can never be picked; it stops if two different exp versions are attached. Each cell deletes its package copy at the end, so
  its own output never carries code.
- Never attach the Cell 9 output (`work_voc_pad00_after_final_split.tar`): it holds the final-split cache. Cells R1-A and R1-B stop if a
  final cache is present. The final split is used only once, at the very end (the new Cell 10).
- Internet On (pip packages; YOLO-World weights for R1-A and R1-B). GPU T4 or P100.
- Each cell is its own notebook. They are independent and can run at the same time.

## Pre-registered reading of round 1 (fixed before any result exists)
- **Rescoring** (`variants`, tune images): a row becomes a candidate if its AP50 diff vs `E5_tuned` is >= +0.005, or its AP50_new diff
  is >= +0.010, with the 95% interval excluding 0. At most 3 candidates go to the eval confirmation (round 1b). The same change on E0
  (`E0__...` rows) is reported next to it: if it lifts E0 as much, it is an inference rule, not a gain of our head.
- **More boxes per image** (Cell R1-C) is run only if, on the TUNE rows of `propdiag` for the base YOLO at NMS 0.7, new-class recall@0.5
  at K=100 is at least 0.02 above K=60.
- **NMS IoU 0.9** becomes an extraction candidate only if its new-class recall@0.5 at K=60 is at least 0.02 above NMS 0.7 (tune rows).
- **YOLO-World** is a diagnostic only (it was pre-trained on data containing the "new" classes). It is never adopted as the method.
- **Class-agnostic detector** (round 3) is trained if the oracle ladder says the boxes (1 - O3, or O3 - O2) cost the new classes more than
  naming (O1 - ref), or if YOLO-World covers the new classes at K=60 at least 0.05 better than the base YOLO.
- All of this is dev-data evidence (tune + eval images). The win rule for adopting anything into the method is unchanged (see the
  round notes): +0.005 AP50 or +0.010 AP50_new, interval above 0, the other metric not down by more than 0.003, base AP50 not down by
  more than 0.005, tune AND eval.

## Cell 0 - checksum of the frozen tune.json (CPU is enough; attach the Cell 7b, Cell 8 and/or Cell 9 notebook outputs)

```
%%bash
for T in $(find /kaggle/input -name 'work_voc_pad00_after_*.tar'); do echo "$T"; tar -xOf "$T" work_voc_pad00/results/tune.json | sha256sum; done
```

**Expect:** one line per attached tar, all with the same 64-character checksum. It only reads `tune.json` out of each tar.

## Cell R1-A - box diagnosis + rescoring screen + proposer coverage (attach: n2l-det-exp, the improvement-run output, the Cell 8 output, the VOC datasets)

```
%%bash
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

echo "=== SETUP ==="
echo "candidate packages:"; find /kaggle/input -name run_pipeline.py
RPS=$(for f in $(find /kaggle/input -name run_pipeline.py); do grep -l N2L_EXP_MARKER_v7 "$f" || true; done)
[ -n "$RPS" ] || { echo "no EXPERIMENTAL package (N2L_EXP_MARKER_v7) found: attach the n2l-det-exp dataset"; exit 1; }
[ "$(for f in $RPS; do sha256sum < "$f"; done | sort -u | wc -l)" = "1" ] || { echo "different exp package versions attached: keep only the newest n2l-det-exp"; exit 1; }
RP=$(echo "$RPS" | head -1)
echo "using: $RP (the frozen v6 code has no marker, so it can never be picked here)"
rm -rf /kaggle/working/n2l_det_exp; cp -r "$(dirname $RP)" /kaggle/working/n2l_det_exp
cd /kaggle/working/n2l_det_exp
pip -q install pycocotools open_clip_torch ultralytics
python -c "import open_clip, ultralytics, pycocotools, torch; print('imports ok | cuda:', torch.cuda.is_available())"
python run_pipeline.py selftest | tail -3

echo "=== RESTORE ==="
if [ -f /kaggle/working/work_voc/results/tune.json ]; then echo "work_voc is already here (same session)"; else
  T=$(find /kaggle/input -name work_voc_after_improve.tar | head -1); echo "tar: $T"
  [ -n "$T" ] || { echo "no work_voc_after_improve.tar found: attach the Notebook Output of the improvement run as Input"; exit 1; }
  tar -xf "$T" -C /kaggle/working
fi
if [ -f /kaggle/working/work_voc_pad00/results/tune.json ]; then echo "work_voc_pad00 is already here (same session)"; else
  T=$(find /kaggle/input -name work_voc_pad00_after_followup.tar | head -1); echo "tar: $T"
  [ -n "$T" ] || { echo "no work_voc_pad00_after_followup.tar found: attach the Notebook Output of the Cell 8 run as Input"; exit 1; }
  tar -xf "$T" -C /kaggle/working
fi
[ ! -d /kaggle/working/work_voc_pad00/cache/final ] || { echo "a FINAL-split cache is present: detach the Cell 9 output; this notebook must not see it"; exit 1; }
W=/kaggle/working/work_voc_pad00
sha256sum $W/results/tune.json | tee /kaggle/working/tune_sha_before.txt

echo "=== IMAGES (the pad 0.0 work dir points at the original work dir's image folder) ==="
python run_pipeline.py fetch --dataset voc --tasks 15+5 --workdir /kaggle/working/work_voc --voc_root /kaggle/input --n_train_per_task 2000 --n_eval 2000 --n_yolo_train 12000 --n_yolo_val 500 --n_tune 500 || echo "stage FETCH FAILED"
python -c "import json; a=json.load(open('/kaggle/working/work_voc_pad00/splits.json')); b=json.load(open('/kaggle/working/work_voc/splits.json')); assert a['eval_ids']==b['eval_ids'] and a['tune_ids']==b['tune_ids'], 'ids changed'; assert not a.get('final_ids'), 'final ids present'; print('eval and tune ids unchanged; no final ids')" || exit 1

echo "=== BOXDIAG TUNE ==="
python run_pipeline.py boxdiag --workdir /kaggle/working/work_voc_pad00 --eval_on tune --tuned /kaggle/working/work_voc_pad00/results/tune.json || echo "stage BOXDIAG TUNE FAILED"
echo "=== BOXDIAG EVAL ==="
python run_pipeline.py boxdiag --workdir /kaggle/working/work_voc_pad00 --eval_on eval --tuned /kaggle/working/work_voc_pad00/results/tune.json || echo "stage BOXDIAG EVAL FAILED"
echo "=== RESCORING SCREEN (tune images) ==="
python run_pipeline.py variants --workdir /kaggle/working/work_voc_pad00 --tuned /kaggle/working/work_voc_pad00/results/tune.json --on tune --n_folds 10 --tag r1 --only gnew_0 gnew_0.1 clipobj_0.5 clipobj_1 yfloor_0.05 yfloor_0.1 E0__gnew_0 E0__gnew_0.1 E0__clipobj_0.5 E0__clipobj_1 E0__yfloor_0.05 E0__yfloor_0.1 || echo "stage RESCORING FAILED"
echo "=== PROPDIAG (fresh proposals, no CLIP) ==="
python run_pipeline.py propdiag --workdir /kaggle/working/work_voc_pad00 --splits tune eval --proposers yolo yoloworld --nms_ious 0.7 0.9 --conf 0.001 --max_det 300 --world_prompts object --world_weights yolov8s-worldv2.pt || echo "stage PROPDIAG FAILED"

echo "=== FROZEN SETTINGS CHECK ==="
sha256sum $W/results/tune.json | tee /kaggle/working/tune_sha_after.txt
cmp <(cut -d' ' -f1 /kaggle/working/tune_sha_before.txt) <(cut -d' ' -f1 /kaggle/working/tune_sha_after.txt) && echo "tune.json unchanged"
echo "=== RESULT FILES ==="
ls -l $W/results
cat $W/results/boxdiag_tune.md $W/results/boxdiag_eval.md $W/results/variants_r1.md $W/results/propdiag.md || true
tar -cf /kaggle/working/r1a_results.tar -C /kaggle/working work_voc_pad00/results tune_sha_before.txt tune_sha_after.txt
rm -rf /kaggle/working/n2l_det_exp
echo "=== ALL DONE ==="
ls -lh /kaggle/working/*.tar
```

**Expect:** banners SETUP, RESTORE (two tars), the tune.json checksum, IMAGES with `eval and tune ids unchanged; no final ids`,
BOXDIAG TUNE, BOXDIAG EVAL, RESCORING SCREEN (12 rows plus 3 references), PROPDIAG (8 lines `yolo_nms0.7_tune: ... s/image` etc.),
`tune.json unchanged`, ALL DONE, `r1a_results.tar` (small: results only). My estimate: 45-75 minutes. Paste everything from
`=== BOXDIAG TUNE ===` down.

## Cell R1-B - YOLO-World boxes through our whole pipeline (DIAGNOSTIC; attach: n2l-det-exp, the improvement-run output, the VOC datasets)

```
%%bash
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

echo "=== SETUP ==="
echo "candidate packages:"; find /kaggle/input -name run_pipeline.py
RPS=$(for f in $(find /kaggle/input -name run_pipeline.py); do grep -l N2L_EXP_MARKER_v7 "$f" || true; done)
[ -n "$RPS" ] || { echo "no EXPERIMENTAL package (N2L_EXP_MARKER_v7) found: attach the n2l-det-exp dataset"; exit 1; }
[ "$(for f in $RPS; do sha256sum < "$f"; done | sort -u | wc -l)" = "1" ] || { echo "different exp package versions attached: keep only the newest n2l-det-exp"; exit 1; }
RP=$(echo "$RPS" | head -1)
echo "using: $RP (the frozen v6 code has no marker, so it can never be picked here)"
rm -rf /kaggle/working/n2l_det_exp; cp -r "$(dirname $RP)" /kaggle/working/n2l_det_exp
cd /kaggle/working/n2l_det_exp
pip -q install pycocotools open_clip_torch ultralytics
python -c "import open_clip, ultralytics, pycocotools, torch; print('imports ok | cuda:', torch.cuda.is_available())"
python run_pipeline.py selftest | tail -3

echo "=== RESTORE ==="
if [ -f /kaggle/working/work_voc/results/tune.json ]; then echo "work_voc is already here (same session)"; else
  T=$(find /kaggle/input -name work_voc_after_improve.tar | head -1); echo "tar: $T"
  [ -n "$T" ] || { echo "no work_voc_after_improve.tar found: attach the Notebook Output of the improvement run as Input"; exit 1; }
  tar -xf "$T" -C /kaggle/working
fi
[ ! -d /kaggle/working/work_voc/cache/final ] || { echo "a FINAL-split cache is present: wrong tar attached"; exit 1; }
echo "=== IMAGES ==="
python run_pipeline.py fetch --dataset voc --tasks 15+5 --workdir /kaggle/working/work_voc --voc_root /kaggle/input --n_train_per_task 2000 --n_eval 2000 --n_yolo_train 12000 --n_yolo_val 500 --n_tune 500 || echo "stage FETCH FAILED"
echo "=== CLONE world ==="
python run_pipeline.py clone --workdir /kaggle/working/work_voc_world --src /kaggle/working/work_voc || echo "stage CLONE FAILED"
python -c "import json; s=json.load(open('/kaggle/working/work_voc_world/splits.json')); assert not s.get('final_ids'), 'final ids present'; print('no final ids')" || exit 1
echo "=== EXTRACT world (YOLO-World 'object' proposals, pad 0.0, 60 boxes) ==="
python run_pipeline.py extract --workdir /kaggle/working/work_voc_world --splits train eval tune --proposer yoloworld --world_prompts object --world_weights yolov8s-worldv2.pt --conf 0.001 --max_det 300 --max_boxes 60 --pad 0.0 || echo "stage EXTRACT FAILED"
echo "=== TUNE world ==="
python run_pipeline.py tune --workdir /kaggle/working/work_voc_world --n_trials 20 || echo "stage TUNE FAILED"
echo "=== RUN world ==="
python run_pipeline.py run --workdir /kaggle/working/work_voc_world --tuned /kaggle/working/work_voc_world/results/tune.json --only E0_zeroshot_tuned E5_tuned E6_gt_ceiling_tuned || echo "stage RUN FAILED"
python compare_runs.py /kaggle/working/work_voc_world --ref E0_zeroshot_tuned --metric AP50 --out /kaggle/working/work_voc_world/results/compare_AP50.md || true
python compare_runs.py /kaggle/working/work_voc_world --ref E0_zeroshot_tuned --metric AP50_new --out /kaggle/working/work_voc_world/results/compare_AP50_new.md || true
echo "=== BOXDIAG world (tune images) ==="
python run_pipeline.py boxdiag --workdir /kaggle/working/work_voc_world --eval_on tune --tuned /kaggle/working/work_voc_world/results/tune.json || echo "stage BOXDIAG FAILED"
cat /kaggle/working/work_voc_world/results/tune.md /kaggle/working/work_voc_world/results/ablation.md /kaggle/working/work_voc_world/results/compare_AP50.md /kaggle/working/work_voc_world/results/compare_AP50_new.md /kaggle/working/work_voc_world/results/boxdiag_tune.md || true
tar --exclude=yolo_data/images --exclude=voc_coco/images -cf /kaggle/working/work_voc_world_after_r1b.tar -C /kaggle/working work_voc_world
rm -rf /kaggle/working/n2l_det_exp
echo "=== ALL DONE ==="
ls -lh /kaggle/working/*.tar
```

**Expect:** as Cell 7b, with `extracted ... images (train, eval, tune)` and the three-row table. My estimate: 60-90 minutes (extraction
dominates). Read it against the pad 0.0 numbers (E0 .671 / E5 .687 / E6 .701 on eval, 5 folds; new .360 / .391 / .410).

## Cell R1-C - 100 boxes per image at pad 0.0 (ONLY if R1-A meets the pre-registered rule above; attach: n2l-det-exp, the improvement-run output, the VOC datasets)

```
%%bash
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

echo "=== SETUP ==="
echo "candidate packages:"; find /kaggle/input -name run_pipeline.py
RPS=$(for f in $(find /kaggle/input -name run_pipeline.py); do grep -l N2L_EXP_MARKER_v7 "$f" || true; done)
[ -n "$RPS" ] || { echo "no EXPERIMENTAL package (N2L_EXP_MARKER_v7) found: attach the n2l-det-exp dataset"; exit 1; }
[ "$(for f in $RPS; do sha256sum < "$f"; done | sort -u | wc -l)" = "1" ] || { echo "different exp package versions attached: keep only the newest n2l-det-exp"; exit 1; }
RP=$(echo "$RPS" | head -1)
echo "using: $RP (the frozen v6 code has no marker, so it can never be picked here)"
rm -rf /kaggle/working/n2l_det_exp; cp -r "$(dirname $RP)" /kaggle/working/n2l_det_exp
cd /kaggle/working/n2l_det_exp
pip -q install pycocotools open_clip_torch ultralytics
python -c "import open_clip, ultralytics, pycocotools, torch; print('imports ok | cuda:', torch.cuda.is_available())"
python run_pipeline.py selftest | tail -3

echo "=== RESTORE ==="
if [ -f /kaggle/working/work_voc/results/tune.json ]; then echo "work_voc is already here (same session)"; else
  T=$(find /kaggle/input -name work_voc_after_improve.tar | head -1); echo "tar: $T"
  [ -n "$T" ] || { echo "no work_voc_after_improve.tar found: attach the Notebook Output of the improvement run as Input"; exit 1; }
  tar -xf "$T" -C /kaggle/working
fi
[ ! -d /kaggle/working/work_voc/cache/final ] || { echo "a FINAL-split cache is present: wrong tar attached"; exit 1; }
echo "=== IMAGES ==="
python run_pipeline.py fetch --dataset voc --tasks 15+5 --workdir /kaggle/working/work_voc --voc_root /kaggle/input --n_train_per_task 2000 --n_eval 2000 --n_yolo_train 12000 --n_yolo_val 500 --n_tune 500 || echo "stage FETCH FAILED"
echo "=== CLONE boxes100 ==="
python run_pipeline.py clone --workdir /kaggle/working/work_voc_pad00_boxes100 --src /kaggle/working/work_voc || echo "stage CLONE FAILED"
echo "=== EXTRACT boxes100 ==="
python run_pipeline.py extract --workdir /kaggle/working/work_voc_pad00_boxes100 --splits train eval tune --conf 0.001 --max_det 300 --max_boxes 100 --pad 0.0 || echo "stage EXTRACT FAILED"
echo "=== TUNE boxes100 ==="
python run_pipeline.py tune --workdir /kaggle/working/work_voc_pad00_boxes100 --n_trials 20 || echo "stage TUNE FAILED"
echo "=== RUN boxes100 ==="
python run_pipeline.py run --workdir /kaggle/working/work_voc_pad00_boxes100 --tuned /kaggle/working/work_voc_pad00_boxes100/results/tune.json --only E0_zeroshot_tuned E5_tuned E6_gt_ceiling_tuned || echo "stage RUN FAILED"
python compare_runs.py /kaggle/working/work_voc_pad00_boxes100 --ref E0_zeroshot_tuned --metric AP50 --out /kaggle/working/work_voc_pad00_boxes100/results/compare_AP50.md || true
python compare_runs.py /kaggle/working/work_voc_pad00_boxes100 --ref E0_zeroshot_tuned --metric AP50_new --out /kaggle/working/work_voc_pad00_boxes100/results/compare_AP50_new.md || true
cat /kaggle/working/work_voc_pad00_boxes100/results/tune.md /kaggle/working/work_voc_pad00_boxes100/results/ablation.md /kaggle/working/work_voc_pad00_boxes100/results/compare_AP50.md /kaggle/working/work_voc_pad00_boxes100/results/compare_AP50_new.md || true
tar --exclude=yolo_data/images --exclude=voc_coco/images -cf /kaggle/working/work_voc_pad00_boxes100_after_r1c.tar -C /kaggle/working work_voc_pad00_boxes100
rm -rf /kaggle/working/n2l_det_exp
echo "=== ALL DONE ==="
ls -lh /kaggle/working/*.tar
```

## Cell R1-D - the 10+5+5 detector (train it now; attach: n2l-det-exp, the VOC datasets; GPU; a separate account is fine)

Classes (standard VOC 10+5+5, VOC order): base = aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow;
task 1 = dining table, dog, horse, motorbike, person; task 2 = potted plant, sheep, sofa, train, tv monitor (the same 5 new classes as 15+5).
A fresh work dir: the detector's 500 validation images are drawn AFTER the eval/tune/final images, so they never overlap.
The final cache is NOT extracted (`--splits train eval tune`).

```
%%bash
set -e
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1

echo "=== SETUP ==="
echo "candidate packages:"; find /kaggle/input -name run_pipeline.py
RPS=$(for f in $(find /kaggle/input -name run_pipeline.py); do grep -l N2L_EXP_MARKER_v7 "$f" || true; done)
[ -n "$RPS" ] || { echo "no EXPERIMENTAL package (N2L_EXP_MARKER_v7) found: attach the n2l-det-exp dataset"; exit 1; }
[ "$(for f in $RPS; do sha256sum < "$f"; done | sort -u | wc -l)" = "1" ] || { echo "different exp package versions attached: keep only the newest n2l-det-exp"; exit 1; }
RP=$(echo "$RPS" | head -1)
echo "using: $RP (the frozen v6 code has no marker, so it can never be picked here)"
rm -rf /kaggle/working/n2l_det_exp; cp -r "$(dirname $RP)" /kaggle/working/n2l_det_exp
cd /kaggle/working/n2l_det_exp
pip -q install pycocotools open_clip_torch ultralytics
python -c "import open_clip, ultralytics, pycocotools, torch; print('imports ok | cuda:', torch.cuda.is_available())"
python run_pipeline.py selftest | tail -3

echo "=== FETCH 10+5+5 ==="
python run_pipeline.py fetch --dataset voc --tasks 10+2x5 --workdir /kaggle/working/work_voc10 --voc_root /kaggle/input --n_train_per_task 2000 --n_eval 2000 --n_yolo_train 12000 --n_yolo_val 500 --n_tune 500 --n_final 2000
python -c "import json; s=json.load(open('/kaggle/working/work_voc10/splits.json')); v=set(s['yolo_val']); o=v & (set(s['eval_ids'])|set(s['tune_ids'])|set(s['final_ids'])); print('tasks', [len(t) for t in s['tasks']], '| detector train', len(s['yolo_train']), '| detector val', len(v), '| overlap with scored images', len(o)); assert not o"
tar --exclude=voc_coco/images -cf /kaggle/working/work_voc10_after_fetch.tar -C /kaggle/working work_voc10
echo "=== YOLO 10 base classes ==="
python run_pipeline.py make-yolo-data --workdir /kaggle/working/work_voc10
python run_pipeline.py train-yolo --workdir /kaggle/working/work_voc10 --yolo_init scratch --yolo_size s --yolo_epochs 40
tar --exclude=yolo_data/images --exclude=voc_coco/images -cf /kaggle/working/work_voc10_after_yolo.tar -C /kaggle/working work_voc10
echo "=== EXTRACT (pad 0.0, 60 boxes; NOT the final split) + GATE-RECALL ==="
python run_pipeline.py extract --workdir /kaggle/working/work_voc10 --splits train eval tune --conf 0.001 --max_det 300 --max_boxes 60 --pad 0.0
python run_pipeline.py gate-recall --workdir /kaggle/working/work_voc10
tar --exclude=yolo_data/images --exclude=voc_coco/images -cf /kaggle/working/work_voc10_after_extract.tar -C /kaggle/working work_voc10
rm -rf /kaggle/working/n2l_det_exp
echo "=== ALL DONE ==="
ls -lh /kaggle/working/*.tar
```

**Expect:** `tasks [10, 5, 5]`, `overlap with scored images 0`, the YOLO training log (about 100 minutes for 40 epochs, as the 15-class
detector), `extracted ... images (train, eval, tune)`, the recall table for task0/task1/task2, three tars. Nothing is tuned or scored yet:
the 10+5+5 runs come after round 1, with whatever method is current then.

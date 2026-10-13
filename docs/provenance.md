# Provenance

Every number in this repository traces to a file that still exists on the working VM
(`~/Desktop/3d_experiments/ShapeR/`, not duplicated into this repo — see
[setup-and-reproduction.md](setup-and-reproduction.md) for why). This page is the map from
claim to source, plus every discrepancy found while writing these docs.

Ground truth used in all metrics below is the **pseudo-ground-truth mesh shipped with the
ShapeR dataset** (captured/reconstructed by the dataset's original authors), not a
physically-measured reference — see [limitations-and-future-work.md](limitations-and-future-work.md).

## A — input ablations

| claim | source file | notes |
|---|---|---|
| A1/A2 per-condition chamfer/f-score means | `experiments/exp1_pointcloud_degradation/outputs_full/*/results.json` (`summary` key), same pattern for exp2 | stored aggregate, read directly, not recomputed |
| A3 per-condition chamfer/f-score means | `experiments/exp3_viewcount_ablation/outputs_full/results_views_{1,4,8,16}.json` | **no stored summary** — these files hold only `per_sample`; means were computed in this session with a direct `statistics.mean` over the 178 `per_sample` entries per file. Recompute with the one-liner in [setup-and-reproduction.md](setup-and-reproduction.md#recomputing-the-exp3-view-count-summary). |
| "4 views is the sweet spot", reused as the mask-guided training default | `updated_shapeR/mask_guided/train.py:65` comment `# Exp 3: 8 is the sweet spot` and `--num_views` default `8` | **note the discrepancy**: the code comment and default say 8, while the raw A3 numbers (best f-score at 4 views, best chamfer at 16, 8 is the middle option) support "diminishing returns after 4" at least as well as "8 is the sweet spot" — both are defensible reads of the same table; the training script's own choice (8) is reported as-is in B, not silently changed to match this doc's framing. |

## B — mask conditioning

| claim | source file | notes |
|---|---|---|
| Architecture / trainable params (305) | `updated_shapeR/mask_guided/architecture.py` | read directly; Option A = widened `proj_in` conv, Option B = `fg_bias_scale` scalar |
| Training config (8 views, 20 epochs, lr 1e-3, AdamW) | `updated_shapeR/mask_guided/train.py` arg defaults | read directly |
| Train set = 18 no-stool objects | `updated_shapeR/train_mask_guided.slurm` | read directly, object list present in the file |
| Single-seed held-out Extra% table | agent memory record `shaper-maskguided-findings`, dated 2026-08-13 session | **not independently re-verified against a results file in this pass** — flagged below as the one number in this repo taken from a prior session's summary rather than a file read in this session. The run's own output files were reported gone from disk as of 2026-08-25 (see next row). |
| Multi-seed verdict (30 paired cells, no CI excludes 0) | `updated_shapeR/occlusion_comparison/paired_stats.py` output, job 160933, per the same memory record | **the `occlusion_comparison/` and `synthetic_distractor_benchmark/` folders were reported gone from disk as of 2026-08-25**, superseded functionally by `updated_shapeR/testing/` (same comparison role: original vs trained, clean/occluded/onobject conditions — confirmed present and current in this session, see `updated_shapeR/testing/README.md`). The specific paired-bootstrap numbers above were **not** re-run in this session; they are carried over from the prior record, not re-verified against a live file. Mark **not yet independently re-verified**. |

## C — occlusion and clutter

| claim | source file | notes |
|---|---|---|
| All of C1–C4 (dose-response table, per-object breakdown, Stage 2/3 method comparison, generalisation-failure follow-up) | `updated_2/README.md` (dated 2026-08-25) and `updated_2/confirm/RESULT.md` (dated 2026-08-27) | read directly and transcribed verbatim from the user's own authored write-up; both files are newer than the result JSONs they describe, confirmed by `stat` in this session |
| C2 synthetic image-only distractor benchmark (null result on the old Extra% metric) | memory record only — `updated_shapeR/synthetic_distractor_benchmark/` reported gone from disk as of 2026-08-25 | **not independently re-verified against a results file in this pass**; kept because `updated_2/README.md` itself explicitly corrects and supersedes part of this claim (Extra% was hiding leakage on woodentable) |
| Stage 1 sweep numbers (dose-response, per-object table) | `updated_2/outputs/stage1/results.json` | file exists and post-dates nothing suspicious; numbers transcribed from `updated_2/README.md`, not independently recomputed from the raw JSON in this session |
| **Stale status line found**: `updated_2/README.md` ends "Stage 1 sweep ... and Stage 2 sweep are queued" | `updated_2/outputs/stage2/results.json` (132 entries, mtime 2026-08-25 20:23) vs `updated_2/README.md` (mtime 2026-08-25 22:00) | the README is actually *newer* than the Stage-2 results file, so "queued" is a leftover sentence, not a sign Stage 2 didn't run. Documented here rather than silently edited in the source file (nothing in `updated_2/` was modified). |

## D — depth-verified masks

| claim | source file | notes |
|---|---|---|
| `oracle_masks()` depth-test implementation, the pre-fix 76–100% contamination bug | `updated_2/mask_vs_leak/oracle_mask.py` | read directly in this session |
| `masks_with_negatives()` support-segmenter alternative | `updated_2/mask_vs_leak/clean_masks.py` | read directly in this session |
| Depth-verified oracle vs SAM2 Leak% means (n=41) | `updated_2/mask_vs_leak/outputs/clean_test/scores.json` | **recomputed in this session** directly from the raw per-object-per-seed JSON (not a stored summary) — see the one-liner in [setup-and-reproduction.md](setup-and-reproduction.md#recomputing-the-d-section-means) |
| Support-segmenter fix vs SAM2 Leak% means (n=24) | `updated_2/mask_vs_leak/outputs/three_arms/scores.json` | **recomputed in this session**, same method as above |

## What is pseudo-GT vs measured, explicitly

- **Pseudo-ground-truth**: every mesh scored against in A, B, C1 comes from the ShapeR dataset's
  own shipped GT mesh, which is itself a reconstruction (not a physically captured reference).
  Chamfer/F-score/Extra%/Missing% against it measure agreement with that mesh, not absolute
  physical accuracy.
- **Measured, not pseudo**: Leak% (C3, C4, D) is scored against the **exact placed distractor
  mesh** that `clutter.py` / the clutter-generation pipeline itself wrote out — this is a direct
  geometric measurement (distance from predicted surface to a known, exactly-placed object), not
  an estimate.
- **Image occlusion vs missing 3D observations**: C1 occludes **pixels only** (SLAM points
  behind the occluder are dropped, but the ones that were already there stay); it is not the
  same manipulation as A1's point-cloud sparsification, which removes points irrespective of
  image visibility. The two sections measure different failure modes and are not directly
  comparable.
- **Target completeness vs clutter rejection**: Missing% (how much of the target the
  reconstruction fails to build) and Leak%/Extra% (how much non-target or hallucinated geometry
  it includes) are reported as separate numbers throughout C and D specifically because a method
  can trade one for the other (see C4's `crop` arm: large Leak% reduction, real Missing% damage).

## Items marked "not yet verified" in this pass

1. B's single-seed held-out Extra% table and the 30-cell multi-seed bootstrap CIs — carried over
   from a prior session's memory record, not re-read from a live results file (the underlying
   `occlusion_comparison/` folder is reported gone from disk).
2. C2's synthetic image-only distractor null result — same situation, source folder reported
   gone from disk; superseded in spirit by `updated_2/README.md`'s correction (image-mode
   leakage does occur on `woodentable`, Extra% just hid it).

Both are reported as claims from a specific prior record, with that record named, rather than
re-stated as independently confirmed facts.

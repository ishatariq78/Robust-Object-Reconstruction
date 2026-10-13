# Setup and reproduction

## What this repository is, concretely

This repo holds **our own experiment code, documentation, small result summaries, and a curated
set of real figures** for the four studies below. It does **not** vendor ShapeR's model code,
checkpoints, or dataset, and it does **not** include the raw experiment outputs (meshes, full
per-sample JSONs, logs) — those run from tens of MB to tens of GB per experiment and stay on the
working VM. See [provenance.md](provenance.md) for exactly which file each reported number comes
from.

To actually run anything here, you need a working clone of upstream ShapeR with the code in
`code/` dropped into it at the **same relative paths** it was copied from (noted per-file below).

## Environment

All experiments ran in one conda environment, `shaper`, on a SLURM cluster:

```bash
conda activate shaper
module load cuda/12.8 2>/dev/null || module load cuda/12.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
```

Verified cluster-specific facts worth knowing before rerunning anything (not guessed — hit and
recorded while the original experiments ran):

- The login node has **no GPU**; every experiment here ran through `sbatch` with `--gres=gpu:1`.
- `--time=10:00:00` was rejected by the QOS in use; `--time=08:00:00` was accepted. Adjust for
  your own cluster's limits rather than assuming either number.
- Mesh rendering for figures (`pyrender`) needs `EGL_PLATFORM=egl` and works on the login node
  used here; `osmesa` was not installed. `open3d.t.geometry.RaycastingScene` was used instead of
  `rtree`/`embreex`, which were not in the `shaper` env.
- SAM2 is pulled from Hugging Face as `facebook/sam2.1-hiera-base-plus` at run time — not
  vendored, not a local checkpoint.
- ShapeR's own `setup_checkpoints()` (called at the top of most scripts here) downloads
  `checkpoints/config.yaml`, `019-0-bfloat16.ckpt`, `vae-088-0-bfloat16.ckpt` on first run. This
  repo assumes that step already works in your ShapeR clone.

## Layout mapping (where each `code/` subfolder came from and goes back to)

| this repo | drop into your ShapeR clone at |
|---|---|
| `code/experiments/exp1_pointcloud_degradation/` | `experiments/exp1_pointcloud_degradation/` |
| `code/experiments/exp2_caption_ablation/` | `experiments/exp2_caption_ablation/` |
| `code/experiments/exp3_viewcount_ablation/` | `experiments/exp3_viewcount_ablation/` |
| `code/root/mask_conditioning.py`, `code/root/infer_shape_masked.py` | ShapeR repo root |
| `code/updated_shapeR/mask_guided/` | `updated_shapeR/mask_guided/` |
| `code/updated_shapeR/testing/` | `updated_shapeR/testing/` |
| `code/updated_shapeR/train_mask_guided.slurm`, `test_mask_guided.slurm` | `updated_shapeR/` |
| `code/updated_2/` (incl. `mask_vs_leak/`, `confirm/`) | `updated_2/` |

## Running each study

**A — input ablations** (point density/noise, captions, view count):
```bash
python experiments/exp1_pointcloud_degradation/prepare_data.py --input_pkls data/*.pkl --output_dir data/exp1_pointcloud
sbatch experiments/exp1_pointcloud_degradation/run_full.slurm
# same pattern for exp2_caption_ablation, exp3_viewcount_ablation
```

**B — mask conditioning** (train then evaluate):
```bash
sbatch updated_shapeR/train_mask_guided.slurm     # -> updated_shapeR/outputs/mask_guided.ckpt
sbatch updated_shapeR/test_mask_guided.slurm       # held-out scoring, asserts disjoint from train set
```

**C — occlusion and clutter**:
```bash
sbatch updated_shapeR/testing/run_test.slurm       # C1: occlusion, clean/occluded/onobject conditions
python updated_shapeR/testing/compare.py           # renders the comparison sheets

sbatch updated_2/run_stage1.slurm                  # C3: clutter dose-response, 3 modes x 3 levels x 3 seeds
sbatch updated_2/run_stage2.slurm                  # C4: crop / gate / refeed on affected objects
sbatch updated_2/run_stage3.slurm                  # C4: support-segmenter variant
python updated_2/confirm/score.py                  # C4 generalisation-failure confirmation (14 fresh objects)
```

**D — depth-verified masks**:
```bash
python updated_2/mask_vs_leak/verify_masks.py --objects <clutter pkl> ...     # mask quality check
python updated_2/mask_vs_leak/clean_masks.py                                  # support-segmenter + SAM2 negatives fix
python updated_2/mask_vs_leak/oracle_mask.py --objects <image-mode clutter pkl> ...  # off/sam2/oracle arms
python updated_2/mask_vs_leak/three_arms.py --objects <pkl> ... --seeds 0 1 2        # control/stock/fixed arms
```

## Recomputing the exp3 view-count summary

`exp3_viewcount_ablation`'s result files store only `per_sample`, no aggregate — this is the
exact aggregation used for the numbers in [results/A_input_ablations.json](../results/A_input_ablations.json):

```python
import json, statistics
for f in ["results_views_1.json", "results_views_4.json", "results_views_8.json", "results_views_16.json"]:
    d = json.load(open(f))
    samples = d["per_sample"]
    chamfer = [s["metrics"]["chamfer"] for s in samples]
    fscore  = [s["metrics"]["f-score"] for s in samples]
    print(f, "n=", len(samples),
          "chamfer_mean=", round(statistics.mean(chamfer), 5),
          "fscore_mean=",  round(statistics.mean(fscore), 4))
```

## Recomputing the D-section means

`clean_test/scores.json` and `three_arms/scores.json` are nested `{sample_key: {arm: metrics}}`
dicts with no aggregate either:

```python
import json, statistics
d = json.load(open("outputs/clean_test/scores.json"))   # or outputs/three_arms/scores.json
by_arm = {}
for key, arms in d.items():
    for arm, m in arms.items():
        by_arm.setdefault(arm, []).append(m)
for arm, rows in by_arm.items():
    leak = [r["leak_pct"] for r in rows]
    print(arm, "n=", len(rows), "leak_pct_mean=", round(statistics.mean(leak), 3))
```

## What is NOT reproducible from this repo alone

- Exact dataset pkls (`data/*.pkl`) — ShapeR dataset, not redistributed here.
- Trained checkpoint `updated_shapeR/outputs/mask_guided.ckpt` — not committed (binary, and
  depends on the frozen base checkpoint, which this repo also does not vendor).
- `updated_shapeR/occlusion_comparison/` and `updated_shapeR/synthetic_distractor_benchmark/` —
  reported gone from the working VM as of 2026-08-25 (see [provenance.md](provenance.md)); the
  occlusion-comparison *role* lives on in `updated_shapeR/testing/`, re-confirmed present in this
  pass, but the specific early-run numbers cannot be regenerated byte-for-byte without rebuilding
  those two folders first.

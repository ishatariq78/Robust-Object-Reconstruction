#!/bin/bash
#SBATCH --job-name=mg_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
cd ~/Desktop/3d_experiments/ShapeR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
module load cuda/12.8 2>/dev/null || module load cuda/12.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running on: $(hostname)"; nvidia-smi | head -12

# --- HELD-OUT TEST set: none of these were in training (stool included on purpose). ---
SAMPLES=(
    data/ADT1292__stool.pkl
    data/BNB1045__wovenstool.pkl
    data/BNB1723__livingroomfurniture.pkl
    data/BNB1045__sofa.pkl
    data/ADT1292__graycouch.pkl
)

# --- safety: assert TRAIN ∩ TEST = empty (fail loudly on any leak) ---
python - "${SAMPLES[@]}" <<'PY'
import re, sys
test = {re.sub(r'.*/', '', p).replace('.pkl','') for p in sys.argv[1:]}
train = set(re.findall(r'data/([A-Za-z0-9_]+)\.pkl',
            open("updated_shapeR/train_mask_guided.slurm").read()))
leak = test & train
assert not leak, f"LEAK: these test samples are also in training -> {sorted(leak)}"
print(f"split OK: {len(train)} train, {len(test)} test, no overlap")
PY

echo "=== BASELINE reconstruction (guidance disabled, same code path) ==="
python updated_shapeR/mask_guided/infer.py \
    --input_pkls "${SAMPLES[@]}" \
    --guided_ckpt updated_shapeR/outputs/mask_guided.ckpt --disable_guidance \
    --config balance --output_dir updated_shapeR/outputs/recon_heldout_baseline

echo "=== MASK-GUIDED reconstruction (trained lambda + mask conv) ==="
python updated_shapeR/mask_guided/infer.py \
    --input_pkls "${SAMPLES[@]}" \
    --guided_ckpt updated_shapeR/outputs/mask_guided.ckpt \
    --config balance --output_dir updated_shapeR/outputs/recon_heldout_masked

echo "=== SCORE held-out set: Extra% baseline vs mask-guided (vs GT from pkl) ==="
python updated_shapeR/mask_guided/score_heldout.py \
    --input_pkls "${SAMPLES[@]}" \
    --baseline_dir updated_shapeR/outputs/recon_heldout_baseline \
    --guided_dir   updated_shapeR/outputs/recon_heldout_masked

echo "Done. Meshes in recon_heldout_{baseline,masked}/ ; scores printed above."

#!/bin/bash
#SBATCH --job-name=shaper_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/isha.tariq/Desktop/3d_experiments/ShapeR/updated_shapeR/testing/logs/%x_%j.out
#SBATCH --error=/home/isha.tariq/Desktop/3d_experiments/ShapeR/updated_shapeR/testing/logs/%x_%j.err
set -e
cd ~/Desktop/3d_experiments/ShapeR
mkdir -p updated_shapeR/testing/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
module load cuda/12.8 2>/dev/null || module load cuda/12.2 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYOPENGL_PLATFORM=egl
echo "Running on: $(hostname)"; nvidia-smi | head -12

# Held-out objects — none was used to train the mask guidance.
OBJECTS=(
    data/BNB1045__wovenstool.pkl
    data/ADT1292__stool.pkl
    data/BNB1045__sofa.pkl
    data/ADT1292__graycouch.pkl
    data/BNB1723__livingroomfurniture.pkl
)
python - "${OBJECTS[@]}" <<'PY'
import re, sys
test = {re.sub(r'.*/','',p).replace('.pkl','') for p in sys.argv[1:]}
train = set(re.findall(r'data/([A-Za-z0-9_]+)\.pkl', open("updated_shapeR/train_mask_guided.slurm").read()))
assert not (test & train), f"LEAK: {sorted(test & train)}"
print(f"split OK: {len(test)} held-out objects, none seen in mask training")
PY

echo "=== run both models on clean / occluded / object-on-it ==="
python updated_shapeR/testing/run_test.py \
    --input_pkls "${OBJECTS[@]}" \
    --guided_ckpt updated_shapeR/outputs/mask_guided.ckpt \
    --config balance --occ_level 0.6 --area_ratio 0.25 --distractor plantpot \
    --skip_existing --output_dir updated_shapeR/testing/outputs

echo "=== build the visual comparison sheets ==="
python updated_shapeR/testing/compare.py

echo "Done. sheets: updated_shapeR/testing/outputs/comparison/"

#!/bin/bash
#SBATCH --job-name=u2_stage1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/isha.tariq/Desktop/3d_experiments/ShapeR/updated_2/logs/%x_%j.out
#SBATCH --error=/home/isha.tariq/Desktop/3d_experiments/ShapeR/updated_2/logs/%x_%j.err
set -e
cd ~/Desktop/3d_experiments/ShapeR
mkdir -p updated_2/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
module load cuda/12.8 2>/dev/null || module load cuda/12.2 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYOPENGL_PLATFORM=egl
echo "Running on: $(hostname)"; nvidia-smi | head -12

# Objects with a real support surface — the paper's example is "tables supporting
# other objects", so every target here is something a second object can stand on.
OBJECTS=(
    data/ADT1292__woodentable.pkl
    data/ADT1292__coffeetable.pkl
    data/ADT1292__nightstand.pkl
    data/BNB1182__sidetable.pkl
    data/HWD0045__desktable.pkl
)

# Pass 1: the three conditioning channels at one clutter size. This is the
# comparison that says WHERE the leakage comes from — pixels, points, or both.
echo "=== pass 1: channel ablation at level 0.25 ==="
python updated_2/demo_limitation.py \
    --input_pkls "${OBJECTS[@]}" \
    --modes image points both --levels 0.25 --seeds 0 1 2 \
    --skip_existing --out_dir updated_2/outputs/stage1

# Pass 2: how leakage scales with the intruder's size, on the realistic channel.
echo "=== pass 2: level sweep, mode=both ==="
python updated_2/demo_limitation.py \
    --input_pkls "${OBJECTS[@]}" \
    --modes both --levels 0.15 0.40 --seeds 0 1 2 \
    --skip_existing --out_dir updated_2/outputs/stage1

echo "Done. results: updated_2/outputs/stage1/results.json"

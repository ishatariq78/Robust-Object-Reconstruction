#!/bin/bash
#SBATCH --job-name=u2_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/isha.tariq/Desktop/3d_experiments/ShapeR/updated_2/logs/%x_%j.out
#SBATCH --error=/home/isha.tariq/Desktop/3d_experiments/ShapeR/updated_2/logs/%x_%j.err
set +e
cd ~/Desktop/3d_experiments/ShapeR
mkdir -p updated_2/logs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
module load cuda/12.8 2>/dev/null || module load cuda/12.2 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYOPENGL_PLATFORM=egl
echo "Running on: $(hostname)"; nvidia-smi | head -12

echo "=== smoke: one object, all three modes, one seed ==="
python updated_2/demo_limitation.py \
    --input_pkls data/ADT1292__woodentable.pkl \
    --modes image points both --levels 0.25 --seeds 0 \
    --n_views_pool 16 --out_dir updated_2/outputs/_gpusmoke

echo "=== smoke: latent probe ==="
python updated_2/latent_probe.py \
    --input_pkl updated_2/outputs/_gpusmoke/clutter/ADT1292__woodentable__clean.pkl \
    --out updated_2/outputs/_gpusmoke/latent_probe.json

echo "=== smoke: latent segmentation, all methods ==="
python updated_2/latent_seg.py \
    --clutter_pkl updated_2/outputs/_gpusmoke/clutter/ADT1292__woodentable__both__L025.pkl \
    --label       updated_2/outputs/_gpusmoke/clutter/ADT1292__woodentable__both__L025__label.npz \
    --methods baseline crop gate refeed --seeds 0 \
    --out_dir updated_2/outputs/_gpusmoke/stage2

echo "SMOKE DONE"

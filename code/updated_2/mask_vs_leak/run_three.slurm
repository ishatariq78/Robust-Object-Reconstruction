#!/bin/bash
#SBATCH --job-name=3arms
#SBATCH --output=updated_2/mask_vs_leak/outputs/three_%j.log
#SBATCH --error=updated_2/mask_vs_leak/outputs/three_%j.log
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
cd /home/isha.tariq/Desktop/3d_experiments/ShapeR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYOPENGL_PLATFORM=egl
python updated_2/mask_vs_leak/three_arms.py --seeds 0 1 2 --objects updated_2/outputs/stage1/clutter/ADT1292__woodentable__both__L025.pkl updated_2/outputs/stage1/clutter/HWD0045__desktable__both__L025.pkl updated_2/outputs/stage1/clutter/ADT1292__coffeetable__both__L025.pkl updated_2/outputs/stage1/clutter/ADT1292__nightstand__both__L025.pkl updated_2/confirm/outputs/clutter/HWD0045__woodentable__both__L025.pkl updated_2/confirm/outputs/clutter/BNB2909__coffeetable__both__L025.pkl updated_2/confirm/outputs/clutter/ADT1292__tvstand__both__L025.pkl updated_2/confirm/outputs/clutter/BNB1182__woodenbox__both__L025.pkl
echo "=== three arms complete ==="

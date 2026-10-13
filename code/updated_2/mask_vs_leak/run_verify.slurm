#!/bin/bash
#SBATCH --job-name=verify
#SBATCH --output=updated_2/mask_vs_leak/outputs/verify_%j.log
#SBATCH --error=updated_2/mask_vs_leak/outputs/verify_%j.log
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
cd /home/isha.tariq/Desktop/3d_experiments/ShapeR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYOPENGL_PLATFORM=egl
python updated_2/mask_vs_leak/verify_masks.py --objects updated_2/outputs/stage1/clutter/*__image__L025.pkl

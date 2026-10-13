#!/bin/bash
#SBATCH --job-name=maskfix
#SBATCH --output=updated_2/mask_vs_leak/outputs/smoke_%j.log
#SBATCH --error=updated_2/mask_vs_leak/outputs/smoke_%j.log
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
cd /home/isha.tariq/Desktop/3d_experiments/ShapeR
python updated_2/mask_vs_leak/clean_masks.py --smoke

#!/bin/bash
#SBATCH --job-name=oracle
#SBATCH --output=updated_2/mask_vs_leak/outputs/oracle_%j.log
#SBATCH --error=updated_2/mask_vs_leak/outputs/oracle_%j.log
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
cd /home/isha.tariq/Desktop/3d_experiments/ShapeR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYOPENGL_PLATFORM=egl
python updated_2/mask_vs_leak/oracle_mask.py --seeds 0 1 2 --objects updated_2/outputs/stage1/clutter/ADT1292__coffeetable__image__L025.pkl updated_2/outputs/stage1/clutter/ADT1292__nightstand__image__L025.pkl updated_2/outputs/stage1/clutter/ADT1292__woodentable__image__L025.pkl updated_2/outputs/stage1/clutter/BNB1182__sidetable__image__L025.pkl updated_2/outputs/stage1/clutter/HWD0045__desktable__image__L025.pkl 
echo "=== oracle mask run complete ==="

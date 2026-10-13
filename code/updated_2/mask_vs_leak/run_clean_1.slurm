#!/bin/bash
#SBATCH --job-name=clean1
#SBATCH --output=updated_2/mask_vs_leak/outputs/clean1_%j.log
#SBATCH --error=updated_2/mask_vs_leak/outputs/clean1_%j.log
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
# Pot painted into the PHOTOS only (never into the scan dots), on tables the mask fine-tune has
# never seen. Three arms: guidance off / SAM2's mask / a verified-clean depth-tested mask.
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
cd /home/isha.tariq/Desktop/3d_experiments/ShapeR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYOPENGL_PLATFORM=egl
python updated_2/clutter.py --input_pkl data/HWD0045__woodentable.pkl data/HWD0045__coffeetable.pkl data/BNB2909__coffeetable.pkl data/ADT1292__kitchenisland.pkl data/ADT1292__tvstand.pkl data/BNB1182__nightstand_2.pkl data/BNB1045__nightstand_2.pkl  \
    --out_dir updated_2/mask_vs_leak/img_clutter --levels 0.25 --modes image
python updated_2/mask_vs_leak/oracle_mask.py --seeds 0 1 2 \
    --out_dir updated_2/mask_vs_leak/outputs/clean_test \
    --objects updated_2/mask_vs_leak/img_clutter/HWD0045__woodentable__image__L025.pkl updated_2/mask_vs_leak/img_clutter/HWD0045__coffeetable__image__L025.pkl updated_2/mask_vs_leak/img_clutter/BNB2909__coffeetable__image__L025.pkl updated_2/mask_vs_leak/img_clutter/ADT1292__kitchenisland__image__L025.pkl updated_2/mask_vs_leak/img_clutter/ADT1292__tvstand__image__L025.pkl updated_2/mask_vs_leak/img_clutter/BNB1182__nightstand_2__image__L025.pkl updated_2/mask_vs_leak/img_clutter/BNB1045__nightstand_2__image__L025.pkl 
echo "=== clean-mask half 1 complete ==="

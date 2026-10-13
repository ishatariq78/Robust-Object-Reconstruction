#!/bin/bash
#SBATCH --job-name=clean2
#SBATCH --output=updated_2/mask_vs_leak/outputs/clean2_%j.log
#SBATCH --error=updated_2/mask_vs_leak/outputs/clean2_%j.log
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
python updated_2/clutter.py --input_pkl data/BNB1723__nightstand.pkl data/BNB1182__barcart.pkl data/ADT1292__utilitycart.pkl data/BNB2909__nightstand.pkl data/ADT1292__shelvingunit.pkl data/HWD0045__storagecabinet.pkl data/BNB1182__woodenbox.pkl  \
    --out_dir updated_2/mask_vs_leak/img_clutter --levels 0.25 --modes image
python updated_2/mask_vs_leak/oracle_mask.py --seeds 0 1 2 \
    --out_dir updated_2/mask_vs_leak/outputs/clean_test \
    --objects updated_2/mask_vs_leak/img_clutter/BNB1723__nightstand__image__L025.pkl updated_2/mask_vs_leak/img_clutter/BNB1182__barcart__image__L025.pkl updated_2/mask_vs_leak/img_clutter/ADT1292__utilitycart__image__L025.pkl updated_2/mask_vs_leak/img_clutter/BNB2909__nightstand__image__L025.pkl updated_2/mask_vs_leak/img_clutter/ADT1292__shelvingunit__image__L025.pkl updated_2/mask_vs_leak/img_clutter/HWD0045__storagecabinet__image__L025.pkl updated_2/mask_vs_leak/img_clutter/BNB1182__woodenbox__image__L025.pkl 
echo "=== clean-mask half 2 complete ==="

#!/bin/bash
#SBATCH --job-name=u2_stage3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=08:00:00
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

C=updated_2/outputs/stage1/clutter
OBJECTS=(
    ADT1292__woodentable
    ADT1292__coffeetable
    ADT1292__nightstand
    BNB1182__sidetable
    HWD0045__desktable
)

# `points` is the mode that actually produces leakage (the image branch does not
# move the geometry), so the method is evaluated there. `both` is included because
# it is the realistic case.
for MODE in points both; do
  for OBJ in "${OBJECTS[@]}"; do
    PKL=$C/${OBJ}__${MODE}__L025.pkl
    LAB=$C/${OBJ}__${MODE}__L025__label.npz
    if [ ! -f "$PKL" ]; then echo "MISSING $PKL — run stage 1 first"; continue; fi
    echo "=== $OBJ / $MODE ==="
    python updated_2/latent_seg.py \
        --clutter_pkl "$PKL" --label "$LAB" \
        --methods baseline crop gate refeed --seeds 0 1 2 \
        --evidence clustered --segmenter support \
        --out_dir updated_2/outputs/stage3_support
  done
done

# Evidence ablation on one object: `assigned` should FAIL (the intruder's points are
# in the set it trusts), `oracle` bounds how much the clustering costs.
echo "=== evidence ablation ==="
for EV in assigned oracle; do
  python updated_2/latent_seg.py \
      --clutter_pkl $C/ADT1292__woodentable__points__L025.pkl \
      --label       $C/ADT1292__woodentable__points__L025__label.npz \
      --methods baseline gate --seeds 0 1 2 --evidence $EV \
      --out_dir updated_2/outputs/stage3_ablation_$EV
done

echo "=== comparison sheets ==="
python updated_2/compare.py --results updated_2/outputs/stage3_support/results.json \
    --group method --out updated_2/outputs/stage3_support/comparison/methods.png

echo "Done. results: updated_2/outputs/stage3_support/results.json"

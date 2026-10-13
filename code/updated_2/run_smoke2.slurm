#!/bin/bash
#SBATCH --job-name=u2_smoke2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/isha.tariq/Desktop/3d_experiments/ShapeR/updated_2/logs/%x_%j.out
#SBATCH --error=/home/isha.tariq/Desktop/3d_experiments/ShapeR/updated_2/logs/%x_%j.err
set +e
cd ~/Desktop/3d_experiments/ShapeR
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
module load cuda/12.8 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYOPENGL_PLATFORM=egl
echo "Running on: $(hostname)"

D=updated_2/outputs/_smoke2
python updated_2/clutter.py --input_pkl data/ADT1292__woodentable.pkl \
    --levels 0.25 --modes points both --n_views 16 --out_dir $D/clutter

echo "=== latent probe: is the latent set spatially segmentable? ==="
python updated_2/latent_probe.py --input_pkl $D/clutter/ADT1292__woodentable__clean.pkl \
    --out $D/latent_probe.json

for EV in clustered assigned oracle; do
  echo "=== latent seg, evidence=$EV ==="
  python updated_2/latent_seg.py \
      --clutter_pkl $D/clutter/ADT1292__woodentable__points__L025.pkl \
      --label       $D/clutter/ADT1292__woodentable__points__L025__label.npz \
      --methods baseline crop gate refeed --seeds 0 --evidence $EV \
      --out_dir $D/stage2_$EV
done
echo "SMOKE2 DONE"

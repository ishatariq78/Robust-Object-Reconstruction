#!/bin/bash
#SBATCH --job-name=mg_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
cd ~/Desktop/3d_experiments/ShapeR              # repo root (needs model/, dataset/, checkpoints/)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
module load cuda/12.8 2>/dev/null || module load cuda/12.2 2>/dev/null || true
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running on: $(hostname)"; nvidia-smi | head -12

# --- STEP 0: verify the surgery is correct BEFORE training (fast) ---
echo "=== self-test (zero-init identity + bias-live) ==="
python updated_shapeR/mask_guided/selftest.py --pkl data/ADT1292__armchair.pkl --num_views 8

# --- TRAIN set: 18 diverse objects with GT meshes.  NO STOOLS (all stools are held out). ---
#     Disjoint from the held-out TEST set in test_mask_guided.slurm (asserted there).
TRAIN_PKLS=(
    data/ADT1292__armchair.pkl
    data/ADT1292__armchair_2.pkl
    data/ADT1292__chair.pkl
    data/ADT1292__chair_2.pkl
    data/ADT1292__metalchair.pkl
    data/BNB1723__chair.pkl
    data/BNB1182__diningchair.pkl
    data/BNB2727__armchair.pkl
    data/ADT1292__coffeetable.pkl
    data/ADT1292__consoletable.pkl
    data/BNB1045__coffeetable.pkl
    data/ADT1292__nightstand.pkl
    data/BNB1182__nightstand.pkl
    data/BNB1045__nightstand.pkl
    data/ADT1292__box.pkl
    data/ADT1292__clock.pkl
    data/ADT1292__coffeemachine.pkl
    data/ADT1292__airpurifier.pkl
)

echo "=== training mask-guidance params on ${#TRAIN_PKLS[@]} samples (Option A + B, backbone frozen) ==="
python updated_shapeR/mask_guided/train.py \
    --data_pkls "${TRAIN_PKLS[@]}" \
    --num_views 8 \
    --epochs 30 \
    --lr 1e-3 \
    --mask_loss_weight 0.0 \
    --out_ckpt updated_shapeR/outputs/mask_guided.ckpt

echo "Done. Trained weights: updated_shapeR/outputs/mask_guided.ckpt"

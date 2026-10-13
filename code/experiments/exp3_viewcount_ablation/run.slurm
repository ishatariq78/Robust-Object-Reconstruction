#!/bin/bash
#SBATCH --job-name=exp3_viewcount
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e
cd ~/Desktop/3d_experiments/ShapeR

source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper

# ── CUDA setup ────────────────────────────────────────────────────────────────
module load cuda/12.8 2>/dev/null || module load cuda/12.2 2>/dev/null || true
if [ -d /usr/local/cuda-12.8 ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
elif [ -d /usr/local/cuda-12.2 ]; then
    export CUDA_HOME=/usr/local/cuda-12.2
else
    export CUDA_HOME=/usr/local/cuda
fi
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_INCLUDE=$CUDA_HOME/include
if [ -d "$CUDA_HOME/lib64" ]; then
    export CUDA_LIB=$CUDA_HOME/lib64
else
    export CUDA_LIB=$CUDA_HOME/lib
fi
export LD_LIBRARY_PATH=$CUDA_LIB:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$CUDA_LIB:${LIBRARY_PATH:-}
export CFLAGS="-I$CUDA_HOME/include"
export CXXFLAGS="-I$CUDA_HOME/include"
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running on: $(hostname)"
nvidia-smi | head -12

# ── Samples to evaluate ───────────────────────────────────────────────────────
SAMPLES=(
    "ADT1292__stool.pkl"
    "BNB2909__pitcher.pkl"
    "ADT1292__chair.pkl"
    "ADT1292__mug.pkl"
    "ADT1292__vase.pkl"
    "BNB1045__sofa.pkl"
    "HWD0045__acousticguitar.pkl"
    "HWD0045__macbook.pkl"
    "ADT1292__tablelamp.pkl"
    "HWD0045__woodenchair.pkl"
    "BNB2909__wickerbasket.pkl"
    "HWD0045__coffeetable.pkl"
    "ADT1292__trashbin.pkl"
    "BNB1182__armchair.pkl"
    "HWD0045__robotvacuum.pkl"
)

# ── Download samples if not already present ───────────────────────────────────
mkdir -p data
for sample in "${SAMPLES[@]}"; do
    if [ ! -f "data/$sample" ]; then
        echo "Downloading $sample ..."
        python -c "from dataset.download import setup_data; setup_data('$sample')"
    else
        echo "Already have: data/$sample"
    fi
done

# ── Run view count ablation ───────────────────────────────────────────────────
# The model is loaded once inside run_eval.py and reused across all view counts.
# All conditions use 25 denoising steps for a fair comparison.

INPUT_PKLS=()
for sample in "${SAMPLES[@]}"; do
    INPUT_PKLS+=("data/$sample")
done

mkdir -p outputs/exp3_viewcount

echo ""
echo "Starting view count ablation: 1, 4, 8, 16 views (25 steps each)..."
python experiments/exp3_viewcount_ablation/run_eval.py \
    --input_pkls "${INPUT_PKLS[@]}" \
    --num_views 1 4 8 16 \
    --num_steps 25 \
    --output_dir outputs/exp3_viewcount \
    --n_points 100000

echo ""
echo "EXPERIMENT 3 COMPLETE"
echo "Results in: outputs/exp3_viewcount/"
ls outputs/exp3_viewcount/

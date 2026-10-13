#!/bin/bash
#SBATCH --job-name=exp1_pc_degrade
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=8:00:00
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
# These are confirmed to exist on facebook/ShapeR-Evaluation HuggingFace dataset.
# Add more sample names here as needed.
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

# ── Generate degraded pkl variants ───────────────────────────────────────────
INPUT_PKLS=()
for sample in "${SAMPLES[@]}"; do
    INPUT_PKLS+=("data/$sample")
done

echo ""
echo "Generating point cloud degradation variants..."
python experiments/exp1_pointcloud_degradation/prepare_data.py \
    --input_pkls "${INPUT_PKLS[@]}" \
    --output_dir data/exp1_pointcloud

echo ""
echo "Variants generated:"
ls -lh data/exp1_pointcloud/

# ── Run evaluation for each condition ────────────────────────────────────────
CONDITIONS=(baseline sparse_50 sparse_25 sparse_10 noisy)
mkdir -p outputs/exp1_pointcloud

for cond in "${CONDITIONS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Evaluating condition: $cond"
    echo "============================================================"

    COND_PKLS=()
    for sample in "${SAMPLES[@]}"; do
        stem="${sample%.pkl}"
        COND_PKLS+=("data/exp1_pointcloud/${stem}__${cond}.pkl")
    done

    python evaluation/eval.py \
        --input_pkls "${COND_PKLS[@]}" \
        --is_local_path \
        --config balance \
        --output_dir "outputs/exp1_pointcloud/${cond}" \
        --eval_output "outputs/exp1_pointcloud/${cond}/results.json" \
        --save_meshes

    echo "Done: $cond"
done

# ── Print summary across all conditions ──────────────────────────────────────
echo ""
echo "============================================================"
echo "EXPERIMENT 1 COMPLETE - Results Summary"
echo "============================================================"
for cond in "${CONDITIONS[@]}"; do
    echo ""
    echo "--- $cond ---"
    python -c "
import json
with open('outputs/exp1_pointcloud/${cond}/results.json') as f:
    r = json.load(f)
s = r.get('summary', {})
cd = s.get('CD(x10^2)', s.get('chamfer', {}))
print(f'  Samples evaluated: {r[\"num_samples\"]}')
if 'f-score' in s:
    print(f'  F-Score: {s[\"f-score\"][\"mean\"]:.4f} +/- {s[\"f-score\"][\"std\"]:.4f}')
if 'normals' in s:
    print(f'  NC:      {s[\"normals\"][\"mean\"]:.4f} +/- {s[\"normals\"][\"std\"]:.4f}')
for k in s:
    if 'chamfer' in k.lower() or 'CD' in k:
        print(f'  {k}:  {s[k][\"mean\"]:.4f} +/- {s[k][\"std\"]:.4f}')
" 2>/dev/null || cat "outputs/exp1_pointcloud/${cond}/results.json"
done

echo ""
echo "All results in: outputs/exp1_pointcloud/"
ls outputs/exp1_pointcloud/

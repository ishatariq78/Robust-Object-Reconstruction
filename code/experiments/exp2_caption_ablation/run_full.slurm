#!/bin/bash
#SBATCH --job-name=exp2_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Runs all 4 conditions sequentially: original, wrong_category, vague, no_caption
# Prepares caption variants once, then evaluates each condition in turn.

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

# ── All 178 samples ───────────────────────────────────────────────────────────
SAMPLES=(
    "ADT1292__airpurifier.pkl"
    "ADT1292__airpurifier_2.pkl"
    "ADT1292__armchair.pkl"
    "ADT1292__armchair_2.pkl"
    "ADT1292__armchair_3.pkl"
    "ADT1292__armchair_4.pkl"
    "ADT1292__banana.pkl"
    "ADT1292__birdhousetoy.pkl"
    "ADT1292__bowl.pkl"
    "ADT1292__box.pkl"
    "ADT1292__chair.pkl"
    "ADT1292__chair_2.pkl"
    "ADT1292__chair_3.pkl"
    "ADT1292__clock.pkl"
    "ADT1292__coffeecup.pkl"
    "ADT1292__coffeemachine.pkl"
    "ADT1292__coffeetable.pkl"
    "ADT1292__consoletable.pkl"
    "ADT1292__cuttingboard.pkl"
    "ADT1292__dishrack.pkl"
    "ADT1292__dishsoap.pkl"
    "ADT1292__donut.pkl"
    "ADT1292__footstool.pkl"
    "ADT1292__graycouch.pkl"
    "ADT1292__jar.pkl"
    "ADT1292__kitchenisland.pkl"
    "ADT1292__knife.pkl"
    "ADT1292__metalbowl.pkl"
    "ADT1292__metalchair.pkl"
    "ADT1292__muffinpan.pkl"
    "ADT1292__mug.pkl"
    "ADT1292__nightstand.pkl"
    "ADT1292__pictureframe.pkl"
    "ADT1292__pot.pkl"
    "ADT1292__scissors.pkl"
    "ADT1292__shelvingunit.pkl"
    "ADT1292__shelvingunit_2.pkl"
    "ADT1292__speaker.pkl"
    "ADT1292__stool.pkl"
    "ADT1292__stool_2.pkl"
    "ADT1292__tablelamp.pkl"
    "ADT1292__tallcabinet.pkl"
    "ADT1292__televisionscreen.pkl"
    "ADT1292__trashbin.pkl"
    "ADT1292__tvstand.pkl"
    "ADT1292__utilitycart.pkl"
    "ADT1292__vase.pkl"
    "ADT1292__vase_2.pkl"
    "ADT1292__vase_3.pkl"
    "ADT1292__woodentable.pkl"
    "BNB1045__barstool.pkl"
    "BNB1045__bed.pkl"
    "BNB1045__bedsidelamp.pkl"
    "BNB1045__chair.pkl"
    "BNB1045__chair_2.pkl"
    "BNB1045__chair_3.pkl"
    "BNB1045__coffeecup.pkl"
    "BNB1045__coffeetable.pkl"
    "BNB1045__decorativevase.pkl"
    "BNB1045__desklamp.pkl"
    "BNB1045__heater.pkl"
    "BNB1045__highchair.pkl"
    "BNB1045__lemon.pkl"
    "BNB1045__map.pkl"
    "BNB1045__map_2.pkl"
    "BNB1045__nightstand.pkl"
    "BNB1045__nightstand_2.pkl"
    "BNB1045__nightstand_3.pkl"
    "BNB1045__paperroll.pkl"
    "BNB1045__soapbottle.pkl"
    "BNB1045__soapdispenser.pkl"
    "BNB1045__soapdispenser_2.pkl"
    "BNB1045__sofa.pkl"
    "BNB1045__tablelamp.pkl"
    "BNB1045__trashcan.pkl"
    "BNB1045__wovenstool.pkl"
    "BNB1182__armchair.pkl"
    "BNB1182__armchair_2.pkl"
    "BNB1182__barcart.pkl"
    "BNB1182__bed.pkl"
    "BNB1182__cabinet.pkl"
    "BNB1182__chair.pkl"
    "BNB1182__coffeecup.pkl"
    "BNB1182__cup.pkl"
    "BNB1182__diningchair.pkl"
    "BNB1182__nightstand.pkl"
    "BNB1182__nightstand_2.pkl"
    "BNB1182__sidetable.pkl"
    "BNB1182__stool.pkl"
    "BNB1182__tablelamp.pkl"
    "BNB1182__topiary.pkl"
    "BNB1182__trashcan.pkl"
    "BNB1182__woodenbox.pkl"
    "BNB1723__armchair.pkl"
    "BNB1723__bathroommirror.pkl"
    "BNB1723__bed.pkl"
    "BNB1723__bedsidelamp.pkl"
    "BNB1723__bench.pkl"
    "BNB1723__ceilinglight.pkl"
    "BNB1723__chair.pkl"
    "BNB1723__diningfurniture.pkl"
    "BNB1723__endofbedbench.pkl"
    "BNB1723__endofbedbench_2.pkl"
    "BNB1723__homeofficefurniture.pkl"
    "BNB1723__lighting.pkl"
    "BNB1723__livingroomfurniture.pkl"
    "BNB1723__microwave.pkl"
    "BNB1723__nightstand.pkl"
    "BNB1723__patiofurniture.pkl"
    "BNB1723__teakettle.pkl"
    "BNB1723__toaster.pkl"
    "BNB1723__woodenbench.pkl"
    "BNB2727__armchair.pkl"
    "BNB2727__armchair_2.pkl"
    "BNB2727__artpiece.pkl"
    "BNB2727__bench.pkl"
    "BNB2727__candlestand.pkl"
    "BNB2727__clock.pkl"
    "BNB2727__decorativesculpture.pkl"
    "BNB2727__footrest.pkl"
    "BNB2727__lighting.pkl"
    "BNB2727__lighting_2.pkl"
    "BNB2727__lighting_3.pkl"
    "BNB2727__lighting_4.pkl"
    "BNB2727__stool.pkl"
    "BNB2909__armchair.pkl"
    "BNB2909__armchair_2.pkl"
    "BNB2909__chair.pkl"
    "BNB2909__chair_2.pkl"
    "BNB2909__coffeetable.pkl"
    "BNB2909__diningchair.pkl"
    "BNB2909__kettle.pkl"
    "BNB2909__nightstand.pkl"
    "BNB2909__pitcher.pkl"
    "BNB2909__plantpot.pkl"
    "BNB2909__trashcan.pkl"
    "BNB2909__wickerbasket.pkl"
    "HWD0045__abstractpainting.pkl"
    "HWD0045__acousticguitar.pkl"
    "HWD0045__barstool.pkl"
    "HWD0045__cabinet.pkl"
    "HWD0045__ceravebottle.pkl"
    "HWD0045__chocolatetin.pkl"
    "HWD0045__coffeecup.pkl"
    "HWD0045__coffeetable.pkl"
    "HWD0045__crochettoy.pkl"
    "HWD0045__cuttingboard.pkl"
    "HWD0045__desktable.pkl"
    "HWD0045__electrickettle.pkl"
    "HWD0045__electrictoothbrush.pkl"
    "HWD0045__gamecontroller.pkl"
    "HWD0045__handsanitizer.pkl"
    "HWD0045__iphonemobile.pkl"
    "HWD0045__laundrybasket.pkl"
    "HWD0045__lightfixture.pkl"
    "HWD0045__lightfixture_2.pkl"
    "HWD0045__macbook.pkl"
    "HWD0045__medicinebottle.pkl"
    "HWD0045__oliveoilbottle.pkl"
    "HWD0045__ottoman.pkl"
    "HWD0045__plasticbottle.pkl"
    "HWD0045__popcorntin.pkl"
    "HWD0045__remotecontrol.pkl"
    "HWD0045__robotvacuum.pkl"
    "HWD0045__scissors.pkl"
    "HWD0045__shampoobottle.pkl"
    "HWD0045__spicebottle.pkl"
    "HWD0045__spicecan.pkl"
    "HWD0045__stackedbooks.pkl"
    "HWD0045__storagecabinet.pkl"
    "HWD0045__wardrobe.pkl"
    "HWD0045__washingmachine.pkl"
    "HWD0045__woodenbench.pkl"
    "HWD0045__woodenchair.pkl"
    "HWD0045__woodenchair_2.pkl"
    "HWD0045__woodenchair_3.pkl"
    "HWD0045__woodentable.pkl"
    "HWD0045__wovenbasket.pkl"
)

# ── Download all samples (existence-checked) ──────────────────────────────────
mkdir -p data logs
for sample in "${SAMPLES[@]}"; do
    if [ ! -f "data/$sample" ]; then
        echo "Downloading $sample ..."
        python -c "from dataset.download import setup_data; setup_data('$sample')"
    fi
done

# ── Prepare all caption variants (once) ──────────────────────────────────────
INPUT_PKLS=()
for sample in "${SAMPLES[@]}"; do
    INPUT_PKLS+=("data/$sample")
done

echo "Generating caption variants for all 178 samples..."
python experiments/exp2_caption_ablation/prepare_data.py \
    --input_pkls "${INPUT_PKLS[@]}" \
    --output_dir data/exp2_full
echo "Data preparation complete."

# ── Evaluate each condition sequentially ──────────────────────────────────────
CONDITIONS=(original wrong_category vague no_caption)
mkdir -p outputs_full/exp2_caption

for COND in "${CONDITIONS[@]}"; do
    COND_PKLS=()
    for sample in "${SAMPLES[@]}"; do
        stem="${sample%.pkl}"
        COND_PKLS+=("data/exp2_full/${stem}__${COND}.pkl")
    done

    mkdir -p "outputs_full/exp2_caption/${COND}"

    echo ""
    echo "============================================================"
    echo "Evaluating condition: $COND  (n=178)"
    echo "============================================================"

    python evaluation/eval.py \
        --input_pkls "${COND_PKLS[@]}" \
        --is_local_path \
        --config balance \
        --output_dir "outputs_full/exp2_caption/${COND}" \
        --eval_output "outputs_full/exp2_caption/${COND}/results.json" \
        --save_meshes

    echo "Done: $COND"
    python -c "
import json
with open('outputs_full/exp2_caption/$COND/results.json') as f:
    r = json.load(f)
s = r.get('summary', {})
print(f'  Samples: {r[\"num_samples\"]}')
if 'f-score' in s:
    print(f'  F-Score: {s[\"f-score\"][\"mean\"]:.4f} +/- {s[\"f-score\"][\"std\"]:.4f}')
if 'normals' in s:
    print(f'  NC:      {s[\"normals\"][\"mean\"]:.4f} +/- {s[\"normals\"][\"std\"]:.4f}')
if 'chamfer' in s:
    print(f'  CD:      {s[\"chamfer\"][\"mean\"]*100:.4f} +/- {s[\"chamfer\"][\"std\"]*100:.4f}')
" 2>/dev/null || cat "outputs_full/exp2_caption/$COND/results.json"
done

echo ""
echo "EXPERIMENT 2 COMPLETE. Results in: outputs_full/exp2_caption/"

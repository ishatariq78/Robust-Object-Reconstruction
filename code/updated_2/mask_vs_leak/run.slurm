#!/bin/bash
#SBATCH --job-name=maskleak
#SBATCH --output=updated_2/mask_vs_leak/outputs/run_%j.log
#SBATCH --error=updated_2/mask_vs_leak/outputs/run_%j.log
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

# Does the mask fine-tune suppress the intruder? Scored with Leak%, not Extra%.
# ONE job on purpose: mask_guided/infer.py writes a fixed /tmp/mg_mesh.obj, so two concurrent
# jobs would overwrite each other's mesh mid-export.
# NOTE: -e but NOT -u — conda's activate scripts reference unbound vars and would abort the job.
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
cd /home/isha.tariq/Desktop/3d_experiments/ShapeR

PKLS=" updated_2/outputs/stage1/clutter/ADT1292__woodentable__image__L025.pkl updated_2/outputs/stage1/clutter/ADT1292__woodentable__both__L025.pkl updated_2/outputs/stage1/clutter/ADT1292__woodentable__points__L025.pkl updated_2/outputs/stage1/clutter/HWD0045__desktable__image__L025.pkl updated_2/outputs/stage1/clutter/HWD0045__desktable__both__L025.pkl updated_2/outputs/stage1/clutter/HWD0045__desktable__points__L025.pkl updated_2/outputs/stage1/clutter/ADT1292__coffeetable__image__L025.pkl updated_2/outputs/stage1/clutter/ADT1292__coffeetable__both__L025.pkl updated_2/outputs/stage1/clutter/ADT1292__coffeetable__points__L025.pkl updated_2/outputs/stage1/clutter/ADT1292__nightstand__image__L025.pkl updated_2/outputs/stage1/clutter/ADT1292__nightstand__both__L025.pkl updated_2/outputs/stage1/clutter/ADT1292__nightstand__points__L025.pkl updated_2/outputs/stage1/clutter/BNB1182__sidetable__image__L025.pkl updated_2/outputs/stage1/clutter/BNB1182__sidetable__both__L025.pkl updated_2/outputs/stage1/clutter/BNB1182__sidetable__points__L025.pkl"

# arm 1: guidance ON (the trained mask conv + trained lambda)
python updated_shapeR/mask_guided/infer.py --input_pkls $PKLS \
    --guided_ckpt updated_shapeR/outputs/mask_guided.ckpt \
    --config balance --num_seeds 3 --seed_base 0 \
    --output_dir updated_2/mask_vs_leak/outputs/guided

# arm 2: identical path, guidance OFF — the control
python updated_shapeR/mask_guided/infer.py --input_pkls $PKLS \
    --guided_ckpt updated_shapeR/outputs/mask_guided.ckpt --disable_guidance \
    --config balance --num_seeds 3 --seed_base 0 \
    --output_dir updated_2/mask_vs_leak/outputs/baseline

python updated_2/mask_vs_leak/score.py
echo "=== mask-vs-leak complete ==="

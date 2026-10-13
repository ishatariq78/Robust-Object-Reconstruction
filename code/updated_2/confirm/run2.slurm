#!/bin/bash
#SBATCH --job-name=abst2
#SBATCH --output=updated_2/confirm/outputs/run2_%j.log
#SBATCH --error=updated_2/confirm/outputs/run2_%j.log
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

# Abstention confirmation, half 2/2. Objects and thresholds are frozen in
# updated_2/confirm/protocol.py and must not be edited while this runs.
# -e but NOT -u: conda's activate scripts reference unbound vars and would abort the job.
set -eo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate shaper
cd /home/isha.tariq/Desktop/3d_experiments/ShapeR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYOPENGL_PLATFORM=egl
module load cuda/12.8 2>/dev/null || module load cuda/12.2 2>/dev/null || true

C=updated_2/confirm/outputs/clutter
mkdir -p $C

# 1) cluttered inputs — same script and settings as the original Stage 1.
#    Re-run is cheap and idempotent; half 1's clutter already succeeded before the loop bug.
python updated_2/clutter.py --input_pkl data/BNB1723__nightstand.pkl data/BNB1182__barcart.pkl data/ADT1292__utilitycart.pkl data/BNB2909__nightstand.pkl data/ADT1292__shelvingunit.pkl data/HWD0045__storagecabinet.pkl data/BNB1182__woodenbox.pkl --out_dir $C --levels 0.25 --modes both

# 2) baseline and refeed, three seeds. Separate out_dir per half so the two jobs
#    cannot overwrite each other's results.json.
for OBJ in BNB1723__nightstand BNB1182__barcart ADT1292__utilitycart BNB2909__nightstand ADT1292__shelvingunit HWD0045__storagecabinet BNB1182__woodenbox ; do
  PKL=$C/${OBJ}__both__L025.pkl
  LAB=$C/${OBJ}__both__L025__label.npz
  if [ ! -f "$PKL" ]; then echo "MISSING $PKL"; continue; fi
  echo "=== $OBJ ==="
  python updated_2/latent_seg.py --clutter_pkl "$PKL" --label "$LAB" \
      --methods baseline refeed --seeds 0 1 2 \
      --evidence clustered --segmenter support \
      --out_dir updated_2/confirm/outputs/stage_half2
done
echo "=== abstention confirmation half 2 complete ==="

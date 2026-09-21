#!/bin/bash
#SBATCH -J pslg-b3-wgan
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00
#SBATCH -o /home/%u/pslg_logs/%x-%j.out
#SBATCH -e /home/%u/pslg_logs/%x-%j.err

# B3-G conditional WGAN-GP (roadmap D3). Adversarial training is the least
# stable route: if it fails, record the negative result and move on — do not
# silently retry with different hyperparameters (roadmap §3.3).
set -euo pipefail

echo "started=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"

module purge
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
module list

source "$(conda info --base)/bin/activate"
conda activate pslg-nilm

: "${PSLG_PROJECT_ROOT:?submit with PSLG_PROJECT_ROOT}"
cd "$PSLG_PROJECT_ROOT"
git rev-parse HEAD
nvidia-smi

: "${PSLG_DONOR_LIBRARY:?submit with PSLG_DONOR_LIBRARY}"
: "${PSLG_SEED:?submit with PSLG_SEED}"
: "${PSLG_EPOCHS:?submit with PSLG_EPOCHS}"
: "${PSLG_COUNT:?submit with PSLG_COUNT}"
RUN_DIR="${PSLG_RUN_ROOT:-$HOME/pslg_artifacts}/b3_wgan_s${PSLG_SEED}_${SLURM_JOB_ID}"

python scripts/train_full_cycle_generator.py \
  --route wgan \
  --real-library-dir "$PSLG_DONOR_LIBRARY" \
  --output-dir "$RUN_DIR/checkpoint" \
  --epochs "$PSLG_EPOCHS" --seed "$PSLG_SEED" \
  --n-critic 5 \
  --device cuda

python scripts/generate_full_cycles.py \
  --route wgan \
  --checkpoint-dir "$RUN_DIR/checkpoint" \
  --real-library-dir "$PSLG_DONOR_LIBRARY" \
  --output-dir "$RUN_DIR/cycles" \
  --count "$PSLG_COUNT" --seed "$PSLG_SEED" --device cuda

python scripts/evaluate_synthetic_quality.py \
  --synthetic-dir "$RUN_DIR/cycles" \
  --real-library-dir "$PSLG_DONOR_LIBRARY" \
  --output "$RUN_DIR/quality_report.json"

python scripts/audit_generation_memorization.py \
  --synthetic-dir "$RUN_DIR/cycles" \
  --real-library-dir "$PSLG_DONOR_LIBRARY" \
  --max-replication-rate 0.01 \
  --output "$RUN_DIR/memorization_report.json"

echo "finished=$(date --iso-8601=seconds)"

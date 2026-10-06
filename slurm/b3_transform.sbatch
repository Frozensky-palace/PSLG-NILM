#!/bin/bash
#SBATCH -J pslg-b3-transform
#SBATCH -p RTX3090
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /home/%u/pslg_logs/%x-%j.out
#SBATCH -e /home/%u/pslg_logs/%x-%j.err

# B3-T real-cycle transform (roadmap D1): no neural training, CPU-only.
# Needs PSLG_PROJECT_ROOT (shared frozen placement comes later from
# build_shared_placement_schedule.py when this route enters C4-style runs).
set -euo pipefail

echo "started=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"

module purge
module load miniconda3/25.5.1-0
module list

source "$(conda info --base)/bin/activate"
conda activate pslg-nilm

: "${PSLG_PROJECT_ROOT:?submit with PSLG_PROJECT_ROOT}"
cd "$PSLG_PROJECT_ROOT"
git rev-parse HEAD

: "${PSLG_DONOR_LIBRARY:?submit with PSLG_DONOR_LIBRARY}"
: "${PSLG_COUNT:?submit with PSLG_COUNT}"
: "${PSLG_SEED:?submit with PSLG_SEED}"
RUN_ID="b3t_s${PSLG_SEED}_${SLURM_JOB_ID}"
RUN_DIR="${PSLG_RUN_ROOT:-$HOME/pslg_artifacts}/$RUN_ID"

python scripts/generate_full_cycles.py \
  --route transform \
  --real-library-dir "$PSLG_DONOR_LIBRARY" \
  --output-dir "$RUN_DIR" \
  --count "$PSLG_COUNT" --seed "$PSLG_SEED"

python scripts/evaluate_synthetic_quality.py \
  --synthetic-dir "$RUN_DIR" \
  --real-library-dir "$PSLG_DONOR_LIBRARY" \
  --output "$RUN_DIR/quality_report.json"

# NOTE: near-replication is a defining property of B3-T (perturbed real
# cycles); the audit is run with the route-appropriate cap and recorded.
python scripts/audit_generation_memorization.py \
  --synthetic-dir "$RUN_DIR" \
  --real-library-dir "$PSLG_DONOR_LIBRARY" \
  --max-replication-rate 1.0 \
  --output "$RUN_DIR/memorization_report.json"

echo "finished=$(date --iso-8601=seconds)"

#!/bin/bash
#SBATCH -J pslg-b5-hsmm
#SBATCH -p RTX3090
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o /home/%u/pslg_logs/%x-%j.out
#SBATCH -e /home/%u/pslg_logs/%x-%j.err

# B5 HSMM constrained composition (roadmap F1-F3): CPU only — fitting and
# composition are pure numpy; the neural primitives come from the B4 pool.
# Runs the default B5 plus the endpoint_match ablation, both through the
# batch-0 gates.
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

: "${PSLG_STATE_LIBRARY:?submit with PSLG_STATE_LIBRARY (= state library v1)}"
: "${PSLG_PRIMITIVES_DIR:?submit with PSLG_PRIMITIVES_DIR (= B4 cycles dir)}"
: "${PSLG_DONOR_LIBRARY:?submit with PSLG_DONOR_LIBRARY (= real cycle library)}"
: "${PSLG_COUNT:?submit with PSLG_COUNT}"
: "${PSLG_SEED:?submit with PSLG_SEED}"
RUN_DIR="${PSLG_RUN_ROOT:-$HOME/pslg_artifacts}/b5_s${PSLG_SEED}_${SLURM_JOB_ID}"

python scripts/fit_hsmm_composer.py \
  --state-library-dir "$PSLG_STATE_LIBRARY" \
  --output "$RUN_DIR/hsmm_model.json"

for mode in none endpoint_match; do
  tag=$(echo "$mode" | tr '_' '-')
  python scripts/compose_b5_cycles.py \
    --primitives-dir "$PSLG_PRIMITIVES_DIR" \
    --hsmm-json "$RUN_DIR/hsmm_model.json" \
    --boundary-mode "$mode" \
    --output-dir "$RUN_DIR/cycles_$tag" \
    --count "$PSLG_COUNT" --seed "$PSLG_SEED"
  python scripts/evaluate_synthetic_quality.py \
    --synthetic-dir "$RUN_DIR/cycles_$tag" \
    --real-library-dir "$PSLG_DONOR_LIBRARY" \
    --output "$RUN_DIR/quality_report_$tag.json"
  python scripts/audit_generation_memorization.py \
    --synthetic-dir "$RUN_DIR/cycles_$tag" \
    --real-library-dir "$PSLG_DONOR_LIBRARY" \
    --output "$RUN_DIR/memorization_report_$tag.json"
done

echo "finished=$(date --iso-8601=seconds)"

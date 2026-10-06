#!/bin/bash
#SBATCH -J pslg-b4diff
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH -o /home/%u/pslg_logs/%x-%j.out
#SBATCH -e /home/%u/pslg_logs/%x-%j.err

# B4WD B4DIFF arm (B4′ second item, prereg reports/b4wd/
# b4wd_protocol_prereg_v1.md §3): segment-level masked diffusion.
# Budgets: epochs 300, diffusion steps 500, width 32, seed 17 (lr not
# prereg-pinned; CLI default 1e-3 = B3-D precedent). Checkpoint rule is
# preregistered: final-epoch model.pt, no curve-based epoch selection.
# Time limit 2 days (B3 sbatch 同规格; wgan arm = b4wd_wgan.sbatch,
# 1 day). Diffusion sampling alone is ~500 ancestral loops per segment —
# never on the login node. Quality/memorization gates + machine assertions
# (incl. the §4 peak-cap evidence) run afterwards in
# scripts/b4wd_server_pipeline.sh on the login node.
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

: "${PSLG_STATE_LIBRARY:?submit with PSLG_STATE_LIBRARY (= state library v1)}"
: "${PSLG_DONOR_LIBRARY:?submit with PSLG_DONOR_LIBRARY (= real cycle library)}"
: "${PSLG_SEED:?submit with PSLG_SEED (= 17, prereg §3)}"
: "${PSLG_COUNT:?submit with PSLG_COUNT (= 246, prereg §3)}"
EPOCHS="${PSLG_EPOCHS:-300}"
RUN_DIR="${PSLG_RUN_ROOT:-$HOME/pslg_artifacts}/b4wd_b4diff_s${PSLG_SEED}_${SLURM_JOB_ID}"

echo "arm=b4diff route=primitive-diffusion epochs=$EPOCHS run_dir=$RUN_DIR"
mkdir -p "$RUN_DIR"
# Prereg §2: record the interpreter build — str-hash determinism is only
# guaranteed within one Python build, and the per-segment rng_seeds in the
# cycles manifest must stay re-derivable.
python --version | tee "$RUN_DIR/python_version.txt"

# PYTHONHASHSEED pinned (prereg §2): the per-segment decode seed formula
# salts with hash(synthetic_cycle_id); without this, regeneration is not
# reproducible across processes.
export PYTHONHASHSEED=0

python scripts/train_full_cycle_generator.py \
  --route primitive-diffusion \
  --state-library-dir "$PSLG_STATE_LIBRARY" \
  --output-dir "$RUN_DIR/checkpoint" \
  --epochs "$EPOCHS" --seed "$PSLG_SEED" \
  --device cuda

python scripts/generate_primitive_cycles.py \
  --primitive-source diffusion \
  --checkpoint-dir "$RUN_DIR/checkpoint" \
  --state-library-dir "$PSLG_STATE_LIBRARY" \
  --output-dir "$RUN_DIR/cycles" \
  --count "$PSLG_COUNT" --seed "$PSLG_SEED" --device cuda

echo "RUN_DIR=$RUN_DIR"
echo "finished=$(date --iso-8601=seconds)"

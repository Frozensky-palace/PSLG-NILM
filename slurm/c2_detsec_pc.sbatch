#!/bin/bash
#SBATCH -J pslg-c2-detsecpc
#SBATCH -p A6000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
#SBATCH -o /home/%u/pslg_logs/%x-%j.out
#SBATCH -e /home/%u/pslg_logs/%x-%j.err

# C2 formal DETSEC-PC state discovery (guide §7.12): train-only, k=3/4/5.
# Submit only after C1 passed. The preflight gate verifies the frozen commit,
# the state-discovery manifest and framework imports; any failure aborts.
set -euo pipefail

echo "started=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

module purge
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
module list

source "$(conda info --base)/bin/activate"
conda activate pslg-detsec

: "${PSLG_PROJECT_ROOT:?submit with PSLG_PROJECT_ROOT}"
cd "$PSLG_PROJECT_ROOT"
git rev-parse HEAD
nvidia-smi

: "${PSLG_FROZEN_COMMIT:?submit with PSLG_FROZEN_COMMIT}"
PSLG_MANIFEST="${PSLG_MANIFEST:-$HOME/pslg_manifests/c1_state_discovery_trainonly_manifest.json}"
PSLG_RUN_ROOT="${PSLG_RUN_ROOT:-$HOME/pslg_artifacts}"
RUN_ID="detsecpc_k345_trainonly_${SLURM_JOB_ID}"
RUN_DIR="$PSLG_RUN_ROOT/$RUN_ID"
STDOUT_LOG="$HOME/pslg_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
STDERR_LOG="$HOME/pslg_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
# Do NOT pre-create RUN_DIR: train_state_discovery refuses existing output
# dirs by design, so it must create the run dir itself. Capture the
# environment to a sibling staging file and move it in after training
# (job 4106 died exactly on this pre-creation, 2026-09-21).
ENVIRONMENT="$PSLG_RUN_ROOT/environment_${SLURM_JOB_ID}.json"

python scripts/server_preflight.py \
  --repo-root . --expected-commit "$PSLG_FROZEN_COMMIT" \
  --manifest "$PSLG_MANIFEST" \
  --output-root "$PSLG_RUN_ROOT" --framework tensorflow

python scripts/capture_server_environment.py \
  --repo-root . --output "$ENVIRONMENT"

python scripts/train_state_discovery.py \
  --config config/experiments/core_wm_state_discovery_detsec_pc.yaml \
  --output-root "$RUN_DIR" \
  --run-id "$RUN_ID" \
  --k 3,4,5 --deterministic-tf

mv "$ENVIRONMENT" "$RUN_DIR/environment.json"

python scripts/finalize_server_run.py \
  --run-dir "$RUN_DIR" --repo-root . --phase C2 \
  --data-manifest "$PSLG_MANIFEST" \
  --environment "$RUN_DIR/environment.json" \
  --stdout-log "$STDOUT_LOG" --stderr-log "$STDERR_LOG"

echo "finished=$(date --iso-8601=seconds)"

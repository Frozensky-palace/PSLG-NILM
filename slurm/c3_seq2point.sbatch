#!/bin/bash
#SBATCH -J pslg-c3-s2p
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH -o /home/%u/pslg_logs/%x-%j.out
#SBATCH -e /home/%u/pslg_logs/%x-%j.err

# C3 Seq2Point first-round validation job (guide §7.13): B0/B1/B2, ratio 0.5,
# test stays locked. Submit only after C2 froze the state library and the
# formal B2 recomposition was rebuilt. One job = one (arm, seed).
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
conda activate pslg-nilm
# deterministic CuBLAS workspace required by torch.use_deterministic_algorithms
export CUBLAS_WORKSPACE_CONFIG=:4096:8

: "${PSLG_PROJECT_ROOT:?submit with PSLG_PROJECT_ROOT}"
cd "$PSLG_PROJECT_ROOT"
git rev-parse HEAD
nvidia-smi

: "${PSLG_FROZEN_COMMIT:?submit with PSLG_FROZEN_COMMIT}"
: "${PSLG_MANIFEST:?submit with PSLG_MANIFEST}"
: "${PSLG_EXPERIMENT_DIR:?submit with PSLG_EXPERIMENT_DIR}"
: "${PSLG_ARM:?submit with PSLG_ARM=B0|B1|B2}"
: "${PSLG_SEED:?submit with PSLG_SEED=17|42|73}"
PSLG_RATIO="${PSLG_RATIO:-0p5}"
PSLG_RUN_ROOT="${PSLG_RUN_ROOT:-$HOME/pslg_artifacts}"
RUN_ID="s2p_${PSLG_ARM}_r${PSLG_RATIO}_s${PSLG_SEED}_${SLURM_JOB_ID}"
RUN_DIR="$PSLG_RUN_ROOT/$RUN_ID"
ENVIRONMENT="$RUN_DIR/environment.json"
STDOUT_LOG="$HOME/pslg_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
STDERR_LOG="$HOME/pslg_logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
mkdir -p "$RUN_DIR"

python scripts/server_preflight.py \
  --repo-root . --expected-commit "$PSLG_FROZEN_COMMIT" \
  --manifest "$PSLG_MANIFEST" \
  --output-root "$PSLG_RUN_ROOT" --framework torch

python scripts/capture_server_environment.py \
  --repo-root . --output "$ENVIRONMENT"

python scripts/train_nilm.py \
  --experiment-dir "$PSLG_EXPERIMENT_DIR" \
  --arm "$PSLG_ARM" --seed "$PSLG_SEED" \
  --device cuda \
  --batch-size 128 --steps-per-epoch 200 --max-epochs 30 --patience 5 \
  --validation-count 20000 \
  --output-dir "$RUN_DIR"

python scripts/predict_nilm.py \
  --experiment-dir "$PSLG_EXPERIMENT_DIR" \
  --checkpoint "$RUN_DIR/best_checkpoint.pt" \
  --device cuda \
  --arm "$PSLG_ARM" --partition validation --limit 20000 \
  --output "$RUN_DIR/validation_predictions.npz"

python scripts/evaluate_nilm_predictions.py \
  --predictions "$RUN_DIR/validation_predictions.npz" \
  --output "$RUN_DIR/validation_metrics.json"

python scripts/finalize_server_run.py \
  --run-dir "$RUN_DIR" --repo-root . --phase C3 \
  --data-manifest "$PSLG_MANIFEST" --environment "$ENVIRONMENT" \
  --stdout-log "$STDOUT_LOG" --stderr-log "$STDERR_LOG"

python scripts/collect_run_artifacts.py --run-dir "$RUN_DIR"

echo "finished=$(date --iso-8601=seconds)"

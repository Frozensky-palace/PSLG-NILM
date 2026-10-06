#!/bin/bash
#SBATCH --job-name=sampling-state-smoke
#SBATCH --partition=RTX3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=log/sampling-state-smoke-%j.out
#SBATCH --error=log/sampling-state-smoke-%j.err

# Reuse existing environments; no package installation or data reconstruction.
set -euo pipefail
: "${SLURM_JOB_ID:?submit using sbatch}"
: "${PSLG_SAMPLING_COMMIT:?submit with the current clean commit}"
cd /home/scnu2024024563/NILM-zzz/PSLG-NILM-zzz
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PSLG_SAMPLING_OUTPUT="/home/scnu2024024563/pslg_artifacts/sampling_state_smoke_${SLURM_JOB_ID}"
echo "job_id=$SLURM_JOB_ID output=$PSLG_SAMPLING_OUTPUT"
/home/scnu2024024563/.conda/envs/pslg-nilm/bin/python -u scripts/run_sampling_state_smoke.py \
  --config config/server/zzz_sampling_state_smoke.json \
  --output-dir "$PSLG_SAMPLING_OUTPUT" \
  --expected-commit "$PSLG_SAMPLING_COMMIT"

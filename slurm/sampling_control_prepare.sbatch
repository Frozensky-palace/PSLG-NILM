#!/bin/bash
#SBATCH --job-name=sampling-control-data
#SBATCH --partition=RTX3090
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=log/sampling-control-data-%j.out
#SBATCH --error=log/sampling-control-data-%j.err

# CPU-only data preparation. No GPU, training, installation or test-array reads.
set -euo pipefail
: "${SLURM_JOB_ID:?submit using sbatch}"
: "${PSLG_SAMPLING_COMMIT:?submit with the current clean commit}"
cd /home/scnu2024024563/NILM-zzz/PSLG-NILM-zzz
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONHASHSEED=0

PSLG_SAMPLING_OUTPUT="/home/scnu2024024563/pslg_artifacts/sampling_control_data_${SLURM_JOB_ID}"
echo "job_id=$SLURM_JOB_ID output=$PSLG_SAMPLING_OUTPUT"
/home/scnu2024024563/.conda/envs/pslg-nilm/bin/python -u scripts/prepare_sampling_control.py \
  --config config/server/zzz_sampling_control.json \
  --output-dir "$PSLG_SAMPLING_OUTPUT" \
  --expected-commit "$PSLG_SAMPLING_COMMIT"

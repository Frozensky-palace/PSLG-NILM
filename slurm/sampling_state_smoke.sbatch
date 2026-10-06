#!/bin/bash
#SBATCH --job-name=sampling-state-smoke
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=log/sampling-state-smoke-%j.out
#SBATCH --error=log/sampling-state-smoke-%j.err

# Reuse existing environments; no package installation or data reconstruction.
# h103 reproduced its documented CUDA-init failure in job 12099. h104 passed
# historical checks; the framework smoke below must still verify it each run.
set -euo pipefail
: "${SLURM_JOB_ID:?submit using sbatch}"
: "${PSLG_SAMPLING_COMMIT:?submit with the current clean commit}"
cd /home/scnu2024024563/NILM-zzz/PSLG-NILM-zzz
module purge
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
module list
source "$(conda info --base)/bin/activate"
conda activate pslg-nilm

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
echo "node=${SLURMD_NODENAME:-unknown} cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset} conda_prefix=$CONDA_PREFIX"
python -u scripts/run_sampling_state_smoke.py \
  --config config/server/zzz_sampling_state_smoke.json \
  --output-dir "$PSLG_SAMPLING_OUTPUT" \
  --expected-commit "$PSLG_SAMPLING_COMMIT"

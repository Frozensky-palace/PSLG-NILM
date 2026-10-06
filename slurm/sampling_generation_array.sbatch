#!/bin/bash
#SBATCH --job-name=sampling-generators
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-13%1
#SBATCH --output=log/sampling-generators-%A_%a.out
#SBATCH --error=log/sampling-generators-%A_%a.err
# One arm/rate per job; failures never silently remove other matrix cells.
set -euo pipefail
: "${SLURM_JOB_ID:?submit with sbatch}"
: "${SLURM_ARRAY_JOB_ID:?requires an array allocation}"
: "${SLURM_ARRAY_TASK_ID:?requires an array task index}"
: "${PSLG_SAMPLING_COMMIT:?submit a frozen commit}"
: "${PSLG_SAMPLING_STATES:?requires a completed paired state run}"
cd /home/scnu2024024563/NILM-zzz/PSLG-NILM-zzz
module purge
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
source "$(conda info --base)/bin/activate"
conda activate pslg-nilm
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u scripts/run_sampling_generation.py --stage generator \
  --expected-commit "$PSLG_SAMPLING_COMMIT" \
  --state-root "$PSLG_SAMPLING_STATES" --task-index "$SLURM_ARRAY_TASK_ID" \
  --output-dir "/home/scnu2024024563/pslg_artifacts/sampling_generation_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

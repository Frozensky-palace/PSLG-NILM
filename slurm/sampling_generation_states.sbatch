#!/bin/bash
#SBATCH --job-name=sampling-gen-states
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=log/sampling-gen-states-%j.out
#SBATCH --error=log/sampling-gen-states-%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?submit with sbatch}"
: "${PSLG_SAMPLING_COMMIT:?submit a frozen commit}"
cd /home/scnu2024024563/NILM-zzz/PSLG-NILM-zzz
module purge
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
source "$(conda info --base)/bin/activate"
conda activate pslg-nilm
export PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u scripts/run_sampling_generation.py --stage states \
  --expected-commit "$PSLG_SAMPLING_COMMIT" \
  --output-dir "/home/scnu2024024563/pslg_artifacts/sampling_states_${SLURM_JOB_ID}"

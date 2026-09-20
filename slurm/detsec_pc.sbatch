#!/bin/bash
#SBATCH -J pslg-detsecpc-k4
#SBATCH -p A6000
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 1-00:00:00
#SBATCH -o /home/<user>/pslg_logs/%x-%j.out
#SBATCH -e /home/<user>/pslg_logs/%x-%j.err

# DETSEC-PC train-only state discovery (doc C2). Formal entry:
# scripts/train_state_discovery.py — local smoke passed before this template
# was enabled. Do not submit until the server data bundle verified OK.
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

cd /home/<user>/projects/PSLG-NILM
git rev-parse HEAD
nvidia-smi

python scripts/train_state_discovery.py \
  --config config/experiments/core_wm_state_discovery_detsec_pc.yaml \
  --output-root pslg_artifacts/state_discovery_detsec_pc_k345 \
  --run-id "detsecpc_k345_trainonly_${SLURM_JOB_ID}" \
  --k 3,4,5

echo "finished=$(date --iso-8601=seconds)"

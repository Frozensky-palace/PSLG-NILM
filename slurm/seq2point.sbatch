#!/bin/bash
#SBATCH -J pslg-s2p-b2m-k4-s17
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH -o /home/<user>/pslg_logs/%x-%j.out
#SBATCH -e /home/<user>/pslg_logs/%x-%j.err

# Seq2Point template (roadmap task 14, doc C1.4). Replace <user> and the
# config path with real values; never put credentials in this file. Module
# names must be re-confirmed with `module avail` at submit time.
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

cd /home/<user>/projects/PSLG-NILM
git rev-parse HEAD
nvidia-smi

python scripts/train_nilm.py --experiment-dir pslg_data/nilm_inputs_b2m_k4 \
  --arm B2 --output-dir pslg_artifacts/"${SLURM_JOB_ID}"_s2p_b2_k4_s17 \
  --seed 17

echo "finished=$(date --iso-8601=seconds)"

#!/bin/bash
#SBATCH -J pslg-cvae-b3-s17
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 3-00:00:00
#SBATCH -o /home/%u/pslg_logs/%x-%j.out
#SBATCH -e /home/%u/pslg_logs/%x-%j.err

# STATUS: TEMPLATE ONLY — Phase D/E entry scripts do not exist yet.
# Do not submit. Per guide §5.5, per-route sbatch files (b3_transform,
# b3_cvae, b3_wgan, b3_diffusion, b4_primitive_cvae, b5_hsmm) are added only
# after their Python entries exist and passed the local smoke.
# B3/B4 CVAE generator template (doc D/E). The generator training script lands
# with Phase D; until then this file only documents resources and environment.
# Never submit before the script exists and passed the local smoke.
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

cd "$HOME/projects/PSLG-NILM"
git rev-parse HEAD
nvidia-smi

# Placeholder: Phase D/E will provide scripts/train_full_cycle_generator.py
echo "generator training script is not implemented yet (Phase D/E)."
echo "finished=$(date --iso-8601=seconds)"

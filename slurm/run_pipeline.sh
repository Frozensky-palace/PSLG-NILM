#!/bin/bash
# ============================================================================
# LEGACY: 旧项目时期的脚本，包含旧用户目录与旧仓库名，禁止在新集群使用。
# Formal C1/C2/C3 jobs use slurm/c1_gpu_smoke.sbatch, c2_detsec_pc.sbatch and
# c3_seq2point.sbatch instead (guide §4.6 / §5.5).
# ============================================================================
#SBATCH -J PSLG-PIPELINE
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /home/scnu2023024258/data/code/PSLG-NILM-ADVANCED/slurm/slurm_log/job-%x-%j.out
#SBATCH -e /home/scnu2023024258/data/code/PSLG-NILM-ADVANCED/slurm/slurm_log/job-%x-%j.err
#SBATCH --time=24:00:00

# 单次完整流水线。所有选择都通过 CLI 参数传入，不改动 config。
# 用法: sbatch run_pipeline.sh
# 覆盖:  sbatch --export=ALL,APPLIANCE=kettle,FEATURE_MODEL=bilstm_ae run_pipeline.sh
source "$(dirname "$0")/env.sh"

APPLIANCE="${APPLIANCE:-fridge}"
SEGMENT_METHOD="${SEGMENT_METHOD:-clasp}"
FEATURE_MODEL="${FEATURE_MODEL:-detsec}"
CLUSTER_METHOD="${CLUSTER_METHOD:-kmeans}"
N_CLUSTERS="${N_CLUSTERS:-3,4,5,6}"
STEPS="${STEPS:-extract,segment,feature,cluster,fewshot,pam,split}"

EXTRA_ARGS=()
if [ "$CLUSTER_METHOD" = "kmeans" ] || [ "$CLUSTER_METHOD" = "kmeans-scan" ]; then
    EXTRA_ARGS+=(--n-clusters "$N_CLUSTERS")
fi

python main.py \
    --appliance "$APPLIANCE" \
    --steps "$STEPS" \
    --segment-method "$SEGMENT_METHOD" \
    --feature-model "$FEATURE_MODEL" \
    --cluster-method "$CLUSTER_METHOD" \
    "${EXTRA_ARGS[@]}"

echo "Job finished on: $(date)"

#!/bin/bash
# ============================================================================
# LEGACY: 旧项目时期的脚本，包含旧用户目录与旧仓库名，禁止在新集群使用。
# Formal C1/C2/C3 jobs use slurm/c1_gpu_smoke.sbatch, c2_detsec_pc.sbatch and
# c3_seq2point.sbatch instead (guide §4.6 / §5.5).
# ============================================================================
#SBATCH -J PSLG-SEG-GRID
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /home/scnu2023024258/data/code/PSLG-NILM-ADVANCED/slurm/slurm_log/job-%x-%j.out
#SBATCH -e /home/scnu2023024258/data/code/PSLG-NILM-ADVANCED/slurm/slurm_log/job-%x-%j.err
#SBATCH --time=48:00:00

# 切分方法网格：每种切分方法各跑一条 extract→segment→feature 流水线。
# 不同切分 → 不同 X → 特征缓存自动失效并重训（内容寻址，无需人工管理）。
source "$(dirname "$0")/env.sh"

APPLIANCE="${APPLIANCE:-fridge}"
FEATURE_MODEL="${FEATURE_MODEL:-detsec}"
# 默认网格；可用位置参数或 --export=ALL,SEG_METHODS_ENV="clasp ggs" 覆盖
if [ $# -gt 0 ]; then SEG_METHODS=("$@"); else SEG_METHODS=(clasp ggs window); fi
if [ -n "${SEG_METHODS_ENV:-}" ]; then read -r -a SEG_METHODS <<< "$SEG_METHODS_ENV"; fi

for seg in "${SEG_METHODS[@]}"; do
    echo "=================================================="
    echo "[grid] segment_method=$seg  feature_model=$FEATURE_MODEL"
    echo "=================================================="
    python main.py \
        --appliance "$APPLIANCE" \
        --steps extract,segment,feature \
        --segment-method "$seg" \
        --feature-model "$FEATURE_MODEL"
done

echo "Job finished on: $(date)"

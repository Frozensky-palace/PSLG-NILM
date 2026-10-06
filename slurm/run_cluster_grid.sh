#!/bin/bash
# ============================================================================
# LEGACY: 旧项目时期的脚本，包含旧用户目录与旧仓库名，禁止在新集群使用。
# Formal C1/C2/C3 jobs use slurm/c1_gpu_smoke.sbatch, c2_detsec_pc.sbatch and
# c3_seq2point.sbatch instead (guide §4.6 / §5.5).
# ============================================================================
#SBATCH -J PSLG-CLUSTER-GRID
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /home/scnu2023024258/data/code/PSLG-NILM-ADVANCED/slurm/slurm_log/job-%x-%j.out
#SBATCH -e /home/scnu2023024258/data/code/PSLG-NILM-ADVANCED/slurm/slurm_log/job-%x-%j.err
#SBATCH --time=12:00:00

# 聚类网格：在同一个已有 run（特征已提取）上遍历聚类方法。
# 必须提供 RUN_ID（该 run 的 manifest 里要有 feature_extract.features）。
#   sbatch --export=ALL,RUN_ID=20260807_120000_fridge run_cluster_grid.sh
# kmeans 一次产出所有候选 k 的带标签结果（kmeans_k3/kmeans_k4/...），
# 无需再为每个 k 单独跑一遍。
source "$(dirname "$0")/env.sh"

if [ -z "${RUN_ID:-}" ]; then
    echo "ERROR: RUN_ID is required (a run that already has feature_extract done)."
    echo "  sbatch --export=ALL,RUN_ID=<run_id> run_cluster_grid.sh"
    exit 1
fi

APPLIANCE="${APPLIANCE:-fridge}"
N_CLUSTERS="${N_CLUSTERS:-2,3,4,5,6,7,8}"
METHODS=("kmeans" "dbscan" "hdbscan")
if [ -n "${METHODS_ENV:-}" ]; then read -r -a METHODS <<< "$METHODS_ENV"; fi

for method in "${METHODS[@]}"; do
    echo "=================================================="
    echo "[grid] cluster_method=$method  run_id=$RUN_ID"
    echo "=================================================="
    EXTRA_ARGS=()
    if [ "$method" = "kmeans" ] || [ "$method" = "kmeans-scan" ]; then
        EXTRA_ARGS+=(--n-clusters "$N_CLUSTERS")
    fi
    # 每次聚类用独立 run-id 后缀，避免不同方法的 variant 互相覆盖；
    # 若希望结果合并进同一 run，把 --run-id 改为 "$RUN_ID" 即可（manifest 会合并）。
    python main.py \
        --appliance "$APPLIANCE" \
        --steps cluster \
        --run-id "$RUN_ID" \
        --cluster-method "$method" \
        "${EXTRA_ARGS[@]}"
done

echo "Job finished on: $(date)"

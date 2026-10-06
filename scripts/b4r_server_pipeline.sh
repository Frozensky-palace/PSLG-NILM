#!/bin/bash
# B4R 服务器一键链（预注册 reports/b4r/b4r_protocol_prereg_v1.md §7 硬门）：
#   S0 校验 → S1 生成+双门+等价断言 → S2 钉放置+starts 断言
#   → S3 de_inputs v1 补丁+哈希门 → 打印三条 sbatch 提交命令
# 在 pslg-nilm 环境内、登录节点直接运行（无需 module）。
# 已存在的有效产物自动跳过再执行（生成/放置）；de_inputs 目录若已存在
# 则拒绝（防静默覆盖）。任何断言失败即退出——作废式，无静默重试。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ART="${ART:-$HOME/pslg_artifacts}"
RUN_DIR="$ART/b4r_s17_gen"
PLACED="$ART/de_placed_r0p5_b4r"
INPUTS="$ART/de_inputs_r0p5_b4r"
DONOR="$ROOT/reports/core_validation/ukdale_b1_washing_machine/real_cycle_library_train_v1"
STATE_LIB="$ART/detsecpc_k345_trainonly_4107/state_library_k4"
export ART RUN_DIR PLACED INPUTS

echo "== S0a repo 校验 =="
git merge-base --is-ancestor 29e4a92 HEAD || { echo "FAIL: 缺 B4R 提交，先 git pull --ff-only origin feature/haojun"; exit 1; }
if [ -n "$(git status --porcelain)" ]; then
  echo "WARN: 工作树不干净（不影响运行，但须记录在案）"
else
  echo "OK: tree clean"
fi
REV="$(git rev-parse HEAD)"; export REV
echo "REV=$REV"

echo "== S0b 状态库 sha256（预注册硬门 1）=="
python - <<'PYEOF'
import hashlib, os, sys
lib = os.path.join(os.environ["ART"],
                   "detsecpc_k345_trainonly_4107", "state_library_k4")
want = {"state_inventory.csv":
        "dfa3e5065630cd80a5afd72854bd48f8b1a78e18e30399687f038fca44296ec0",
        "state_waveforms.npz":
        "5b81d52422dbd6e59872a040b58fe1d8234e3124dffff1712a8677e5d5f6b312"}
for name, digest in want.items():
    with open(os.path.join(lib, name), "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    print(f"{name}: {got[:16]} {'OK' if got == digest else 'MISMATCH'}")
    if got != digest:
        sys.exit(1)
PYEOF

echo "== S1a 生成（已有产物则跳过）=="
if [ -f "$RUN_DIR/cycles/generation_summary.json" ]; then
  echo "OK: $RUN_DIR/cycles 已存在，跳过再生成"
else
  python scripts/generate_primitive_cycles.py --primitive-source real \
    --state-library-dir "$STATE_LIB" \
    --output-dir "$RUN_DIR/cycles" --count 246 --seed 17
fi

echo "== S1b 质量门 + 记忆化审计（重跑并机判）=="
python scripts/evaluate_synthetic_quality.py \
  --synthetic-dir "$RUN_DIR/cycles" --real-library-dir "$DONOR" \
  --output "$RUN_DIR/quality_report.json"
python scripts/audit_generation_memorization.py \
  --synthetic-dir "$RUN_DIR/cycles" --real-library-dir "$DONOR" \
  --max-replication-rate 0.01 --output "$RUN_DIR/memorization_report.json"
python - <<'PYEOF'
import json, os, sys
run = os.environ["RUN_DIR"]
q = json.load(open(f"{run}/quality_report.json"))
assert q["passed"], "quality gate FAILED"
warn = {k for k, v in q["flags"].items() if v == "WARN"}
assert warn == {"duration_distribution", "diversity"}, \
    f"unexpected WARN set: {warn}（已追认的只有这两项，出现新 WARN 即停）"
m = json.load(open(f"{run}/memorization_report.json"))
assert m["passed"] and m["exact_duplicate_count"] == 0, "memorization FAILED"
print(f"gates OK: WARN={sorted(warn)}, "
      f"replicated={m['replicated_count']}/{m['n_synthetic']}, exact=0")
PYEOF

echo "== S1c 等价断言 vs 归档 B4 s17（硬门 2）=="
python - <<'PYEOF'
import json, os
run, art = os.environ["RUN_DIR"], os.environ["ART"]
b4r = json.load(open(f"{run}/cycles/generation_summary.json"))
try:
    b4 = json.load(open(f"{art}/b4_s17_4133/cycles/generation_summary.json"))
except FileNotFoundError:
    print("NOTE: 服务器无 b4_s17_4133 归档；以本地断言为准（执行记录注明）")
else:
    r = {x["synthetic_cycle_id"]: x for x in b4r["records"]}
    a = {x["synthetic_cycle_id"]: x for x in b4["records"]}
    assert set(r) == set(a), "cycle id sets differ"
    bad = [cid for cid in r
           if r[cid]["state_path"] != a[cid]["state_path"]
           or [s["target_samples"] for s in r[cid]["segments"]]
           != [s["target_samples"] for s in a[cid]["segments"]]]
    assert not bad, f"mismatches: {bad[:5]}"
    print(f"equivalence OK: {len(r)}/{len(r)} match")
PYEOF

echo "== S2 钉住放置 + starts 断言（硬门 3）=="
if [ -f "$PLACED/placement_summary.json" ]; then
  echo "OK: 已有放置产物，仅复验 starts"
else
  python scripts/place_synthetics_on_background.py \
    --aligned-dir reports/core_validation/ukdale_b1_washing_machine/aligned_partitions_v2 \
    --synthetic-dir "B4R=$RUN_DIR/cycles" \
    --output-dir "$PLACED" \
    --seed 17 --envelope-samples 2372
fi
python - <<'PYEOF'
import json, os
art, placed = os.environ["ART"], os.environ["PLACED"]
v1 = json.load(open(f"{art}/de_placed_r0p5/placement_summary.json"))
new = json.load(open(f"{placed}/placement_summary.json"))
b4 = [p["start_global_index"] for p in v1["arms"]["B4"]["placements"]]
b4r = [p["start_global_index"] for p in new["arms"]["B4R"]["placements"]]
assert b4 == b4r, f"starts mismatch（{sum(a == b for a, b in zip(b4, b4r))}/{len(b4)} 相同）"
print(f"starts OK: {len(b4r)}/{len(b4)}")
PYEOF

echo "== S3 de_inputs v1 补丁 + 哈希门（硬门 4）=="
if [ -d "$INPUTS" ]; then
  echo "ABORT: $INPUTS 已存在。若为失败后重跑，先移开并记录原因："
  echo "  mv $INPUTS $INPUTS.aborted.\$(date +%Y%m%d)"
  exit 1
fi
python scripts/prepare_nilm_b0_b2_inputs.py \
  --aligned-dir reports/core_validation/ukdale_b1_washing_machine/aligned_partitions_v2 \
  --placed-dir reports/core_validation/ukdale_b1_washing_machine/b2_policy_ablation_seed17_r0p5/k4_feat_duration_ratio_0p67_1p5/placed \
  --output-dir "$INPUTS" \
  --extra-arm B3T="$ART/de_placed_r0p5" \
  --extra-arm B3CVAE="$ART/de_placed_r0p5" \
  --extra-arm B3WGAN="$ART/de_placed_r0p5" \
  --extra-arm B3DIFF="$ART/de_placed_r0p5" \
  --extra-arm B4="$ART/de_placed_r0p5" \
  --extra-arm B5="$ART/de_placed_r0p5" \
  --extra-arm B5EP="$ART/de_placed_r0p5" \
  --extra-arm B4R="$PLACED" \
  --skip-test
python - <<'PYEOF'
import hashlib, json, os, sys
art = os.environ["ART"]
v1dir = f"{art}/de_inputs_r0p5"
newdir = f"{art}/de_inputs_r0p5_b4r"
names = sorted(n for n in os.listdir(v1dir) if n != "dataset_manifest.json")
def digest(directory, name):
    with open(os.path.join(directory, name), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()
bad = [n for n in names if digest(v1dir, n) != digest(newdir, n)]
assert not bad, f"MISMATCH（冻结源被改动，作废+查因）: {bad}"
v1 = json.load(open(f"{v1dir}/dataset_manifest.json"))
new = json.load(open(f"{newdir}/dataset_manifest.json"))
for arm, pool in v1["train_sampling_pools"].items():
    assert new["train_sampling_pools"][arm] == pool, f"pool drift: {arm}"
extra = set(new["train_sampling_pools"]) - set(v1["train_sampling_pools"])
assert extra == {"B4R"}, f"unexpected new arms: {extra}"
assert new["validation_monitor"] == v1["validation_monitor"]
assert new["window_counts"] == v1["window_counts"]
print(f"hash gate OK: {len(names)} 个同名文件逐字节相等；B4R 是唯一新增臂")
PYEOF

echo ""
echo "== ALL GATES PASSED =="
echo "逐条复制下面三行提交下游训练（层 2 预注册终判口径）："
for SEED in 17 42 73; do
  echo "sbatch --export=ALL,PSLG_PROJECT_ROOT=$ROOT,PSLG_FROZEN_COMMIT=$REV,PSLG_MANIFEST=$HOME/pslg_manifests/c1_nilm_b0_b2_trainval_manifest.json,PSLG_EXPERIMENT_DIR=$INPUTS,PSLG_ARM=B4R,PSLG_SEED=$SEED slurm/c3_seq2point.sbatch"
done
echo "提交后用 squeue -u \$USER 监控；完成后把三个 validation_metrics.json 的 MAE 发回。"

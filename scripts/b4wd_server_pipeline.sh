#!/bin/bash
# B4WD 服务器一键链（预注册 reports/b4wd/b4wd_protocol_prereg_v1.md §7 硬门；
# b4r_server_pipeline.sh 的按臂克隆，克隆差异即预注册 §7 清单）：
#   S0 校验 → S1 双门+坍缩门机判+等价断言 → S2 钉放置+starts 断言
#   → S3 de_inputs v1 补丁+哈希门 → 打印三条 sbatch 提交命令
# 训练+生成在 slurm/b4wd_wgan.sbatch / slurm/b4wd_diffusion.sbatch
# （GPU 作业，本脚本之外先行；prereg §3 时限 1 天/2 天 = B3 分文件同规格）；
# 本脚本在 pslg-nilm 环境内、登录节点直接运行，消费其 RUN_DIR。
# 已存在的有效产物自动跳过再执行（放置）；de_inputs 目录若已存在
# 则拒绝（防静默覆盖）。任何断言失败即退出——作废式，无静默重试。
# 用法：ARM=B4WGAN|B4DIFF B4WD_RUN_DIR=~/pslg_artifacts/b4wd_<arm>_s17_<jobid> \
#       bash scripts/b4wd_server_pipeline.sh
set -euo pipefail

: "${ARM:?set ARM=B4WGAN or B4DIFF}"
case "$ARM" in
  B4WGAN|B4DIFF) ;;
  *) echo "FAIL: ARM must be B4WGAN or B4DIFF"; exit 1 ;;
esac
ARM_LC="$(echo "$ARM" | tr 'A-Z' 'a-z')"
: "${B4WD_RUN_DIR:?set B4WD_RUN_DIR to the b4wd_primitive.sbatch RUN_DIR (contains cycles/)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ART="${ART:-$HOME/pslg_artifacts}"
RUN_DIR="$B4WD_RUN_DIR"
PLACED="$ART/de_placed_r0p5_${ARM_LC}"
INPUTS="$ART/de_inputs_r0p5_${ARM_LC}"
DONOR="$ROOT/reports/core_validation/ukdale_b1_washing_machine/real_cycle_library_train_v1"
STATE_LIB="$ART/detsecpc_k345_trainonly_4107/state_library_k4"
# Prereg §2: rng_seed 复现的机检前提——与生成作业同一哈希盐约定。
export PYTHONHASHSEED=0
export ART ARM ARM_LC RUN_DIR PLACED INPUTS

echo "== S0a repo 校验 =="
# 锚 = B4WD 实现提交（解码分派 + 段级训练函数 + 脚本层尺度修复 + 管线克隆；
# 预注册 §3 代码锚，实现提交合入时回填）。占位符不是合法哈希 → git 报错
# 即 exit 1，fail-closed，不会静默放行。
B4WD_IMPL_COMMIT="__B4WD_IMPL_COMMIT__"
git merge-base --is-ancestor "$B4WD_IMPL_COMMIT" HEAD || { echo "FAIL: 缺 B4WD 实现提交 $B4WD_IMPL_COMMIT，先 git pull --ff-only origin feature/haojun"; exit 1; }
if [ -n "$(git status --porcelain)" ]; then
  echo "WARN: 工作树不干净（不影响运行，但须记录在案）"
else
  echo "OK: tree clean"
fi
REV="$(git rev-parse HEAD)"; export REV
echo "REV=$REV"

if [ ! -f "$RUN_DIR/cycles/generation_summary.json" ]; then
  echo "FAIL: $RUN_DIR/cycles 不存在——先提交 slurm/b4wd_${ARM_LC}.sbatch 完成训练+生成"
  exit 1
fi

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

echo "== S1b 质量门 + 记忆化审计 + 坍缩门机判（预注册 §4）=="
python scripts/evaluate_synthetic_quality.py \
  --synthetic-dir "$RUN_DIR/cycles" --real-library-dir "$DONOR" \
  --diversity-max-cycles 246 \
  --output "$RUN_DIR/quality_report.json"
python scripts/audit_generation_memorization.py \
  --synthetic-dir "$RUN_DIR/cycles" --real-library-dir "$DONOR" \
  --max-replication-rate 0.01 --output "$RUN_DIR/memorization_report.json"
python - <<'PYEOF'
import json, os, sys
run, arm = os.environ["RUN_DIR"], os.environ["ARM"]
q = json.load(open(f"{run}/quality_report.json"))
assert q["passed"], "quality gate FAILED"
warn = {k for k, v in q["flags"].items() if v == "WARN"}
# B4R 管线的 {'duration_distribution','diversity'} 是 donor 生日碰撞的追认
# 集合：diversity WARN 在本协议=坍缩门触发，克隆不改会误杀健康运行、
# 放行坍缩运行（预注册 §4 机判承接）。
assert warn == {"duration_distribution"}, \
    f"unexpected WARN set: {warn}（期望仅 duration_distribution；diversity WARN=坍缩门触发即作废）"
div = q["diversity"]
assert div["sampled_cycles"] == 246, \
    f"diversity 未覆盖全量：sampled_cycles={div['sampled_cycles']} != 246"
assert div["identical_pairs"] == 0, \
    "坍缩门：identical_pairs>=1 ⇒ 本次运行作废 + 书面查因（预注册 §4，无豁免）"
assert div["min_pairwise_distance"] >= 1e-3, \
    f"坍缩门：min_pairwise={div['min_pairwise_distance']:.6f} < 1e-3 ⇒ 作废 + 书面查因"
m = json.load(open(f"{run}/memorization_report.json"))
assert m["passed"] and m["exact_duplicate_count"] == 0, "memorization FAILED"
# rng_seed 机检：manifest 落盘值须与本解释器（PYTHONHASHSEED=0）重推导一致。
summary = json.load(open(f"{run}/cycles/generation_summary.json"))
seed = summary["seed"]
checked = 0
for record in summary["records"]:
    rng_seeds = record["conditions"].get("rng_seeds")
    assert rng_seeds is not None, f"{record['synthetic_cycle_id']}: rng_seeds 未落盘"
    expected = [seed * 1_000_003 + hash(record["synthetic_cycle_id"]) % 1_000 + o
                for o in range(len(record["state_path"]))]
    assert rng_seeds == expected, f"{record['synthetic_cycle_id']}: rng_seeds 与公式不符"
    checked += 1
if arm == "B4DIFF":
    cap_path = f"{run}/cycles/peak_cap_report.json"
    assert os.path.isfile(cap_path), \
        "B4DIFF 缺 prereg §4 证据 peak_cap_report.json（贴上限占比/截断触发率）"
    cap = json.load(open(cap_path))
    print(f"B4DIFF 贴上限占比={cap['cap_pinned_share']:.4f} "
          f"截断触发率={cap['cap_trigger_rate']:.4f}（§4 证据列，非门）")
print(f"gates OK: WARN={sorted(warn)}, identical=0, "
      f"min_pairwise={div['min_pairwise_distance']:.4f}, "
      f"replicated={m['replicated_count']}/{m['n_synthetic']}, exact=0, "
      f"rng_seeds 复核 {checked} 条")
PYEOF

echo "== S1c 等价断言 vs 归档 B4 s17（硬门 2；rng 序列断言对生成后端同样适用）=="
python - <<'PYEOF'
import json, os
run, art = os.environ["RUN_DIR"], os.environ["ART"]
new = json.load(open(f"{run}/cycles/generation_summary.json"))
try:
    b4 = json.load(open(f"{art}/b4_s17_4133/cycles/generation_summary.json"))
except FileNotFoundError:
    print("NOTE: 服务器无 b4_s17_4133 归档；以本地断言为准（执行记录注明）")
else:
    r = {x["synthetic_cycle_id"]: x for x in new["records"]}
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
    --synthetic-dir "$ARM=$RUN_DIR/cycles" \
    --output-dir "$PLACED" \
    --seed 17 --envelope-samples 2372
fi
python - <<'PYEOF'
import json, os
art, placed, arm = os.environ["ART"], os.environ["PLACED"], os.environ["ARM"]
v1 = json.load(open(f"{art}/de_placed_r0p5/placement_summary.json"))
new = json.load(open(f"{placed}/placement_summary.json"))
b4 = [p["start_global_index"] for p in v1["arms"]["B4"]["placements"]]
got = [p["start_global_index"] for p in new["arms"][arm]["placements"]]
assert b4 == got, f"starts mismatch（{sum(a == b for a, b in zip(b4, got))}/{len(b4)} 相同）"
print(f"starts OK: {len(got)}/{len(b4)}")
PYEOF

echo "== S3 de_inputs v1 补丁 + 哈希门（硬门 4；每臂独立目录，哈希基线=冻结 v1）=="
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
  --extra-arm "$ARM=$PLACED" \
  --skip-test
python - <<'PYEOF'
import hashlib, json, os, sys
art, arm, arm_lc = (os.environ["ART"], os.environ["ARM"],
                    os.environ["ARM_LC"])
v1dir = f"{art}/de_inputs_r0p5"
newdir = f"{art}/de_inputs_r0p5_{arm_lc}"
names = sorted(n for n in os.listdir(v1dir) if n != "dataset_manifest.json")
def digest(directory, name):
    with open(os.path.join(directory, name), "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()
bad = [n for n in names if digest(v1dir, n) != digest(newdir, n)]
assert not bad, f"MISMATCH（冻结源被改动，作废+查因）: {bad}"
v1 = json.load(open(f"{v1dir}/dataset_manifest.json"))
new = json.load(open(f"{newdir}/dataset_manifest.json"))
for arm_key, pool in v1["train_sampling_pools"].items():
    assert new["train_sampling_pools"][arm_key] == pool, f"pool drift: {arm_key}"
extra = set(new["train_sampling_pools"]) - set(v1["train_sampling_pools"])
assert extra == {arm}, f"unexpected new arms: {extra}"
assert new["validation_monitor"] == v1["validation_monitor"]
assert new["window_counts"] == v1["window_counts"]
print(f"hash gate OK: {len(names)} 个同名文件逐字节相等；{arm} 是唯一新增臂")
PYEOF

echo ""
echo "== ALL GATES PASSED =="
echo "逐条复制下面三行提交下游训练（层 2 预注册终判口径）："
for SEED in 17 42 73; do
  echo "sbatch --export=ALL,PSLG_PROJECT_ROOT=$ROOT,PSLG_FROZEN_COMMIT=$REV,PSLG_MANIFEST=$HOME/pslg_manifests/c1_nilm_b0_b2_trainval_manifest.json,PSLG_EXPERIMENT_DIR=$INPUTS,PSLG_ARM=$ARM,PSLG_SEED=$SEED slurm/c3_seq2point.sbatch"
done
echo "提交后用 squeue -u \$USER 监控；完成后把三个 validation_metrics.json 的 MAE 发回。"

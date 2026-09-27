# B4R 服务器 s17 执行手册（runbook）

对应预注册：`reports/b4r/b4r_protocol_prereg_v1.md`（判读规则 §5，硬门 §7）。
本地侧已全过（`reports/b4r/local_s17_execution_record.md`：等价性 246/246、
starts 246/246、审计 0/246）。本手册为服务器侧剩余步骤，按序执行，
**任何硬门不过即停，不静默重试**。

约定：`ART=$HOME/pslg_artifacts`（`source slurm/project_paths.sh` 后即
`$PSLG_ARTIFACT_ROOT`），repo = `~/projects/PSLG-NILM-c1`。

## 第 0 步：代码同步 + 哈希前置条件

```bash
cd ~/projects/PSLG-NILM-c1
git status --short        # 期望干净；若有本地改动，先记录报告，勿覆盖（G-5 备查过 protocol-config 差异）
git pull --ff-only origin feature/haojun
git merge-base --is-ancestor a35fe30 HEAD && echo "OK: 含 runbook 修正提交"
# 上一行必须输出 OK；再记下当前提交号，第 5 步 PSLG_FROZEN_COMMIT 用同一值
REV=$(git rev-parse HEAD) && echo "REV=$REV"
sha256sum $HOME/pslg_artifacts/detsecpc_k345_trainonly_4107/state_library_k4/state_inventory.csv
# 期望 dfa3e5065630cd80a5afd72854bd48f8b1a78e18e30399687f038fca44296ec0
sha256sum $HOME/pslg_artifacts/detsecpc_k345_trainonly_4107/state_library_k4/state_waveforms.npz
# 期望 5b81d52422dbd6e59872a040b58fe1d8234e3124dffff1712a8677e5d5f6b312
```

不等即停（预注册硬门 1）。

## 第 1 步：B4R 生成 + 双门（无训练，CPU 数分钟）

```bash
source slurm/project_paths.sh && cd "$PSLG_PROJECT_ROOT"
source "$(conda info --base)/bin/activate" && conda activate pslg-nilm
ART=$PSLG_ARTIFACT_ROOT
RUN_DIR=$ART/b4r_s17_gen
export ART RUN_DIR
DONOR=$PSLG_PROJECT_ROOT/reports/core_validation/ukdale_b1_washing_machine/real_cycle_library_train_v1

python scripts/generate_primitive_cycles.py \
  --primitive-source real \
  --state-library-dir $ART/detsecpc_k345_trainonly_4107/state_library_k4 \
  --output-dir $RUN_DIR/cycles --count 246 --seed 17

python scripts/evaluate_synthetic_quality.py \
  --synthetic-dir $RUN_DIR/cycles --real-library-dir "$DONOR" \
  --output $RUN_DIR/quality_report.json

python scripts/audit_generation_memorization.py \
  --synthetic-dir $RUN_DIR/cycles --real-library-dir "$DONOR" \
  --max-replication-rate 0.01 --output $RUN_DIR/memorization_report.json
```

期望与本地逐位一致：quality `passed=true`（WARN 应只有 W1 duration、W2
diversity 两项，同本地）；audit `replicated=0/246, exact_duplicate_count=0`。

**等价性断言（硬门 2，vs 服务器冻结的 b4_s17_4133）：**

```bash
python - <<'EOF'
import json, os
run_dir = os.environ["RUN_DIR"]; art = os.environ["ART"]
b4r = json.load(open(f"{run_dir}/cycles/generation_summary.json", encoding="utf-8"))
b4  = json.load(open(f"{art}/b4_s17_4133/cycles/generation_summary.json", encoding="utf-8"))
r = {x["synthetic_cycle_id"]: x for x in b4r["records"]}
a = {x["synthetic_cycle_id"]: x for x in b4["records"]}
assert set(r) == set(a), "cycle id sets differ"
bad = [cid for cid in r if r[cid]["state_path"] != a[cid]["state_path"]
       or [s["target_samples"] for s in r[cid]["segments"]]
       != [s["target_samples"] for s in a[cid]["segments"]]]
print(f"equivalence: {len(r) - len(bad)}/{len(r)} match, mismatches={bad[:5]}")
assert not bad
EOF
```

（若 `$ART/b4_s17_4133` 不在服务器，说明归档已清理——改用本机回传的
`generation_summary.json` 上传后比对，或跳过服务器侧断言、以本地断言为准
并在执行记录注明。）

## 第 2 步：钉住放置（硬门 3）

```bash
ART=${ART:-$HOME/pslg_artifacts}; export ART; cd ~/projects/PSLG-NILM-c1
python scripts/place_synthetics_on_background.py \
  --aligned-dir reports/core_validation/ukdale_b1_washing_machine/aligned_partitions_v2 \
  --synthetic-dir B4R=$RUN_DIR/cycles \
  --output-dir $ART/de_placed_r0p5_b4r \
  --seed 17 --envelope-samples 2372

python - <<'EOF'
import json, os
art = os.environ["ART"]
v1 = json.load(open(f"{art}/de_placed_r0p5/placement_summary.json", encoding="utf-8"))
new = json.load(open(f"{art}/de_placed_r0p5_b4r/placement_summary.json", encoding="utf-8"))
b4_starts = [p["start_global_index"] for p in v1["arms"]["B4"]["placements"]]
b4r_starts = [p["start_global_index"] for p in new["arms"]["B4R"]["placements"]]
print(f"starts: {b4r_starts == b4_starts} ({len(b4r_starts)}/{len(b4_starts)})")
assert b4r_starts == b4_starts
EOF
```

## 第 3 步：de_inputs v1 补丁重建 + 哈希门（硬门 4）

```bash
ART=${ART:-$HOME/pslg_artifacts}; export ART; cd ~/projects/PSLG-NILM-c1
python scripts/prepare_nilm_b0_b2_inputs.py \
  --aligned-dir reports/core_validation/ukdale_b1_washing_machine/aligned_partitions_v2 \
  --placed-dir reports/core_validation/ukdale_b1_washing_machine/b2_policy_ablation_seed17_r0p5/k4_feat_duration_ratio_0p67_1p5/placed \
  --output-dir $ART/de_inputs_r0p5_b4r \
  --extra-arm B3T=$ART/de_placed_r0p5/B3T \
  --extra-arm B3CVAE=$ART/de_placed_r0p5/B3CVAE \
  --extra-arm B3WGAN=$ART/de_placed_r0p5/B3WGAN \
  --extra-arm B3DIFF=$ART/de_placed_r0p5/B3DIFF \
  --extra-arm B4=$ART/de_placed_r0p5/B4 \
  --extra-arm B5=$ART/de_placed_r0p5/B5 \
  --extra-arm B5EP=$ART/de_placed_r0p5/B5EP \
  --extra-arm B4R=$ART/de_placed_r0p5_b4r \
  --skip-test

cd $ART
for f in $(ls de_inputs_r0p5 | grep -v '^dataset_manifest.json$'); do
  a=$(sha256sum "de_inputs_r0p5/$f" | cut -d' ' -f1)
  b=$(sha256sum "de_inputs_r0p5_b4r/$f" | cut -d' ' -f1)
  [ "$a" = "$b" ] || echo "MISMATCH: $f"
done; echo "hash-gate: done (只有 MISMATCH 行才是问题)"

python - <<'EOF'
import json
v1 = json.load(open("de_inputs_r0p5/dataset_manifest.json", encoding="utf-8"))
new = json.load(open("de_inputs_r0p5_b4r/dataset_manifest.json", encoding="utf-8"))
for arm, pool in v1["train_sampling_pools"].items():
    assert new["train_sampling_pools"][arm] == pool, f"pool drift: {arm}"
assert "B4R" in new["train_sampling_pools"], "B4R pool missing"
assert new["validation_monitor"] == v1["validation_monitor"]
assert new["window_counts"] == v1["window_counts"]
print("manifest subset check: OK;",
      "B4R pool:", new["train_sampling_pools"]["B4R"])
EOF
cd "$PSLG_PROJECT_ROOT"
```

说明：唯一允许差异的文件是 `dataset_manifest.json`（合法新增 B4R 条目）；
其余 23 个同名文件（20 个 indices npy + window_ranges.csv +
normalization.json + validation_monitor_indices.npy）必须逐字节相等——
这些文件在 prepare 中只依赖 B0/共享窗口表/各臂自己的放置 shards，与新增
臂无关（代码可溯源），不等即说明冻结源被动过 ⇒ 作废 + 查因。

## 第 4 步：WARN 接受确认（下游 sbatch 前置，预注册 §4）

本地两 WARN（W1 duration：与 B4 s17 逐位同构 102/246；W2 diversity：内部
同 donor 碰撞 1 对）已记录在 `local_s17_execution_record.md`，**接受人需
用户追认**。追认后再提交第 5 步。

## 第 5 步：下游训练（c3_seq2point，与 B4 作业 4162 同规格）

```bash
cd ~/projects/PSLG-NILM-c1
ART=${ART:-$HOME/pslg_artifacts}
PSLG_PROJECT_ROOT=${PSLG_PROJECT_ROOT:-$HOME/projects/PSLG-NILM-c1}
REV=${REV:-$(git rev-parse HEAD)}   # 须与第 0 步记下的值一致，且彼时工作树干净
sbatch --export=ALL,\
PSLG_PROJECT_ROOT=$PSLG_PROJECT_ROOT,\
PSLG_FROZEN_COMMIT=$REV,\
PSLG_MANIFEST=$HOME/pslg_manifests/c1_nilm_b0_b2_trainval_manifest.json,\
PSLG_EXPERIMENT_DIR=$ART/de_inputs_r0p5_b4r,\
PSLG_ARM=B4R,PSLG_SEED=17 \
slurm/c3_seq2point.sbatch
```

（RTX3090 分区、30 epoch 上限、val 20k 窗口，与 B4/B5 系完全同口径。）

## 第 6 步：判读（预注册 §5 层 1）+ 回传

- 回读：`validation_metrics.json`（run 目录 `s2p_B4R_r0p5_s17_<jobid>`）。
- x = val MAE：x ≤ 12.1 → 生成器模糊信号；x ≥ 16.5 → 组合天花板信号；
  灰区 → 升 s42/s73（层 2 终判，预注册阈值 15.64 / 19.03）。
- 回传打包（照 DE 惯例）：`$RUN_DIR`（cycles + 双报告）、
  `de_placed_r0p5_b4r/placement_summary.json`、`de_inputs_r0p5_b4r/`
  （manifest + 校验输出）、s2p run 目录、slurm 日志。

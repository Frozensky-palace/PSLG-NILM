# B4WD 本地 CPU 冒烟执行记录（2026-09-29）

预注册 `reports/b4wd/b4wd_protocol_prereg_v1.md`（冻结）实现完成后的
本地验证链。产物全部在 `reports/b4wd/local_smoke/`（**数据不进 git**，
本记录为唯一入库文档）。

**记录两轮**：第 1 轮（§1–§3 命令与结果）在实现直后执行；随后 20-agent
对抗审查确认 3 项缺陷（§3.5），修复后**全链重跑**——下表数字为第 2 轮
（修复后代码）结果；重跑原因 = 审查修复改变了被测代码（B4WGAN lr
1e-3→1e-4 路由默认 + B4DIFF 新增 §4 证据落盘）。两处交叉验证：扩散臂
noise_mse 轨迹两轮逐位一致（0.97908→0.75768，构建前播种修复的直接
证据）；扩散臂 min_pairwise 两轮一致（0.9093，解码路径未变）。

## 0. 范围与口径

- 实现内容：解码分派（`PrimitiveComposer.model_backend` ∈ cvae/wgan/
  diffusion，numpy rng 序列不动）、段级 masked 训练（WGAN 输入侧 mask
  于 critic/GP/监控全路径；扩散 masked noise-MSE 仅有效点）、CLI
  （`train_full_cycle_generator.py --route primitive-wgan|primitive-diffusion`、
  `generate_primitive_cycles.py --primitive-source wgan|diffusion`）、
  服务器管线（`slurm/b4wd_primitive.sbatch` + `scripts/b4wd_server_pipeline.sh`）。
- 冒烟目的：**机制正确性**（分派/等价/门机判），非模型质量——3 epoch
  欠训练属预期；正式预算（150/300 epoch）在服务器 GPU。
- 对照归档：`reports/server_de/2026-09-22/b4_s17_4133/`（B4 CVAE，
  246 条，seed 17）。
- 状态库：真实 k4 冻结 v1 本地副本
  `reports/server_c2/2026-09-21/results/detsecpc_k345_trainonly_4107/state_library_k4/`。
- 全程 `PYTHONHASHSEED=0`（rng_seed 公式机检前提）。

## 1. 命令（顺序执行，均 0 退出）

```bash
STATE_LIB=reports/server_c2/2026-09-21/results/detsecpc_k345_trainonly_4107/state_library_k4
PYTHONHASHSEED=0 python scripts/train_full_cycle_generator.py --route primitive-wgan \
  --state-library-dir $STATE_LIB --output-dir reports/b4wd/local_smoke/ckpt_wgan \
  --epochs 3 --seed 17 --n-critic 5 --device cpu
PYTHONHASHSEED=0 python scripts/train_full_cycle_generator.py --route primitive-diffusion \
  --state-library-dir $STATE_LIB --output-dir reports/b4wd/local_smoke/ckpt_diffusion \
  --epochs 3 --seed 17 --diffusion-steps 500 --device cpu
PYTHONHASHSEED=0 python scripts/generate_primitive_cycles.py --primitive-source wgan \
  --checkpoint-dir reports/b4wd/local_smoke/ckpt_wgan --state-library-dir $STATE_LIB \
  --output-dir reports/b4wd/local_smoke/cycles_wgan --count 20 --seed 17 --device cpu
PYTHONHASHSEED=0 python scripts/generate_primitive_cycles.py --primitive-source diffusion \
  --checkpoint-dir reports/b4wd/local_smoke/ckpt_diffusion --state-library-dir $STATE_LIB \
  --output-dir reports/b4wd/local_smoke/cycles_diffusion --count 20 --seed 17 --device cpu
PYTHONHASHSEED=0 python reports/b4wd/local_smoke/assert_smoke.py
# 质量门 + 记忆化（S1b 机制，覆盖参数按冒烟规模取 20）
for arm in wgan diffusion; do
  PYTHONHASHSEED=0 python scripts/evaluate_synthetic_quality.py \
    --synthetic-dir reports/b4wd/local_smoke/cycles_$arm \
    --real-library-dir reports/core_validation/ukdale_b1_washing_machine/real_cycle_library_train_v1 \
    --diversity-max-cycles 20 --output reports/b4wd/local_smoke/quality_report_$arm.json
  PYTHONHASHSEED=0 python scripts/audit_generation_memorization.py \
    --synthetic-dir reports/b4wd/local_smoke/cycles_$arm \
    --real-library-dir reports/core_validation/ukdale_b1_washing_machine/real_cycle_library_train_v1 \
    --max-replication-rate 0.01 --output reports/b4wd/local_smoke/memorization_report_$arm.json
done
```

## 2. 结果

| 检查 | B4WGAN | B4DIFF |
|---|---|---|
| 训练收敛迹象（3 epoch） | w_distance=2.093（lr=1e-4 路由默认） | noise_mse 0.979→0.758 |
| 每态尺度打印（§3 冒烟检查） | 全局 q99 峰值 3901.8W | 957.4/144.5/737.7/270.1W，缩后中位 std 0.94/0.87/0.80/0.47 |
| 等价门（state_path+target_samples vs b4_s17_4133） | 20/20 逐位相等 | 20/20 逐位相等 |
| rng_seeds 公式机检 | 20/20 | 20/20 |
| route 标签 | B4WGAN 全部 | B4DIFF 全部 |
| 三臂波形哈希两两不等（B4/双新臂） | 20/20 | 20/20 |
| 波形健全（有限/非负/非全零） | 20/20 | 20/20 |
| 坍缩门 identical_pairs / min_pairwise | 0 / 0.2872 | 0 / 0.9093 |
| 记忆化 replicated / exact | 0/20 / 0 | 0/20 / 0 |
| §4 证据（贴上限占比 / 截断触发率） | 不适用（无截断） | 0.2319 / 1.0000（3 epoch 欠训贴顶，机制按设计工作） |
| 质量 flags | duration_distribution=WARN（期望集） | duration_distribution+energy_scale=WARN（3 epoch 欠训练，预期内） |

WGAN 冒烟 WARN 集恰为预注册期望 `{duration_distribution}`；扩散多出的
`energy_scale` 归因欠训练（正式 300 epoch 预算的对象即此），冒烟不设豁免
也不据此改判——正式判读只在服务器硬门（S1b 断言 WARN 集恰为
`{duration_distribution}`）。

## 3. 过程中发现并修复的两个根因级缺陷（先于提交）

1. **训练脚本缺构建前播种**：`train_full_cycle_generator.py` 原在
   `--seed` 只传入训练函数（函数内部 `torch.manual_seed`），模型权重
   初始化消费进程默认 RNG（OS 熵）→ 同种子跨进程权重漂移。证据：同种子
   6 进程权重 sha256 全不同（单线程化后仍全不同，排除线程调度）；补
   `main()` 内构建前 `torch.manual_seed(args.seed)` 后 4/4 一致。该缺陷
   同时解释此前测试两类闪断（见 2）。归档 B3/B4/B5 运行不受影响（产物
   已冻结，不重训）；B4WD 双臂自此满足"种子语义完整"。
2. **`generate_dataset` 缺非有限值守卫**：`NaN < 0` 与 `NaN > cap` 均为
   False，NaN 波形可静默通过负功率门/峰值门/质量门全部比较口径。补
   `np.isfinite` 全量守卫（RuntimeError，作废式），并加单测钉住
   （`FiniteGuardTests`）。

测试稳定性修复（tests/test_primitive_backends.py `_trained_backends`）：
夹具改为对齐生产语义——构建前 `torch.manual_seed(1)` + 训练输入先除
尺度（/800，生产路径恒先除 power scale）。修复前跨进程权重漂移导致
两类间歇失败（NaN 轨迹：`nan != nan` 破坏确定性断言；零塌缩轨迹：两
后端解码同被 clip 至全零 → 哈希碰撞破坏"跨臂内容不等"断言）。修复后
该模块 10/10 稳定，全量 `python -m unittest discover -s tests` =
**218/218 OK**（203 旧 + 15 新）。

## 3.5 对抗审查（提交前，ultracode 工作流）

5 镜头（预注册保真 / 解码分派 / masked 训练 / CLI-scripts / bash-回归）
× 每项发现独立对抗验证：13 agent，8 项声明 → 5 项 CONFIRMED（3 项驳回），
去重后 **3 项真实缺陷，全部修复并回归**：

1. **B4WGAN 正式训练将以 lr=1e-3 违反冻结 §3 Adam(1e-4)**（3 个镜头
   独立命中，high）：CLI 默认 `--learning-rate 1e-3` 被臂包装透传、
   sbatch 未显式传参、且 config.json 不记录 lr——偏差完全静默。修复：
   `--learning-rate` 默认改 None，按路由解析（primitive-wgan→1e-4，
   其余→1e-3），双新臂 config.json 落盘 `learning_rate`；单测钉住
   （不传参时 wgan=1e-4 / diffusion=1e-3）。
2. **§4 预注册证据未实现**（medium）：贴上限占比与截断触发率全链无人
   计算。修复：`generate_primitive_cycles.py` 对带 peak_caps 的臂在
   生成后落盘 `peak_cap_report.json`（全局+分态 pinned_share /
   trigger_rate；`power >= cap` 计数——截断后不可能越限，故该谓词
   恰为被钉样本集）；管线 S1b 对 B4DIFF 断言文件存在并回显两数；
   单测钉住字段与取值范围。
3. **sbatch 时限违反 §3 分臂规格**（low）：单一文件 `-t 2-00:00:00`
   使 WGAN 臂拿到 2 天。修复：按 B3 同规格拆分
   `slurm/b4wd_wgan.sbatch`（1 天）/`slurm/b4wd_diffusion.sbatch`
   （2 天），臂参数随文件固化；管线提示语同步。

驳回 3 项（记录在案不修）：sbatch 2 天上限盖住 1 天预算（文件内已书面
注明）；以及两项经代码溯源不成立的声明。

## 4. 清单与去向

- 本地冒烟产物：`reports/b4wd/local_smoke/`（ckpt_*/cycles_*/四份
  report JSON/peak_cap_report.json/assert_smoke.py，均不入库）。
- 已完成：对抗审查（§3.5，3 缺陷修复 + 全量 218/218）。
- 待办：里程碑提交 C1（实现）→ C2 回填预注册 §3 代码锚与管线
  `__B4WD_IMPL_COMMIT__` 占位（锚=C1）→ 服务器 GPU 修复后
  `slurm/b4wd_wgan.sbatch` / `slurm/b4wd_diffusion.sbatch` 正式运行 +
  `scripts/b4wd_server_pipeline.sh` 硬门。

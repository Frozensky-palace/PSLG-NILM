# D/E 本机任务执行清单（TODO）

## 服务器执行（✅ 生成阶段 5/5 完成 2026-09-22）

| 路线 | quality | memorization | replication_rate |
|---|---|---|---|
| B3-T 变换 | true | true | 0.992（定义属性，exact=0） |
| B3-V CVAE | true | true | 0.0 |
| B4 基元拼接 | true | true | 0.004 |
| B3-G WGAN | true | true | 0.0 |
| B3-D 扩散 | true | true | 0.0 |

- 扩散经历三次修复后过门：时间嵌入设备跟随（fd3a657）、std 尺度替代
  峰值尺度（学习信噪比）、方法级峰值截断（坐标系换算后生效，d83f200）
- 服务器运行目录：`~/pslg_artifacts/{b3t_s17, b3_cvae_s17, b4_s17,
  b3_wgan_s17, b3_diffusion_s17}_*`（各含 quality/memorization 报告）

## 待办（下一会话）

- [ ] 五个 run 目录打包回传本机 `reports/server_de/2026-09-22/`
- [ ] 共享放置表 + 四路线合成周期统一放置 + 同预算 Seq2Point 重训
      （自动化脚本待本机补齐：schedule → place → prepare → train 串联）
- [ ] B3-T/V/G/D/B4 五路线 + B0/B1 基准的 validation 对照表（D/E 结论）
- [ ] 批次 4：B5/HSMM 组本机开发（Phase F）
- [ ] C1 记录补账（pack_c1.sh）；B2-random 输入准备（协议诊断组）

> 依据：路线图附录 A。本文件是**执行追踪器**，随进度更新勾选；
> 研究方案本身见路线图 §Phase D/E/F，验收标准见附录 A.1。
> 目标：完成 D/E 阶段除服务器正式实验外的全部本机任务。

## 前置门槛

- [x] G-1 C2 库格式兼容验证（零代码，31 列同构）
- [x] G-2 validation 选 k 复核 → v1 = C2-detsec-k4 冻结
- [ ] G-3 生成质量评价管线（批次 0）

## 批次 0：评价管线（✅ 完成 2026-09-21）

- [x] `src/validation/synthetic_quality.py`（物理合法性/分布/多样性/边界检查）
- [x] `src/validation/memorization.py`（最近邻复制审计，分块距离计算）
- [x] `scripts/evaluate_synthetic_quality.py`
- [x] `scripts/audit_generation_memorization.py`
- [x] `scripts/build_shared_placement_schedule.py`（复用 idle_runs/schedule_lengths，
      输出冻结 schedule CSV + SHA-256）
- [x] 单元测试 15 项（含 CLI 失败路径、确定性、排他自距离）
- [x] ConstantGenerator + 真实周期库端到端冒烟：质量门全 PASS +
      diversity=WARN（常量生成器被正确标记零多样性）；复制审计 0/20
- [x] 提交

## 批次 1：B3-T 真实周期变换（✅ 完成 2026-09-21）

- [x] `src/generation/full_cycle_transform.py`（受限时间/功率缩放 + 守卫包络
      + 有界重试；重试上限 64，不可行包络确定性 RuntimeError）
- [x] `scripts/generate_full_cycles.py`（统一生成入口，未实现路线快速失败）
- [x] 单元测试 6 项（缩放边界、负功率、包络重试、确定性、provenance、CLI+质量门）
- [x] 本机端到端闭环：生成 246 条 → 质量门全 PASS（含 diversity=PASS）→
      记忆审计 replication=99.2%（**B3-T 定义属性**：形变真实周期本就近邻；
      exact 复制 0；形状中位距离 0.0451）。神经路线必须接近 0，两口径分开记录
- [x] `slurm/b3_transform.sbatch`（CPU 作业，无 GPU）
- [x] 提交

## 批次 2：CVAE 族（B3-V + B4）（✅ 完成 2026-09-21）

- [x] `src/generation/full_cycle_cvae.py`（条件 CVAE + 长度桶/mask + 梯度裁剪
      + 功率归一化 + CVAESamplingGenerator）
- [x] `src/generation/primitive_cvae.py`（状态段加载、经验路径模型、
      PrimitiveComposer 基础拼接）
- [x] `scripts/train_full_cycle_generator.py`（--route cvae|primitive）
- [x] `scripts/generate_primitive_cycles.py`
- [x] `scripts/compose_generated_cycles.py`（从既有基元池重组）
- [x] 单元测试 13 项（桶倍数/覆盖、损失下降、采样形状、mask、确定性、
      schema 有效、CLI 路由）
- [x] CPU 冒烟：B4 基元 CVAE 10 epoch（loss 0.0342→0.0064，功率归一化
      修复 NaN）；组合 246 条 → 质量门 PASS（duration WARN 为欠训预期）+
      复制审计 0/246；B3-V 5 epoch + 采样 20 条；compose 100 条
- [x] `slurm/b3_cvae.sbatch`、`slurm/b4_primitive_cvae.sbatch`
- [ ] 提交

## 批次 3：B3-G 条件 WGAN（✅ 完成 2026-09-21）

- [x] `src/generation/full_cycle_wgan.py`（Critic/Generator、梯度惩罚、
      w_distance/penalty 历史曲线）
- [x] 训练与生成入口接入 wgan 路由（--n-critic 可调）
- [x] 单元测试 3 项 + CPU 冒烟：3 epoch 训练 → 采样 30 条 →
      质量门 6 项全 PASS + 复制 0/30
- [x] `slurm/b3_wgan.sbatch`
- [x] 提交

## 批次 4：B5 HSMM 组（可与批次 3 并行）

- [ ] `src/composition/transition_model.py`（Markov 顺序）
- [ ] `src/composition/duration_model.py`
- [ ] `src/composition/hsmm_sequence.py`
- [ ] `src/composition/boundary_handler.py`（端点/斜率匹配，可消融）
- [ ] `src/composition/constrained_composer.py`
- [ ] `scripts/fit_hsmm_composer.py`（仅 train 状态序列拟合）
- [ ] 消融原型（B4+Markov → +HSMM → +endpoint），本机可完整验证
- [ ] `slurm/b5_hsmm.sbatch`
- [ ] 提交

## 批次 5：B3-D 条件扩散（✅ 本机部分完成 2026-09-21）

- [x] `src/generation/full_cycle_diffusion.py`（线性调度 DDPM、正弦时间
      嵌入、噪声预测训练、祖先采样）
- [x] 单元测试 4 项（嵌入形状、加噪尺度、噪声 MSE 下降、采样有限）
- [x] CLI 路由（--diffusion-steps 可调）
- [x] `slurm/b3_diffusion.sbatch`
- [x] 提交
- [ ] **服务器正式训练**（3 epoch/50 步冒烟的生成峰值越界，质量门
      FAIL 属预期；正式 epoch 数 + 全步调度后才可能过门——诚实记录，
      不放宽门槛）

## 收尾

- [ ] 全套测试 + compileall + YAML 检查
- [ ] 附录 A 勾选与里程碑更新（M2：管线闭环）
- [ ] 服务器执行清单整理（哪些 sbatch 何时提交）

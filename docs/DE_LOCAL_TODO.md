# D/E 本机任务执行清单（TODO）

## B5 修复轮（✅ 全部完成 2026-09-24）

- [x] F-a 功率补偿实现：`ConstrainedComposer(power_compensate=True)`，
      拼接点对齐前段终点（clip≥0），条件字段记录补偿开关；
      `compose_b5_cycles --boundary-mode power_compensate` 接入；
      b5_hsmm.sbatch 三模式循环（none/endpoint_match/power_compensate）
- [x] 回归测试：拼接点 |diff|<1W 断言（195 项全绿）
- [x] 服务器：重跑 b5_hsmm（三模式，作业 4165）→ 双门 PASS
- [x] 服务器：de_inputs_r0p5_v2 重建（8 臂 ← de_placed_r0p5_v2 完整重放置）
- [x] 服务器：B5PC 下游重训（作业 4178；4176 因提交顺序错误失败，进展报告问题 #17）
- [x] 判定：B5PC 15.96 > B4 15.39 → 预注册门槛失败，如实记负面结果；
      但干预成功（44.51→15.96，-64%）证实接缝因果；修复后 HSMM ≈ B4 打平
      （0.57W 差距 ≪ B4 自身 ±3.74），天花板在基元模糊性 → F-b/F-c 取消
- [x] 多种子扩展：B3T/B3CVAE/B3WGAN/B3DIFF/B4 × seed 42/73（10 作业
      4166–4175）✅；同预算 bootstrap：B3WGAN/B3DIFF 显著优于 B2@r0.5
      （p≈1e-5），生成三强臂与 B2@r2.0 平台打平；详见进展报告 §8.1/§8.2/§8.4

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

## 批次 4：B5/HSMM 组（✅ 本机完成 2026-09-22）

- [x] `src/composition/`：transition_model / duration_model / hsmm_sequence /
      boundary_handler / constrained_composer
- [x] `scripts/fit_hsmm_composer.py` + `scripts/compose_b5_cycles.py`
- [x] 18 项新测试；全套 194 项通过
- [x] 真实 v1 库实战：HSMM 拟合 + B5 默认/endpoint 双消融 246 条过双门；
      端点选择平均边界代价 807.5W → 537.5W（-33%）
- [x] `slurm/b5_hsmm.sbatch`（CPU 双消融组）
- [x] 手册 §13 同步实际接口
- [x] 提交

## 待办（下一会话）

## Phase G：test 解锁（预注册，2026-09-24 用户拍板）

**选组规则（先于任何 test 访问写入）**：进入 test 的是
「无增强锚点 + 重放最优比例 + 真实重组两端 + 全部过门的生成路线」，
即 7 组 × 3 seeds = 21 run（B2 两个比例合计 6 run，arm 标签 6 个）：

| 组 | ratio | jobid (s17/s42/s73) |
|---|---|---|
| B0 无增强 | 0.5 | 4130 / 4121 / 4122 |
| B1 重放 | 1.0 | 4141 / 4142 / 4143 |
| B2-matched | 0.5 | 4126 / 4127 / 4128 |
| B2-matched | 2.0 | 4153 / 4154 / 4155 |
| B3-T 变换 | 0.5 | 4158 / 4166 / 4167 |
| B3-G WGAN | 0.5 | 4160 / 4170 / 4171 |
| B3-D 扩散 | 0.5 | 4161 / 4172 / 4173 |

排除规则（同等预注册）：B3-V（未过质量门）、B4/B5/B5-EP/B5-PC（已按
预注册门槛判负并归档，见进展报告 §8.4）不进 test；B1/B2 未列出的比例
不进 test。冻结后不可追加组，加组需新版本冻结并书面说明。

- [x] G-2 服务器无 test 访问审计（2026-09-24 all_clean=true，证据归档
      `reports/server_g/2026-09-24/`）
- [x] G-3 服务器冻结（2026-09-24T10:50:18Z，21 run，commit f4a737b，
      `reports/protocol_freeze/g3_protocol_freeze_v1.md`）
- [x] G-4 人工签核（用户 2026-09-24 回复"确认"；
      `reports/protocol_freeze/g4_signoff_2026-09-24.md`）
- [x] G-5 test 解锁（2026-09-24）：test shard 上传（7/7 校验）→
      `de_inputs_r0p5_g5test` 补丁（4436 查因，见 g4 签核 md 补记）→
      作业重投 21/21 全量推理（2,580,292 窗口 × 21 checkpoint）→
      **test 终报 `reports/test/2026-09-24_final_test_report.md`**：
      B3-D 显著优于 B2@r0.5（Δ−3.46W，3/3 种子方向一致，CI 不含 0）；
      生成臂与 B2@r2.0 未检测到显著差异（等价性无法确认，CI 宽 ±2~4W）；
      B1@r1.0 优于 B2@r2.0（Δ−0.98W，CI 不含 0，效应量 <1W 意义待商榷）；
      衰减排序 B0 最小（+0.15）< 生成臂（+0.4~+1.4）< B2@r0.5 最大（+2.43），
      "合成更稳"仅相对 B2@r0.5；F1/SAE 描述性（无检验）。§4 定案：D 路线成立。

## B4′：机制探究（G 后新阶段，用户 2026-09-24 定向）

- [ ] B4-real 基元消融（零训练）：组合器不动，真实状态段替换生成基元
      → 回落 B2 平台（~10.4W）则坐实"生成器模糊"元凶；
      仍 ~19W 则组合路径是天花板，B4-WGAN/扩散不值得训
- [ ] 视 B4-real 结果决定：训练 B4-WGAN / B4-扩散（基元级替换 CVAE）
- [ ] 若过 validation 门槛且需进 test：新版本冻结 + 书面原因
- [ ] g5_test 产物回传归档（predictions ≈20MB×21 + metrics + access log）

- [x] 五个 run 目录打包回传本机 `reports/server_de/2026-09-22/`
- [x] 共享放置表 + 路由放置 + extra-arm 窗口输入 + 路由 arm 全链
      （place_synthetics_on_background / prepare --extra-arm / validate_arm）
- [x] 服务器：de_inputs 构建 + 五路线（含 B5 两个消融组）下游重训
      （4158–4164 七臂首轮 + 4178 B5PC，见进展报告 §8.1）
- [x] B0/B1/B2/B3-T/V/G/D/B4/B5(±endpoint/B5PC) 对照表
      （进展报告 §8.1 三种子终版 + `~/pslg_scripts/final_compare_table.txt`）
- [ ] C1 记录补账：审计发现旧 `pack_c1.sh` 归档混入内核镜像（vmlinuz/initrd
      约 145MB）且 `c1_nilm_smoke_rerun` 为空 → 服务器用
      `scripts/pack_run_records.py` 白名单重打（非阻塞 G）
- [ ] `de_inputs_r0p5_v2` 明细（indices npy/normalization）小包回传
      （仅归档完整性，G 不依赖）
- [ ] B2-random 输入准备（协议诊断组，可选）

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

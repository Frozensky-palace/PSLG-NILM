# B4R 本地 s17 执行记录（2026-09-27）

对应预注册：`b4r_protocol_prereg_v1.md`（修订版，含审查发现 2–7）。
本记录覆盖 §7 硬门清单中可在本机完成的各项；服务器侧硬门（de_inputs 哈希门）
在服务器执行时补记。

## 运行命令（本机，Windows / CPU）

```
python scripts/generate_primitive_cycles.py \
  --primitive-source real \
  --state-library-dir reports/server_c2/2026-09-21/results/detsecpc_k345_trainonly_4107/state_library_k4 \
  --output-dir reports/b4r/local_s17/b4r_gen \
  --count 246 --seed 17
python scripts/evaluate_synthetic_quality.py \
  --synthetic-dir reports/b4r/local_s17/b4r_gen \
  --real-library-dir reports/core_validation/ukdale_b1_washing_machine/real_cycle_library_train_v1 \
  --output reports/b4r/local_s17/quality_report.json
python scripts/audit_generation_memorization.py \
  --synthetic-dir reports/b4r/local_s17/b4r_gen \
  --real-library-dir reports/core_validation/ukdale_b1_washing_machine/real_cycle_library_train_v1 \
  --max-replication-rate 0.01 \
  --output reports/b4r/local_s17/memorization_report.json
python scripts/place_synthetics_on_background.py \
  --aligned-dir reports/core_validation/ukdale_b1_washing_machine/aligned_partitions_v2 \
  --synthetic-dir B4R=reports/b4r/local_s17/b4r_gen \
  --output-dir reports/b4r/local_s17/de_placed_b4r \
  --seed 17 --envelope-samples 2372
```

## 硬门结果

| # | 门（预注册 §7） | 结果 |
|---|---|---|
| 1 | 状态库 sha256（本地副本 vs C2 冻结三方记录） | **PASS**：inventory `dfa3e506…96ec0`、waveforms `5b81d524…b312`，与 `artifact_manifest.json`、k-check pilot、库内自声明一致 |
| 2 | 等价性断言（vs `b4_s17_4133` 归档 246 条记录） | **PASS**：`state_path` + 逐段 `target_samples` 全量相等，0 不匹配；单状态记录 19 条全为 state 0（与归档一致） |
| 3 | 放置 starts 断言（envelope 钉 2372） | **PASS**：246/246 `start_global_index` 相等、246/246 `samples` 相等（vs `de_placed_r0p5` 归档 B4 臂）；overlap=0；背景不变性误差 2.44e-4 W（float32 量级） |
| 4 | de_inputs 哈希门 | **待服务器执行**（v1 补丁构建时） |
| 5a | 质量门 | PASS（passed=true），2 个 WARN 见下 |
| 5b | 记忆化审计 | **PASS**：replicated=0/246、exact=0、阈值 0.1409（min 距离 0.1441 > 阈值）——预声明的"定义性复刻"未触发 |

## WARN 接受记录（预注册 §4 要求：下游 sbatch 前记录）

**W1 duration_distribution=WARN**（out_of_duration=102/246）
- 数值：synthetic 时长分位 331.5 / 5301 / 10446 s（q05/q50/q95），
  real 参考 2721.6 / 5460 / 6132 s。
- 理由：B4R 的周期时长由 path + donor 长度决定，而二者与已接受的 B4 s17
  运行**逐位相同**（硬门 2）——归档 B4 质量报告的 out_of_duration 同为
  102/246、时长分位逐位一致（331.5/5301/10446），同 WARN 当时已被接受并
  进入下游。该 WARN 反映路径模型的长度展布，与基元来源无关，消融不应
  （也无法）改变它。
- 接受人：用户（2026-09-27 会话内书面追认："1.追认"）。

**W2 diversity=WARN**（identical_pairs=1，min_pairwise_distance=0）
- 数值：`synthetic_0168` 与 `synthetic_0192` 完全相同——均为单状态 state-0
  记录、选中同一 donor 块 `1621`（纯内部生日碰撞；19 条单状态记录 vs 467
  个 state-0 donor，期望碰撞数 ≈0.4，观测 1 例在期望内）。B4 归档报告该项
  为 PASS（min_pairwise 0.0257>0），差异机理是 CVAE 解码噪声此前把同
  (path, donor) 抽取拆开了不同波形。
- 理由：donor 直取的定义属性，属集内多样性而非对真实数据的复制；对真实
  数据的记忆化门全过（0/246、exact=0）。1/246 不构成训练信号泄露风险
  （同一波形出现两次 ≈ 该 donor 权重翻倍，与 donor 重采样的正常波动同级）。
- 接受人：待用户追认（同上）。

## 附带观察（非门项，供判读参考）

- **peak 分位**：B4R q50=3167.5W vs B4 q50=1929.8W——真实基元恢复了 CVAE
  被 MSE 训练削平的峰值（§8.4"基元模糊性"论断的定量注脚）。
- **记忆化距离**：B4R 的 min=0.1441 > B4 的 0.1293，replicated 0/246 <
  B4 的 1/246——真实 donor 拼接体在整周期形状空间反而更远离真实周期。
- energy q50：773.5 vs 762.3 Wh（接近）。

## 产物清单（本机，数据不入 Git）

- `reports/b4r/local_s17/b4r_gen/`（246 周期 + generation_summary.json）
- `reports/b4r/local_s17/quality_report.json` / `memorization_report.json`
- `reports/b4r/local_s17/de_placed_b4r/`（B4R 臂放置 shards + summary）

## 服务器执行补记（2026-09-28）

- 服务器 S0–S2 全过（两次运行复现）：库哈希 OK；双门机判
  `WARN=['diversity','duration_distribution'], replicated=0/246, exact=0`；
  **等价断言 246/246（服务器侧 vs `$ART/b4_s17_4133` 归档，硬门 2 双侧闭合）**；
  starts 246/246。
- S3 首次失败，查因：助手脚本 `--extra-arm` 误指按臂子目录
  （`de_placed_r0p5/B3T/`）；归档证实 v1 构建是把合并目录
  `de_placed_r0p5` 传七次（shard 路径自带 `LABEL/` 前缀，与 v1 manifest
  绝对路径吻合）。失败点在 prepare 的 extra-arm 校验（L118），先于
  output_dir.mkdir（L170）→ **无半成品，冻结源未动**。属命令规格修正，
  非再生成，不触发新协议版本。
- 修正提交 0c86245 因服务器到 GitHub 网络中断（GnuTLS -110 / 拉取挂起）
  未能拉取；以仓外脚本（sed 派生，仅改 extra-arm 指向与 ROOT 定位）执行，
  repo 保持 579ad5e 干净，下游 preflight 冻结于 579ad5e。网络恢复后
  补拉即可（服务器侧无本地提交，无分叉风险）。
- **网络随即恢复，pull 至 c60b580 后全链重跑通过**：S3 以合并目录方式
  重建 `de_inputs_r0p5_b4r` 成功——哈希门 23 个同名文件逐字节相等、
  manifest 十臂子集校验通过、B4R 唯一新增臂；prepare 窗口数
  （train 1,309,408 / validation 2,178,332 / B0 active 60,022）与 v1
  manifest 完全一致。**硬门 1–4 全部闭合（本地+服务器双侧）**。
  下游三个 sbatch（SEED=17/42/73，冻结 c60b580）待提交。
- 下游首轮提交（作业 5345/5346/5347）：**5346（s42）正常训练**
  （config/history/checkpoint 齐全）；**5345（s17）死于 train_nilm 启动时**
  `CUDA initialization: CUDA unknown error`（preflight 已过、环境已记录、
  无训练结果；其日志内 nvidia-smi 显示 GPU 空闲，同节点同环境的 5346
  正常——判为共享 GPU 瞬时状态问题）；**5347（s73）死于 nvidia-smi 阶段**
  `Unable to determine the device handle for GPU0: 0000:02:00.0:
  Unknown Error`（比 5345 更早，与 5345 同族：三作业并发启动时间窗内
  节点 GPU0 驱动坏状态）。两作业均无任何训练结果 → 重提不构成
  预注册不重跑条款意义上的重跑（该条款针对看过结果后的再生成）；死作业
  run 目录（仅环境文件）保留作证据，不删除。
- 重提轮（作业 5348=s17、5349=s73）：提交后 13 秒内两作业即离开队列
  （训练需数小时，不可能正常完成）——**日志查证为再次秒死，且定性升级
  为分区级 GPU/驱动故障**：
  - 5348＝5345 完整复刻（.err/.out 逐字节同大小）：nvidia-smi 见卡空闲 →
    preflight 过 → CUDA init `unknown error` 死——落在 h104 的第二块卡
    （卡可见、新 CUDA 上下文建不起来；h104 共 2×3090，5346 持健康卡）
  - 5349＝5347 复刻，.out 暴露 **host=h103-slurm-a**：作业内 nvidia-smi
    "No devices were found" + 驱动拿不到 GPU0（0000:02:00.0）句柄——
    h103 的卡整个不可见
  - 分区拓扑（sinfo）：h102 down\*、h103 idle（卡不可见）、h104 mix；
    全分区仅 5346 所持一卡确认健康。同脚本 5346 正常训练 → 排除
    脚本/数据/协议因素；s73 两轮均死于 nvidia-smi 步骤（无 run 目录、
    无任何结果产生）
  - 处置：1 分钟 GPU 探针作业复测 h103/h104 当前状态；探针恢复 →
    就地单作业串行重提；仍坏 → 报管理员做驱动级检查/GPU reset
    （h104 reset 须等 5346 结束）。5346 不受影响继续跑 s42。
- **用户确认（2026-09-28）：分区仅 h104 可用**（h103/h102 均不可用）。
  h104 两卡中 5346 持好卡、空闲卡状态未知 → 以 s17 作业本身为探针
  （`-w h104-slurm-a` 直接提交，1 分钟内见生死，死亡零结果合规）：
  活则 s73 排队跟进（两卡占满自动串行）；死则停提，等 5346 交出
  s42 的 MAE 并释放好卡后再试；若两卡全空仍连续落死卡 → 报管理员
  （草稿含 h103 卡不可见、h102 down、h104 第二卡 CUDA 死三项）。
- **结局（2026-09-28）**：探针 s17 作业（5350）**并非秒死而是完整跑完**——
  助手一度按两张 squeue 间隔误判"消失＝死"，实为第二张 squeue 拍在
  完成之后；.err 无任何 CUDA 错误，NNPACK 时间戳 16:58:17（train）
  / 17:32:12（predict）证明完整生命周期 ~34 分钟。**h104 恢复可用**
  （5346 结束、两卡空闲后新 CUDA 上下文成功；死卡闲置自愈或调度换卡）。
- **首个下游结果：5346（s42）完整跑完**（16:10→16:38，28 分钟；25 epoch
  早停，best epoch 20，val_mae 10.025W 监控口径；最终 evaluate 20k 窗口）：
  **val MAE = 10.024954504919052W**（F1 0.860、SAE 0.0797）。
  对照锚点（记录用，判读等三种子齐）：B4 s42 = 22.761、
  B2@r0.5 = 11.84±1.61、B2@r2.0 平台 = 10.38±0.08。
- **s17 结果（5350，完整确认）**：finalize/finished=17:32:14 齐全，
  **val MAE = 11.104258522701263W**（F1 0.731、SAE 0.168）。
  两种子在手：s17=11.104 / s42=10.025（vs B4 同种子 15.387 / 22.761）。
  s73＝作业 5351（h104，提交于 17:3x，预计 ~30 分钟）。
  算术边界（非预判）：s73 ≥ 25.79W 才会使 x̄ > 15.64 离开
  "生成器模糊"区间（比 B4 最差种子 22.761 还高）；≥ 35.96W 才达
  "组合天花板"线。待 5351 实测后按预注册 §5 层 2 终判并回填 §8。
- **s73 结果（5351，完整确认）**：finalize/finished=20:17:07 齐全，
  **val MAE = 12.163531839227677W**（F1 0.641、SAE 0.214）。
- **层 2 终判（预注册 §5，2026-09-28）**：x̄ = (11.1043+10.0250+12.1635)/3
  = **11.098W**（种子 sd 1.07）≤ 15.64，距档边界 4.54W ≥ 1.5W 噪声边际
  → **【生成器模糊为主因】成立，无噪声带限定**：B4 validation 判负的
  元凶是 CVAE 基元经 MSE 训练的模糊性，**不是**组合路径天花板
  （B4R 11.10±1.07 vs B4 19.43±3.74，Δ̄=−8.33W，3/3 种子同向；消融
  唯一变量＝基元波形来源，硬门 1–4 双侧闭合）。预注册后续动作触发：
  **值得训练 B4-WGAN / B4-扩散基元**。相对锚点：优于同预算 B2@r0.5
  （11.84±1.61）0.74W；距 B2@r2.0 平台（10.38±0.08）0.72W（自身
  sd 内，描述性）。层 1 初筛同方向但距界 0.996W<1.5W 带噪声带限定，
  与层 2 无冲突。结论已回填预注册 §8。
- 打包清单已发用户（b4r_results_20260928.tar.gz：cycles+双报告+
  placement summary+de_inputs manifest+哈希清单+三个完整 run 目录+
  两个死作业空目录 5345/5348+全部作业日志），回传本机
  reports/server_b4r/ 后校验归档。

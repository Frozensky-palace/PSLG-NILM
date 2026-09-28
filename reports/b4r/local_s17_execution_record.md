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

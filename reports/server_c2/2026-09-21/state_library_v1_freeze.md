# 状态库 v1 冻结报告（M1 里程碑）

- 日期：2026-09-21
- 结论：**冻结 state library v1 = C2-detsec-k4**
  （`reports/server_c2/2026-09-21/results/detsecpc_k345_trainonly_4107/state_library_k4/`，
  `library_status=formal_candidate_detsec_pc`，git_commit `88b3269`）

## G-1 格式兼容验证（零代码通过）

B2 构建器消费的 pilot 库格式与 C2 库逐项比对：

| 项 | 结果 |
|---|---|
| `state_inventory.csv` 列 | 31 列逐列一致 |
| `state_waveforms.npz` 键 | 一致（offsets/power_w/state_block_id/timestamp） |
| 构建器关键列非空 | state_label / duration_seconds / mean_power_w / waveform_offset_* / cycle_id 全部通过 |

原因：两类库由同一生产者（`build_state_library.py`）产出，格式天然同构，
无需适配器。

## G-2 validation 选 k 复核

方法：Phase A 冻结策略（B2-matched：duration-only donor + 限长 [0.67,1.5] +
top-5）分别作用于三个 C2 库，seed 17、ratio 0.5、validation CPU smoke
（train-per-class 25,000 / validation 50,000）。管线：
build → place（246 事件、0 重叠、guard 300s、idle 20W）→ prepare → smoke。
每臂窗口数与冻结 C3 输入一致（train 1,309,408 / validation 2,178,332）。

| 库 | B2-matched MAE (W) | F1 |
|---|---:|---:|
| C2-k3 | 24.305 | 0.4887 |
| **C2-k4** | 24.434 | 0.4855 |
| C2-k5 | 24.334 | 0.4874 |
| （参考）pilot-k4 | 23.855 | 0.4890 |
| （参考）B1 / B0 | 21.277 / 19.920 | — |

**判读**：三个 k 的差异 ≤ 0.13W（约 0.5%），在该树模型噪声范围内，
validation 不构成对任何 k 的显著偏好。因此按结构证据决定：

1. **k=5 排除**（生成视角硬伤）：状态 4 仅 2 块/2 周期，不可训练；
2. **k=3 次选**：合并中功率与高功率， primitive 语义少一档，转移图仅 6 边；
3. **k=4 冻结为 v1**：结构与 pilot 物理态交叉印证、无退化态、11 条转移边
   最利于 B4/B5 的基元与顺序建模，且 validation 无惩罚。

方向性观察（供 C5 参考，非本报告结论）：三个 C2 库的 B2-matched 均落在
24.3–24.4W，弱于 B1（21.277）与 B0（19.920）的既有参考，与 Phase A 的
方向一致——真实基元重组尚未超过完整周期重放。

## 过程记录

- k5 首次放置曾因命令笔误误用 k3 的配对产物，在生成任何指标前发现，
  已删除错误产物并用正确 k5_build 重跑；上表 k5 数字来自修正后管线。
- 中间产物（k{3,4,5}_build/placed/inputs/smoke）保留于
  `reports/core_validation/ukdale_b1_washing_machine/c2_state_library_k_check/`，
  如遇磁盘压力可删 placed/inputs（保留 build 的 provenance 与 smoke 指标）。

## 影响与后续

- **v1 立即服务对象**：D/E 批次 1–5（B3 条件生成、B4 基元生成、B5 受约束拼接）。
- C3 的 B2 arm 不变：仍用 Phase A 冻结的 B2-matched 输入（bundle 内，
  pilot-k4 基），这是协议定义的 C3 冻结基线。
- 后续若 C5 判决要求"detsec 库版 B2"对照，可基于 v1 追加（登记为新 arm，
  不改动本冻结）。

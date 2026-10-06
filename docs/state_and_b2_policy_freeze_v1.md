# 状态粒度与 B2 策略冻结 v1（state_and_b2_policy_freeze_v1）

> 冻结日期：2026-09-20。
> 依据：本机 validation-only CPU smoke（单 seed=17、ratio=0.5 pilot），输入全部为
> train-only 状态库与共同背景实验；最终 test 未访问。
> 本文档冻结"真实基元拼接策略与状态粒度候选"。服务器 DETSEC-PC 正式状态发现后可按
> C2/C5 协议更新为 v2，但不得在不留记录的情况下回退到与本文件冲突的规则。

## 1. 冻结决定总表

| 项目 | 冻结值 | 依据 |
|---|---|---|
| B2 donor 匹配特征 | **duration-only**（时长对数差） | A3：功率匹配项一致有害 |
| 长度比例限制 | **[0.67, 1.5]**（target/donor） | A3：MAE 最优 |
| 长度限制 fallback | 过滤后无跨 cycle 候选时回退全池，计数入 provenance | 17 链中仅 1 次 fallback |
| 匹配 top-k | **5** | top-1/top-10 均略差，影响温和 |
| 边界处理 | **none**（不平滑、不校正） | A4：cross-fade 与端点校正均使下游变差 |
| 边界短窗范围 | 如未来必须平滑：6–30 样本内，禁止长窗 | A4：10 样本已致 MAE 变差 |
| 状态粒度首选 | **k=4**（primary） | duration-only 后 MAE 与 F1 双优 |
| 状态粒度对照 | **k=5**（secondary，仅作粒度对照） | 与服务器 DETSEC-PC 对照用 |
| 落选规则 | 功率匹配、cross-fade、endpoint-offset、k=3 | 负结果记录于消融报告，不静默丢弃 |

## 2. 关键证据（validation-only，单 seed=17 pilot）

| 比较 | MAE (W) | F1 | 结论 |
|---|---:|---:|---|
| B0 纯真实 train | 19.920 | 0.5608 | 仍优于全部增强组（疑似域偏移，见 §6） |
| B1 完整真实周期重放 | 21.277 | 0.5535 | 真实基元重组仍未超过它 |
| B2-random | 31.909 | 0.3703 | 无约束基元拼接严重有害 |
| B2-matched full（k4） | 23.8555 | 0.4886 | 功率匹配项引入噪声 |
| **B2 冻结策略（k4）** | **22.007** | 0.5163 | 较 full 匹配改善 7.8% MAE |
| B2 duration-only（k4，无限长） | 22.129 | 0.5245 | 变体中 F1 最高 |

注：最后一行在严格意义上不是与上行的单因素对照（限长是它的增量因素），两行都
进入候选；服务器首轮用冻结策略（含限长），如需归因可回退比较。

## 3. 冻结策略的精确规则（供服务器复现）

1. **donor 池**：同状态、跨 cycle 的 train 状态块。
2. **长度比例过滤**：`target_samples / donor_samples ∈ [0.67, 1.5]`；
   过滤后若无跨 cycle 候选，回退为不限长的同状态跨 cycle 全池，并在 provenance
   中把 `ratio_limited_fallback_donors` 加一。
3. **评分**：`|log(donor_samples / target_samples)|`，升序取 top-5，等概率随机选一
   （rng 为 `np.random.default_rng(seed)`，按合成序号顺序消耗）。
4. **拼接**：donor 线性重采样到模板块长度后直接连接；不做平滑或端点校正；
   不改变周期总长度；脚本断言无负功率。
5. **背景放置**：与冻结 pilot 相同（seed=17、idle 20 W、guard 300 s、共同背景、
   事件重叠 0）。
6. **粒度**：主用 `state_library_pilot_k4_v1`；k=5 仅作对照，不得混用不同 k 的
   状态块于同一合成 cycle。

## 4. 与旧产物的关系

- 原始 B2-matched（full 特征）目录保持冻结不动；
- 新策略产物在 `b2_policy_ablation_seed17_r0p5/`，不覆盖任何基线目录；
- `aligned_partitions_v1` 仍然禁用。

## 5. 边界处理负结果的记录义务

A4 的两个负结果必须随协议传递，不得因为"看起来更平滑"而在 B5 中重新引入：

- 线性 cross-fade（10 样本窗）把边界跳变从约 620 W 降到约 75 W，但 k4/k5 的
  validation MAE 一致变差（k4：23.855→24.780；k5：24.183→25.528）；
- 端点偏移校正（endpoint offset + clip）能量漂移达数百 Wh 且不稳定。

对 B5 的直接指导：**边界约束通过 donor/生成状态的端点选择实现，不通过波形平滑**；
任何未来平滑必须以 validation 显著改善为前提并留档。

## 6. 已知限制与服务器阶段义务

- 全部结论来自单 seed、树模型 CPU smoke，只能作方向筛选；
  正式结论以服务器 Seq2Point 3 seeds 为准（C3）。
- B0 仍优于全部增强组：服务器必须先核查合成放置/采样机制是否引入域偏移（C5），
  再进入生成阶段。
- 状态库仍是 physical-stats pilot 版本（`pilot_physical_stats_not_final`）；
  本文件冻结的是 k 候选与拼接规则，不是最终状态定义；DETSEC-PC 正式库完成后按
  C2 重新评估粒度，与 k=4/5 对照。
- 消融共 18 个变体链（8 个单因素 × k=4/5，加 2 个 duration+限长组合），
  **全部一次成功**，无失败、无静默重跑；`ablation_results.csv` 共 18 行数据。

## 6.1 引用产物路径

```text
reports/core_validation/ukdale_b1_washing_machine/b2_policy_ablation_seed17_r0p5/
├─ ablation_results.csv
├─ report.md
└─ <tag>/{paired,placed,inputs,smoke}/
reports/core_validation/ukdale_b1_washing_machine/cpu_nilm_smoke_b2matched_k{3,4,5}_vec2/
reports/core_validation/ukdale_b1_washing_machine/cpu_nilm_smoke_b2random_vec2/
reports/state_quality/k3_k4_k5_exchangeability.md
```

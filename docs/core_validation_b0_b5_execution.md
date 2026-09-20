# B0–B5 核心验证第一阶段执行方案

> 版本：v1  
> 日期：2026-09-20  
> 第一研究对象：UK-DALE House 1 washing machine

## 1. 这一步在做什么

先把所有可用洗衣过程列成清单，再把完整洗衣过程分成训练、验证和测试三组。
测试组保持完全真实，后续所有方法都在同一套测试题上比赛。

这一阶段本地电脑即可完成，不需要 GPU 服务器。

## 2. 当前已经确认的数据范围

- 洗衣机支路：2012-11 至 2017-04；
- House 1 总表：2013-03 至 2017-04；
- 两者共同可用范围：约 2013-03 至 2017-04；
- House 1 在 2015-09 更换过物理洗衣机；
- 之前完整流程只使用 2013-04 一个月，共提取 28 个活动周期；
- 已有完整支路 CSV：`input/ukdale_house1_washing_machine.csv`。

因此先扫描完整支路 CSV，建立两台物理洗衣机的总清单。第一轮 B0–B5 只选择
2015-09 更换前的 instance 1；instance 2 留作后续跨设备泛化验证。

## 3. 第一阶段文件

### 输入

```text
input/ukdale_house1_washing_machine.csv
datasets/ukdale/ukdale.h5
```

### 输出

```text
reports/core_validation/ukdale_b1_washing_machine/
├─ cycle_inventory.csv
├─ cycle_inventory_summary.json
├─ cycle_inventory_with_split.csv
├─ partition_manifest.json
└─ leakage_report.json
```

简单解释：

- `cycle_inventory.csv`：每一次候选洗衣过程的清单；
- `cycle_inventory_summary.json`：一共有多少周期、时长和功率的大致分布；
- `cycle_inventory_with_split.csv`：每个周期被分到训练、验证还是测试；
- `partition_manifest.json`：后续程序使用的正式分组名单；
- `leakage_report.json`：检查是否有同一周期混入多个分组。

## 4. 周期判定规则

第一版沿用 UK-DALE 元数据和仓库现有逻辑：

```text
功率达到 20 W                 → 视为活跃点
相邻活跃点间隔不超过 150 秒   → 仍属于同一次洗衣过程
完整活动持续至少 1800 秒       → 视为候选完整周期
周期位于总表有效时间范围内     → 可以参加正式实验
```

这只是第一版固定规则。后续会通过周期清单和波形抽检判断是否需要调整，但不能在
查看最终测试结果之后为了提高分数而修改。

## 5. 无泄漏划分

周期按照时间先后排序后分配：

```text
前 60% 完整周期 → train
中间 20%        → validation
最后 20%        → test
```

一个完整 cycle 只能属于一个分组。后续：

- train：建立状态库、训练生成器和 NILM；
- validation：选择模型和参数；
- test：只做最终真实评价。

## 6. B0–B5 固定定义

| 组别 | 简单解释 |
|---|---|
| B0 | 只使用原始真实训练数据 |
| B1 | 完整真实周期不拆分，只重新放置 |
| B2 | 从不同训练周期选择同类真实状态片段进行拼接 |
| B3 | 一次生成完整洗衣周期 |
| B4 | 分别生成各状态，再按真实状态顺序拼接 |
| B5 | B4 加入状态顺序、持续时间和边界约束 |

核心比较：

```text
B2 vs B1：基元重组有没有价值
B4 vs B3：基元生成是否优于普通整段生成
B4 vs B2：生成新基元是否优于重组真实基元
B5 vs B4：加入物理约束是否有价值
```

## 7. 统一比赛规则

所有方法必须使用：

- 相同真实训练周期；
- 相同训练背景负载；
- 相同启动时间表；
- 相同合成周期数量和近似总时长；
- 相同 NILM 模型；
- 相同训练轮数和随机种子；
- 相同 validation 规则；
- 相同的完全真实 test。

第一轮随机种子固定为 17、42、73，真实与合成比例测试 1:0.5、1:1、1:2。

## 8. 执行命令

### 8.1 生成完整周期清单

```powershell
.\.venv\Scripts\python.exe scripts\build_cycle_inventory.py `
  --input input\ukdale_house1_washing_machine.csv `
  --output-dir reports\core_validation\ukdale_b1_washing_machine `
  --dataset UK-DALE --building 1 --appliance washing_machine `
  --threshold-w 20 --max-inactive-seconds 150 `
  --min-duration-seconds 1800 --sample-seconds 6 `
  --valid-start 2013-03-17T19:12:43Z `
  --valid-end 2017-04-26T17:35:58Z `
  --device-instance-cutover 2015-09-07T23:00:00Z `
  --before-instance 1 --after-instance 2
```

### 8.2 生成无泄漏划分

```powershell
.\.venv\Scripts\python.exe scripts\create_research_split.py `
  --inventory reports\core_validation\ukdale_b1_washing_machine\cycle_inventory.csv `
  --output-dir reports\core_validation\ukdale_b1_washing_machine `
  --train-ratio 0.6 --validation-ratio 0.2 --test-ratio 0.2 `
  --device-instance 1
```

## 9. 完成条件

第一阶段只有满足以下条件才算完成：

- [x] 完整 CSV 扫描成功；
- [x] 周期清单已生成，可人工打开检查；
- [x] 每个 eligible 周期只属于一个 partition；
- [x] `leakage_report.json` 中 `passed=true`；
- [x] train、validation、test 都有足够的完整周期；
- [x] 第一轮分层抽取18个周期查看波形，未发现明显的大规模误检；
- [x] B0–B5 试运行协议已经冻结；正式实验前只允许根据波形质检结果做有记录的修订。

## 10. 完成后下一步

1. [x] 按 partition 分批提取严格对齐的 branch、mains 和 background；
2. [x] 执行周期级对齐覆盖质检，并在模型运行前记录排除规则；
3. [x] 从对齐数据生成仅包含 train 的真实完整周期库；
4. [x] 只对 train 完成 pilot 状态发现和 `state_merge`；
5. [x] 生成 pilot `state_inventory.csv`、波形库和转移图；
   正式冻结前仍需与服务器 `detsec_pc` 结果比较；
6. [x] 完成 seed=17、ratio=0.5 的 B1/B2 周期级配对和共同背景放置；
7. [ ] 使用统一 NILM 基线训练 B0/B1/B2，并只用 validation 选择参数；
   本机 CPU 轻量检查已完成，但它不是最终 Seq2Point 结果；
8. [ ] 完成全部种子和比例后，再开始 B3/B4 生成模型。

## 11. 本机 B2 约束诊断（2026-09-20）

原始随机 B2 的早期验证明显弱于 B1。检查发现，776 个基元中有183个发生超过
2倍的压缩或拉伸。为分清“基元思想有问题”还是“随机拼接方式有问题”，新增了
不覆盖原始 B2 的诊断组 `B2-matched`。

`B2-matched` 按持续时间、平均功率、起点功率和终点功率选择较接近的同状态
train 基元。它与原始 B2 使用完全相同的模板、背景位置和训练/验证预算。

CPU 轻量 validation 结果：原始 B2 的 MAE/F1 为 31.909 W/0.370，匹配后为
23.949 W/0.484；MAE 降低24.9%，F1 提高30.6%。不过 B1 仍为
21.277 W/0.553，因此目前不能宣称基元拼接优于完整周期拼接。

通俗结论：限制不合理的强行拉伸是有效的，但还需要细化状态、改善边界和持续时间
建模。详细记录见 `reports/core_validation/ukdale_b1_washing_machine/
b2_constraint_diagnostic.md`。本轮未读取 test 指标。

## 12. k=3/4/5 状态粒度诊断（2026-09-20）

已完成 k=4 和 k=5 的 train-only pilot 状态库，并在完全相同的246个模板、背景位置、
训练预算和真实 validation 子集上进行 B2-matched 比较。

| k | MAE (W) | F1 | 平均边界跳变 |
|---:|---:|---:|---:|
| 3 | 23.949 | 0.484 | 884 W |
| 4 | 23.855 | 0.489 | 626 W |
| 5 | 24.183 | 0.500 | 553 W |

k=4 的功率回归误差略好，k=5 的状态识别和边界更好。当前保留 k=4、k=5 两个
候选进入服务器正式 DETSEC-PC + Seq2Point，不用本机单 seed 树模型强行决定最终
状态数。详细报告见 `state_granularity_k3_k5_pilot.md`。test 仍未访问。

对齐后的正式数据位于 `aligned_partitions_v2/`。`v1` 是修正分片边界问题前的
中间产物，不能用于实验。每个 NPZ 中包含：

- `timestamp`：统一的 6 秒时间点；
- `mains_w`：家庭总功率；
- `appliance_w`：洗衣机功率；
- `background_signed_w`：原始背景，严格等于总功率减洗衣机功率；
- `background_clipped_w`：负数截为 0 后的背景，供拼接实验使用；
- `segment_id`：连续数据段编号，训练窗口不允许跨编号。

覆盖质检结果：train 493/493 完整，validation 164/164 完整，test 165/166
完整。测试集中唯一不完整周期只保留作审计，不进入模型评估。

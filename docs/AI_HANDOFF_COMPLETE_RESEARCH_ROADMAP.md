# PSLG-NILM 基元状态生成与拼接研究：完整后续执行计划与 AI 交接说明

> 文档版本：v1.2  
> 整理日期：2026-09-20  
> 仓库：`D:\zhj\vscode\PSLG-NILM`  
> 第一研究对象：UK-DALE House 1 washing machine  
> 主要用途：供后续 AI 模型、开发者和服务器实验直接接手执行  
> 当前状态：Phase A 与 Phase B 本机部分已完成（B2 策略与粒度候选已冻结、Seq2Point
> 训练器已实现且本机冒烟跑通、全套 106 项测试通过）；服务器 DETSEC-PC、正式
> Seq2Point 多 seed 网格与 B3–B5 生成模型尚未开始。B2 冻结策略见
> `docs/state_and_b2_policy_freeze_v1.md`

---

## 0. 给接手 AI 的最短说明

本项目原本是一个 NILM 电器活动的“提取、切分、表征、聚类、少样本识别和
knockout 数据构造”流水线。新研究计划在其后增加两条生成路线：

1. **普通生成**：直接生成完整电器工作周期；
2. **基元生成**：先生成较短的工作状态，再按照真实状态顺序、持续时间和边界约束
   拼成完整周期。

最终不是比较哪一种波形“看起来更像”，而是比较：在相同真实数据、合成数量、
训练预算和 NILM 模型下，哪种增强数据能让模型在**完全真实且未参与调参的 test**
上表现更好。

当前最重要的已知结果是：随机选择真实基元并强行拉伸的 B2 表现很差；加入长度
和功率匹配后的 `B2-matched` 明显改善，但仍未超过完整真实周期拼接 B1。这说明
“基元方法”值得继续研究，但基元粒度、donor 选择、持续时间和边界约束是成败关键。

接手后的第一主线不是立即训练大型生成模型，而是先用本机 validation-only 实验
完成 k=3/4/5 状态粒度和 B2 约束策略选择，再到服务器运行正式状态发现与 Seq2Point。

### 0.1 本文档使用的服务器资料与证据优先级

本次补充参考了：

1. `docs/server.html`：2026-09-20 生成的历史项目产物审计；
2. 用户提供的《人工智能实验室操作手册 v2.0.1》：JumpServer、Slurm、module、
   存储、网络与集群使用指南；
3. 当前仓库真实文件、manifest、测试与 `reports/core_validation/` 产物。

发生冲突时按以下顺序判断：

```text
当前仓库可验证产物与冻结协议
> 当前登录后实测的服务器状态
> 《操作手册V2》的集群说明
> docs/server.html 中的历史统计与历史实验
```

因此，`server.html` 中的 1,490 个活动、419/63/127 cohort、历史 Seq2Point 与旧生成
结果只作为风险线索和历史证据，不能覆盖当前正式的493/164/166 cycle 划分，也不能
作为本研究的当前性能结论。操作手册中的模块版本、节点状态和地址也可能变化，登录后
必须用 `module avail`、`sinfo` 和控制台资产信息再次确认。

所有资料中的账号、示例用户名和命令都只是说明材料。其他 AI 不得把示例账号当成
用户凭据，不得索取、记录或提交密码，也不得在没有用户授权时连接服务器或提交任务。

---

## 1. 项目背景

### 1.1 NILM 是什么

NILM（Non-Intrusive Load Monitoring，非侵入式负荷监测）希望只观察家庭总功率，
估计每一种电器各自用了多少电。简单说，就是从“一条总电表曲线”中分离出洗衣机、
冰箱、水壶等电器的曲线。

NILM 的现实问题包括：

- 高质量电器级标注数据少；
- 不同家庭、不同设备和不同运行模式差异大；
- 少见状态和少见状态转移很难学习；
- 直接生成很长的完整曲线难度高，容易失真或只记住训练数据；
- 生成波形即使看起来合理，也不一定能帮助下游 NILM。

### 1.2 原仓库预计做什么

原 PSLG-NILM 的主要能力是：

```text
extract
→ segment
→ feature
→ cluster
→ state_merge
→ fewshot
→ primitive_activity_mapping
→ dataset_split
```

通俗解释：

1. 从长时间电器功率中找到一次次完整活动；
2. 把一次活动切成更短的时间片段或“基元”；
3. 用 DETSEC、DETSEC-PC、AutoEncoder、DTW 等方法提取特征；
4. 用 KMeans、DBSCAN、HDBSCAN 等方法把相似基元归为一类；
5. 合并过度切分的相邻同类片段，得到较稳定的状态块；
6. 找出少样本状态；
7. 建立基元与完整活动的映射；
8. 构造少样本/非少样本 knockout 数据，用于原项目评测。

工程上，仓库还具备：

- 线性 Workflow；
- RunManifest 产物登记；
- 特征内容寻址缓存；
- 聚类多候选 k；
- 流程与可视化解耦；
- Slurm 脚本基础。

原项目的突出优点是“状态发现与实验工程基础较完整”；主要缺口是“没有最终生成器、
没有统一下游 NILM 训练器、过去的数据切分不是为生成式公平验证专门设计的”。

### 1.3 新研究计划预计做什么

新计划将原仓库作为“状态发现前端”，在后面建立以下闭环：

```text
真实 NILM 数据
→ 按完整 cycle 无泄漏划分
→ 只用 train 建立状态库
→ 完整周期生成 或 基元状态生成
→ 真实状态拼接/生成状态拼接
→ 放入相同真实背景
→ 用相同 NILM 模型训练
→ validation 选择方案
→ 冻结后只在真实 test 上评价一次
```

计划回答五个研究问题：

1. 洗衣机内部是否存在稳定、可复用的工作状态？
2. 不同 cycle 的同类真实状态是否可以交换和拼接？
3. 生成基元再拼接是否优于直接生成完整周期？
4. 状态顺序、持续时间和边界约束是否真正有效？
5. 基元方法是否在低数据量、少样本状态或计算效率方面更有优势？

### 1.4 当前第一研究对象为什么是 washing machine

早期文档曾建议用 kettle 快速验证流程，但当前正式第一对象已经冻结为
UK-DALE House 1 washing machine，原因是：

- 洗衣机具有多个工作阶段，更适合验证状态分解和拼接创新；
- 数据量足够大；
- 2015 年 9 月前后存在物理设备更换，可在后续做跨设备泛化；
- 简单电器不能充分证明受约束状态拼接的价值。

---

## 2. 核心创新点与验证逻辑

### 2.1 核心创新点

核心创新不是“把任意短片段接起来”，而是：

> 将完整电器工作周期分解为可复用状态，分别学习或生成这些状态，再通过状态顺序、
> 持续时间、边界和运行上下文约束重建完整周期，以提高数据利用率、生成可控性和
> 下游 NILM 效用。

### 2.2 B0–B5 核心组

| 组别 | 数据方案 | 要回答的问题 |
|---|---|---|
| B0 | 只用原始真实 train | 不增强时的基准性能 |
| B1 | 完整真实 cycle 重新放置到背景 | 完整真实周期拼接本身的收益 |
| B2 | 从不同 train cycle 选择同类真实状态并拼接 | 不训练生成器时，基元结构是否有价值 |
| B3 | 一次直接生成完整 cycle | 普通整段生成基线 |
| B4 | 分状态生成基元，再进行基础拼接 | 基元生成是否优于整段生成 |
| B5 | B4 加状态顺序、持续时间、边界和上下文约束 | 完整受约束方法是否进一步提升 |

补充诊断组：

| 组别 | 定义 | 当前用途 |
|---|---|---|
| B2-random | 原始随机 donor 的真实基元拼接 | 暴露无约束拼接问题 |
| B2-matched | 按时长、均值、起点和终点功率匹配 donor | 验证约束选择是否有效，不替代冻结的原始 B2 |

### 2.3 必须完成的核心比较

```text
B2 vs B1       真实基元重组是否优于完整真实周期拼接
B2-matched vs B2-random
               donor 匹配和限制时间缩放是否有效
B4 vs B3       生成基元再拼接是否优于直接整段生成
B4 vs B2       生成新基元是否优于只重组真实基元
B5 vs B4       顺序、持续时间和边界约束是否有效
B5 vs B3       最终创新方法是否优于普通生成
B0/B1 vs 其他  复杂生成是否真的比不生成或简单重放更值得
```

---

## 3. 不可违反的实验纪律

### 3.1 数据泄漏红线

1. train、validation、test 必须按完整 cycle 和连续时间划分。
2. 同一个 cycle 不得拆开进入多个分区。
3. 状态聚类、聚类中心、归一化、状态顺序、持续时间分布、donor 池和生成模型只能
   使用 train。
4. validation 只用于模型和超参数选择。
5. test 在所有方案、超参数、阈值和统计方法冻结前不得用于比较方案。
6. 当前 test 中唯一覆盖不完整的 cycle 必须保持排除状态，不可为了数量补入。
7. 每个合成 cycle 必须保存 provenance，至少包含来源 partition、模板 cycle、
   donor block、状态、随机种子、生成器版本和配置哈希。

### 3.2 公平比较红线

所有核心组必须尽量保持：

- 相同真实训练数据；
- 相同真实背景；
- 相同事件启动时间表；
- 相同合成 cycle 数量或相同合成总时长；
- 相同真实与合成比例；
- 相同 NILM 架构、训练步数、batch size、早停规则和随机种子；
- 相同 validation/test 窗口；
- 相同指标实现；
- 生成模型比较时使用相同或明确可比的参数量、训练时间或计算预算。

### 3.3 禁止行为

- 不得使用 `aligned_partitions_v1`；该目录是修复分片边界前的旧产物。
- 不得根据 test 结果修改状态数、阈值、模型或拼接规则。
- 不得把本机 CPU smoke 当成最终 Seq2Point 结论。
- 不得只报告生成曲线视觉效果或 FID 类指标，而不做真实 test 下游评价。
- 不得删除或覆盖原始 B2；新的优化策略使用新名称和新目录。
- 不得在脏工作树中使用 `git reset --hard` 或覆盖不相关的用户修改。
- 不得因为某组结果不好而静默重跑；失败和重跑原因必须记录。

---

## 4. 已完成的数据与实现

### 4.1 数据范围和 cycle inventory

已经扫描完整洗衣机 CSV：

- 扫描行数：19,555,935；
- 候选活动组：1,821；
- eligible cycle：1,346；
- 排除：475；
- 物理设备 instance 1：823 个 eligible cycle；
- instance 2：523 个 eligible cycle；
- 活跃阈值：20 W；
- 最大活动间断：150 秒；
- 最短完整周期：1,800 秒；
- 采样间隔：6 秒。

第一轮 B0–B5 只使用 2015-09 设备更换前的 instance 1。instance 2 保留作后续
跨设备泛化，不得混入第一轮训练和调参。

### 4.2 无泄漏 research split

已按时间顺序和完整 cycle 完成 60%/20%/20% 划分：

| 分区 | cycle 数 | 时间用途 |
|---|---:|---|
| train | 493 | 状态发现、生成器训练、NILM 训练 |
| validation | 164 | 选择状态粒度、约束和超参数 |
| test | 166 | 最终冻结后评价一次 |

泄漏检查：

```text
passed = true
duplicate_cycle_ids = 0
eligible_unassigned = 0
chronological_order_ok = true
```

### 4.3 三路功率对齐

正式使用目录：

```text
reports/core_validation/ukdale_b1_washing_machine/aligned_partitions_v2/
```

每个 NPZ 包含：

- `timestamp`：6 秒时间网格；
- `mains_w`：家庭总功率；
- `appliance_w`：洗衣机功率；
- `background_signed_w = mains_w - appliance_w`：用于精确审计；
- `background_clipped_w = max(0, mains_w - appliance_w)`：用于合成；
- `segment_id`：连续段编号，任何训练窗口不得跨连续段。

覆盖结果：

| 分区 | 有效时间点比例 | 完整 cycle |
|---|---:|---:|
| train | 97.90% | 493/493 |
| validation | 99.98% | 164/164 |
| test | 96.72% | 165/166 |

### 4.4 train-only 真实 cycle 库

目录：`real_cycle_library_train_v1/`

- cycle：493；
- 总样本：429,335；
- 总洗衣机能量：359,643.31 Wh；
- 中位周期时长约 5,460 秒；
- validation/test cycle 数：0；
- 每个 cycle 已检查样本数、时间顺序和功率恒等式。

### 4.5 pilot 状态库

目录：`state_library_pilot_k3_v1/`

状态发现流程：train cycle → prim-glr 切分 → physical statistics 特征 →
KMeans → state merge。

- 原始 train cycle：493；
- 初始基元：2,982；
- k=3 合并后状态块：1,538；
- 全部状态记录只来自 train；
- 状态块对 cycle 的覆盖精确完整。

| 状态 | 状态块 | 覆盖 cycle | 中位功率 | 中位时长 | 当前解释 |
|---:|---:|---:|---:|---:|---|
| 0 | 476 | 444 | 1,014 W | 324 s | 中功率过渡/运行 |
| 1 | 587 | 493 | 211 W | 4,548 s | 长时低功率综合状态 |
| 2 | 475 | 367 | 1,866 W | 636 s | 高功率加热 |

当前 k=3 仅为 `pilot_physical_stats_not_final`。状态1很宽，可能混合洗涤、漂洗、
脱水等阶段，是下一阶段首要问题。

### 4.6 B1/B2 配对与共同背景

已经完成 seed=17、synthetic ratio=0.5 的 pilot：

- 配对 cycle：246；
- B1/B2 每对持续时间一致；
- B2 donor 全部来自其他 train cycle；
- 使用共同真实 train 背景；
- 事件重叠为0；
- 每个事件前后 guard 为300秒；
- 背景功率恒等关系误差接近浮点精度。

### 4.7 NILM 输入层

已实现框架独立的 lazy sharded window reader 和窗口清单：

- 窗口长度：599 点，即约 3,594 秒；
- train stride：6 点；
- validation/test stride：1 点；
- 窗口不跨缺失段或排除区间；
- B0/B1/B2 使用相同窗口中心；
- 归一化只来自 B0 real train；
- validation/test 均为真实数据；
- test 不完整 cycle 已连同窗口上下文排除。

窗口数量：

| 分区 | 窗口数 |
|---|---:|
| train | 1,309,408 |
| validation | 2,178,332 |
| test | 2,580,292 |

### 4.8 B2-matched 诊断

原始随机 B2 中，776 个 donor block 有183个发生超过2倍的压缩或拉伸。新增
`B2-matched`，在同状态内根据以下信息匹配：

- 持续时间；
- 平均功率；
- 起点功率；
- 终点功率。

改进后：

- 长度比例范围从约 0.058–17.861 收敛到 0.801–1.759；
- 超过2倍的压缩/拉伸从183个降为0；
- 合成能量从 B1 的97.2%提高到100.5%；
- 平均边界跳变从976 W降到884 W。

CPU validation-only smoke：

| 训练组 | MAE (W) | RMSE (W) | SAE | F1 | FP |
|---|---:|---:|---:|---:|---:|
| B0 | 19.920 | 87.817 | 0.309 | 0.561 | 4,870 |
| B1 | 21.277 | 95.535 | 0.354 | 0.553 | 5,029 |
| B2-random | 31.909 | 114.337 | 0.618 | 0.370 | 10,614 |
| B2-matched | 23.949 | 101.469 | 0.415 | 0.484 | 6,655 |

相对随机 B2，matched 的 MAE 降24.9%、F1升30.6%、误报降37.3%。但它仍
弱于 B1。因此当前只能说“约束 donor 有效”，不能说“基元拼接已经优于完整周期”。

### 4.8.1 k=4/k=5 状态粒度更新

已复用同一个 train-only run 中现有的 k=4/k=5 聚类与 state merge 结果，导出两个
独立状态库。k=4 将 k=3 的宽泛长状态拆成约153 W/3,300秒和400 W/1,266秒；
k=5 又进一步细分低功率时间结构。

| k | MAE (W) | F1 | 平均边界跳变 |
|---:|---:|---:|---:|
| 3 | 23.949 | 0.484 | 884 W |
| 4 | 23.855 | 0.489 | 626 W |
| 5 | 24.183 | 0.500 | 553 W |

k=4 的 MAE 最好，k=5 的 F1、误报和边界最好。差异不足以让树模型单 seed pilot
决定最终 k，因此保留 k=4、k=5 两个候选进入服务器 DETSEC-PC + Seq2Point。
详细报告见 `state_granularity_k3_k5_pilot.md`。

### 4.9 代码和测试

已经实现的研究脚本包括：

| 脚本 | 状态 | 用途 |
|---|---|---|
| `build_cycle_inventory.py` | 已实现 | 扫描完整 cycle |
| `create_research_split.py` | 已实现 | 无泄漏时间划分 |
| `audit_cycle_inventory.py` | 已实现 | cycle 波形审计 |
| `build_aligned_research_partitions.py` | 已实现 | 对齐 mains、appliance、background |
| `build_real_cycle_library.py` | 已实现 | 建立 train-only cycle 库 |
| `export_cycle_library_segments.py` | 已实现 | 为状态发现导出片段 |
| `build_state_library.py` | 已实现 | 建立状态库与转移图 |
| `build_b1_b2_pilot_cycles.py` | 已实现 | B1、B2-random、B2-matched |
| `place_b1_b2_on_train_background.py` | 已实现 | 公平放入共同背景 |
| `prepare_nilm_b0_b2_inputs.py` | 已实现 | 构建 NILM 窗口与采样池 |
| `run_cpu_nilm_smoke.py` | 已实现 | 本机轻量方向检查 |
| `evaluate_nilm_predictions.py` | 已实现 | MAE/RMSE/SAE/P/R/F1 |
| `build_server_transfer_manifest.py` | 已实现 | 最小服务器包及 SHA-256 |

全套本地测试：106 项通过（2026-09-20 更新；原 58 项，新增 Phase A/B 覆盖与状态
发现入口测试，并已
修复第三方 `tests` 包遮蔽导致的导入问题）。当前警告仅涉及 Windows 物理核心探测，
不影响结果。

---

## 5. 当前还没有完成的内容

以下内容不得被接手 AI 误认为已完成：

- k=4/k=5 的正式状态库和可交换性报告；
- 使用服务器 `detsec_pc` 特征的正式状态发现；
- Seq2Point 服务器正式多随机种子训练（本机训练器/检查点/预测代码已实现并通过
  小数据冒烟；多 seed 正式网格只在服务器执行，尚无结果）；
- B3 完整周期生成模型；
- B4 基元状态生成模型；
- B5 顺序、持续时间、边界和上下文约束生成；
- 生成模型的记忆/复制检查与分布质量评价；
- B0–B5 全部 seed、全部比例实验；
- 最终真实 test 结果；
- instance 2 跨设备泛化；
- 多电器 aggregate 合成；
- 自采集数据和 PCB 系统接入。

任何新脚本在实际落盘和测试前，都只能在文档中标为“计划实现”。

---

## 6. 本机与服务器的分工原则

### 6.1 适合在本机完成

本机适合数据整理、规则实验、小规模 CPU 模型、审计和开发：

- cycle 和状态库统计；
- k=3/4/5 的 physical-stats 状态诊断；
- B2 donor 规则、权重、top-k 和边界处理消融；
- validation-only CPU smoke；
- 数据读取、采样、指标和 provenance 单元测试；
- Seq2Point/生成器代码实现与小批量冒烟；
- 配置生成、实验矩阵检查、结果表汇总；
- 服务器传输包和 SHA-256 校验；
- 报告、图表和论文表格生成。

本机不适合长时间训练多个深度模型组合，也不应为了节约服务器使用而降低正式实验
预算。

### 6.2 必须或更适合在服务器完成

- GPU 版 DETSEC-PC 特征训练；
- 正式 Seq2Point B0–B5 多 seed、多比例训练；
- B3/B4/B5 生成模型训练；
- 大规模生成采样和质量评估；
- 多状态数、多超参数的受控网格；
- 多电器或跨住宅扩展；
- 需要重复训练才能计算置信区间的实验。

### 6.3 迁移规则

本机生成的 transfer manifest 是文件清单和校验基准。上传后必须：

1. 按 manifest 检查文件数、字节数和 SHA-256；
2. 使用项目相对路径，不把 Windows 绝对路径写进服务器配置；
3. 记录 Git commit、环境、GPU、CUDA、随机种子和开始/结束时间；
4. 服务器产物先写独立 run 目录，再同步回本机报告目录；
5. 禁止只复制最终指标而丢失配置、日志、检查点和 provenance。

当前 B2-matched 最小服务器包约278 MiB，不需要上传约5.9 GiB原始 HDF。

### 6.4 指南记录的集群资源

《操作手册V2》记录的 Slurm 计算资源如下。这里是规划参考，实际状态以登录后
`sinfo`、`scontrol show node` 和 `nvidia-smi` 为准。

| Slurm 分区 | 节点 | 每节点 GPU | 单卡显存 | 指南记录的可用内存 | 逻辑 CPU |
|---|---|---:|---:|---:|---:|
| RTX3090（默认） | h102-slurm-a | 2 × RTX3090 | 24 GB | 116 GB | 40 |
| RTX3090（默认） | h103-slurm-a | 2 × RTX3090 | 24 GB | 116 GB | 40 |
| RTX3090（默认） | h104-slurm-a | 2 × RTX3090 | 24 GB | 116 GB | 32 |
| A6000 | h107-slurm-a | 1 × A6000 | 48 GB | 116 GB | 32 |
| A6000 | h108-slurm-a | 1 × A6000 | 48 GB | 116 GB | 32 |

集群合计为6张 RTX3090 和2张 A6000。指南记录单次作业最长7天，RTX3090 是默认
分区。不得使用 `--exclusive` 独占节点。

建议资源映射：

- DETSEC-PC 单次正式训练：优先1张 A6000；若24GB足够，也应在 RTX3090 做一次
  可运行性验证；
- Seq2Point 单组训练：1张 RTX3090、8 CPU、32GB内存；
- B3/B4/B5 第一版 CVAE：先用1张 RTX3090，显存不足再切换 A6000；
- 不要用多 GPU 作为默认方案；只有单 GPU 基线稳定后再考虑分布式训练；
- 多 seed 通过 Slurm 独立作业并行，不在一个进程里争抢多张 GPU。

### 6.5 登录、远程开发与安全边界

指南记录的优先入口是 JumpServer：公网门户 `http://pub.lab.networkctl.cn:2000/`，
校内入口 `http://lab.networkctl.cn:2000/`。SSH 使用2222端口和组合用户名：

```text
ssh -p 2222 "<jumpserver_user>@<system_user>@172.28.255.242@lab.networkctl.cn"
```

其中 `172.28.255.242` 是指南记录的 `slogin` 资产地址。实际用户名、资产授权和地址
必须从当前控制台读取，不把手册示例原样当成用户凭据。

安全与使用规则：

- 只使用实验室批准的 JumpServer/VPN 路径；
- 登录节点 `slogin` 只用于编辑、传输、提交和查看作业，不运行长时间训练；
- 计算必须通过 `srun` 或 `sbatch` 获取资源；
- 禁止 `reboot`、`shutdown` 和 `--exclusive`；
- 禁止利用集群网络搭建未授权代理或绕过网络策略；
- VS Code 远程开发受支持；指南明确表示不支持 JetBrains Gateway 类远程开发；
- 端口转发只用于当前获批计算会话，不直接暴露计算节点端口。

### 6.6 存储与传输策略

指南说明用户 home 位于 `/home/<user>`，在所有计算节点可见并持久保存。集群有共享
存储，但配额、可写目录和清理策略应在登录后确认。

建议服务器目录：

```text
/home/<user>/projects/PSLG-NILM/       Git 工作树
/home/<user>/pslg_data/                不进 Git 的输入数据
/home/<user>/pslg_artifacts/<run_id>/  模型、预测与指标
/home/<user>/pslg_logs/                Slurm 日志
```

数据传输分两阶段：

1. **开发包**：只包含 train、validation、配置、索引和 SHA-256；
2. **最终 test 包**：协议冻结后再单独传输或解锁。

当前 `build_server_transfer_manifest.py` 生成的包仍包含 test shard。它可用于最终服务器
环境，但在开发阶段应新增 `--exclude-test` 或构建独立 dev manifest，减少误访问 test
的风险。完成该改造前，不得把“代码没有主动评价 test”等同于“test 在服务器上物理
隔离”。

约278 MiB的候选包可以通过 JumpServer SFTP，但手册建议更大的数据使用实验室认可的
HTTP/对象存储中转，再在集群用 `curl`/`wget` 获取，避免网页 SFTP 波动。任何方式都
必须在服务器复算 SHA-256；密码、令牌和对象存储密钥不得写入 Git、Slurm 脚本或日志。

### 6.7 module 与环境规则

指南编写时可见的模块包括 Miniconda、CUDA 12.1.1/12.9.1 和多版本 GCC，但模块名与
版本可能变化。作业开头必须：

```bash
module purge
module avail
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
module list
```

以上版本只是指南示例；如果 `module avail` 显示不同名称，应使用现场真实名称。手册
部分示例写成 `cuda/12.1`，而模块列表写成 `cuda-toolkit/12.1.1`，不能盲目复制。

项目环境必须额外冻结：Python、NumPy（当前要求 `<2`）、深度学习框架、CUDA、
cuDNN、GPU 驱动和关键依赖。建议建立独立 `environment_server.yml`，不要直接改坏
本机 `.venv`，也不要在每个作业中临时无版本安装依赖。

---

## 7. 后续分阶段执行计划

## Phase A：本机状态粒度与真实基元可交换性冻结

### 目标

在训练生成模型前，确认状态应分成几类，以及真实基元在什么条件下可以安全互换。

### A1. 生成 k=4 和 k=5 pilot 状态库

任务：

- 复用相同 train cycle、相同 segmentation 和 physical statistics；
- 分别生成 k=4、k=5 状态标签、state merge、inventory、transition graph；
- 不使用 validation/test 拟合聚类；
- 保存独立目录，例如 `state_library_pilot_k4_v1/` 和 `...k5_v1/`；
- 检查每个状态的样本数、覆盖 cycle、功率、持续时间和能量；
- 重点判断 k=4/5 是否把宽泛的状态1分成有意义的洗涤、漂洗或脱水类状态。

验收：

- 每个库都为 train-only；
- 每个 cycle 样本覆盖精确；
- 无空状态、无极小到不可训练的状态；
- 输出可人工查看的典型/边缘波形；
- 形成 k=3/4/5 对照报告。

### A2. 同状态重复性与可交换性分析

建议实现：`scripts/evaluate_state_exchangeability.py`。

每个状态计算：

- 时长、均值、峰值、能量分布；
- 起点/终点功率和斜率分布；
- 同状态内距离、不同状态间距离；
- 最近邻来自同 cycle 的比例；
- 不同 cycle 间替换后的边界跳变；
- 状态在前驱/后继上下文中的条件分布；
- 长状态内部是否仍有明显多模态。

验收：形成 `reports/state_quality/k3_k4_k5_exchangeability.md`，明确哪些状态可直接
交换、哪些状态必须带上下文条件、哪些状态需要继续细分。

### A3. B2-matched 规则消融

固定 seed=17、ratio=0.5、相同模板和背景，只在 validation 上依次改变一个因素：

1. duration only；
2. duration + mean power；
3. duration + start/end power；
4. duration + mean + start/end；
5. top-k ∈ {1, 3, 5, 10}；
6. 是否限制长度比例，例如 `[0.75, 1.33]`、`[0.67, 1.5]`、`[0.5, 2]`；
7. k ∈ {3, 4, 5}；
8. 是否加入前驱/后继状态条件。

不要一次同时改变多项，否则无法知道哪个因素有效。

输出：

- 每个方案的数据质量统计；
- 相同 CPU smoke 指标；
- 运行时间和有效 donor 覆盖率；
- 统一排序表。

### A4. 边界处理最小消融

比较：

- 不处理；
- 线性 cross-fade；
- 端点偏移校正；
- 短窗幅值/斜率匹配。

边界处理不能改变 cycle 总长度，不得引入负功率，并应记录能量变化。建议先测试
6–30个采样点的短窗口，不要用过长平滑掩盖真实状态。

### A5. Phase A 决策门槛

使用 validation-only CPU smoke 选择最多2个候选进入服务器：

- 相比 B2-random 必须稳定改善；
- 优先选择接近或超过 B1 的候选；
- 不能靠明显改变总能量获得虚假收益；
- 状态样本量必须足够训练生成器；
- 规则必须可解释、可复现、计算成本合理。

如果所有 k 和约束都明显弱于 B1，也不要停止整个研究。应记录负结果，并把核心问题
调整为“什么条件下基元可交换”以及“生成方法是否在低数据量下更有价值”。

预计：3–7个本机工作日。

---

## Phase B：本机完成正式训练代码和实验基础设施

### 目标

在上服务器前，保证正式模型代码、配置、恢复训练、日志和指标都能用小数据跑通。

### B1. 正式 Seq2Point 实现

计划增加：

```text
src/nilm/seq2point.py
src/nilm/trainer.py
src/nilm/checkpoint.py
scripts/train_nilm.py
scripts/predict_nilm.py
```

必须支持：

- 读取现有 `ShardedWindowDataset`；
- center-point 回归；
- 固定随机种子；
- B0/B1/B2/B2-matched 相同采样预算；
- active/inactive 平衡采样；
- validation MAE 早停；
- 断点恢复；
- 保存最佳模型、最后模型、训练曲线、配置和环境信息；
- 推理输出与 `evaluate_nilm_predictions.py` 对接；
- test 模式需要显式冻结标记，防止误运行。

本机只运行数百到数千 batch 的冒烟，不追求最终性能。

### B2. 生成数据统一 schema

计划实现：

```text
src/generation/schema.py
src/generation/provenance.py
src/generation/base_generator.py
```

每个生成样本至少保存：

- `synthetic_cycle_id`；
- 生成路线 B3/B4/B5；
- 随机种子；
- 模型 checkpoint 哈希；
- 条件变量；
- 状态序列；
- 每个状态目标/实际时长；
- 边界处理方式；
- 生成时间；
- 波形 SHA-256；
- 是否通过物理和复制检查。

### B3. 统一实验配置和注册表

建议增加：

```text
config/experiments/b0_b5_protocol_v2.yaml
config/generation/full_cycle_cvae.yaml
config/generation/primitive_cvae.yaml
config/composition/b2_matched.yaml
config/composition/b5_constrained.yaml
reports/experiment_registry.csv
```

注册表每行记录：run id、组别、状态库版本、seed、ratio、Git commit、配置哈希、
训练状态、最佳 epoch、validation 结果、test 是否访问。

### B4. 结果统计代码

计划增加：

- 按 cycle 计算 MAE、RMSE、SAE、precision、recall、F1；
- paired bootstrap 95% CI；
- 多 seed mean/std/CI；
- 每组训练耗时、峰值显存、模型参数量和生成速度；
- 自动检查不同组是否使用相同 validation/test 索引。

建议主指标：MAE；关键辅助指标：F1、SAE。RMSE、precision、recall 用于解释。

### B5. Phase B 验收

- 小数据端到端训练、保存、恢复、推理、评价成功；
- CPU 或小 GPU 冒烟结果可重复；
- 所有新模块有单元测试；
- test 默认锁定；
- 失败运行不会覆盖成功产物；
- server transfer manifest 可完整生成和校验。

预计：4–8个本机工作日。

---

## Phase C：服务器正式状态发现与 B0/B1/B2 验证

### 目标

用正式特征和正式 Seq2Point 验证“真实基元重组”的上限，再决定生成路线的状态定义。

### C1. 服务器环境冻结

#### C1.1 登录后只读检查

在提交任何任务前记录：

```bash
hostname
date
sinfo
squeue -u "$USER"
module avail
module list
```

进入一次短时交互资源后再记录：

```bash
nvidia-smi
python --version
git rev-parse HEAD
git status --short
```

不要在 `slogin` 直接运行 GPU 训练或长时间数据处理。

#### C1.2 环境与数据冻结

记录并保存：

- OS、Python、PyTorch/TensorFlow、CUDA、cuDNN和驱动；
- GPU 型号、显存和实际分配节点；
- Git commit和工作树是否干净；
- `conda env export` 与 `pip freeze`；
- module 列表；
- 数据 manifest 文件数、总字节数和逐文件 SHA-256；
- train/validation/test 是否物理分离；
- Slurm job id、提交时间、开始时间、结束时间和退出码。

#### C1.3 启动门槛

顺序必须是：

1. 全套单元测试；
2. CPU 数据读取冒烟；
3. 单 batch GPU 前向/反向；
4. 小样本短 epoch；
5. checkpoint 保存与恢复；
6. validation 推理；
7. 才启动正式长任务。

任一步失败都不得直接提交完整网格。

#### C1.4 推荐 Slurm 作业骨架

```bash
#!/bin/bash
#SBATCH -J pslg-s2p-b2m-k4-s17
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH -o /home/<user>/pslg_logs/%x-%j.out
#SBATCH -e /home/<user>/pslg_logs/%x-%j.err

set -euo pipefail

echo "started=$(date --iso-8601=seconds)"
echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

module purge
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
module list

source "$(conda info --base)/bin/activate"
conda activate pslg-nilm

cd /home/<user>/projects/PSLG-NILM
git rev-parse HEAD
nvidia-smi

# train_nilm.py 尚属计划脚本；实现并本机冒烟通过后再启用。
python scripts/train_nilm.py --config config/experiments/<frozen-config>.yaml

echo "finished=$(date --iso-8601=seconds)"
```

`<user>` 和配置名必须替换为当前真实值；模块版本以现场 `module avail` 为准。禁止把
密码、访问令牌或私钥写入脚本。

### C2. DETSEC-PC 状态发现

- 输入仅为493个 train cycle；
- 运行正式 `detsec_pc` 特征；
- 比较 k=3/4/5；
- 与 physical-stats pilot 比较状态稳定性、可解释性和可交换性；
- validation 只用于选择最终状态粒度，不重建 train 聚类；
- 输出正式 state library v1 和完整 provenance。

### C3. 正式 B0/B1/B2 第一轮

第一轮使用：

- seeds：17、42、73；
- synthetic ratio：0.5；
- 固定 Seq2Point 架构；
- 每组相同 steps per epoch、最大 epoch、patience；
- validation 选 checkpoint；
- 暂不访问 test。

组别至少包含 B0、B1、B2-random、最佳 B2-matched。

### C4. 比例实验

在第一轮状态和规则冻结后，再跑 ratio：

```text
1:0.5
1:1
1:2
```

如果 1:2 明显因合成数据过多而退化，应保留结果，而不是修改采样使其消失。

### C5. Phase C 决策

- 如果 B2-matched ≥ B1：说明真实基元结构和约束具有直接价值，可进入生成阶段；
- 如果 B2-matched < B1 但显著优于 B2-random：约束有效，生成路线仍可继续，但论文
  不能声称真实基元重组已胜过完整周期；
- 如果 B2-matched 与 B2-random 接近：重新检查状态定义或转向低数据量场景；
- 如果 B0 始终最好：说明当前合成放置或采样机制可能造成域偏移，先解决域偏移再训练
  生成器。

预计：依服务器资源3–7天。

---

## Phase D：B3 普通完整周期生成

### 目标

建立一个公平、足够强但成本可控的直接整段生成基线。

### D1. 第一版模型建议

优先使用 conditional VAE，原因是实现稳定、训练成本低、条件控制直接。不要一开始
同时开发 GAN、Diffusion 和 Transformer。只有在 CVAE 明显失败且问题明确时，才
增加第二种模型。

可能的条件：

- cycle 长度或长度桶；
- 总能量桶；
- 峰值功率桶；
- 可选运行模式代理变量。

变长序列可先采用长度桶、mask 和插值到标准长度，采样后恢复目标长度。但任何变换
必须同样接受真实性和能量检查。

### D2. 训练与选择

- 只使用493个 train cycle；
- validation 只用于生成质量和下游效果选 checkpoint；
- 模型规模、训练时长、调参次数需记录；
- 生成数量、时长分布和放置时间表与 B4/B5 对齐；
- 生成 cycle 先通过物理和复制检查，再进入 NILM。

### D3. 必须的生成质量检查

- 非负功率、最大功率范围、总能量、持续时间；
- 起止功率是否合理；
- 自相关、频谱、功率分布；
- 与最近 train cycle 的距离；
- 完全相同片段或高相似复制检测；
- 多样性和 mode collapse；
- 合法放入背景后的总表恒等关系。

### D4. 输出

```text
checkpoints/full_cycle_generator/<run_id>/
reports/generation_quality/B3_<run_id>/
reports/synthetic_cycles/B3_<run_id>/
```

预计：服务器3–7天，代码准备在本机完成。

---

## Phase E：B4 基元状态生成

### 目标

对冻结状态库中的每个状态分别建模，生成新的状态波形，再按真实模板路径做基础拼接。

### E1. 训练单元

第一版建议共享编码器并以 `state_label` 为条件，避免为每个状态训练完全独立的大模型
导致预算不公平。也可训练每状态小模型，但必须报告总参数量和总训练成本。

条件建议包括：

- state label；
- 目标持续时间或长度桶；
- 能量/均值功率条件；
- 可选前驱和后继状态。

### E2. B4 基础拼接定义

- 状态路径从 train 的真实路径分布中采样，或复用与 B3 相同模板条件；
- 生成每个状态；
- 只做最基本的长度恢复和连接；
- 不加入完整 B5 边界优化；
- 保留足够差异，使 B5 vs B4 能回答“约束是否有效”。

### E3. 公平性

B3 和 B4 必须：

- 使用相同 train cycle；
- 生成相同数量和近似总时长；
- 使用相同背景和放置时间表；
- 使用相同 NILM 训练预算；
- 尽量匹配生成模型总参数或总 GPU 时间；
- 使用相同 validation 选择次数。

预计：服务器4–10天。

---

## Phase F：B5 受约束生成与消融

### 目标

将 B2-matched 得到的经验正式用于“生成基元拼接”，并逐项证明每种约束的价值。

### F1. 状态顺序约束

- 从 train transition graph 学习合法转移；
- 禁止 train 中完全未出现且缺少物理依据的跳转；
- 对稀有但合法的路径使用平滑概率，避免完全消失；
- 保存每条合成路径的概率和来源统计。

### F2. 持续时间约束

- 按状态学习经验分布或参数分布；
- 可按前驱/后继条件建模；
- 限制极端缩放；
- 报告生成时长分布与真实 train/validation 的差异。

### F3. 边界约束

- donor/生成状态选择时匹配端点功率和斜率；
- 使用短窗 cross-fade 或可学习 transition；
- 控制边界跳变、能量改变和局部频谱异常；
- 任何平滑不得泄漏 validation/test 波形。

### F4. 上下文约束

可选条件：

- 前驱和后继状态；
- cycle 总能量/总时长；
- 背景负载区间；
- 温度、时间等外部条件仅在数据完整且公平时加入。

### F5. 逐级消融

建议服务器正式组：

```text
B4                  generated primitive + basic composition
B4 + transition     只加顺序
B4 + duration       顺序 + 时长
B4 + boundary       顺序 + 时长 + 边界
B5                  再加最终上下文/困难样本策略
```

一次只增加一个约束，才能判断贡献来源。

预计：服务器5–12天。

---

## Phase G：核心 NILM 公平比较与最终 test

### G1. validation 阶段

完成全部 B0–B5、seed 和 ratio 的 validation。预先冻结：

- 最终状态库版本；
- 最终 B2/B5 规则；
- B3/B4 checkpoint 选择规则；
- NILM checkpoint 选择规则；
- 阈值20 W是否保持；
- 主要/辅助指标；
- 异常运行和失败重跑规则；
- 统计检验方法。

形成 `protocol_freeze_before_test.md` 并记录 Git commit 与配置哈希。

### G2. 建议预注册的主要成功标准

在查看 test 前正式写定。建议：

- 主要比较：B5 vs B3 的真实 test cycle-level MAE；
- 关键辅助比较：F1 和 SAE；
- 使用相同 test cycle 做 paired bootstrap 95% CI；
- 报告3个 seed 的 mean/std/CI；
- 若 MAE 改善但 F1 明显恶化，不视为无条件成功；
- 同时报告 GPU 时间、模型参数量、生成速度和存储成本。

具体最小改善阈值必须在 test 前由研究者确认。可参考“MAE 相对改善至少5%，或
F1 绝对提高至少0.02且另一主指标不明显退化”，但这只是建议，不是当前已冻结标准。

### G3. 最终 test

只在方案冻结后执行一次：

1. 校验 test exclusions；
2. 对所有冻结组生成预测；
3. 使用统一评价脚本；
4. 按 cycle 和整体报告；
5. 保存原始预测、指标、bootstrap 样本和报告；
6. 不再根据 test 修改方案；
7. 如发现代码错误，记录错误、修复和全部组统一重跑，不允许只重跑某一组。

预计：服务器2–5天。

---

## Phase H：效率、泛化与扩展实验

### H1. 数据效率

使用 train 的10%、25%、50%、100%，仍按完整 cycle 抽样，比较：

- B0、B3、B5；
- 达到相同 MAE/F1 所需的真实 cycle 数；
- 每增加一个真实 cycle 或合成 cycle 的边际收益。

这可能是基元方法最有优势的场景，即使全数据下 B5 只与 B3 接近。

### H2. 跨物理设备泛化

保持 instance 2 为外部设备测试域。不得把其数据用于第一轮模型选择。可以比较：

- instance 1 训练 → instance 2 测试；
- 少量 instance 2 适配；
- B3 与 B5 哪种增强更有利于跨设备。

### H3. 其他电器/住宅

优先顺序建议：

1. kettle：简单单状态控制组；
2. fridge：周期型中等复杂度；
3. 另一住宅 washing machine：跨住宅验证。

### H4. 自采集数据

自采集系统和未来 PCB 可作为：

- 新数据源接入演示；
- 不同地区/硬件/采样方式的外部验证；
- 状态分布变化和条件生成实验。

该扩展不应阻塞 UK-DALE 主线论文闭环。

---

## 8. 建议的近期执行顺序

### 接下来立即做（本机）

1. [x] 把 matched donor 评分函数从脚本内部提取到可单测模块；
2. [x] 增加 donor 匹配、长度约束和确定性测试；
3. [x] 生成 k=4、k=5 pilot 状态库；
4. [x] 输出每个状态的典型、边缘和异常波形（`scripts/evaluate_state_exchangeability.py`）；
5. [x] 运行 donor policy × top-k 的受控小网格（18 变体链，结果见 `b2_policy_ablation_seed17_r0p5/report.md`）；
6. [x] 边界消融：cross-fade 与端点偏移校正均为负结果，冻结为不处理；
7. [x] 使用同一 validation CPU smoke 选最多2个候选（k4+duration+ratio、k4+duration）；
8. [x] 写 `docs/state_and_b2_policy_freeze_v1.md`；
9. [x] 实现 Seq2Point 训练器并完成小数据冒烟（训练/恢复/推理/评价/test 锁定）；
10. [x] 更新 matched 服务器传输 manifest（冻结候选的 dev bundle，263MB）；
11. [x] 为传输脚本增加 `--exclude-test`，生成 train+validation 开发包；
12. [x] 增加服务器端 manifest 校验脚本 `scripts/verify_server_manifest.py`；
13. [x] 增加并冻结 `environment_server.yml`；
14. [x] 为 DETSEC-PC、Seq2Point 和生成器分别增加 Slurm 模板（`slurm/*.sbatch`）。

### 随后做（服务器）

1. 校验传输文件和环境；
2. 用 train+validation 开发包完成测试，保持最终 test 物理隔离；
3. 运行 train-only DETSEC-PC；
4. 比较 physical-stats 与 DETSEC-PC 的 k=3/4/5；
5. 冻结正式 state library；
6. 跑 B0/B1/B2 候选的3 seed、ratio=0.5 Seq2Point validation；
7. 冻结真实基元拼接策略；
8. 扩展 ratio=1和2；
9. 开始 B3/B4 生成器；
10. 完成 B5 约束消融；
11. 冻结全部协议并形成签名记录；
12. 最后传输/解锁 test，并只运行一次最终评价。

---

## 9. 计划中的文件与目录

建议逐步增加，而不是一次创建空目录：

```text
src/
├─ generation/
│  ├─ schema.py
│  ├─ provenance.py
│  ├─ base_generator.py
│  ├─ full_cycle_cvae.py
│  └─ primitive_cvae.py
├─ composition/
│  ├─ donor_matching.py
│  ├─ transition_model.py
│  ├─ duration_model.py
│  ├─ boundary_handler.py
│  └─ constrained_composer.py
├─ nilm/
│  ├─ window_dataset.py          # 已有
│  ├─ metrics.py                 # 已有
│  ├─ seq2point.py               # 计划
│  ├─ trainer.py                 # 计划
│  └─ checkpoint.py              # 计划
└─ validation/
   ├─ state_exchangeability.py
   ├─ synthetic_quality.py
   ├─ memorization.py
   ├─ statistics.py
   └─ fairness_audit.py

scripts/
├─ evaluate_state_exchangeability.py
├─ run_b2_ablation.py
├─ train_nilm.py
├─ predict_nilm.py
├─ train_full_cycle_generator.py
├─ train_primitive_generator.py
├─ compose_generated_cycles.py
├─ evaluate_synthetic_quality.py
└─ freeze_protocol_before_test.py

config/
├─ generation/
├─ composition/
└─ experiments/

reports/
├─ state_quality/
├─ generation_quality/
├─ downstream/
├─ efficiency/
└─ paper_tables/
```

---

## 10. 指标与报告规范

### 10.1 下游 NILM 指标

- MAE：平均每个时间点偏差多少瓦，主要指标候选；
- RMSE：对大误差更敏感；
- SAE：总能量估计偏差；
- precision：预测为开机时有多少是真的；
- recall：真实开机时有多少被发现；
- F1：precision 和 recall 的综合；
- TP/FP/FN：用于解释 F1，尤其关注大量误报。

### 10.2 生成质量指标

- 功率、能量、时长分布；
- 起点、终点、边界跳变和斜率；
- 状态顺序合法率；
- 自相关和频域特征；
- train 最近邻距离与复制率；
- 状态覆盖、多样性和 mode collapse；
- 放入背景后的功率恒等关系。

### 10.3 效率指标

- 训练 GPU 小时；
- 峰值显存；
- 参数量；
- 每1,000个 cycle 生成耗时；
- 每个合成 cycle 存储量；
- 每单位真实 cycle 带来的下游性能增益。

### 10.4 报告最小内容

每次正式运行必须报告：

- run id；
- Git commit；
- 配置文件和哈希；
- 输入 manifest 和哈希；
- seed；
- 数据比例；
- 硬件与运行时间；
- 最佳 epoch 和选择指标；
- validation/test 是否访问；
- 完整指标；
- 失败或警告；
- 输出路径。

---

## 11. 风险、判断方式与备选路线

| 风险 | 当前证据 | 应对 |
|---|---|---|
| k=3 状态过粗 | 状态1中位时长4,548秒且内部复杂 | 比较k=4/5，分析多模态和上下文 |
| 随机基元严重失真 | 原始B2有183个极端缩放 | 时长/功率/端点匹配，限制缩放 |
| 基元方法仍弱于完整周期 | B2-matched仍弱于B1 | 加顺序/时长/边界；检查低数据量优势 |
| CPU smoke 与正式深度模型结论不同 | 当前模型只是树模型 | 只用于筛选，正式结论来自Seq2Point |
| 生成器复制训练数据 | cycle数仅493 | 最近邻、重复片段、membership式审计 |
| 合成数据造成域偏移 | B0当前优于B1/B2 | 检查背景放置、采样比例和校准 |
| 模型预算不公平 | B4可能多个子模型 | 比较总参数、GPU时间和调参次数 |
| test 被过早使用 | 研究周期长、方案多 | 代码锁、freeze文档、访问日志 |
| 依赖环境脆弱 | nilmtk、numpy、GPU框架复杂 | 冻结环境，不随意升级核心依赖 |
| 历史上游表征可能泄漏 | `server.html` 对旧 DETSEC/聚类标为 holdout 未认证 | 正式 DETSEC-PC 只 fit train，validation/test 只 transform，并记录输入哈希 |
| 总表物理语义不明确 | 历史审计质疑旧 channel 是否为有功功率 | 核实 UK-DALE meter key、字段、单位和元数据；代数恒等式不能替代物理语义核验 |
| 设备换机混淆结论 | 2015-09 前后为不同物理洗衣机 | 主实验只用 instance 1；instance 2 单独作为跨设备实验 |
| 基元截断或覆盖错误 | 旧配置存在 `max_seg_len=1536` 风险 | 正式 run 审计每个 cycle 起止、缺口、长度和重建覆盖，禁止只看总长度 |
| 连续背景误报 | cycle 数据开机比例高，可能掩盖 OFF 场景错误 | 所有组共享背景，报告 FPR、FP、OFF 能量误差和事件指标 |
| 服务器结果错绑版本 | 同名 metrics 可能来自不同协议 | 结果绑定 commit、config hash、data hash、job id、环境和 run id |
| 登录节点被误用 | 交互方便，容易直接启动重任务 | slogin 只编辑/传输/提交，训练必须经 srun/sbatch |
| 模块名称漂移 | 手册示例与模块列表已有名称差异 | 每次先 `module avail`，保存 `module list`，不硬编码未经验证的别名 |
| 工作范围过大 | 多数据集、多电器、硬件并行 | 先完成washing machine B0–B5闭环 |

如果 B5 最终没有超过 B3，研究仍可形成有价值结论：

- 说明基元生成在哪些约束下才不退化；
- 量化随机拼接和极端缩放造成的损害；
- 展示低数据量、少样本状态或跨设备条件下是否存在优势；
- 给出明确的失败边界，而不是只报告正结果。

---

## 12. 阶段完成定义

### 本机准备完成

- [x] k=3/4/5 第一轮状态粒度与下游 smoke 报告完成；
- [x] 最佳 B2 规则在 validation 上冻结（duration-only + 限长 [0.67,1.5]，k=4 首选）；
- [x] Seq2Point 小数据训练/恢复/推理跑通（192,225 参数，test 默认锁定）；
- [x] 生成 schema 和 provenance 测试通过；
- [x] 服务器包 SHA-256 校验清单完成（dev bundle 73 文件校验通过）；
- [x] 全套单元测试通过（106 项；已为 `tests/` 增加 `__init__.py` 包定义并让
  `test_dpc_kmeans` 的跨模块导入兼容两种发现模式，消除第三方 `tests` 包遮蔽
  导致的导入错误——修复前该环境仅 97 项可运行）。

### 服务器 B0–B2 完成

- [ ] DETSEC-PC 状态库冻结；
- [ ] 3 seeds × 3 ratios 的 B0/B1/B2 计划执行；
- [ ] validation 报告和置信区间完成；
- [ ] 未访问最终 test；
- [ ] 真实基元策略冻结。

### 生成研究完成

- [ ] B3 和 B4 使用公平预算训练；
- [ ] B5 各约束有逐项消融；
- [ ] 生成质量与复制检查通过；
- [ ] B0–B5 使用统一 NILM 训练与评价；
- [ ] protocol freeze 文档签定。

### 核心论文闭环完成

- [ ] test 只在冻结后运行；
- [ ] 报告所有组、所有 seed 和不利结果；
- [ ] 完成 B5 vs B3、B4 vs B3、B5 vs B4、B2 vs B1；
- [ ] 完成效率统计；
- [ ] 代码、配置、manifest、日志和表图可复现。

---

## 13. 重要现有路径

```text
docs/core_validation_b0_b5_execution.md
docs/nilm_state_composition_execution_plan.md
docs/nilm_research_progress_and_plan.md

reports/core_validation/ukdale_b1_washing_machine/
├─ cycle_inventory.csv
├─ cycle_inventory_summary.json
├─ cycle_inventory_with_split.csv
├─ partition_manifest.json
├─ leakage_report.json
├─ aligned_partitions_v2/
├─ real_cycle_library_train_v1/
├─ state_library_pilot_k3_v1/
├─ b1_b2_pilot_seed17_r0p5/
├─ b1_b2_placed_seed17_r0p5/
├─ b1_b2matched_pilot_seed17_r0p5/
├─ b1_b2matched_placed_seed17_r0p5/
├─ nilm_inputs_seed17_r0p5_v1/
├─ nilm_inputs_b2matched_seed17_r0p5_v1/
├─ cpu_nilm_smoke_seed17_r0p5/
├─ cpu_nilm_smoke_b2matched_seed17_r0p5/
└─ b2_constraint_diagnostic.md
```

配置：

```text
config/experiments/nilm_b0_b2_pilot.yaml
```

正式数据只认 `aligned_partitions_v2`。如果文档、旧脚本或缓存指向 v1，应立即停止并
修正路径。

---

## 14. 给后续 AI 的工作协议

接手 AI 每次开始工作前应：

1. 阅读本文件、`core_validation_b0_b5_execution.md` 和最新阶段报告；
2. 检查 Git 状态，保留用户已有修改；
3. 明确本次任务属于本机筛选、服务器正式训练还是最终 test；
4. 确认输入 partition 和 manifest；
5. 在修改前记录计划，修改后运行相关测试；
6. 新实验使用新目录，不覆盖冻结基线；
7. 用简单中文解释做了什么、为什么做、结果意味着什么；
8. 明确区分 pilot 结果、validation 结果和最终 test 结果；
9. 如果发现协议冲突，以“无泄漏、公平比较、test 最后一次使用”为最高优先级；
10. 未经研究者确认，不自行改变核心成功标准或扩大研究对象。

每次交付至少包含：

- 修改/生成了哪些文件；
- 使用了哪些输入；
- 是否访问 validation/test；
- 测试是否通过；
- 当前能得出什么结论；
- 不能得出什么结论；
- 下一步最小可执行动作。

### 14.1 Git 与本地产物规则

详细政策见 `docs/GIT_TRACKING_POLICY.md`。简要规则：

- Git 追踪源码、脚本、测试、配置和 Markdown 结论；
- `reports/` 只追踪 Markdown，不追踪 NPZ/NPY/CSV/JSON/PNG；
- 原始数据、模型权重、运行日志、缓存和服务器传输数据不进 Git；
- 不追踪不等于可以删除，重要实验数据必须用 manifest、SHA-256 和外部备份保存；
- 不得为了整理工作树而丢弃用户已有修改；
- `docs/server.html` 是旧快照，不是当前协议事实来源；
- `.git` 的历史大对象需要单独备份和授权后再清理。

---

## 15. 当前一句话结论

项目已经建立了可信的无泄漏数据基础，并证明“限制基元的错误匹配与极端时间缩放”
能够显著改善真实基元拼接；下一步应先在本机冻结状态粒度和基元约束，再在服务器用
正式 DETSEC-PC、Seq2Point 和 B3/B4/B5 生成模型回答最终问题：受约束的基元状态
生成与拼接，是否比普通完整周期生成具有更好的真实 NILM 效果或数据效率。

---

## 16. 当前可直接运行的命令附录

以下命令对应已经实现的脚本。未出现的 `train_nilm.py`、生成器训练脚本等仍是计划，
在实现前不要假设它们存在。

### 16.1 运行全套测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

当前基准：106 项通过。

### 16.2 重建 B2-matched 配对周期

```powershell
.\.venv\Scripts\python.exe scripts\build_b1_b2_pilot_cycles.py `
  --cycle-library-dir reports\core_validation\ukdale_b1_washing_machine\real_cycle_library_train_v1 `
  --state-library-dir reports\core_validation\ukdale_b1_washing_machine\state_library_pilot_k3_v1 `
  --output-dir reports\core_validation\ukdale_b1_washing_machine\b1_b2matched_pilot_seed17_r0p5 `
  --seed 17 --synthetic-ratio 0.5 --sample-seconds 6 `
  --b2-policy matched --matched-top-k 5
```

### 16.3 放入与原 B2 相同规则的真实背景

注意：冻结 pilot 的 idle threshold 是20 W，不是1 W。

```powershell
.\.venv\Scripts\python.exe scripts\place_b1_b2_on_train_background.py `
  --aligned-dir reports\core_validation\ukdale_b1_washing_machine\aligned_partitions_v2 `
  --paired-cycle-dir reports\core_validation\ukdale_b1_washing_machine\b1_b2matched_pilot_seed17_r0p5 `
  --output-dir reports\core_validation\ukdale_b1_washing_machine\b1_b2matched_placed_seed17_r0p5 `
  --seed 17 --sample-seconds 6 --idle-threshold-w 20 --guard-seconds 300
```

### 16.4 生成统一 NILM 窗口清单

```powershell
.\.venv\Scripts\python.exe scripts\prepare_nilm_b0_b2_inputs.py `
  --aligned-dir reports\core_validation\ukdale_b1_washing_machine\aligned_partitions_v2 `
  --placed-dir reports\core_validation\ukdale_b1_washing_machine\b1_b2matched_placed_seed17_r0p5 `
  --output-dir reports\core_validation\ukdale_b1_washing_machine\nilm_inputs_b2matched_seed17_r0p5_v1 `
  --window-length 599 --train-stride 6 --eval-stride 1 --sample-seconds 6
```

### 16.5 运行 CPU validation-only smoke

```powershell
.\.venv\Scripts\python.exe scripts\run_cpu_nilm_smoke.py `
  --experiment-dir reports\core_validation\ukdale_b1_washing_machine\nilm_inputs_b2matched_seed17_r0p5_v1 `
  --output-dir reports\core_validation\ukdale_b1_washing_machine\cpu_nilm_smoke_b2matched_seed17_r0p5 `
  --seed 17 --train-per-class 25000 --validation-count 50000
```

该命令不应读取 test。它只用于开发检查和 validation 候选筛选。

### 16.6 生成服务器传输清单

```powershell
.\.venv\Scripts\python.exe scripts\build_server_transfer_manifest.py `
  --experiment-dir reports\core_validation\ukdale_b1_washing_machine\nilm_inputs_b2matched_seed17_r0p5_v1 `
  --output reports\core_validation\ukdale_b1_washing_machine\nilm_inputs_b2matched_seed17_r0p5_v1\server_transfer_manifest.json
```

### 16.7 命令执行后的必查项目

每次重建后至少确认：

- `all_sources_train=true`；
- `all_b2_donors_cross_cycle=true`；
- paired cycle 数和每对持续时间一致；
- event overlap 为0；
- 新旧方案的模板和背景放置位置一致；
- validation/test 窗口对不同 arm 完全相同；
- `test_accessed=false`；
- transfer manifest 中所有 SHA-256 可复算一致。

---

## 17. 实验室服务器操作附录

> 来源：《人工智能实验室操作手册 v2.0.1》。本节是项目化整理，不取代实验室最新
> 管理规定。地址、节点、模块和资源随时可能调整，以 JumpServer 控制台和登录后的
> 实测为准。

### 17.1 第一次连接

优先通过 JumpServer 网页确认自己的资产授权。指南记录：

```text
公网门户：http://pub.lab.networkctl.cn:2000/
校内门户：http://lab.networkctl.cn:2000/
slogin 资产：172.28.255.242
SSH 端口：2222
```

SSH 格式：

```bash
ssh -p 2222 "<jumpserver_user>@<system_user>@172.28.255.242@lab.networkctl.cn"
```

首次登录可能要求修改密码。密码只在交互提示中输入，不能写入命令历史、配置、文档、
Git、Slurm 日志或聊天消息。

### 17.2 登录后的状态检查

```bash
sinfo
squeue -u "$USER"
module avail
df -h "$HOME"
quota -s 2>/dev/null || true
```

简单解释：先确认哪些 GPU 空闲、自己是否已有作业、模块是否变化，以及 home 是否有
足够空间。不要一登录就创建环境或上传全部数据。

### 17.3 交互式调试

RTX3090 示例：

```bash
srun -p RTX3090 --gres=gpu:1 -c 8 --mem=32G --pty /bin/bash
```

A6000 示例：

```bash
srun -p A6000 --gres=gpu:1 -c 8 --mem=32G --pty /bin/bash
```

进入计算节点后确认：

```bash
hostname
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi
```

交互式资源只用于环境安装、单元测试和短冒烟。退出 shell 即释放资源。不要在
`slogin` 上用普通 `python train.py` 启动训练。

### 17.4 Slurm 常用命令

| 命令 | 用途 | 本项目使用方式 |
|---|---|---|
| `sinfo` | 查看分区和节点 | 提交前检查 RTX3090/A6000 |
| `srun` | 获取交互资源 | 环境验证、单 batch 冒烟 |
| `sbatch` | 提交后台任务 | 正式 DETSEC-PC、Seq2Point、生成器 |
| `squeue -u $USER` | 查看自己的队列 | 判断 Running/Pending |
| `scontrol show job <id>` | 查看作业细节 | 排查资源、节点、退出状态 |
| `scancel <id>` | 取消自己的作业 | 配置错误或异常时停止 |

禁止用 `scancel -u $USER` 作为日常操作，因为它会取消自己全部任务。只有明确确认所有
任务都应停止时才使用。

### 17.5 环境建立建议

先在交互式计算节点执行：

```bash
module purge
module avail
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
module list
source "$(conda info --base)/bin/activate"
```

然后根据未来的 `environment_server.yml` 创建 `pslg-nilm` 环境。环境冻结前至少验证：

```bash
python --version
python -c "import numpy; print(numpy.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

如果正式实现使用 TensorFlow，则另外验证 TensorFlow 与 GPU，不要默认 PyTorch 成功就
代表 TensorFlow 可用。当前仓库要求 NumPy `<2`；未经完整测试不要升级。

### 17.6 代码与数据传输

代码建议通过 Git 获取，数据通过独立通道传输：

```text
代码：Git clone / pull
开发数据：train + validation bundle
最终测试：冻结协议后独立传输或解锁
模型与预测：artifact 目录，不回写 Git
```

当前候选包约278 MiB，可以使用 JumpServer SFTP。更大的数据按手册建议使用获批对象
存储或 HTTP 中转。传输完成后必须执行项目校验脚本；在该脚本实现前，至少对照 JSON
manifest 逐文件复算 SHA-256。

不要上传：

- 本机 `.venv`；
- `.git` 中的历史垃圾对象；
- `aligned_partitions_v1`；
- 完整31GB数据集目录，除非服务器阶段确实需要重新读取原始 HDF；
- 与正式候选无关的旧日志和缓存。

### 17.7 服务器 run 目录与命名

建议 run id：

```text
<stage>_<arm>_<state>_r<ratio>_s<seed>_<date>
```

例如：

```text
s2p_b2m_k4_r0p5_s17_20260920
detsecpc_k4_trainonly_s17_20260920
cvae_b3_fullcycle_s17_20260920
```

每个 run 独立保存：

```text
config.yaml
environment.txt
git_commit.txt
data_manifest.json
slurm_job.json 或 job id
stdout.log
stderr.log
history.csv
best checkpoint
predictions 或 prediction manifest
metrics.json
run_summary.md
```

不得让不同 seed、ratio 或状态库版本共用同一个可覆盖目录。

### 17.8 推荐作业资源矩阵

| 任务 | 首选分区 | GPU | CPU | 内存 | 首轮时限 | 说明 |
|---|---|---:|---:|---:|---:|---|
| 单元测试/数据读取 | 交互 CPU 或短 srun | 0 | 4 | 8–16GB | 30分钟 | 不占 GPU |
| GPU 单 batch 冒烟 | RTX3090 | 1 | 4 | 16GB | 30分钟 | 验证环境 |
| DETSEC-PC | A6000 | 1 | 8 | 32–64GB | 1–2天 | 显存优先，稳定后再网格 |
| Seq2Point 单组 | RTX3090 | 1 | 8 | 32GB | 1–2天 | seed 独立任务 |
| B3/B4 CVAE | RTX3090 | 1 | 8 | 32GB | 1–3天 | 显存不足转 A6000 |
| B5 采样/拼接 | RTX3090或CPU | 0–1 | 8 | 32GB | 1天 | 视生成器推理而定 |

这是保守起点，不是固定配额。根据第一轮日志中的显存、CPU、I/O和耗时再调整，避免
一开始过度申请资源。所有任务不得超过指南记录的7天上限。

### 17.9 端口转发与远程开发

指南支持 VS Code 远程开发。Jupyter 或可视化服务应运行在已分配的计算节点，再从
计算节点反向映射到 `slogin`，最后使用 VS Code 端口转发到本机。指南示例形式：

```bash
ssh -fNTR <slogin_port>:localhost:<compute_port> <user>@slogin
```

转发通常依赖当前 `srun --pty` 会话，退出会话后映射可能失效。端口号、主机别名和
许可方式必须按当前实验室规定设置；不要直接暴露 Jupyter 到公网，也不要把 token
写入共享日志。

### 17.10 作业失败处理

```bash
squeue -u "$USER"
scontrol show job <job_id>
tail -n 200 /home/<user>/pslg_logs/<job>.err
tail -n 200 /home/<user>/pslg_logs/<job>.out
```

按以下顺序判断：

1. Pending：查看 `Reason`，不要反复重复提交；
2. OOM：区分 GPU OOM 与系统内存 OOM，再调整 batch 或资源；
3. 模块/动态库错误：检查 `module list` 与环境导出；
4. 数据错误：先核对 manifest/hash，不重新生成 test；
5. 代码错误：本机修复并测试，通过 Git 同步；不要只在服务器热改而不回传；
6. 超时：从 checkpoint 恢复，不把失败 run 当成新的随机种子。

### 17.11 最终 test 解锁流程

在 test 包传入服务器或解除权限前，必须存在：

- `protocol_freeze_before_test.md`；
- 冻结 Git commit；
- 冻结配置哈希；
- B0–B5 的最佳 checkpoint 清单；
- 主要指标、辅助指标和统计方法；
- test exclusion 清单；
- 全部 validation 结果；
- 用户对最终 test 执行的明确确认。

执行 test 后只做统一评价和报告。若发现实现错误，需要记录原因并对所有受影响组统一
重跑；不得只重跑表现不佳的组，也不得继续根据 test 调参。

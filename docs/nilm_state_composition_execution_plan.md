# PSLG-NILM 基元状态生成与受约束拼接研究执行计划

> 文档版本：v1.0  
> 制定日期：2026-09-15  
> 适用仓库：PSLG-NILM-ADVANCED  
> 文档用途：作为后续代码开发、数据准备、实验执行、阶段验收和论文整理的统一工作依据。

---

## 1. 项目目标

### 1.1 当前代码基础

当前仓库已经具备以下完整处理链：

```text
extract
→ segment
→ feature
→ cluster
→ state_merge
→ fewshot
→ pam
→ split
```

现有能力主要用于：

1. 从长时间电器功率序列中提取完整运行周期；
2. 将完整运行周期切分为较小的时间基元；
3. 提取基元特征并进行无监督聚类；
4. 将过度切分的相邻基元合并为工作状态块；
5. 识别少样本状态，并建立基元与完整活动之间的映射；
6. 构造少样本/非少样本 knockout 评测数据；
7. 使用 RunManifest、内容寻址缓存和独立可视化管理实验产物。

当前仓库还没有完成：

- 正式论文使用的无泄漏 train/validation/test 划分；
- 可检索、可追溯的状态片段库；
- 整段工作周期生成器；
- 基元状态生成器；
- 随机、顺序约束和物理约束拼接器；
- 多电器 synthetic aggregate 生成；
- 真正的 NILM 训练、推理与下游评价；
- “基元生成拼接”和“普通生成”的统一公平评测。

### 1.2 最终研究目标

在现有状态发现能力之后建立完整闭环：

```text
真实 NILM 数据
→ 无泄漏数据划分
→ 工作周期提取
→ 基元切分与状态发现
→ 状态/转移片段库
→ 普通整段生成 或 基元状态生成拼接
→ 多电器叠加与精确标签生成
→ NILM 模型训练
→ 完全真实测试集评价
→ 精度、泛化、真实性和效率比较
```

最终要回答的核心问题是：

> 在相同真实数据、合成数量、生成模型规模、训练预算和下游 NILM 模型条件下，基元状态生成与受约束拼接是否比普通整段生成获得更好的真实测试性能、数据效率或计算效率？

### 1.3 研究结论的优先级

结论优先级从高到低如下：

1. **下游有效性**：是否提升完全真实、未见测试集上的 NILM 性能；
2. **数据效率**：是否能用更少真实周期达到相同或更高性能；
3. **跨场景泛化**：是否改善跨时间、跨住宅或跨设备实例表现；
4. **物理合理性**：生成状态、转移、持续时间和能量是否合理；
5. **生成真实性与多样性**：是否既接近真实分布，又没有复制训练数据；
6. **计算与存储效率**：是否能以合理成本生成和使用数据。

生成波形“看起来更像”不能单独作为方法有效的结论。

---

## 2. 研究问题与待验证假设

### RQ1：状态是否存在且稳定？

- H1：同一电器的同一工作状态在不同真实运行周期中具有可重复结构。
- 验证：类内/类间距离、跨周期状态识别、持续时间与功率统计、人工抽检。

### RQ2：状态片段是否可交换？

- H2：来自不同运行周期的同类状态片段在一定上下文条件下可以互换。
- 验证：替换后的边界、能量、转移合法性、分布变化和下游效果。

### RQ3：基元拼接是否优于普通生成？

- H3：在公平条件下，基元生成拼接比整段工作周期生成具有更好的真实 NILM 测试性能。
- 验证：统一数据、统一预算、统一下游模型的配对实验。

### RQ4：约束是否真正有价值？

- H4：状态顺序、持续时间、边界和上下文约束能显著改善随机拼接结果。
- 验证：逐项消融，而不是只比较最终完整方法。

### RQ5：基元方法是否更高效？

- H5：基元方法在低数据量、少样本状态或复杂多状态电器上具有更高的单位数据收益。
- 验证：低数据量曲线、每个真实周期的性能收益、每个合成样本的边际收益、生成耗时和 GPU/CPU 成本。

---

## 3. 必须遵守的实验原则

### 3.1 数据泄漏红线

1. 必须先按完整 cycle、连续时间块、session、设备实例或住宅划分数据，再建立状态库和训练生成模型。
2. 测试 cycle 的任何基元、聚类中心、归一化统计或边界信息不得进入训练流程。
3. 同一个真实 cycle 不得拆散后同时进入 train、validation 和 test。
4. 归一化参数、状态聚类中心、duration 分布、transition graph 和生成模型只能使用训练集拟合。
5. validation 只用于选模型和超参数；最终 test 在方案冻结前不得反复查看。
6. 每个合成样本必须保存完整 provenance，能够追踪到来源 partition、cycle、segment、状态和生成参数。

### 3.2 公平比较原则

基元方法与普通整段生成必须尽量控制以下变量：

- 相同的真实训练数据；
- 相同的训练/验证/测试划分；
- 相同的合成周期数量或合成总时长；
- 相同的真实与合成数据比例；
- 相同的条件信息；
- 相近的模型参数量或相同的计算预算；
- 相同的调参次数和 validation 选择规则；
- 相同的下游 NILM 模型、训练轮数和随机种子；
- 相同的真实测试集和评价代码；
- 相同的失败重跑规则。

不能使用一个经过充分调优的基元方法去比较一个明显较弱、未调优的整段生成器。

### 3.3 结果报告原则

1. 所有核心实验至少运行 3 个随机种子；正式结论建议使用 5 个随机种子。
2. 报告 mean、standard deviation 和 95% confidence interval。
3. 按 appliance 分项报告，不只给总体平均值。
4. 同时报告正向结果、无明显差异结果和失败案例。
5. 在查看最终测试结果前预先确定主要指标和成功标准。
6. 论文核心结论以真实测试集下游结果为准，生成质量指标作为解释证据。

---

## 4. 统一对照实验矩阵

| 编号 | 训练数据方案 | 目的 |
|---|---|---|
| E0 | Real only | 真实数据基础性能 |
| E1 | Real + noise/scale/shift/crop 等普通增强 | 判断简单增强是否已经足够 |
| E2 | Real + 直接整段周期生成 | 基元方法最重要的普通生成对照 |
| E3 | Real + 真实基元随机替换/拼接 | 分离“基元结构”与“生成模型”的贡献 |
| E4 | Real + 生成基元随机拼接 | 判断生成基元本身是否有效 |
| E5 | E4 + 合法状态顺序约束 | 判断 transition topology 的贡献 |
| E6 | E5 + 状态持续时间约束 | 判断 duration model 的贡献 |
| E7 | E6 + transition/boundary 约束 | 判断边界处理的贡献 |
| E8 | E7 + operating context/困难样本采样 | 完整候选方法 |

其中 E2、E3 和 E8 是第一轮必须完成的关键对照：

- E2 vs E8：回答基元方法是否优于普通整段生成；
- E3 vs E8：回答生成新基元是否比重组真实片段更有价值；
- E3 vs E5/E6/E7：回答各种约束是否真正必要；
- E0/E1 vs 其他组：回答复杂生成方案是否值得其额外成本。

---

## 5. 目标软件架构

### 5.1 建议增加的目录

```text
src/
├─ data/
│  ├─ canonical_schema.py
│  ├─ dataset_registry.py
│  ├─ research_split.py
│  └─ leakage_check.py
├─ state_library/
│  ├─ schema.py
│  ├─ builder.py
│  ├─ retrieval.py
│  └─ validator.py
├─ generation/
│  ├─ base_generator.py
│  ├─ full_cycle_generator.py
│  ├─ primitive_generator.py
│  └─ checkpoints.py
├─ composition/
│  ├─ random_composer.py
│  ├─ transition_graph.py
│  ├─ duration_model.py
│  ├─ boundary_handler.py
│  ├─ constrained_composer.py
│  └─ provenance.py
├─ aggregate_synthesis/
│  ├─ placement.py
│  ├─ mixer.py
│  └─ label_export.py
├─ nilm_baselines/
│  ├─ base.py
│  ├─ seq2point.py
│  └─ evaluation.py
└─ validation/
   ├─ state_quality.py
   ├─ synthetic_quality.py
   ├─ physical_validity.py
   ├─ downstream_utility.py
   ├─ efficiency.py
   └─ statistics.py

configs/
├─ data/
├─ generation/
├─ composition/
├─ nilm/
└─ experiments/

experiments/
├─ E0_real_only/
├─ E1_standard_aug/
├─ E2_full_cycle_generation/
├─ E3_real_primitive_composition/
├─ E4_generated_primitive_random/
├─ E5_transition_constraint/
├─ E6_duration_constraint/
├─ E7_boundary_constraint/
└─ E8_full_method/

reports/
├─ data_quality/
├─ state_quality/
├─ generation_quality/
├─ downstream/
├─ efficiency/
└─ paper_tables/
```

目录可在实现阶段根据现有 `src/steps/` 结构适当合并，但模块职责不能混在一个大脚本中。

### 5.2 建议增加的流水线步骤

```text
extract
→ research_split
→ segment
→ feature
→ cluster
→ state_merge
→ build_state_library
→ validate_states
→ train_generator
→ compose
→ synthesize_aggregate
→ train_nilm
→ evaluate
```

原有 `fewshot → pam → split` 路线保留，用于原项目的少样本/knockout 研究。

新步骤必须继续使用现有 Workflow 和 RunManifest，不建立第二套互不兼容的实验管理体系。

---

## 6. 分阶段工作计划总览

| 阶段 | 名称 | 建议工期 | 核心产出 | 阶段门槛 |
|---|---|---:|---|---|
| P0 | 范围、基线与协议冻结 | 3–5 天 | 基线清单、数据与指标协议 | 能复现当前流程，研究定义无歧义 |
| P1 | 正式数据层与无泄漏划分 | 1–2 周 | canonical schema、research split、泄漏检查 | 任一 cycle 只属于一个 partition |
| P2 | 状态库建设 | 1–2 周 | state library、transition library、provenance | 可从状态记录追溯到原始数据 |
| P3 | 状态有效性验证 | 1–2 周 | 重复性、可分性、可交换性报告 | 至少主电器存在可复用状态 |
| P4 | 真实基元拼接 MVP | 1–2 周 | E3、E5、E6、E7 初版 | 生成路径合法、结果可复现 |
| P5 | 普通整段与基元生成器 | 2–3 周 | E2、E4 生成器和检查点 | 两条生成路线可公平训练与采样 |
| P6 | 多电器聚合合成 | 1 周 | synthetic aggregate 和精确标签 | 叠加关系与标签数值完全一致 |
| P7 | NILM 下游闭环 | 1–2 周 | 统一训练、推理和指标 | E0 可稳定复现，所有组用同一评测 |
| P8 | 核心公平比较 | 2 周 | E0–E8 结果、置信区间、结论 | 能回答“是否优于普通生成” |
| P9 | 基元效率优化 | 2 周 | 数据/生成/计算效率曲线 | 找到至少一个可验证的效率优化点 |
| P10 | 鲁棒性与论文整理 | 2–3 周 | 跨电器/跨住宅、论文表图与复现包 | 第三方可按文档复现实验 |

完整路线预计 16–20 周。若以尽快形成投稿闭环为目标，应优先完成 P0–P8，并把 P9、P10 中的部分内容作为扩展实验。

---

## 7. 各阶段详细任务

## P0：范围、基线与协议冻结

### 目标

明确研究对象、比较方法、主要指标和算力范围，记录当前仓库可复现基线。

### 任务清单

- [ ] 保存当前 Git commit、Python 版本、依赖版本和硬件信息。
- [ ] 运行当前全部单元测试，并保存结果。
- [ ] 复现现有 UK-DALE washing machine 完整流水线。
- [ ] 固定第一轮研究电器：washing machine 为核心，kettle 为简单控制组，fridge 为周期型中间组。
- [ ] 明确定义“普通生成”：第一版为直接整段条件生成，而不是只指噪声增强。
- [ ] 第一版固定一个生成模型家族，建议先使用训练稳定、成本可控的 conditional VAE。
- [ ] 固定一个主要 NILM baseline；建议先使用 Seq2Point，后续可增加第二个结构验证结论稳健性。
- [ ] 预先确定主要指标：建议功率回归以 MAE 为主、状态识别以 F1 为主。
- [ ] 确定最小随机种子数、最大训练轮数和单组算力预算。
- [ ] 建立实验命名规则和配置模板。

### 产物

- `reports/baseline_environment.md`
- `reports/baseline_test_results.md`
- `configs/experiments/protocol_v1.yaml`
- 第一版数据集、电器、模型和指标选择记录。

### 验收条件

- 当前测试全部通过；
- 现有 UK-DALE 流程可复现；
- E0–E8 定义没有歧义；
- 所有后续实验共享同一份 protocol 配置；
- 在正式测试前已经写明主要成功标准。

### 阻断/转向条件

- 如果当前基线无法稳定复现，先修复基线，不进入新功能开发。

---

## P1：正式数据层与无泄漏划分

### 目标

建立适用于公开数据和自采集数据的统一格式，并从流程入口阻止数据泄漏。

### 任务清单

- [ ] 定义统一数据字段：timestamp、mains、appliance power、appliance_id、house_id、session_id、cycle_id、sampling_rate。
- [ ] 建立 dataset registry，统一登记 UK-DALE、REDD、REFIT、ECO 和自采集数据。
- [ ] 为每个数据集实现适配器，将数据转成 canonical schema。
- [ ] 新增 `research_split`，支持按连续时间、cycle、session、device instance 和 house 划分。
- [ ] 添加边界隔离区，避免同一活动跨越 train/test。
- [ ] 生成不可修改的 partition manifest，记录每个 cycle 所属集合。
- [ ] 新增 leakage checker，检查重复 cycle、时间重叠、来源片段越界和测试统计被训练使用。
- [ ] 让后续 feature、cluster、state library 和 generator 显式读取 partition。
- [ ] 保留原 DatasetSplitStep，但在文档和代码命名中标明其为 knockout split。

### 第一轮建议划分

1. 同住宅时间泛化：连续时间按约 60%/20%/20% 划分 train/validation/test；
2. 跨住宅泛化：在设备可对应的情况下，使用不同 house 作为 train/validation/test；
3. 活动跨边界时整段归入一侧或丢弃，不允许拆成两部分。

比例不是固定真理，可根据有效活动数量调整，但调整原因必须记录。

### 产物

- `partition_manifest.json`
- `data_quality_report.json`
- `leakage_report.json`
- 每个 partition 的 cycle 索引与时间范围。

### 必须增加的测试

- [ ] 同一个 `cycle_id` 不会跨 partition。
- [ ] train/validation/test 时间范围不重叠。
- [ ] 隔离区正确生效。
- [ ] test 片段无法进入状态库。
- [ ] 归一化参数仅由 train 计算。
- [ ] 固定随机种子时划分完全可复现。

### 验收条件

- 泄漏检查报告为 0 个违规项；
- 每个样本都能确定唯一 partition；
- 划分可由相同配置和种子复现；
- train、validation、test 的真实活动数量足以支持计划实验。

### 阻断/转向条件

- 若某电器的有效 cycle 太少，不进行复杂生成比较；先延长数据范围、增加住宅或改用其他电器。

---

## P2：状态库与转移库建设

### 目标

把当前 `state_merge` 结果变成训练集专属、可检索、可追溯的状态与转移片段库。

### 状态记录字段

- [ ] dataset、house_id、device_instance、appliance；
- [ ] session_id、cycle_id、segment_id、state_id；
- [ ] source_partition、source_file、原始起止位置；
- [ ] raw waveform、feature vector；
- [ ] duration、mean、std、RMS、peak、energy、slope；
- [ ] previous_state、next_state；
- [ ] transition_type 和可用的 transition waveform；
- [ ] mode、load level、环境等可获得 context；
- [ ] segmentation、feature、cluster、state_merge 的版本和配置哈希。

### 任务清单

- [ ] 定义稳定的 state library schema 和版本号。
- [ ] 从 `blocks.json`、`state_sequences.json` 和原始波形重建完整状态记录。
- [ ] 给状态 ID 增加作用域，避免不同实验的 cluster 0 被误认为相同状态。
- [ ] 建立按 appliance/state/context 查询的索引。
- [ ] 从真实 state sequence 统计合法 transition graph。
- [ ] 单独保存 transition segment，而不是只记录两个状态标签。
- [ ] 统计每个状态的 duration、amplitude 和 energy 分布。
- [ ] 计算重复片段和近重复片段，标记但暂不删除。
- [ ] 为每个库生成 summary 和可视化。

### 产物

- `state_library/metadata.parquet` 或等价结构化索引；
- `state_library/waveforms.npz` 或分片存储；
- `transition_library/`；
- `transition_graph.json`；
- `duration_statistics.json`；
- `state_library_summary.json`。

### 必须增加的测试

- [ ] 任意状态记录可追溯至唯一原始 cycle 和时间范围。
- [ ] 状态波形长度与 duration 一致。
- [ ] 所有库记录均来自 train。
- [ ] transition 的前后状态与原始序列一致。
- [ ] 保存和重新加载后数值不变。

### 验收条件

- 所有记录 provenance 完整；
- 测试分区污染数量为 0；
- 每个主状态拥有足够的跨 cycle 实例；
- 状态库构建过程可以通过 RunManifest 复现。

---

## P3：状态重复性、可分性和可交换性验证

### 目标

在训练生成器之前判断基元状态是否具有研究基础，并确定合适的状态粒度。

### 任务清单

- [ ] 计算每个状态的类内距离和状态之间的类间距离。
- [ ] 比较同状态跨 cycle、跨 session、跨 house 的稳定性。
- [ ] 使用简单分类器执行 train-cycle → unseen-cycle 状态识别。
- [ ] 分析状态持续时间、功率、能量和频域分布。
- [ ] 人工抽检一小批状态，记录其可能的物理含义或明显错误。
- [ ] 比较不同分段方法、聚类数和 state_merge 参数。
- [ ] 建立真实基元替换试验：保持真实状态路径，只替换一个或多个同类状态。
- [ ] 对替换点计算 ΔP、斜率、短窗能量和 transition feature 距离。
- [ ] 训练真实/替换边界判别器，检查是否存在明显 splice artifact。
- [ ] 形成状态粒度选择报告，不以单一 silhouette 指标决定最终粒度。

### 需要回答的问题

1. 哪些状态稳定且可以直接复用？
2. 哪些状态需要加入 mode、load level 等 context 才能稳定？
3. 哪些聚类只是相似波形分组，没有明确时序功能？
4. 当前是否过切分或过度合并？
5. 真实 transition 是否必须随状态一起保留？

### 产物

- `reports/state_quality/state_repeatability.*`
- `reports/state_quality/state_separability.*`
- `reports/state_quality/state_exchangeability.*`
- `reports/state_quality/granularity_selection.md`
- 推荐的 segmentation/feature/cluster/state_merge 配置。

### 阶段门槛

满足以下条件才进入生成阶段：

- 核心多状态电器中存在跨 cycle 可重复的状态；
- 同状态类内差异总体小于关键状态间差异；
- 替换实验没有普遍产生明显非物理边界；
- 能说明哪些状态可交换、哪些只能在特定 context 下交换。

### 阻断/转向条件

- 如果状态不稳定：增加 context 条件、调整切分粒度或使用更合适的表征。
- 如果 cluster label 缺乏时序意义：从“聚类状态”退回更细 primitive，或加入人工/规则辅助状态定义。
- 如果大部分片段不可交换：转向状态条件生成，而不是继续增加拼接规则。

---

## P4：真实基元拼接 MVP

### 目标

先不生成新波形，使用真实训练片段完成可追溯的 A1/A2/A3 拼接器，验证“状态结构和拼接”本身是否有价值。

### A1：同位置随机替换

- [ ] 选择一个训练集真实 cycle 作为状态路径模板。
- [ ] 对每个状态从其他训练 cycle 的同状态池中采样。
- [ ] 禁止使用原位置自身片段，防止直接复制。
- [ ] 保存来源 cycle、segment 和随机种子。

### A2：顺序约束拼接

- [ ] 从 transition graph 采样合法状态路径。
- [ ] 支持允许/禁止拓扑和概率转移两种模式。
- [ ] 对低样本转移增加平滑策略，但不得自动允许未观察到的物理路径。
- [ ] 输出非法转移检查结果。

### A3：物理约束拼接

- [ ] 加入状态 duration 分布。
- [ ] 加入 amplitude、energy 和 context 条件。
- [ ] 优先选用匹配前后状态的真实 transition segment。
- [ ] 实现无处理、offset alignment、短 cross-fade 等边界策略作为消融。
- [ ] 不能默认越平滑越真实；必须和真实 transition 分布比较。
- [ ] 对候选片段计算兼容性得分，并在高分候选中随机采样。

### 每个 synthetic cycle 必须保存

```text
synthetic_id
template_cycle
state_path
source_segments
source_cycles
transition_segments
duration_parameters
boundary_method
context
random_seed
generator_version
validation_flags
```

### 产物

- E3、E5、E6、E7 的真实基元拼接数据；
- `synthetic_manifest.jsonl`；
- 拼接前后可视化；
- 非法路径、duration 和边界质量报告。

### 必须增加的测试

- [ ] 固定 seed 时结果一致。
- [ ] 不会采样 validation/test 片段。
- [ ] 每个状态转移合法。
- [ ] duration 在允许范围内。
- [ ] provenance 字段完整。
- [ ] 拼接后时间索引连续。
- [ ] 合成周期总能量可由各片段精确求和复核。

### 验收条件

- 能稳定生成指定数量的合成周期；
- 非法状态路径率为 0；
- 所有样本均可追溯；
- E3、E5、E6、E7 之间只改变指定约束，满足消融要求。

---

## P5：普通整段生成器与基元状态生成器

### 目标

建立 E2 和 E4，使“整段生成”和“基元生成”可以在统一接口和公平预算下比较。

### 第一版模型建议

- 使用同一生成模型家族，例如 conditional VAE；
- 整段生成器输入/输出完整 cycle，并支持 padding mask 和 length condition；
- 基元生成器按 appliance/state/context 生成单个状态片段；
- 后续只有在第一版闭环稳定后，再增加 GAN 或 diffusion 作为第二模型家族。

### 任务清单

- [ ] 定义统一 `BaseGenerator`：fit、sample、save、load、metadata。
- [ ] 实现 FullCycleGenerator。
- [ ] 实现 PrimitiveGenerator。
- [ ] 两者共享尽可能一致的归一化、条件编码和训练记录格式。
- [ ] 记录参数量、训练时间、峰值显存/内存和采样速度。
- [ ] 支持变长序列和 padding mask。
- [ ] 生成数据反归一化后进行非负功率、能量和幅值检查。
- [ ] 建立 checkpoint 和 early stopping。
- [ ] 防止同一数据被错误重复计权。
- [ ] 对生成样本执行最近邻检查，识别训练样本复刻。

### 公平性检查表

- [ ] 训练周期来源相同；
- [ ] 合成周期数或合成总时长相同；
- [ ] 模型参数量或总训练预算相近；
- [ ] validation 选择规则相同；
- [ ] 超参数搜索次数相同；
- [ ] 相同随机种子集合；
- [ ] 相同的数据后处理规则；
- [ ] 相同的失败样本过滤规则。

### 产物

- E2 完整周期生成数据；
- E4 生成基元随机拼接数据；
- 模型 checkpoint；
- training history；
- 参数量、训练耗时、采样吞吐量报告；
- 真实/生成最近邻报告。

### 验收条件

- 两类生成器均可训练、保存、恢复和批量采样；
- 生成样本没有大规模 NaN、负功率或长度错误；
- 两条路线满足预先规定的公平预算；
- 任一实验均可由 config 和 manifest 完整复现。

### 阻断/转向条件

- 若完整周期生成器因序列过长无法稳定训练，先使用固定窗口/阶段条件生成，而不是直接判定基元方法获胜。
- 若基元生成质量明显差于真实基元重组，保留 E3 作为主要方法候选，并分析生成器瓶颈。

---

## P6：多电器 synthetic aggregate 生成

### 目标

将真实或合成的单电器周期在时间轴上放置并求和，生成带精确 appliance ground truth 的总负荷。

### 任务清单

- [ ] 定义参与叠加的 appliance 集合。
- [ ] 支持随机启动、经验使用时段和困难样本三种 placement 策略。
- [ ] 控制同时启动、功率接近设备重叠和多状态交叉。
- [ ] 支持加入真实背景负载或受控噪声。
- [ ] 输出 aggregate、每台设备支路、ON/OFF 标签和状态标签。
- [ ] 检查每个时间点 `aggregate = sum(appliances) + background`。
- [ ] 记录所有单电器来源和时间偏移。
- [ ] 统一采样率、时间戳和缺失值策略。

### 产物

- `synthetic_mains.npy`；
- `appliance_ground_truth/*.npy`；
- `state_ground_truth/*.npy`；
- `aggregate_manifest.json`；
- 重叠难度和设备占比摘要。

### 必须增加的测试

- [ ] 每个时间点的叠加误差小于预设数值容差。
- [ ] 各支路和总表时间戳严格一致。
- [ ] 状态标签与对应支路片段一致。
- [ ] 相同 seed 可复现放置结果。
- [ ] 所有来源仍然只来自 train。

### 验收条件

- 可以生成普通、重叠和困难三类 aggregate；
- 标签与功率数值完全对齐；
- 合成数据可直接被下游 NILM loader 读取。

---

## P7：NILM 下游训练和评价闭环

### 目标

建立真正用于评价数据增强价值的 NILM 训练、推理和指标系统。

### 任务清单

- [ ] 选择并实现/接入第一版 Seq2Point baseline。
- [ ] 建立统一的训练窗口、归一化和标签格式。
- [ ] 所有实验组使用完全相同的下游训练代码。
- [ ] 实现 TRTR、TSTR、Real+Synthetic→Real。
- [ ] 实现 10%、25%、50%、100% real-data curve。
- [ ] 实现同住宅时间测试和跨住宅测试。
- [ ] 实现 MAE、RMSE、SAE/energy error、F1、Precision、Recall 和事件指标。
- [ ] 按 appliance、house、状态和事件类型保存结果。
- [ ] 保存训练历史、最佳 checkpoint、推理结果和失败窗口。
- [ ] 增加第二个 NILM backbone 之前先完成第一版全部实验。

### 产物

- E0 的稳定真实数据基线；
- 每个实验组的 NILM checkpoint；
- `metrics_by_appliance.csv`；
- `metrics_by_seed.csv`；
- 预测与真值序列；
- 错误案例可视化。

### 验收条件

- E0 在相同种子下可复现；
- 所有实验组通过同一入口训练和评价；
- 测试集始终保持 100% 真实且未参与生成；
- 指标可以从保存的预测结果重新计算得到。

---

## P8：核心公平比较与统计分析

### 目标

完成 E0–E8，并直接回答“基元状态生成拼接是否优于普通生成”。

### 第一轮实验顺序

1. E0：Real only；
2. E1：普通增强；
3. E2：完整周期生成；
4. E3：真实基元拼接；
5. E4：生成基元随机拼接；
6. E5：加入 transition；
7. E6：加入 duration；
8. E7：加入 boundary/transition segment；
9. E8：完整方法。

### 每组必须运行

- [ ] 相同 real data fraction；
- [ ] 相同 synthetic ratio；
- [ ] 相同 seed 集合；
- [ ] 相同 NILM backbone；
- [ ] 相同下游训练预算；
- [ ] 相同 validation 和 early-stop 规则。

### 统计分析

- [ ] 报告均值、标准差和 95% CI。
- [ ] 使用相同测试窗口进行 paired comparison。
- [ ] 对 E8−E2、E8−E3、E7−E6、E6−E5、E5−E4 计算差值和置信区间。
- [ ] 同时报告效果量，不能只报告 p-value。
- [ ] 对不同 appliance 分别解释，避免平均值掩盖失败电器。
- [ ] 检查提升是否仅来自增加 ON-state 比例或改变类别分布。

### 建议的成功标准

在正式运行前将最终阈值写入 protocol。推荐采用以下分层判断：

1. **最低结论**：基元方法在至少一个预先指定的核心场景中稳定优于 E0 和 E1；
2. **主要结论**：E8 在核心多状态电器的主要指标上稳定优于 E2，并且没有以明显恶化其他关键指标为代价；
3. **强结论**：上述优势在多个电器、多个种子和低数据量/跨住宅场景中保持；
4. **效率结论**：基元方法在相同效果下使用更少真实数据、合成样本或计算时间。

“稳定”至少表示大多数随机种子方向一致；正式论文应进一步结合置信区间判断。

### 产物

- `reports/downstream/core_benchmark.csv`；
- `reports/downstream/paired_effects.csv`；
- `reports/paper_tables/main_results.*`；
- `reports/paper_tables/ablation.*`；
- 低数据量曲线、跨住宅结果图和失败案例图；
- 核心结论草稿。

### 验收条件

- 能在公平条件下直接比较 E2 和 E8；
- 每个提升都能通过消融定位来源；
- 结果可重复计算；
- 能明确回答“更好、相当还是更差”，而不是只展示有利样本。

### 停止/转向条件

- 如果 E8 多次不能优于 E1：复杂生成路线可能不值得，转向简单增强和困难样本采样。
- 如果 E3 优于 E4–E8：生成器是瓶颈，优先保留真实片段重组并优化采样。
- 如果 E2 与 E8 性能相当但 E8 成本更低：研究重点转向效率和可解释性。
- 如果 E2 明显优于 E8：分析切分误差、状态定义、边界伪影和信息损失，不继续盲目增加拼接约束。

---

## P9：基元状态效率优化

### 目标

根据 P8 的结果寻找性能提升来源和成本瓶颈，形成可量化的基元效率优化方法。

### 9.1 数据效率

- [ ] 绘制 10%、25%、50%、100% 真实数据曲线。
- [ ] 计算达到目标性能所需的真实 cycle 数和标注时长。
- [ ] 比较每增加一个真实 cycle 带来的性能收益。
- [ ] 优先测试复杂多状态和少样本状态。

### 9.2 合成样本效率

- [ ] 测试 synthetic ratio：0%、25%、50%、100%、200%、400%。
- [ ] 计算单位合成样本的性能增益。
- [ ] 找到边际收益开始饱和的位置。
- [ ] 超过饱和点后自动停止继续生成。

### 9.3 状态库效率

- [ ] 使用最近邻或聚类删除近重复片段。
- [ ] 尝试使用 medoid/代表片段代替完整状态池。
- [ ] 比较保留 25%、50%、75%、100% 状态片段时的结果。
- [ ] 记录状态库大小、检索耗时和下游性能。

### 9.4 状态粒度效率

- [ ] 比较粗、中、细三种切分粒度。
- [ ] 比较不同 cluster k 和 state_merge 阈值。
- [ ] 同时评价状态稳定性、生成成本和 NILM 性能。
- [ ] 以最终下游收益选择粒度，不单独追求聚类指标。

### 9.5 目标导向生成

- [ ] 从 NILM 错误中识别漏检、误检和混淆状态。
- [ ] 只对困难状态或少样本状态增加生成配额。
- [ ] 比较均匀生成、按频率反比生成、按下游错误生成。
- [ ] 主动生成同时启动、相近功率和多状态交叉等困难组合。

### 9.6 计算效率

- [ ] 记录每种方法训练时间、采样时间、参数量、峰值显存/内存。
- [ ] 缓存 transition 候选、duration 模型和 boundary compatibility matrix。
- [ ] 为状态特征建立快速索引。
- [ ] 复用现有内容寻址缓存机制。
- [ ] 计算每 GPU-hour 或 CPU-hour 获得的性能增益。

### 效率指标

```text
数据效率 = 下游性能 / 真实训练周期数
样本效率 = 下游性能增益 / 合成样本数
时间效率 = 下游性能增益 / 训练与生成总小时数
存储效率 = 下游性能 / 状态库大小
边际收益 = 新增一批合成数据后的性能变化 / 新增成本
```

### 产物

- 数据量—性能曲线；
- 合成比例—性能曲线；
- 状态库压缩—性能曲线；
- 切分粒度—性能曲线；
- 时间/显存/存储成本表；
- 推荐的高效配置和停止规则。

### 验收条件

- 至少识别一个经过对照实验验证的效率优化点；
- 优化结果不是通过改变测试集、训练预算或评价规则获得；
- 可以说明在哪些设备和场景下效率改进成立。

---

## P10：鲁棒性验证、复现与论文整理

### 目标

确认主要结论不只存在于一个数据片段或一个模型，并整理为可复现的论文实验包。

### 任务清单

- [ ] 扩展到至少三种状态复杂度不同的电器。
- [ ] 在条件允许时完成跨住宅验证。
- [ ] 增加第二个 NILM backbone，确认结论不依赖单一模型。
- [ ] 增加第二个生成模型家族，仅用于验证方法结论的稳健性。
- [ ] 在 UK-DALE 外增加一个公开数据集或自采集数据。
- [ ] 汇总失败案例和适用边界。
- [ ] 固定最终配置、随机种子和 checkpoint。
- [ ] 整理一键运行脚本和最小复现数据。
- [ ] 更新 README、教程、环境安装和依赖说明。
- [ ] 生成论文主表、消融表、效率表和主要图。

### 论文建议结构

1. Introduction：NILM 数据稀缺与多状态电器问题；
2. Related Work：普通增强、整段生成、状态/Markov 合成和 NILM 生成研究；
3. Method：状态发现、状态生成、约束拼接和 aggregate 构造；
4. Experimental Protocol：无泄漏划分与公平预算；
5. Results：真实性、下游效果、低数据量、跨住宅；
6. Ablation：random、transition、duration、boundary、context；
7. Efficiency：真实数据、样本、算力和状态库效率；
8. Limitations：状态定义、边界伪影、复杂设备和跨设备限制。

### 最终产物

- 完整代码和配置；
- 数据 manifest 与 provenance；
- 固定 checkpoint；
- 主实验和消融结果；
- 环境锁定文件；
- 复现实验说明；
- 论文图表与结果说明。

### 最终验收条件

- 从原始数据到最终指标可以按文档重新执行；
- 所有主要结果能够找到对应 config、run manifest、seed 和 checkpoint；
- test 数据从未进入生成和模型选择过程；
- 论文结论和实验数据一一对应；
- 能明确说明方法的优势、失败场景和适用范围。

---

## 8. 最小投稿闭环与完整研究闭环

### 8.1 最小投稿闭环

在时间有限的情况下，优先完成：

1. 一个核心多状态电器：washing machine；
2. 一个清晰的无泄漏 temporal split；
3. 一个整段生成器；
4. 真实基元拼接和生成基元拼接；
5. transition、duration、boundary 三项消融；
6. 一个固定 NILM baseline；
7. E0、E1、E2、E3、E4、E5、E6、E7/E8；
8. 3 个随机种子；
9. 真实测试集上的 MAE、F1 和能量误差；
10. 一组低数据量实验；
11. 训练时间和采样速度比较。

满足以上内容即可形成一条完整、可解释的研究闭环。

### 8.2 完整研究闭环

在最小闭环基础上增加：

- 三种不同状态复杂度的电器；
- 5 个随机种子；
- 跨住宅和跨数据集；
- 两个 NILM backbone；
- 两个生成模型家族；
- 自采集数据；
- 状态库压缩和错误导向生成；
- 完整效率与鲁棒性分析。

---

## 9. 每周执行和记录制度

### 每周开始

- [ ] 从本计划选择本周 task ID；
- [ ] 写清输入、输出和验收条件；
- [ ] 确认依赖任务已经完成；
- [ ] 确认不会修改已冻结的 test protocol。

### 每次实验前

- [ ] 保存 config；
- [ ] 保存 git commit；
- [ ] 保存数据 manifest；
- [ ] 保存 seed；
- [ ] 估算运行时间和算力；
- [ ] 确认实验只改变一个计划变量。

### 每次实验后

- [ ] 检查 manifest 和产物完整性；
- [ ] 检查 NaN、异常功率和标签错位；
- [ ] 记录成功或失败原因；
- [ ] 不手工删除不利结果；
- [ ] 更新阶段结果表；
- [ ] 将关键图表和结论放入 `reports/`。

### 每周结束

- [ ] 汇总完成任务、失败任务和阻塞项；
- [ ] 更新下一周优先级；
- [ ] 判断是否满足当前 phase gate；
- [ ] 若不满足 gate，不盲目进入下一阶段。

---

## 10. 风险登记表

| 风险 | 可能表现 | 检查方式 | 应对措施 |
|---|---|---|---|
| 数据泄漏 | 测试指标异常高 | provenance 和时间重叠检查 | 先划分再建库；自动 leakage test |
| 状态不稳定 | 同状态跨 cycle 差异过大 | 类内/类间距离 | 增加 context、调整粒度或更换表征 |
| 聚类无物理意义 | label 难以解释且顺序混乱 | 时序重构和人工抽检 | 规则/人工辅助、重新定义 state |
| 过度切分 | 单一功能状态被拆成许多段 | state_merge 与粒度实验 | 调整切分和短状态吸收 |
| 拼接伪影 | 模型利用接缝而非设备规律 | 边界判别和 ΔP 分析 | 使用真实 transition、对齐或边界约束 |
| 整段基线过弱 | 基元方法获得不公平优势 | 参数量和预算审计 | 同模型家族、相同调参预算 |
| 合成像但无用 | 分布接近但 NILM 不提升 | 真实测试集下游评价 | 错误导向采样、降低无效样本比例 |
| 模式坍塌/复制 | 样本高度重复训练数据 | NN 距离、覆盖率 | 多样性约束、去重、调整生成器 |
| 合成比例过高 | 真实性能下降 | synthetic ratio curve | 使用边际收益停止规则 |
| 算力不足 | 完整网格无法执行 | 时间/显存预算记录 | 先 CVAE、单 backbone、3 seeds |
| 公开数据不足 | 某状态样本太少 | cycle/state count | 扩展时间、住宅或自采集数据 |
| 创新重合 | 与已有 Markov/state synthesis 相似 | 系统文献矩阵 | 强化真实基元、边界兼容、效率和下游证据 |

---

## 11. 阶段完成定义（Definition of Done）

一个阶段只有同时满足以下条件才视为完成：

- [ ] 代码已实现并通过相关单元测试；
- [ ] 现有测试没有回归；
- [ ] 输入、输出和配置已经登记到 RunManifest；
- [ ] 关键产物可以从干净运行重新生成；
- [ ] 阶段报告已写入 `reports/`；
- [ ] 验收指标达到 phase gate；
- [ ] 已记录失败样本和已知限制；
- [ ] README 或对应教程已经更新；
- [ ] 下一阶段依赖已经明确。

仅“代码能运行”不代表阶段完成；必须同时具备测试、产物、报告和验收证据。

---

## 12. 近期最优先任务清单

以下任务建议作为接下来第一批实际工作：

### 优先级 P0

- [ ] T001：冻结第一版实验协议和成功标准。
- [ ] T002：保存当前 39 个测试的基线结果。
- [ ] T003：复现并固化 UK-DALE washing machine 完整运行。
- [ ] T004：确定第一版整段生成器和 NILM baseline。

### 优先级 P1

- [ ] T101：定义 canonical data schema。
- [ ] T102：实现 cycle/time/house 级 research split。
- [ ] T103：实现 partition manifest。
- [ ] T104：实现自动 leakage checker。
- [ ] T105：为 split 增加单元测试和小数据 smoke test。

### 优先级 P2

- [ ] T201：定义 state library schema。
- [ ] T202：从现有 state_merge 产物构建训练状态库。
- [ ] T203：构建 transition graph 和 duration statistics。
- [ ] T204：实现 provenance 完整性检查。

### 优先级 P3/P4

- [ ] T301：生成状态重复性与可分性报告。
- [ ] T302：完成单状态跨 cycle 替换实验。
- [ ] T401：实现真实基元随机拼接 E3。
- [ ] T402：实现 transition constrained E5。
- [ ] T403：实现 duration constrained E6。
- [ ] T404：实现 boundary/transition constrained E7。

只有 T001–T404 的结果证明状态具有可重复性和可交换性后，才应投入较多时间训练复杂生成模型。

---

## 13. 最终决策逻辑

完成核心实验后，按下列逻辑选择后续方向：

```text
E3 真实基元拼接是否优于 E0/E1？
├─ 否 → 优先修正状态定义、切分和采样；暂缓复杂生成
└─ 是
   ↓
E4 生成基元是否优于 E3？
├─ 否 → 生成器是瓶颈；保留真实片段重组作为主路线
└─ 是
   ↓
E5/E6/E7 是否逐步优于 E4？
├─ 否 → 删除无贡献约束，保持方法简单
└─ 是
   ↓
E8 是否优于 E2 整段生成？
├─ 精度更高 → 主张下游效果优势
├─ 精度相当但成本更低 → 主张效率与可解释性优势
├─ 只在低数据/多状态场景更高 → 收窄适用范围并形成条件性结论
└─ 全面更差 → 分析信息损失与边界伪影，考虑状态条件整段生成
```

本项目不预设基元生成拼接必然胜出。研究价值来自严格地证明它在什么条件下有效、为什么有效、成本是多少，以及在哪些条件下不应使用。


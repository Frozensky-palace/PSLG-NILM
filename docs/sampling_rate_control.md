# 原采样率与两点均值降频对照

本实验只检验降频对原有生成增强和 NILM 效果的影响，不替换论文主线，也不将降频包装成新的生成模型。原始数据、原 6 秒结果及师弟分支保持不变；12 秒数据和结果放入新的目录。

## 代码基线和现有工作

固定的师弟代码版本为 `feature/haojun@ef0ebbf96933cfaf8bd11292d03c28471efd9aa7`，已合入本人 `zzz` 分支并保留提交历史。两者没有在师弟分支上互相覆盖。

现有流程包括训练周期划分、6 秒总表/支路对齐、PrimGLR 切分、DETSEC-PC 或物理统计表示、聚类与状态合并、B0 无增强、B1 周期重放、B2 真实基元重组、B3 整周期变换/CVAE/WGAN/扩散、B4 基元 CVAE/WGAN/扩散与真实供体对照、B5 状态路径/时长/边界拼接，以及 Seq2Point 下游验证。

师弟已经有测试集报告，本轮使用同一个测试时期时属于重复评估，不能称为新的完全未见测试。历史结果来自不同提交和增强比例时，只列作历史参考；严格的 6 秒与 12 秒对照须核对相同代码、划分、种子、训练设置和增强预算，必要时补跑同版本的 6 秒参照。

本轮核对到的继承限制：

- `src/generation/primitive_cvae.py` 的训练条件仍拼接全零状态 one-hot，生成时却启用状态 one-hot，影响共用该函数的基元生成路线。本轮不同时修改模型算法，报告必须披露这一点，不能由此宣称某类生成模型无效。
- 基元生成仍用 Python `hash` 派生部分种子。记录式启动器固定子进程 `PYTHONHASHSEED=0`；旧运行若未固定，不保证逐位重现。
- B2 历史路线的 physical-stats 状态库与 B4/B5 的 DETSEC 状态库不同。重跑须保留并记录该差异，不能默默换成同一种库后仍称只有降频变化。
- 合成质量门的 WARN 不会令 `passed` 为 false；“通过”不是全部分布指标都良好。
- 三种子重采样结果不能自动解读成稳健的显著性结论，也不能把未发现差异等同于方法等效。

## 评价指标

| 指标 | 当前实现的定义 | 汇报注意事项 |
| --- | --- | --- |
| MAE W | 预测和真值截为非负后，逐评价点绝对误差均值 | 主回归指标；12 秒目标更平滑，不能只比两个原生 MAE |
| RMSE W | 同一评价点集合上的均方根误差 | 对大误差更敏感 |
| SAE | 预测功率和与真值功率和之差的绝对值，除以真值功率和 | 真值和为零时为 null；监控子集不能当整段能耗；等间隔共同支持上才可解释为对应能量相对误差 |
| Precision Recall F1 | 真值和预测均按功率 ≥20 W 判开机，再逐点统计 TP FP FN | 是逐采样点开关指标，不是周期/事件匹配 F1 |
| 合成质量 | 负功率、峰值上限、时长、能量分位数、多样性、边界统计 | hard FAIL、WARN 分开记录；原边界斜率单位为 W/sample，跨采样率不可直接当 W/s 比较 |
| 近复制 | 与真实训练周期的形状距离及逐点复制检查 | B1 重放本就重复真实周期，不把它作为生成器记忆失效 |
| 工程成本 | 命令执行时间、模型参数量、训练 epoch、输入/输出大小 | 记录实际值，不假定降频必然带来某个速度提升 |

训练监控 MAE 使用归一化后再还原的原始预测误差；正式指标由预测入口截断负预测后计算，两者口径不能混写。训练采样池目前使用 `>20` 而指标使用 `>=20`，也作为继承口径记录，不在本次悄悄改动。

## 数据处理规则

必须先确定原实验的 6 秒对齐数据。不能把不规则时间戳的 DAT 直接隔行抽样，也不能把旧 HDF 构建命令的 `--sample-seconds` 改成 12 就称为逐对合并。

对完整的相邻两点：`P12(t) = (P6(t) + P6(t+6)) / 2`。输出时间戳为 Unix 12 秒网格的左端点 t。总表与目标设备使用完全相同的配对索引。功率取均值，不求和；对应 12 秒区间的能量等于保留的两条 6 秒样本能量之和。

- 不跨训练/验证/测试分区、实际时间缺口或同文件内的连续段边界配对。已确认的 `aligned_research_partitions_v1` 源协议中，连续段 ID 是文件内编号；同分区相邻文件的两点若严格相差 6 秒且左点位于 12 秒网格，可以跨存储文件配对。
- 不足两点或相位不完整的网格舍弃，不复制、不补零、不新增插值。逐逻辑处理块记录点数与能量：跨文件配对时，下一文件首点归当前块、下一块跳过该点，整个分区每个源点恰好计入一次。
- 连续总表上的开关边沿仍正常取均值，不人为制造缺口。周期库仅取两个源点都位于原周期内的完整区间，原周期边界另外保存，记录边界收缩和周期准入变化。
- 类别/状态标签不平均。重建 12 秒周期库后，继续运行原切分/表示/聚类入口；原始标签不直接贴到新采样点上。
- 有符号背景保持线性平均；供拼接使用的裁剪背景按 `max(mean(background_signed_w),0)` 重算，并在审计中注明它不等于先裁剪再平均。
- 保存源/目标文件 SHA-256，以及每个输出点左右两侧的原文件索引与物理行号。跨文件配对另存两侧哈希、时间戳和原连续段 ID；原始文件不覆盖。

`scripts/downsample_aligned_pairs.py` 支持生成 12 秒数据，也可用 `--factor 1` 建立另存的 6 秒参照。服务器准备配置为 12 秒组启用 `--pair-across-shards` 和 v2 输出协议；不带此参数保留旧版不跨文件行为，便于复核。默认只转换 train/validation；`--include-test` 会记录测试数据被转换，但不执行模型训练或选模。

## 哪些参数保持不变

本对照默认保留原模型输入点数、网络结构、优化器、增强策略和种子，Seq2Point 均为 599 点。因此窗口区间长度从 `599×6=3594` 秒变为 `599×12=7188` 秒。它是“相同离散模型配置下的采样率对照”，不是物理上下文完全相等的实验。

本轮不同时加入 299 点模型；改变点数还会改变现有 Seq2Point 全连接层的参数量，应另列实验。所有以秒表达的量及能量计算必须使用实际采样间隔 12 秒；状态合并 `fs` 应为 `1/12`，不能沿用 `0.1666667`。原生点数型超参数保持原值并记录其物理跨度变化。

原先的 Slurm 脚本有硬编码账户/环境和大量默认 6 秒参数，不能不经配置核对直接运行。旧命令和参数值必须逐项抄录到新运行记录，而不是执行一个未审计的全局替换。

## 服务器配置和第一步

配置文件：`config/server/zzz_sampling_control.json`。

- 项目：`/home/scnu2024024563/NILM-zzz/PSLG-NILM-zzz`
- 原始数据：`/home/scnu2024024563/dataset`
- 下游与生成环境：复用 `pslg-nilm`，具体依赖以预检为准。
- 状态发现环境：复用 `pslg-detsec`，具体依赖以预检为准。

先在登录节点执行轻量清单检查：

```bash
cd /home/scnu2024024563/NILM-zzz/PSLG-NILM-zzz
conda run --no-capture-output -n pslg-nilm python scripts/inspect_sampling_control.py
conda run --no-capture-output -n pslg-detsec python scripts/inspect_server_setup.py --output-dir log/sampling_control_setup/detsec
tar -czf log/sampling_control_setup.tar.gz -C log sampling_control_setup
```

返回 `log/sampling_control_setup.tar.gz`。第一条只读 labels/metadata、候选通道前三行及已有产物位置，不读取完整波形，不解压、不安装、不训练。既有 HDF 中 `meter54` 的含义不能未经核实就套用到原始目录的同编号文件；标签、计量类型和列结构确认后才能选择总表通道。

若找到原 aligned 数据和周期划分，应优先复用并验证哈希。如果只有原始 DAT，则先按原协议构建一次 6 秒对齐数据，再从同一份数据派生 12 秒；不能把重新划分的数据与历史表直接作单变量比较。

## 2026 年 10 月 6 日服务器清单结论

回传的 `sampling_control_setup.tar.gz` 确认服务器为 `zzz@761b1c3`，已跟踪文件无改动。设备元数据确认：洗衣机为 `house_1/channel_5.dat` 的有功功率；`meter54` 对应 `house_1/mains.dat`，其四列为时间戳、有功功率、视在功率、电压。`channel_54.dat` 不存在并不表示总表缺失。`channel_1.dat` 为视在功率，且元数据标为 disabled，不能用它代替原有功总表。

两套环境均已有包版本记录：`pslg-nilm` 为 Python 3.12.14、PyTorch 2.5.1+cu121；`pslg-detsec` 为 Python 3.12.14、TensorFlow 2.18.1。Slurm 返回 RTX3090 和 A6000 分区可用。登录节点无 `nvidia-smi` 路径不能用于判定计算节点 GPU 不可用；框架导入和 GPU 前后向仍须在分配到的计算节点上验证。

在 `pslg_artifacts` 找到六份历史下游 `dataset_manifest.json`：`de_inputs_r0p5`、`de_inputs_r0p5_v2`、`de_inputs_r0p5_g5test`、`de_inputs_r0p5_b4r`、`de_inputs_r0p5_b4wgan`、`de_inputs_r0p5_b4diff`。第一次清单只记录路径，且搜索范围漏掉师弟操作手册中的 `~/projects/`，因此不能据此认定原对齐数据已丢失。

现改用来源追溯：读取这些清单、归一化和窗口范围，以及可定位的对齐清单/周期划分，检查 B0 实际文件位置和元数据哈希。不读取 NPZ 波形、不修改旧清单、不自动选择多个同名数据副本，也不重建划分。

```bash
conda run --no-capture-output -n pslg-nilm python scripts/inspect_sampling_sources.py
tar -czf log/sampling_source_trace.tar.gz -C log sampling_source_trace
```

回传 `log/sampling_source_trace.tar.gz`。若找到与历史记录哈希一致的原 6 秒对齐清单，仍须在正式计算作业中校验波形 SHA-256 后才进行两点降频；元数据哈希一致不等于波形验证通过。来源缺失或歧义将保留在报告中，不自动换通道或重划训练集。

来源追溯新增 6 项本地测试，验证旧仓库定位、哈希匹配与不匹配、路径歧义、路径越界/符号链接、读取大小限制，以及不会打开波形数组。连同下述原 69 项测试，来源追溯阶段共 75 项通过。

## 来源追溯结果与数据准备作业

第二次回传 `sampling_source_trace.tar.gz` 定位到原数据：

```text
/home/scnu2024024563/projects/PSLG-NILM-c1/reports/core_validation/ukdale_b1_washing_machine/aligned_partitions_v2
```

其对齐清单 SHA-256 为 `158dbb3333539c695225b64627a0786e23a0ac3d9f79235339c3618ceba77ce4`，与发现的九份历史下游清单记录一致。原周期划分 CSV 的实际哈希也与对齐清单引用一致；清单中的 Windows 路径是历史来源记录，不必修改原文件。配置已固定服务器实际路径和对齐清单、coverage、inventory、`de_inputs_r0p5_v2` 清单四份哈希。

| 原 6 秒分区 | 分片数 | 有效连续数据点 | analysis eligible 周期 |
| --- | --- | --- | --- |
| train | 19 | 7,896,997 | 493 |
| validation | 5 | 2,187,295 | 164 |

以上为原清单记录；作业 12097 已验证相应 24 个波形文件，降频结果见下节。

`slurm/sampling_control_prepare.sbatch` 复用 `pslg-nilm`，申请 RTX3090 分区的 4 个 CPU 和 16 GB 内存，不申请 GPU。2 小时是作业时间上限，不是耗时承诺。运行顺序如下：

1. 校验冻结元数据、原周期 ID/边界/划分和 24 个 train/validation 波形分片的 SHA-256。
2. 在新运行目录另存 factor1 的 6 秒参照及 factor2 的 12 秒数据；不读取测试集数组，不修改源文件。
3. 两组分别建立 train/validation 周期库；只有 train 周期导出状态发现用 CSV。
4. 输出逐周期能量/时长/峰值/边缘位移、短周期最小长度、eligible 增减 ID、逐逻辑处理块丢弃点数与能量。
5. 再校验源文件哈希，生成不含大波形的 `diagnostics.tar.gz`。异常尝试也保留记录和已有诊断，不覆盖、不删除后重跑。

提交前先更新到本次发布的提交，再执行：

```bash
cd /home/scnu2024024563/NILM-zzz/PSLG-NILM-zzz
mkdir -p log
PSLG_SAMPLING_COMMIT="$(git rev-parse HEAD)" \
  sbatch --export=ALL slurm/sampling_control_prepare.sbatch
```

排队和运行期间不要切换分支、拉取或修改该仓库；作业会在每步前后检查提交和已跟踪文件。输出在 `/home/scnu2024024563/pslg_artifacts/sampling_control_data_<JOBID>/`；完成后回传该目录的 `diagnostics.tar.gz`。Slurm 启动失败或强制超时时可能来不及生成包，此时回传仓库 `log/sampling-control-data-<JOBID>.out` 和 `.err`。

数据阶段完成不代表生成器训练完成或效果改善。若 12 秒周期库少了某些周期，先审查 `cycle_comparison.csv` 和 `eligibility_changes`，不得直接宣称两组训练周期集合完全相同；若全部周期失效则作业停止。GPU 框架检查、切分/聚类、生成器和 Seq2Point 的真实重训属于后续阶段。

数据作业初版另新增 8 项本地测试：完整小样本流水线、波形/元数据变更拒绝、登录节点保护、运行前后代码冻结、失败包和短周期退出报告。初版与前述测试合计 83 项通过；Slurm 脚本通过 `bash -n` 检查。

## 作业 12097 的结果与存储边界修正

回传 `sampling_control_12097_diagnostics.tar.gz` 记录作业在 `zzz@665add621d812ac11b8b91f09c4b07232fb5eed0` 完成全部 8 个数据步骤，用时 157.58 秒。四份冻结元数据及 24 个 train/validation 波形文件的前后校验均通过；未读取 test 数组、未训练模型。产物保留在 `/home/scnu2024024563/pslg_artifacts/sampling_control_data_12097`，不覆盖重跑。

| 分区与采样率 | 输出点数 | 丢弃源点数 | 可用周期数 |
| --- | --- | --- | --- |
| train 6 秒 | 7,896,997 | 0 | 493 |
| train 12 秒初版 | 3,948,426 | 145 | 491 |
| validation 6 秒 | 2,187,295 | 0 | 164 |
| validation 12 秒初版 | 1,093,641 | 13 | 164 |

两条训练周期失去准入均源于初版不跨存储文件配对，各少一个完整 bin，而不是周期太短：

| 周期 ID | 缺失的两个源时间戳 | 12 秒已观测与应有点数 |
| --- | --- | --- |
| uk-dale_b1_washing_machine_1396013542_1396019572 | 1396017600 和 1396017606 | 501 / 502 |
| uk-dale_b1_washing_machine_1406730817_1406736557 | 1406731200 和 1406731206 | 476 / 477 |

修正仅允许已验证连续的存储文件边界配对，不放宽真实缺测、同文件内部连续段边界或分区边界。每个跨文件 bin 同时记录两侧来源，借入行只消费一次。6 秒数据、周期划分、生成模型和下游算法不改。

按初版记录的文件首尾时间推算，新作业应在训练分区恢复 17 个跨文件 bin：12 秒点数预计为 3,948,443，弃点由 145 降至 111，可用周期由 491 恢复到 493；验证分区应保持不变。这些是待服务器验证的验收预期，不是已完成结果。时间戳 1387982382 到 1387982544 的 162 秒真实缺口必须保留。

初版连续数据的保留样本能量误差为零，但丢弃源点仍损失少量能量：训练支路丢弃 0.950833 Wh、验证支路丢弃 0 Wh。周期库的边缘收缩是另一项损失：共同可用的 491 条训练周期能量差合计为 −62.228333 Wh，164 条验证周期为 −20.767361 Wh。不能把连续保留样本守恒表述为每个周期能量完全不变；修正版须重新报告这些统计。

## 作业 12098 数据验收通过

回传 `sampling_control_12098_diagnostics.tar.gz` 确认 `zzz@5add0e1ac5a843cbfc59f05bbe47d19e076c1ee6` 在 2026 年 10 月 6 日完成全部 8 个数据步骤，用时 157.85 秒。24 个原始 train/validation 分片的前后校验通过且两份校验记录完全一致；未读取 test 数组，未训练生成模型或 NILM。后续输入固定为 `/home/scnu2024024563/pslg_artifacts/sampling_control_data_12098`，不再重做降频或重划周期。

| 验收项 | 6 秒 | 12 秒 |
| --- | --- | --- |
| 训练周期 | 493 | 493 |
| 验证周期 | 164 | 164 |
| 训练连续数据点 | 7,896,997 | 3,948,443 |
| 验证连续数据点 | 2,187,295 | 1,093,641 |
| 训练周期 CSV 总采样点 | 429,335 | 214,417 |
| 训练支路丢弃源点能量 Wh | 0 | 0.028333 |

两组可用周期 ID 集合完全相同、无增减。12 秒训练数据恢复 17 个跨存储文件 bin，弃点由 145 降至 111；验证数据仍丢弃 13 个源点，其支路丢弃能量为零。跨文件两点的时间差、网格相位、原文件行号和两侧哈希审计一致。1387982382 到 1387982544 的 162 秒真实缺口保留；两组清单中的缺口均为上一分片网格终点 1387982388 到下一分片起点 1387982544。

连续保留样本的能量误差记录为零，但两组完整周期库仍有边缘收缩：493 条训练周期能量差合计 −62.505 Wh，164 条验证周期为 −20.767361 Wh。该差值须与连续数据弃点损失分开报告。以上是数据正确性验收，不是降频提升模型效果的证据。

## 配对状态发现链路测试

下一作业 `slurm/sampling_state_smoke.sbatch` 申请 RTX3090 分区的 h104-slurm-a 节点、1 张 GPU、4 个 CPU、32 GB 内存，2 小时为上限。h104 是历史通过检查的节点，当前可用性仍由本次框架检查判定。复用现有环境，不安装包，不覆盖 12098 输入。具体流程如下：

1. 固定已验收运行的 8 份元数据 SHA-256，复核两组 train 周期 NPZ、CSV 及清单的 1,980 个文件，确认周期 ID 和顺序相同。只读取 train 数据，validation/test 数组不进入此阶段。
2. Slurm 脚本加载 `miniconda3/25.5.1-0`、`cuda-toolkit/12.1.1`，激活 `pslg-nilm` 启动编排器。每个子进程通过 `scripts/run_in_sampling_env.sh` 完整激活目标环境：TensorFlow 用 `pslg-detsec`，PyTorch 用 `pslg-nilm`。记录实际 Python 路径、Conda 前缀、模块、GPU 和提交号，分别运行两框架 GPU 前向与反向检查，任一失败即停止。保留 Slurm 的 `CUDA_VISIBLE_DEVICES`，不手动改卡号。
3. 取 CSV 顺序中的前 8 个相同训练周期，各跑 2 个 epoch 的 PrimGLR → DETSEC-PC → KMeans k=3/4/5 → 状态合并与状态库导出。只检查接口，不选 k、不冻结状态库、不进入生成器或 Seq2Point 训练。
4. 逐周期核对状态块从首点到末点连续覆盖、无重叠、无遗漏，并检查时长使用正确采样间隔。原入口会吞掉的 PrimGLR 后端异常、截断造成的覆盖失败均使本作业失败，不记为成功。
5. 再校验源文件，保存两组配置、所选周期、状态统计、状态路径、训练历史、耗时、命令、输入输出哈希和失败日志。大张量与波形留在服务器；成功或 Python 异常均尝试打包小型诊断文件。

保持原分段/特征/聚类/合并算法及离散点数配置，状态合并 `fs` 分别为 `1/6` 和 `1/12`；`min_block_seconds=90` 不变。KMeans seed 沿用 42；两组另显式固定 Python/NumPy/TensorFlow 初始化 seed=42。继承的 DETSEC-PC 内部分段留出与 batch 随机种子仍为 0，并非同一个 KMeans 参数。原历史入口未固定 TensorFlow 初始化，故此次不能声称逐位复现历史特征。禁用旧特征缓存，避免未固定种子的旧特征替代新训练。两组状态编号各自有效，不直接把同号状态当作同一物理状态。

入口为 `scripts/run_sampling_state_smoke.py`，配置为 `config/server/zzz_sampling_state_smoke.json`。提交前更新到本次发布的干净提交，然后执行：

```bash
cd /home/scnu2024024563/NILM-zzz/PSLG-NILM-zzz
mkdir -p log
PSLG_SAMPLING_COMMIT="$(git rev-parse HEAD)" \
  sbatch --export=ALL slurm/sampling_state_smoke.sbatch
```

排队或运行时不要修改该仓库。返回 `/home/scnu2024024563/pslg_artifacts/sampling_state_smoke_<JOBID>/diagnostics.tar.gz`；若 Slurm 启动失败、进程被强制终止而无包，返回 `log/sampling-state-smoke-<JOBID>.out` 和 `.err`。链路通过后才扩展到全部 493 个训练周期和正式 epoch，生成/NILM 先导的 seed 17 属于更下游阶段；8 周期、2 轮结果不得用于论文效果比较。

## 作业 12099 的 GPU 初始化失败与启动修正

2026 年 10 月 6 日，12099 在 `ace4af6` 上运行约 19.35 秒后失败。诊断包显示 Slurm 已在 h103-slurm-a 分配 `gres/gpu=1`，`CUDA_VISIBLE_DEVICES=0`，`nvidia-smi` 能列出 RTX 3090。但 TensorFlow 2.18.1 的 `cuInit` 返回 `CUDA_ERROR_UNKNOWN`，GPU 检查失败，尚未运行 PyTorch 检查或任何 6 秒/12 秒状态训练。cuFFT/cuDNN/cuBLAS 的重复注册信息不能单独当作本次失败原因；直接阻断点是 CUDA 初始化失败。

这一现象与 [9 月 21 日节点报告](../reports/server_c1/2026-09-21/cluster_node_health_report_2026-09-21.md) 中 h103 的记录一致：调度与 `nvidia-smi` 正常，但两个框架无法初始化 CUDA；当时相同环境在 h104 通过。该证据支持优先避开 h103，不能据此确定内核模块或硬件的具体故障。没有重装包、修改驱动、重置 GPU 或更改集群节点状态。

旧启动脚本虽调用了正确的 `pslg-detsec/bin/python`，但未完整激活环境，子进程仍继承 base 的 `CONDA_PREFIX`，且没有加载任何 module。这是相对原 C1/C2 启动流程的遗漏，不能仅凭它断言“不激活就是 cuInit 故障的唯一原因”。修正版同时恢复原模块/激活流程并指定历史验证通过的 h104，作为恢复运行措施，不用于单独归因某一变量。若 h104 新作业仍失败，保留诊断并进一步区分节点和运行库问题。

12099 的失败产物原样保留。新作业继续使用已验收的 12098 数据与相同小规模配置；运行记录中的 `environment` 明确表示父启动进程，真正的框架环境以子进程 `environment_tensorflow.json`、`environment_torch.json` 为准。环境初始化若在 Python 编排器启动前失败，诊断包可能尚未生成，此时回传 Slurm `.out` 和 `.err`。

## 执行阶段和全程留档

| 阶段 | 内容 | 需要留档 |
| --- | --- | --- |
| 输入核验 | 通道、环境、原 aligned 数据、原周期划分定位 | 清单 JSON、原协议与提交号 |
| 降频 | 6 秒数据两点均值派生 12 秒 | downsampling_audit、输入/输出哈希、配对行号、丢弃与能量统计 |
| 上游复跑 | train 周期库、CSV 导出、原切分/表示/聚类 | 周期数、状态数、时长/功率/能量、配置、状态库指纹 |
| 生成复跑 | 沿用原生成路线与增强预算 | 每轮训练历史、模型、生成来源、质量与近复制报告 |
| NILM | 同种子、同离散网络设置训练并验证 | 配置、参数量、训练时间、checkpoint、预测、全部指标 |
| 对照汇总 | 同方法/种子的原生指标与共同 12 秒网格指标 | 原始逐种子数据、比较 JSON、共同预测数组、失败与缺失项 |

训练作业在 Slurm 计算节点运行。首批先做 seed 17 的链路和耗时检查；它只能是先导，不能冒充三种子完整结论。是否直接复用某个历史 6 秒结果，取决于其原始配置与预测是否可核验。

`scripts/run_recorded_command.py` 可以包裹每条既有命令，记录提交、命令、指定输入/输出的 SHA-256、时间、退出码、环境、标准输出和错误。输入、输出和记录目录需独立，失败尝试保留，不原地覆盖；计算作业使用 `--require-slurm`。它只记录指定的产物，调用时必须列齐 checkpoint、预测、配置和统计目录。大波形和权重保留服务器，报告包附索引和指纹，不提交 Git。

## 跨采样率比较

新预测入口会额外保存时间戳、源分片/中心行号、采样间隔和训练种子。`scripts/compare_sampling_predictions.py` 将 6 秒预测和真值按对应时间点两两平均，与 12 秒预测在共同网格比较；若两边真值不匹配则拒绝比较。

共同网格评估应在两侧推理时使用 `--sample-source first_n`（不设 `--limit` 即全部可用窗口），或另行冻结相同物理时期。默认 monitor 是稀疏训练监控子集，不保证包含完整相邻两点，不能拿它冒充全量共同支持。检查点没有可核验种子时也拒绝配对。

```bash
python scripts/compare_sampling_predictions.py \
  --native-predictions /path/to/6s/validation_predictions.npz \
  --coarse-predictions /path/to/12s/validation_predictions.npz \
  --output-dir /path/to/new/comparison
```

汇报分别列出 6 秒原生指标、12 秒原生指标、共同 12 秒目标上的两模型指标，以及实际比较点数。原生误差变小不自动意味着模型变好；相同输入点数导致的时间上下文变化仍应单独说明。没有时间戳的历史预测不能直接用于该比较，可在原模型与原数据上重新推理补齐定位信息。

## 当前完成边界

已接入固定师弟代码、实现配对降频和审计、周期库接口验证、带时间戳的预测记录、共同网格比较、服务器路径配置和记录式命令入口。12098 真实数据准备与存储边界修正已验收通过；12099 在 h103 的 TensorFlow GPU 检查阶段失败，修正版待 h104 新作业验证。正式切分/聚类、生成器重训和 NILM 效果实验尚未完成；不存在可汇报的 12 秒模型效果数值。

2026-10-06 当前版本地验证：以下 118 项测试通过，另已执行 Python 语法编译、`bash -n` 和 `git diff --check`。其中包括完整准备流程的跨文件恢复测试和 13 项边界测试，覆盖 29 种文件切法、单行/空文件、来源行号唯一性、输入/能量总账、坏哈希、真实缺口和分区隔离。状态链路及启动检查共 16 项，另纳入原有 5 项状态发现辅助测试。其余测试覆盖周期库/CSV 接口、预测中心定位、B1/B2 适配、数据划分、调度和质量指标。测试数据为小型构造样本；本地缺少 PyTorch/TensorFlow，没有验证神经网络训练或 GPU 可用性。

```bash
python -m unittest tests.test_sampling_state_smoke tests.test_state_discovery \
  tests.test_sampling_shard_boundaries tests.test_sampling_preparation \
  tests.test_sampling_source_trace tests.test_sampling_rate_control tests.test_sampling_server_tools \
  tests.test_server_setup_inventory tests.test_research_data \
  tests.test_shared_placement tests.test_synthetic_quality
```

状态链路本地测试覆盖输入变更/越界拒绝、两组周期顺序、采样单位配置、显式初始化种子、缺周期/重叠/截断拒绝、代码漂移、失败包和完整编排。新增 4 项启动测试验证两个子环境分别激活、GPU 分配变量保留、参数安全传递、激活失败停止与 Slurm 节点/模块配置；测试使用模拟 Conda 函数和本地 Python，不证明服务器激活成功。GPU 与训练子进程的编排测试也使用模拟结果，真实结果须以新作业回传为准。

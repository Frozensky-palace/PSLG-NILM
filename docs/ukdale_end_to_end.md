# UK-DALE 数据准备、切分与端到端运行

## 1. 本仓库中的数据

数据文件为 `datasets/ukdale/ukdale.h5`，大小约 5.90 GiB，是 NILMTK 格式的
UK-DALE，包含 building 1–5。

洗衣相关通道如下：

| 住宅 | 支路 | 总表 | 说明 |
|---|---:|---:|---|
| building 1 | meter5 | meter54 | 独立 washer dryer；总表从 2013-03-17 开始 |
| building 2 | meter12 | meter20 | 独立 washing machine |
| building 4 | meter6 | meter1 | 与 microwave、breadmaker 共用，不建议作为干净标签 |
| building 5 | meter24 | meter26 | 独立 washer dryer |

building 3 没有洗衣机支路。第一次跑通建议使用 building 1。

### 分离 House 1 的全部物理通道

```powershell
.\.venv\Scripts\python.exe scripts\extract_ukdale_house.py `
  --building 1 --allow-washer-dryer
```

输出目录为 `datasets/ukdale/house1/`：

- `meter01.h5` 至 `meter54.h5`：每个文件保存一个物理通道，数据表键为 `/data`；
- `channels.json`：完整通道、列、设备与原始元数据；
- `channels.csv`：便于人工查看的通道索引；
- 洗衣机 `meter5` 同时导出至 `input/ukdale_house1_washing_machine.csv`。

完整流程使用 2013-04-01 至 2013-05-01 的总表共同覆盖区间：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_ukdale_pair.py `
  --building 1 --appliance "washing machine" --allow-washer-dryer `
  --start 2013-04-01 --end 2013-05-01 `
  --out-prefix input\ukdale_house1_washing_machine_pipeline `
  --branch-csv input\ukdale_house1_washing_machine_pipeline.csv

.\.venv\Scripts\python.exe main.py `
  --config config\config_ukdale_house1_e2e.yaml `
  --run-id ukdale_house1_wm_complete --steps all `
  --segment-method fluss --feature-model dtw `
  --cluster-method kmeans --n-clusters 3 `
  --cluster-tag kmeans_k3_merged
```

## 2. 先生成严格对齐的支路/总表切片

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_ukdale_pair.py `
  --building 1 `
  --appliance "washing machine" `
  --allow-washer-dryer `
  --start 2013-03-18 `
  --end 2013-03-25 `
  --out-prefix input\ukdale_b1_wm_week
```

输出：

- `input/ukdale_b1_wm_week_branch.npy`
- `input/ukdale_b1_wm_week_mains.npy`
- `input/ukdale_b1_wm_week_metadata.json`

两个 NPY 均为 `(n, 2)`，列为 `[unix_timestamp, power]`，时间戳逐行一致。
脚本把两路数据统一到 6 秒网格；只插值不超过 150 秒的短缺口；无法恢复的
总表点从两路同时删除。

## 3. 分步骤跑通项目流水线

所有命令必须使用相同的 `run-id`、分段方法和特征模型。

### 3.1 活动提取

```powershell
.\.venv\Scripts\python.exe main.py --config config\config_ukdale_smoke.yaml `
  --run-id ukdale_b1_wm_e2e --steps extract
```

### 3.2 活动内部时间分段

```powershell
.\.venv\Scripts\python.exe main.py --config config\config_ukdale_smoke.yaml `
  --run-id ukdale_b1_wm_e2e --steps segment --segment-method fluss
```

### 3.3 DTW 特征

```powershell
.\.venv\Scripts\python.exe main.py --config config\config_ukdale_smoke.yaml `
  --run-id ukdale_b1_wm_e2e --steps feature `
  --segment-method fluss --feature-model dtw
```

### 3.4 聚类

```powershell
.\.venv\Scripts\python.exe main.py --config config\config_ukdale_smoke.yaml `
  --run-id ukdale_b1_wm_e2e --steps cluster `
  --segment-method fluss --feature-model dtw `
  --cluster-method kmeans --n-clusters 2
```

### 3.5 时序状态合并

```powershell
.\.venv\Scripts\python.exe main.py --config config\config_ukdale_smoke.yaml `
  --run-id ukdale_b1_wm_e2e --steps state_merge `
  --segment-method fluss --feature-model dtw --cluster-method kmeans
```

它会生成新的聚类标签 `kmeans_k2_merged`。后面的步骤应显式使用该标签。

### 3.6 few-shot 识别

```powershell
.\.venv\Scripts\python.exe main.py --config config\config_ukdale_smoke.yaml `
  --run-id ukdale_b1_wm_e2e --steps fewshot `
  --segment-method fluss --feature-model dtw --cluster-method kmeans `
  --cluster-tag kmeans_k2_merged
```

### 3.7 基元到完整活动的映射

```powershell
.\.venv\Scripts\python.exe main.py --config config\config_ukdale_smoke.yaml `
  --run-id ukdale_b1_wm_e2e --steps pam `
  --segment-method fluss --feature-model dtw --cluster-method kmeans `
  --cluster-tag kmeans_k2_merged
```

### 3.8 生成项目定义的 train/test_a/test_b

```powershell
.\.venv\Scripts\python.exe main.py --config config\config_ukdale_smoke.yaml `
  --run-id ukdale_b1_wm_e2e --steps split `
  --segment-method fluss --feature-model dtw --cluster-method kmeans `
  --cluster-tag kmeans_k2_merged
```

关键结果位于：

- `log/ukdale_b1_wm_e2e/run_manifest.json`
- `log/ukdale_b1_wm_e2e/DatasetSplit/dataset_split_summary.json`
- `log/ukdale_b1_wm_e2e/DatasetSplit/*_branch.npy`
- `log/ukdale_b1_wm_e2e/DatasetSplit/*_mains.npy`

确认分步骤执行无误后，也可以换一个 `run-id` 一次执行：

```powershell
.\.venv\Scripts\python.exe main.py --config config\config_ukdale_smoke.yaml `
  --run-id ukdale_b1_wm_e2e_all --steps all `
  --segment-method fluss --feature-model dtw `
  --cluster-method kmeans --n-clusters 2 `
  --cluster-tag kmeans_k2_merged
```

## 4. “项目切分”与“论文评估切分”不要混用

当前 `DatasetSplitStep` 是事件级随机划分，并在同一条完整时间线上执行 knockout：

- train 保留训练事件；
- test_a 保留测试 few-shot 与 non-few-shot 事件；
- test_b 只保留测试 few-shot 事件；
- 被移除的支路功率同时从 mains 中扣除。

这是项目的少样本/消融实验构造，不是标准 NILM 泛化评估。三套数据共享相同的
背景负载和时间上下文，不能把它们当成完全独立的常规训练集和测试集。

正式论文实验建议另做以下两层切分：

1. **同住宅泛化**：先按时间连续切成 train/validation/test（例如 60%/20%/20%），
   不随机打散窗口；归一化、特征模型、聚类中心和阈值只能在 train 上拟合。
2. **跨住宅泛化**：使用 building 1 训练、building 2 验证、building 5 测试；
   避免 building 4 的混合通道。每栋住宅都只使用支路与总表共同覆盖的时间范围。

在连续块边界至少留出一个模型窗口长度的隔离区，活动跨越边界时应整段归入一侧
或丢弃，避免同一次洗衣活动同时出现在训练与测试中。

本页的一周数据只用于工程冒烟测试。它只有 4 个活动，不足以评价 few-shot
识别效果；正式实验应扩展日期范围后再选择聚类数和模型超参数。

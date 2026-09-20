# PSLG-NILM-ADVANCED

> 新手入口：**[docs/TUTORIAL.md](docs/TUTORIAL.md)** —— 从部署到出图的完整教程
> （含 UK-DALE washing machine 冒烟测试）。

> 当前生成式 NILM 研究入口：
> **[完整研究路线与 AI 交接](docs/AI_HANDOFF_COMPLETE_RESEARCH_ROADMAP.md)**；
> 仓库整理规则见 **[Git 文件追踪政策](docs/GIT_TRACKING_POLICY.md)**。

PSLG-NILM 的重构版：单一线性流水线引擎、内容寻址特征缓存、聚类全候选 k、
出图与流程完全解耦。相对旧版（`PSLG-NILM`）的四个核心改动：

| 旧版问题 | 本版方案 |
|---|---|
| 每次实验要改 config / 改 slurm 脚本 | 步骤与变体全部走 CLI 参数，config 只放固定参数 |
| 缓存体系复杂（按轨迹多级缓存） | 只有特征提取（最慢）一个内容寻址缓存，其余步骤不缓存 |
| 出图逻辑耦合在流程步骤里；先 scan 选 best-k 再聚一次 | 步骤零出图；kmeans 一次产出**所有**候选 k 的结果，scan 降级为可选诊断 |
| 冗余中间产物（同一矩阵多处拷贝） | 产物按 manifest 单点登记、单点存储，下游全部经 manifest 解析 |

## 目录结构

```
main.py                  # CLI 入口 + 线性 Workflow 引擎装配
config/config.yaml       # 固定参数（路径、模型超参、切分超参、出图设置）
models/                  # 特征提取模型（沿用旧版，未改动）
src/
  framework/             # Step 基类 / Workflow / RunManifest / feature_cache
  steps/                 # 7 个流水线步骤
  utils/                 # 聚类指标等纯函数
visualize/               # 独立出图脚本（只读 manifest + config 的 visualization 块）
slurm/                   # 纯循环 slurm 脚本（无 sed，无临时 config）
tests/                   # M1–M4 单测（28 个）
log/<run_id>/            # 每次运行的全部产物 + run_manifest.json
output/<run_id>/figure/  # 出图脚本的统一输出
.cache/features/         # 特征提取内容寻址缓存
```

## 流水线

```
extract → segment → feature → cluster → fewshot → pam → split
```

| 步骤 | step_type | 说明 |
|---|---|---|
| extract | `extract_active_data` | 从原始序列切出活动段 CSV |
| segment | `time_segmentation` | 活动段 → 基元（clasp / ggs / window …） |
| feature | `feature_extract` | 基元 → 潜特征（detsec / bilstm_ae / …），**唯一带缓存的步骤** |
| cluster | `time_clustering` | kmeans（全候选 k）/ kmeans-scan（诊断）/ dbscan / hdbscan |
| fewshot | `few_shot_cluster_extract` | 按簇规模识别少样本簇并导出 |
| pam | `primitive_activity_mapping` | 基元↔活动映射（索引对齐），划分少样本/非少样本活动 |
| split | `dataset_split` | 生成 train / test_a / test_b（knockout：支路置零、总线扣减） |

## 用法

```bash
# 完整流水线（步骤与变体全部在命令行选择）
python main.py --appliance fridge \
    --steps extract,segment,feature,cluster,fewshot,pam,split \
    --segment-method clasp --feature-model detsec \
    --cluster-method kmeans --n-clusters 3,4,5,6

# 只重跑聚类，复用已有 run 的特征（manifest 解析上游产物）
python main.py --steps cluster --run-id <已有run_id> \
    --cluster-method dbscan

# 下游：选定某个聚类结果做 fewshot → pam → split
python main.py --steps fewshot,pam,split --run-id <run_id> --cluster-tag kmeans_k4
```

CLI 参数一览：`--steps`、`--segment-method`、`--feature-model`、
`--cluster-method`、`--n-clusters`、`--cluster-tag`、`--appliance`、
`--run-id`、`--raw-series`、`--config`。

## RunManifest：产物的唯一事实来源

每次运行写 `log/<run_id>/run_manifest.json`。每个步骤登记自己的产物路径
（相对 run 目录），下游步骤与出图脚本一律经 manifest 按 `(step_type, key)`
解析，不再拼接目录名猜测上游输出。

- `add_step`：单结果步骤整体登记；
- `add_cluster_result(tag, ...)`：聚类按 tag 登记多个并存结果（`kmeans_k3`、
  `kmeans_k4`、…），这就是"全候选 k"的落点；
- `add_step_artifact`：增量合并单个产物（kmeans-scan 诊断用），不覆盖已有
  `results`；
- 用同一 `--run-id` 重跑部分步骤时会加载已有 manifest，窄范围重跑也能解析
  上游产物。

## 特征缓存语义

特征提取是唯一缓存的步骤，key 为纯内容寻址：

```
SHA256( X 内容+shape+dtype | lengths | 模型名 | 全部超参 | 模型源码指纹 )
```

- 与 run-id、目录结构、跑了哪些步骤**无关**；
- 上游选择（数据集、切分方法、切分超参）通过 X 的字节级哈希**传递性**覆盖——
  上游任何改变都会改变 X，从而改变 key；
- 模型源码（模型文件 + `models/base_model.py`）改动同样使缓存失效；
- 命中时打印 `cache HIT (<key12>...)`，meta.json 记录 provenance
  （appliance / segment_method / 超参 / 源码指纹），provenance 不进 key：
  内容相同的 X 天然共享缓存，这是设计使然。

## 出图（与流程完全解耦）

出图脚本在 `visualize/`，只读 manifest 和 config 的 `visualization:` 块，
输出统一在 `output/<run_id>/figure/<kind>/`：

```bash
python -m visualize.visualize_segments      --run-id <id>
python -m visualize.visualize_separation    --run-id <id>
python -m visualize.visualize_clustering    --run-id <id> [--cluster-tag kmeans_k4]
python -m visualize.visualize_cluster_reconstruction --run-id <id>
python -m visualize.visualize_feature_history        --run-id <id>
```

`--cluster-tag` 缺省时对 run 内**所有**聚类结果出图。

## Slurm

`slurm/` 下 4 个纯循环脚本，全部通过 CLI 参数选择变体，不再 sed 改 config：

```bash
sbatch slurm/run_pipeline.sh                      # 单次完整流水线
sbatch slurm/run_segmentation_grid.sh             # 切分方法网格
sbatch slurm/run_feature_models.sh                # 特征模型网格（缓存去重）
sbatch --export=ALL,RUN_ID=<id> slurm/run_cluster_grid.sh   # 聚类方法网格
```

可用 `--export=ALL,VAR=...` 覆盖脚本顶部变量（APPLIANCE、FEATURE_MODEL、
N_CLUSTERS 等）。公共环境 preamble 在 `slurm/env.sh`。

## 测试

```bash
python tests/test_m1_framework.py    # 引擎 + 前置步骤 (8)
python tests/test_m2_cache.py        # 特征缓存 (10)
python tests/test_m3_clustering.py   # 聚类 + fewshot (7)
python tests/test_m4_downstream.py   # PAM + DatasetSplit (3)
```

## UK-DALE 小型冒烟测试（nilmtk）

`datasets/ukdale/ukdale.h5` 是 nilmtk 格式的 HDF5。用 nilmtk 截一段
100 行的 washing machine 功率切片（锚定在首个 >20W 活动点之前），
再跑通 提取→切分→特征→聚类：

```bash
# 1. 生成 100 行测试切片 -> input/ukdale_washing_machine_100.csv
python scripts/prepare_ukdale.py

# 2. 完整跑一遍（小切片上 clasp 的自动窗口过大，用 fluss；
#    无 GPU/TF 环境用 dtw 特征模型）
python main.py --config config/config_ukdale_test.yaml \
    --appliance washing_machine --run-id ukdale_wm_test \
    --steps extract,segment,feature,cluster \
    --raw-series input/ukdale_washing_machine_100.csv \
    --segment-method fluss --feature-model dtw \
    --cluster-method kmeans --n-clusters 2
```

预期：1 个活动段 → fluss 切出 3 个基元 → dtw 特征 (3×3) →
`kmeans_k2` 结果（2/1 两簇），产物与 `run_manifest.json` 在
`log/ukdale_wm_test/`。重跑同一命令可观察特征缓存 `cache HIT`。

注意：`config/config_ukdale_test.yaml` 是为 100 行小切片调参的测试配置
（window_size 10、n_regimes 3、excl_factor 2），真实规模实验请用
`config/config.yaml` 的参数。

## 环境

Python 3.12；依赖见 `requirements.txt`（`pip install -r requirements.txt`）。
要点：numpy 必须 <2；nilmtk 无 py3.12 的 PyPI 包，从 GitHub 源码安装；
claspy/fluss 为 `models/time_segmentation/` 下的 vendored 包，不要再
pip 安装 claspy（会拖入 numpy 2.x 破坏环境）；tensorflow（神经特征模型）、
hdbscan 按需懒加载。

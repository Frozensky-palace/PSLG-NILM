# PSLG-NILM Git 文件追踪政策

> 更新日期：2026-09-20  
> 目标：Git 保存“重建实验所需的逻辑与结论”，大数据和可再生产物使用外部存储。

## 1. 必须追踪

以下内容是项目的可复现核心：

| 类型 | 路径 | 原因 |
|---|---|---|
| 程序入口 | `main.py` | 工作流入口 |
| 核心源码 | `src/` | 数据、NILM、工作流与算法实现 |
| 模型源码 | `models/` | 切分、特征与聚类模型 |
| 执行脚本 | `scripts/` | 数据准备、审计和实验执行 |
| 测试 | `tests/` | 防止科研逻辑被改坏 |
| 配置 | `config/` | 冻结实验参数 |
| 服务器脚本 | `slurm/` | 正式实验提交方式 |
| 可视化源码 | `visualize/` | 可重建图表 |
| 核心文档 | `README.md`、`docs/*.md` | 背景、协议、进度和交接 |
| 人工结论 | `reports/**/*.md` | 小型、可审阅的实验结论 |
| 依赖 | `requirements.txt` | 环境复现 |
| CI | `.github/` | 自动检查 |

当前新增且应该进入 Git 的研究实现包括：

```text
config/config_ukdale_house1_e2e.yaml
config/config_ukdale_smoke.yaml
config/experiments/nilm_b0_b2_pilot.yaml
models/feature_extract/physical_stats.py
src/data/
src/nilm/
scripts/audit_cycle_inventory.py
scripts/build_aligned_research_partitions.py
scripts/build_b1_b2_pilot_cycles.py
scripts/build_cycle_inventory.py
scripts/build_real_cycle_library.py
scripts/build_server_transfer_manifest.py
scripts/build_state_library.py
scripts/create_research_split.py
scripts/evaluate_nilm_predictions.py
scripts/export_cycle_library_segments.py
scripts/extract_ukdale_house.py
scripts/place_b1_b2_on_train_background.py
scripts/prepare_nilm_b0_b2_inputs.py
scripts/prepare_ukdale_pair.py
scripts/run_cpu_nilm_smoke.py
tests/test_research_data.py
```

这些文件体积很小，却包含当前无泄漏数据层和 B0–B2 验证的全部关键逻辑。

## 2. 建议追踪，但提交前需要人工审阅

- 已修改的旧源码与旧测试；
- `determined/` 中真正用于论文复现的历史配置；
- 小型、人工维护的示例数据；
- 论文最终固定的表格或图片；
- 环境锁文件，例如未来生成的 `requirements-lock.txt`。

当前工作树中以下已追踪文件存在修改，应保留追踪，但提交前检查它们是否属于同一
功能提交：

```text
main.py
scripts/prepare_ukdale.py
src/steps/dataset_split_step.py
src/steps/feature_extract_step.py
src/steps/time_segmentation.py
tests/test_dpc_kmeans.py
tests/test_m4_downstream.py
```

不要为了得到“干净状态”而丢弃这些修改；它们可能包含用户已有工作。

## 3. 不应追踪

| 类型 | 当前路径或扩展名 | 原因 |
|---|---|---|
| 原始数据集 | `datasets/` | 当前约31GB，且可能受数据许可限制 |
| 预处理输入 | `input/` | 当前约323MB，可由数据脚本重建 |
| 运行日志与 manifest | `log/` | 当前约154MB，运行相关且数量持续增长 |
| 自动输出 | `output/` | 图表和 few-shot 导出可重建 |
| 特征缓存 | `.cache/` | 内容寻址缓存，不是源代码 |
| 虚拟环境 | `.venv/`、`venv/` | 平台相关且约679MB |
| 报告二进制 | `reports/**/*.npz`、`*.npy` | 当前占报告目录绝大部分 |
| 自动表格/清单 | `reports/**/*.csv`、`*.json` | 可由冻结脚本与输入重新生成 |
| 模型权重 | `*.pt`、`*.pth`、`*.ckpt`、`*.keras` | 大且需外部版本管理 |
| HDF 数据 | `*.h5`、`*.hdf5` | 大型原始或中间数据 |
| 实验平台目录 | `wandb/`、`tensorboard/`、`lightning_logs/` | 自动产生 |
| 本机 IDE/缓存 | `.vscode/`、`.idea/`、`__pycache__/` | 与项目逻辑无关 |

## 4. reports 的特殊规则

`reports/.gitignore` 使用白名单，只开放 Markdown：

```text
reports/**/*.md       追踪
reports/**/*.npz      忽略
reports/**/*.npy      忽略
reports/**/*.csv      忽略
reports/**/*.json     忽略
reports/**/*.png      忽略
```

这样可以保存研究结论，又不会把约922MB的本地实验产物推入 Git。

关键 JSON/CSV 虽然不进 Git，仍必须保存。建议方法：

1. 用 `build_server_transfer_manifest.py` 记录文件路径、大小和 SHA-256；
2. 数据保存在本机、服务器和至少一份独立备份；
3. Markdown 报告记录关键统计、输入目录、配置和随机种子；
4. 论文冻结时把最终 manifest 发布到外部 artifact storage，而不是普通 Git。

## 5. 当前不应追踪的特殊文件

`docs/server.html` 是旧的本地 HTML 汇总快照，其中 cohort 数量和无泄漏状态与当前
冻结协议不一致。它可以留在本机作历史审计，但不能作为当前事实来源，也不建议提交。
当前事实应以以下文档为准：

```text
docs/AI_HANDOFF_COMPLETE_RESEARCH_ROADMAP.md
docs/core_validation_b0_b5_execution.md
reports/core_validation/ukdale_b1_washing_machine/*.md
```

## 6. Git 仓库体积异常

只读检查发现 `.git` 约8.6GB，其中约6.86GB是 loose objects，另有约1.52GB的
临时 garbage objects。当前已追踪工作树中没有超过5MB的大文件，因此这些对象很可能
来自过去中断或撤销的大文件暂存操作，而不是当前版本必须内容。

不要直接删除 `.git/objects`。在确认没有重要未引用对象且做好仓库备份后，可由用户
另行授权执行 Git 官方清理流程。清理 Git 对象与本次 `.gitignore` 整理是两件事：
`.gitignore` 防止以后误加大文件，但不会自动缩小已有 `.git`。

## 7. 推荐提交拆分

不要把所有修改塞进一个无法审阅的大提交。建议：

1. `chore: define git tracking policy and ignore generated artifacts`
2. `feat: add leakage-free UK-DALE research data pipeline`
3. `feat: add B1 B2 composition and NILM input preparation`
4. `test: add research data and composition coverage`
5. `docs: add validation protocol and AI handoff roadmap`

这里只给出拆分建议，未自动暂存或提交，避免混入用户尚未审阅的旧修改。

## 8. 提交前检查

```powershell
git status --short
git diff --check
git diff --stat
git ls-files | Select-String -Pattern '\.(h5|hdf5|npz|npy|pt|pth|ckpt|keras)$'
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

预期：最后一条大文件搜索不应返回新研究产物；全套测试应通过。

## 9. 2026-09-20 整理后的实际审计

应用当前忽略规则后：

```text
尚未追踪但应该追踪的文件：48
总大小：约 0.28 MB
其中：
  config    5 个
  docs      6 个
  models    1 个
  reports  10 个（含 reports/.gitignore，只保留 Markdown）
  scripts  15 个
  src      10 个
  tests     1 个
```

最大的候选文件约44KB，没有大型数据或模型权重混入。`reports/` 中约922MB的
NPZ/NPY/CSV/JSON/PNG 已从拟提交集合中排除，但文件仍保留在本机，未被删除。

本次整理只调整追踪规则和文档，没有执行 `git add`、`git commit`、历史重写或
数据清理。现在 `git status` 中剩余的未追踪项基本就是应该由研究者审阅后加入 Git
的源码、配置、测试与 Markdown。

## 10. 本地文件保留与清理建议

这些目录都不进 Git，但本机仍需分级管理：

### 必须保留或先完成外部备份

```text
datasets/ukdale/ukdale.h5
reports/.../aligned_partitions_v2/
reports/.../real_cycle_library_train_v1/
reports/.../state_library_pilot_k3_v1/
reports/.../state_library_pilot_k4_v1/
reports/.../state_library_pilot_k5_v1/
```

它们是当前正式数据和状态研究的基础。即使可以重建，也不应在没有校验备份时删除。

### 当前实验仍在使用，暂时保留

```text
b1_b2_placed_seed17_r0p5/
b1_b2matched_placed_seed17_r0p5/
b1_b2matched_k4_placed_seed17_r0p5/
b1_b2matched_k5_placed_seed17_r0p5/
nilm_inputs*_seed17_r0p5_v1/
cpu_nilm_smoke*_seed17_r0p5/
```

这些目录用于服务器候选比较和复核。完成上传、SHA-256校验并有外部备份后，可以
依照生成脚本重新创建，因此属于“可再生产物”。

### 明确不能用于实验、可在备份后清理

```text
reports/.../aligned_partitions_v1/    约129 MB
```

它是修复分片边界前的旧产物。当前保留 `DO_NOT_USE.md` 进入 Git 已足够表达禁用规则；
二进制目录只有历史审计价值。是否删除仍由用户决定，本次没有删除。

### 最大空间问题：`.git`

`.git` 约8.6GB，比所有当前报告产物更大。应先制作仓库完整备份，再单独审计和清理
未引用对象。不要用资源管理器直接删除 `.git/objects`，也不要在没有备份时运行激进
prune。该项预计比删除任何一个 report 目录释放更多空间。

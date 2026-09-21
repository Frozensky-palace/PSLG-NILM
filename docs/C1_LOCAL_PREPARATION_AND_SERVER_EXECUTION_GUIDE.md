# PSLG-NILM 进入 C1 前的本机准备与服务器执行手册

> 版本：v1.1  
> 日期：2026-09-20  
> 适用范围：C1 服务器环境冻结、C2 正式状态发现、C3 B0/B1/B2 第一轮验证  
> C0 冻结：以 `phase-c0-freeze` 标签为准（含本手册 §5.2 全部 P0 脚本）；离线归档
> 与 SHA-256 记录在本机 `server_transfer/` 目录（不进 Git），服务器优先走 Git 远程。
> 核心原则：train 用于学习，validation 用于选择，最终 test 暂不上传、不读取、不评价。

本文件用于解释“进入 C1 前准备什么”。准备完成后，实际从 C1 执行到 Phase G 请使用
`docs/C1_TO_G_SERVER_MANUAL.md`。

---

## 1. 先用一句话说明现在要做什么

C1 不是马上跑大模型，而是先证明服务器环境、GPU、代码、数据和恢复流程都可靠。通俗地
说，就是先把“厨房、食材和秤”校准好，再正式做实验；否则跑几天后才发现环境或数据错了，
结果无法相信。

本手册将工作分为三层：

1. **进入 C1 前必须完成**：本机整理代码、制作无 test 数据包、准备服务器环境方案；
2. **C1 在服务器完成**：验证硬件和两个深度学习框架、冻结环境、跑短冒烟；
3. **C1 通过后才能做**：C2 DETSEC-PC 和 C3 Seq2Point 正式任务。

后续 B3/B4/B5 生成模型的文件也在本文列出，但它们不阻塞 C1，不应为了“目录看起来完整”
而提前创建空文件。

---

## 2. 2026-09-20 当前复核结论

### 2.1 已经通过的部分

| 检查项 | 当前结果 | 简单解释 |
|---|---|---|
| Git 工作树 | 检查前干净；本文修改后只应出现文档变更 | 先前代码修复已提交 |
| 当前分支 | `feature/haojun`，提交 `9dd3786` | 这是目前可追溯的代码版本 |
| 冻结标签 | `phase-b-freeze` 指向 `9dd3786` | Phase B 修订版标签存在 |
| 测试 | 本轮再次运行，131 项全部通过 | 含自定义仓库目录、打包、GPU设备和早停逻辑测试 |
| DETSEC-PC 冒烟 | 8 个 train cycle，k=3/4/5，`test_accessed=false` | 新增状态库元数据修复有效 |
| 状态库元数据 | feature、segment、commit、status 已正确透传 | 产物能说明自己如何生成 |
| validation 预测索引 | 与冻结的 validation monitor 完全相同 | 没有偷偷换评价样本 |
| B0/B1/B2 无 test 包 | `--exclude-test` 可生成并通过逐文件哈希 | 开发阶段可物理隔离 test |
| YAML | 40 个现有 YAML 已解析通过 | 含新增 C1/C2/C3 服务器矩阵 |

### 2.2 进入 C1 前仍有四个硬门槛

2026-09-20 更新：以下四项的本机部分已全部完成（见 §9 执行顺序第 1–6 项的产物），
遗留动作只剩"推送到远程"需要用户凭据，见 §2.3。

1. **DETSEC-PC GPU 环境** → 已拆分：`environment_detsec_server.yml`
   （TensorFlow 2.18.1 GPU + numpy<2）与 `environment_nilm_server.yml`
   （PyTorch cu121），均标注"初版参考"，服务器 GPU 枚举冒烟通过后才冻结；
   原 `environment_server.yml` 降级为本机 CPU 参考并记录本机版本基线。
2. **状态发现包** → `scripts/build_c1_server_bundle.py --profile state-discovery`
   生成 `manifests/c1_state_discovery_trainonly_manifest.json`
   （503 个文件，train-only 守卫 + 零 test 路径断言）；
   `--profile nilm-b0b2` 生成 `manifests/c1_nilm_b0_b2_trainval_manifest.json`；
   两者均通过 `verify_server_manifest.py` 本地逐文件复算。
3. **Slurm 旧模板** → `env.sh` 与 5 个旧 wrapper 已打 LEGACY 禁用标记；
   正式模板改为 `c1_gpu_smoke.sbatch`、`c2_detsec_pc.sbatch`（含 preflight 门槛与
   `--deterministic-tf`）、`c3_seq2point.sbatch`，全部使用 `%u`/`$HOME`，
   无 `<user>`、无旧用户目录。
4. **分支上游** → 远程 `origin` 已配置；推送分支与标签需要用户执行（凭据），
   命令见 §4.1；离线回退方案为 `server_transfer/phase-c0-freeze.bundle`
   （git bundle，含标签与提交哈希）。

这四项没有解决前，可以登录服务器做只读调查，但不应提交正式 C2/C3 长任务。

### 2.3 推送与归档（需要用户执行）

远程已存在：`origin = https://github.com/Frozensky-palace/PSLG-NILM.git`。
`feature/haojun` 尚未推送到 origin；`phase-b-freeze`、`phase-c0-freeze` 标签也只在
本机。请在本机执行：

```powershell
git push -u origin feature/haojun
git push origin phase-b-freeze phase-c0-freeze
```

若 GitHub 不可达，改用离线归档：`server_transfer/phase-c0-freeze.bundle` 已由
`git bundle create` 生成并记录 SHA-256；服务器端 `git clone <bundle路径>` 后
`git checkout phase-c0-freeze`，即可获得与完全一致的提交哈希。

---

## 3. 阶段边界：C1、C2、C3 分别是什么

| 阶段 | 做什么 | 不做什么 | 完成标志 |
|---|---|---|---|
| C1 | 冻结服务器环境、数据、代码；完成 CPU/GPU/恢复冒烟 | 不跑正式多 seed 网格 | 环境记录齐全，短任务成功且可恢复 |
| C2 | 只用 train cycle 跑正式 DETSEC-PC，比较 k=3/4/5 | 不用 validation 重训聚类，不碰 test | 正式 state library v1 冻结 |
| C3 | B0/B1/B2，3 seeds，ratio=0.5 的 validation 比较 | 不开始最终 test | 冻结真实周期/基元重组基线 |

推荐顺序：

```text
本机准备 → C1 环境和冒烟 → C2 状态发现 → 冻结状态库 → C3 B0/B1/B2
         → B3 四种完整周期生成 → B4 基元生成 → B5 HSMM 受约束拼接
```

---

## 4. 进入 C1 前，本机必须完成的事项

### 4.1 冻结 Git 代码身份

执行并保存结果：

```powershell
git status --short
git branch -vv
git log -3 --oneline --decorate
git tag -n
```

要求：

- 工作树只允许存在本轮有意修改的文档或准备脚本；
- 不把 `.venv/`、模型 checkpoint、大型 `.npz/.npy` 数据和日志加入 Git；
- 文档和脚本完成、测试通过后再提交；
- 建议新建不移动旧标签的 `phase-c0-freeze`，记录 C1 实际使用的最终提交；
- 标签和分支要推送到服务器能够访问的远程；若服务器不能访问 Git 远程，则制作代码归档，
  同时保存归档 SHA-256 和原始 Git commit；
- 不要在服务器手工改代码后不回传。本机仓库应始终是唯一可维护来源。

标签建议命令应在最终提交后执行，本文不自动创建标签：

```powershell
git tag -a phase-c0-freeze -m "C0 frozen before server C1"
git push <remote> feature/haojun
git push <remote> phase-c0-freeze
```

### 4.2 运行本机回归检查

最低检查：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall src scripts
```

再做配置解析：

```powershell
@'
from pathlib import Path
import yaml
for path in Path("config").rglob("*.yaml"):
    yaml.safe_load(path.read_text(encoding="utf-8"))
print("all yaml parsed")
'@ | .\.venv\Scripts\python.exe -
```

通过标准：测试全绿、脚本可编译、所有 YAML 可解析。若测试数量不再是131，必须解释是
新增了哪些测试，不能只把文档中的数字机械改掉。

### 4.3 再审计一次数据泄漏

必须确认：

- 状态发现输入只来自 `research_split=train`；
- 生成模型训练 donor 只来自 train；
- validation 只用于选 k、策略和 checkpoint；
- `validation_monitor_indices.npy` 与实际预测索引一致；
- 开发服务器包内不存在 `aligned_partitions_v2/test/`；
- manifest 和日志中 `test_accessed=false`；
- test 数据仍留在本机受控位置，C1–C3 不上传。

“脚本没有调用 test”不等于“没有泄漏”。最可靠的方式是 C1–C3 的服务器上根本没有 test
文件。

### 4.4 制作两个互相独立的数据包

不要再用一个大包承担所有任务。建议拆成：

#### 包 A：C1/C2 状态发现包

建议 manifest 名：

```text
manifests/c1_state_discovery_trainonly_manifest.json
```

必须包含：

```text
config/experiments/core_wm_state_discovery_detsec_pc.yaml
reports/core_validation/ukdale_b1_washing_machine/real_cycle_library_train_v1/
  real_cycle_library.csv
  real_cycle_library_manifest.json
  segments/segment_source_map.csv
  segments/segment_export_manifest.json
  segments/*.npy
研究 split / cycle inventory 中用于证明 train-only 的元数据
```

不得包含 validation/test 原始波形。C2 的模型选择摘要可以回到本机或使用单独 validation
评价包，但不能把 validation 片段混入聚类训练输入。

#### 包 B：C1/C3 NILM 开发包

建议 manifest 名：

```text
manifests/c1_nilm_b0_b2_trainval_manifest.json
```

使用现有 `scripts/build_server_transfer_manifest.py --exclude-test` 生成。必须包含 train、
validation、B1/B2 放置结果、索引、归一化参数和冻结配置；不得包含 test shard。

当前复核过的候选规模是73个文件、263,845,964 bytes，即263.8 MB或251.6 MiB。重新生成
后应以新 manifest 的数字为准，不要把这个数字写死到程序。

#### 每个包的验收

- manifest 使用项目相对路径；
- 每个文件有 bytes 和 SHA-256；
- 服务器端可用 `scripts/verify_server_manifest.py` 逐个复算；
- test 路径数量为0；
- 同一文件不要同时以两个不同路径重复打包；
- 数据不进 Git，只追踪打包/校验脚本和小型 manifest；
- 大型 manifest 若包含本机敏感绝对路径，应先改成相对路径再追踪。

### 4.5 准备服务器环境方案

推荐将环境拆成两个，而不是强行把 TensorFlow 和 PyTorch 全塞进一个环境：

| 环境 | 主要用途 | 核心框架 |
|---|---|---|
| `pslg-detsec` | C2 DETSEC-PC | 能识别服务器 GPU 的 TensorFlow |
| `pslg-nilm` | C3 Seq2Point，以及后续大多数生成器 | CUDA 匹配的 PyTorch |

这样做的好处很简单：两个框架的 CUDA/依赖要求不同，拆开更容易定位问题，也不会为了修
TensorFlow 破坏已经工作的 PyTorch。

本机现在应完成的是环境文件和验证脚本，而不是猜测服务器安装命令。服务器驱动、CUDA、
可用 module 必须现场查看。特别注意：

- 正式 `pslg-detsec` 不应继续使用 `tensorflow-cpu`；
- Linux 上 TensorFlow GPU 支持方式必须以安装后的 GPU 枚举结果为准；
- `numpy<2` 是当前仓库约束，升级前必须全量回归；
- PyTorch 需安装与现场驱动兼容的 CUDA 构建；
- 环境的任何偏离都写进 `environment.txt`，不能只在终端历史里留下痕迹。

建议最终增加：

```text
environment_detsec_server.yml
environment_nilm_server.yml
```

在这两个文件经服务器验证前，当前 `environment_server.yml` 只能标为“初版参考”，不能
标为“正式冻结 GPU 环境”。

### 4.6 整理 Slurm 模板

进入 C1 前必须逐个检查：

- `slurm/env.sh`：当前含旧项目路径，应标记 legacy/do-not-use，或改成不写死用户目录；
- `slurm/detsec_pc.sbatch`：替换 `/home/<user>`，加入环境记录和 GPU 检查；
- `slurm/seq2point.sbatch`：同上；
- 每个作业开头 `module purge`，再 load 现场存在的准确模块；
- 禁止 `--exclusive`；
- 训练不在 `slogin` 执行；
- 输出目录含 run id 和 Slurm job id，避免覆盖；
- 正式作业先检查 manifest，校验失败立即退出；
- DETSEC-PC GPU 冒烟要测试 `--deterministic-tf`。若该模式在现场报不支持，则记录原因并
  用固定 seed 的重复运行衡量波动，不能悄悄去掉。

不要把用户名、密码、私钥、访问令牌写进 Slurm 文件。

### 4.7 建立唯一的 run id 规则

建议：

```text
<task>_<method>_k<k>_r<ratio>_s<seed>_<yyyymmdd>
```

例如：

```text
detsecpc_k4_trainonly_s17_20260920
s2p_b2m_k4_r0p5_s17_20260920
```

每个 run 独立保存：配置、commit、数据 manifest、环境、job id、stdout/stderr、history、
checkpoint、预测、指标和简短总结。失败运行也保留失败原因，不覆盖成“成功”。

---

## 5. 所有预计需要的脚本文件

### 5.1 已存在且 C1–C3 会直接使用

| 文件 | 用途 | 当前判断 |
|---|---|---|
| `scripts/train_state_discovery.py` | 跑 DETSEC-PC/physical-stats 状态发现 | 已存在；正式前做 GPU 冒烟 |
| `scripts/build_state_library.py` | 把聚类结果整理成状态库 | 已存在 |
| `scripts/compare_detsec_vs_pcdetsec.py` | 比较特征方案 | 已存在 |
| `scripts/compare_cluster_methods.py` | 比较聚类结果 | 已存在 |
| `scripts/evaluate_state_exchangeability.py` | 检查状态是否适合互换 | 已存在 |
| `scripts/prepare_nilm_b0_b2_inputs.py` | 准备 B0/B1/B2 输入 | 已存在 |
| `scripts/place_b1_b2_on_train_background.py` | 把周期放入共同背景 | 已存在 |
| `scripts/train_nilm.py` | 训练 Seq2Point | 已存在并完成本机冒烟 |
| `scripts/predict_nilm.py` | 生成 validation 预测 | 已存在 |
| `scripts/evaluate_nilm_predictions.py` | 统一评价预测 | 已存在 |
| `scripts/run_cpu_nilm_smoke.py` | 小规模端到端检查 | 已存在 |
| `scripts/build_server_transfer_manifest.py` | 制作 B0/B1/B2 train+validation 包 | 已存在；不覆盖状态发现包 |
| `scripts/verify_server_manifest.py` | 服务器逐文件验 SHA-256 | 已存在 |

### 5.2 进入 C1 前应新增或扩展

这些是真正的近期开发任务，建议按顺序完成：

| 优先级 | 预计文件 | 需要完成的功能 | 验收方法 |
|---:|---|---|---|
| P0 | `scripts/build_c1_server_bundle.py` | 支持 `state-discovery` 与 `nilm-b0b2` 两种 profile，默认排除 test | 临时目录构建；路径/字节/哈希测试 |
| P0 | `scripts/server_preflight.py` | 检查目录、commit、manifest、test 路径、可写输出目录和依赖导入 | 任一硬门槛失败时非0退出 |
| P0 | `scripts/gpu_framework_smoke.py` | 分别验证 TensorFlow/PyTorch GPU，做极小前向与反向 | 输出设备名、显存、框架版本和 PASS/FAIL |
| P0 | `scripts/capture_server_environment.py` | 把 Python、包、环境变量白名单、Git、CUDA/GPU、Slurm 信息写成 JSON/TXT | 同一 run 可重现、无密钥 |
| P1 | `scripts/collect_run_artifacts.py` | 检查一个 run 是否含配置、日志、checkpoint、预测、指标 | 缺文件时列清单并非0退出 |
| P1 | `scripts/audit_no_test_access.py` | 汇总配置、manifest、运行记录中的 test 访问证据 | 生成可审计 JSON 报告 |

实现建议：先扩展已有 manifest 代码中的公共函数，不要复制两套 SHA-256 逻辑。所有脚本都应
支持 `--help`、明确输入/输出、失败返回非0，并有单元测试。

### 5.3 C1 通过后、进入 B3 前需要

| 文件 | 对应方案 | 简单作用 |
|---|---|---|
| `src/generation/schema.py` | B3–B5 共用 | 定义每条合成数据必须记录什么 |
| `src/generation/provenance.py` | B3–B5 共用 | 记录样本来源、参数和哈希 |
| `src/generation/full_cycle_transform.py` | B3-T | 真实周期受限时间/功率变换 |
| `src/generation/full_cycle_cvae.py` | B3-V | 完整周期条件 CVAE |
| `src/generation/full_cycle_wgan.py` | B3-G | TraceGAN-style/SGAN-inspired 条件 WGAN |
| `src/generation/full_cycle_diffusion.py` | B3-D | 完整周期条件扩散 |
| `scripts/train_full_cycle_generator.py` | B3-V/G/D | 统一训练入口 |
| `scripts/generate_full_cycles.py` | B3-T/V/G/D | 统一采样与 provenance 输出 |
| `scripts/evaluate_synthetic_quality.py` | B3–B5 | 物理、分布、多样性和边界质量 |
| `scripts/audit_generation_memorization.py` | B3–B5 | 检查复制和过近训练样本 |
| `scripts/build_shared_placement_schedule.py` | B1–B5 | 冻结共同背景和放置位置 |

对应配置建议：

```text
config/generation/full_cycle_transform.yaml
config/generation/full_cycle_cvae.yaml
config/generation/full_cycle_wgan.yaml
config/generation/full_cycle_diffusion.yaml
```

### 5.4 进入 B4/B5 前需要

| 文件 | 对应方案 | 简单作用 |
|---|---|---|
| `src/generation/primitive_cvae.py` | B4 | 按状态生成短基元 |
| `scripts/train_primitive_generator.py` | B4 | 训练共享条件基元 CVAE |
| `scripts/generate_primitive_cycles.py` | B4 | 生成状态并做基础拼接 |
| `src/composition/transition_model.py` | B5 消融 | 简单 Markov 顺序模型 |
| `src/composition/duration_model.py` | B5 | 学习各状态持续多久 |
| `src/composition/hsmm_sequence.py` | B5 | 同时采样状态顺序和持续时间 |
| `src/composition/boundary_handler.py` | B5 | 可消融的端点/斜率约束 |
| `src/composition/constrained_composer.py` | B5 | 把 HSMM 和生成基元组合起来 |
| `scripts/fit_hsmm_composer.py` | B5 | 只用 train 状态序列拟合 HSMM |
| `scripts/compose_generated_cycles.py` | B5 | 生成最终受约束完整周期 |

对应配置建议：

```text
config/generation/primitive_cvae.yaml
config/composition/b4_basic.yaml
config/composition/b5_markov.yaml
config/composition/b5_hsmm.yaml
config/composition/b5_hsmm_endpoint.yaml
```

### 5.5 预计的 Slurm 文件

```text
slurm/c1_gpu_smoke.sbatch
slurm/c2_detsec_pc.sbatch
slurm/c3_seq2point.sbatch
slurm/b3_transform.sbatch
slurm/b3_cvae.sbatch
slurm/b3_wgan.sbatch
slurm/b3_diffusion.sbatch
slurm/b4_primitive_cvae.sbatch
slurm/b5_hsmm.sbatch
```

这些模板要在对应 Python 入口存在且本机冒烟通过后再添加。空 sbatch 文件没有价值。

---

## 6. 上传服务器前的本机最终验收清单

只有全部勾选，才进入服务器 C1：

- [ ] 全量测试和 compileall 通过；
- [ ] YAML 全部可解析；
- [ ] Git 工作树干净；
- [ ] 最终 commit 和 `phase-c0-freeze` 标签已记录；
- [ ] 分支/标签可被服务器取得，或代码归档哈希已保存；
- [ ] 状态发现包只含 train；
- [ ] NILM 开发包只含 train+validation；
- [ ] 两个包的 test 路径数量都是0；
- [ ] 两个 manifest 都在本机完成逐文件校验；
- [ ] TensorFlow 与 PyTorch 环境文件分开，或已有充分理由维持单环境；
- [ ] 旧 `slurm/env.sh` 不会被正式脚本 source；
- [ ] sbatch 中没有 `/home/<user>`、旧仓库名、密码或 token；
- [ ] 所有输出目录按 run id 隔离；
- [ ] 本机保留数据包、manifest、哈希和冻结代码的备份。

---

## 7. 上服务器后的全部操作

### 7.1 第一步：通过批准的入口登录

使用实验室 JumpServer。指南记录的形式为：

```text
ssh -p 2222 "<jumpserver_user>@<system_user>@172.28.255.242@lab.networkctl.cn"
```

用户名和资产地址以当前控制台为准，不复制手册示例凭据。不要把密码写入本文、Git、终端
脚本或日志。

### 7.2 在 slogin 只做只读调查

```bash
hostname
date --iso-8601=seconds
id
pwd
sinfo
squeue -u "$USER"
module avail
module list
df -h "$HOME"
quota -s 2>/dev/null || true
```

记录分区名、可用节点、时间限制、home 配额和准确 module 名。不要在 slogin 运行训练、
大规模解压或长时间哈希任务。

### 7.3 建立目录

```bash
PROJECT_ROOT="$HOME/projects/PSLG-NILM"
DATA_ROOT="$HOME/pslg_data"
ARTIFACT_ROOT="$HOME/pslg_artifacts"
LOG_ROOT="$HOME/pslg_logs"
MANIFEST_ROOT="$HOME/pslg_manifests"

mkdir -p "$PROJECT_ROOT" "$DATA_ROOT" "$ARTIFACT_ROOT" "$LOG_ROOT" "$MANIFEST_ROOT"
```

这里是创建专用目录，不删除服务器已有内容。如果目录已存在，先检查归属和内容。

### 7.4 取得冻结代码

Git 可用时：

```bash
git clone <approved-repository-url> "$PROJECT_ROOT"
cd "$PROJECT_ROOT"
git fetch --tags
git checkout phase-c0-freeze
git rev-parse HEAD
git status --short
```

若使用代码归档，则在专用空目录解压，复算归档 SHA-256，并确认 `git_commit.txt`。服务器
看到的 commit 必须与本机冻结记录一致。

### 7.5 上传两个数据包

小包可走 JumpServer SFTP；更大数据按实验室批准的 HTTP/对象存储方式传输。上传到
`$DATA_ROOT` 或保持 manifest 中的项目相对结构。不要上传最终 test。

完成后不要凭文件大小“看起来差不多”就开始训练，必须复算哈希：

```bash
cd "$PROJECT_ROOT"
python scripts/verify_server_manifest.py \
  --manifest "$MANIFEST_ROOT/c1_state_discovery_trainonly_manifest.json" \
  --repo-root "$PROJECT_ROOT" \
  --report "$MANIFEST_ROOT/c1_state_discovery_verify.json"

python scripts/verify_server_manifest.py \
  --manifest "$MANIFEST_ROOT/c1_nilm_b0_b2_trainval_manifest.json" \
  --repo-root "$PROJECT_ROOT" \
  --report "$MANIFEST_ROOT/c1_nilm_verify.json"
```

如果数据放在项目外，需要让 bundle builder/verify 脚本支持明确的数据根目录，不要靠符号
链接掩盖路径差异后不记录。

### 7.6 申请短时交互 GPU

先用默认 RTX3090：

```bash
srun -p RTX3090 --gres=gpu:1 -c 8 --mem=32G -t 00:30:00 --pty /bin/bash
```

若 DETSEC-PC 显存不够，再用 A6000：

```bash
srun -p A6000 --gres=gpu:1 -c 8 --mem=64G -t 00:30:00 --pty /bin/bash
```

进入计算节点后记录：

```bash
hostname
echo "$CUDA_VISIBLE_DEVICES"
nvidia-smi
```

禁止 `--exclusive`。如果分区名和指南不同，以 `sinfo` 现场结果为准。

### 7.7 加载模块并创建环境

先查看现场模块，然后使用真实存在的版本：

```bash
module purge
module avail
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
module list
```

指南中的版本只是参考。若现场不同，使用现场版本并记录。推荐分别创建：

```bash
conda env create -f environment_detsec_server.yml
conda env create -f environment_nilm_server.yml
```

如果环境已存在，不要直接覆盖；先导出、比较，再决定更新或新建带版本后缀的环境。

### 7.8 验证 TensorFlow GPU

```bash
source "$(conda info --base)/bin/activate"
conda activate pslg-detsec
python - <<'PY'
import tensorflow as tf
print("tensorflow", tf.__version__)
print("gpus", tf.config.list_physical_devices("GPU"))
assert tf.config.list_physical_devices("GPU"), "TensorFlow did not detect a GPU"
PY
```

仅看到 `nvidia-smi` 不算通过；必须是 TensorFlow 自己列出 GPU，并完成一个极小前向/反向。

### 7.9 验证 PyTorch GPU

```bash
conda activate pslg-nilm
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
assert torch.cuda.is_available(), "PyTorch did not detect a GPU"
print("device", torch.cuda.get_device_name(0))
x = torch.randn(8, 16, device="cuda", requires_grad=True)
(x.square().mean()).backward()
print("forward_backward=PASS")
PY
```

### 7.10 保存环境快照

每个环境至少保存：

```bash
module list 2> "$ARTIFACT_ROOT/c1_module_list.txt"
conda env export --no-builds > "$ARTIFACT_ROOT/c1_environment.yml"
python -m pip freeze > "$ARTIFACT_ROOT/c1_pip_freeze.txt"
nvidia-smi -q > "$ARTIFACT_ROOT/c1_nvidia_smi.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD > "$ARTIFACT_ROOT/c1_git_commit.txt"
```

两个 conda 环境应分别导出，文件名中写清 `detsec` 或 `nilm`。输出中如意外包含密钥、代理
凭据或私有 token，必须先清理再回传和追踪。

### 7.11 C1 冒烟阶梯

严格按下列顺序：

1. manifest 校验；
2. 全量单元测试；
3. CPU 读取一个 train shard 和一个 validation shard；
4. TensorFlow 单 batch GPU 前向/反向；
5. PyTorch 单 batch GPU 前向/反向；
6. 8-cycle DETSEC-PC，k=3/4/5，确认 `test_accessed=false`；
7. Seq2Point 短 epoch；
8. 保存 checkpoint；
9. 从 checkpoint 恢复并继续；
10. 只对冻结 validation indices 推理和评价；
11. 重跑同 seed，检查结果波动和确定性记录。

任何一步失败都停在该层修复，不直接提交正式网格。

### 7.12 提交 C2 正式 DETSEC-PC

C1 全部通过后再提交：

```bash
sbatch slurm/c2_detsec_pc.sbatch
squeue -u "$USER"
```

C2 要求：

- 只读 train-only state-discovery manifest；
- 正式 DETSEC-PC 特征；
- k=3/4/5；
- 每个 run 输出独立目录；
- 保存随机种子、deterministic 设置、环境、commit 和数据哈希；
- validation 只负责选择最终 k，不回流重训聚类；
- 产出 state library v1、状态统计、可交换性报告和 comparison report。

### 7.13 C2 完成后提交 C3

状态库冻结后，先运行 ratio=0.5：

```text
组：B0、B1、B2-random、最佳 B2-matched
seed：17、42、73
模型：同一 Seq2Point
评价：同一 validation monitor
```

第一轮稳定后才扩展 ratio=1 和2。不要在第一轮失败时一口气提交更多作业掩盖问题。

### 7.14 查看、取消和诊断作业

```bash
squeue -u "$USER"
scontrol show job <job_id>
tail -n 200 "$LOG_ROOT/<job>.out"
tail -n 200 "$LOG_ROOT/<job>.err"
scancel <job_id>
```

判断顺序：

- Pending：看 `Reason`，不要重复提交相同作业；
- GPU OOM：先降 batch/序列长度，再考虑 A6000；
- 系统内存 OOM：检查数据复制和 worker 数；
- 动态库错误：核对 `module list`、框架 CUDA 和驱动；
- 数据错误：重验 manifest，不临时复制未知文件；
- 超时：从 checkpoint 恢复，同一 run id 标记续跑；
- 代码错误：回本机修复、测试、提交，再同步服务器，避免服务器产生“幽灵版本”。

### 7.15 下载产物回本机

每个完成 run 至少带回：

```text
config.yaml
git_commit.txt
data_manifest.json
environment.txt / environment.yml
module_list.txt
slurm_job.json 或 job id
stdout.log / stderr.log
history.csv
best checkpoint 或 checkpoint hash
predictions 或 prediction manifest
metrics.json
run_summary.md
```

大 checkpoint 可先只传 SHA-256 和存储位置，但论文最终复现所需 checkpoint 必须进入可靠
备份。回传后本机再次复算哈希。

---

## 8. C1 正式完成的判定标准

以下全部满足，才能把 C1 标成完成：

- [ ] 服务器 commit 与本机冻结 commit 一致；
- [ ] 工作树干净；
- [ ] 两个数据包哈希全部通过；
- [ ] 开发服务器上没有最终 test shard；
- [ ] TensorFlow 能看到 GPU 并完成反向传播；
- [ ] PyTorch 能看到 GPU 并完成反向传播；
- [ ] 131项或更新后的全部测试通过；
- [ ] DETSEC-PC 8-cycle 冒烟通过；
- [ ] Seq2Point 训练、保存、恢复、validation 推理通过；
- [ ] 环境、module、GPU、Slurm、Git 和数据 manifest 已保存；
- [ ] 无密码、token、私钥进入日志或 Git；
- [ ] 已记录真实耗时和峰值显存，用于调整正式资源；
- [ ] C2 sbatch 在短样本模式下至少成功完成一次。

### C1 完成后立即冻结的记录

```text
reports/server_c1/<date>/
  c1_summary.md
  git_commit.txt
  module_list.txt
  detsec_environment.yml
  nilm_environment.yml
  nvidia_smi.txt
  state_discovery_verify.json
  nilm_verify.json
  gpu_smoke.json
  smoke_run_ids.txt
```

---

## 9. 现在最合理的实际执行顺序

1. 实现 `build_c1_server_bundle.py`，先补齐状态发现 train-only 包；
2. 实现 `server_preflight.py` 和 `gpu_framework_smoke.py`；
3. 拆分 TensorFlow/PyTorch 服务器环境文件；
4. 清理或禁用旧 `slurm/env.sh`，现场化两个 C1/C2/C3 sbatch；
5. 本机全测并生成两个 bundle/manifest；
6. 提交、建立新的 `phase-c0-freeze` 标签并同步远程；
7. 登录服务器完成只读调查和目录建立；
8. 传代码与无 test 数据包，复算哈希；
9. 申请30分钟交互 GPU，按冒烟阶梯逐项通过；
10. 冻结 C1 环境和记录；
11. 提交 C2 DETSEC-PC；
12. 冻结状态库后提交 C3 B0/B1/B2。

最先应写的是第1项，而不是 B3 的 GAN 或扩散模型。原因很简单：没有正式状态发现包，
C2 无法可靠运行；而 B4/B5 又依赖 C2 冻结出的状态定义。

---

## 10. 与后续 B3/B4/B5 的衔接

C3 完成后执行：

- B3-T：真实完整周期受限变换；
- B3-V：完整周期条件 CVAE；
- B3-G：TraceGAN-style/SGAN-inspired 条件 1D WGAN；
- B3-D：完整周期条件扩散；
- B4：共享条件基元 CVAE + 基础拼接；
- B5：Markov 消融、HSMM 顺序/持续时间、可选端点约束。

Markov/HSMM 不与 GAN、扩散争夺“波形生成器”的位置。它负责决定“先出现哪个状态、持续
多久”；CVAE/GAN/扩散负责生成“这个状态或完整周期具体长什么样”。这个职责划分正是本项目
创新点能否被清楚验证的关键。

最终核心比较保持不变：在完全相同的真实 train 数据、合成数量、背景、放置位置、NILM
预算和真实 test 上，比较普通完整周期生成与基元生成/拼接谁能带来更好的下游 NILM 效果，
再分析 HSMM、持续时间、端点和上下文约束各自贡献了多少。

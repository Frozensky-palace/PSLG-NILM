# PSLG-NILM 从 C1 到 Phase G 的人工操作指南

> 版本：v1.1  
> 日期：2026-09-21  
> 冻结起点：`phase-c0-freeze` / `e5f0ae3`  
> 用途：指导研究者在本机与 Slurm 服务器之间完成 C1、C2、C3、C4、C5、D、E、F、G。  
> 重要边界：本文中的 `test` 始终指最终真实测试集。在 G 冻结前不得上传、读取或评价。

---

## 1. 先回答：C1 到 G 是否全程都需要服务器

不需要全程都在服务器，但正式重计算最好都在服务器完成。

| 工作 | 推荐位置 | 原因 |
|---|---|---|
| 修改配置、写代码、单元测试、小数据冒烟 | 本机 | 反馈快，不占 GPU 排队时间 |
| C1 查询服务器、装环境、GPU 冒烟 | 服务器 | 必须验证真实驱动、CUDA、GPU 和 Slurm |
| C2 DETSEC-PC 正式状态发现 | 服务器 GPU | TensorFlow 训练量大 |
| C3/C4 Seq2Point 多组、多 seed | 服务器 GPU | 重复训练多，适合 Slurm 并行 |
| C5 比较、作图、决定是否进入生成 | 本机为主 | 主要是读取结果和做研究判断 |
| D 的 B3-T 真实周期变换 | 本机或服务器 CPU | 不训练神经网络；数据已在服务器时可直接在那里做 |
| D 的 CVAE/GAN/扩散 | 服务器 GPU | 神经生成模型训练较重 |
| E 基元生成 | 服务器 GPU | 要训练共享条件基元生成器 |
| F HSMM 拟合 | 本机或服务器 CPU | HSMM 本身较轻；大批量生成可放服务器 |
| F 生成数据后的 NILM 重训 | 服务器 GPU | 仍然是多组 Seq2Point |
| G validation 汇总与协议冻结 | 本机为主 | 不需要 GPU，且方便人工审阅 |
| G 最终 test 推理 | 服务器 GPU 推荐 | 统一环境、速度快；只能在冻结后执行一次 |
| 论文表格、统计、图表、归档 | 本机 | 便于审查和写作 |

最稳妥的协作方式是：**本机负责做决定和保存主记录，服务器负责重计算。** 每轮服务器结果
都同步回本机，再决定下一阶段，而不是一直在服务器目录里临时修改。

---

## 2. 当前真实可运行范围

截至本版本：

- C1 辅助工具、双数据包、双环境和 GPU 冒烟已准备；
- C2 正式状态发现入口已经存在；
- C3 B0/B1/B2 Seq2Point 入口已经存在；
- C4 可复用 C3 入口，但需要为 ratio=1、2 重新准备冻结输入和矩阵；
- C5 是分析决策，不需要新训练器；
- D–F 的研究方案已经冻结，但具体 CVAE/WGAN/扩散、基元生成和 HSMM 入口尚未实现；
- G 的冻结辅助脚本已经具备，但 B3–B5 最终训练入口完成后还要扩展 NILM arm 抽象。

因此，现在可以真正执行 C1 → C2 → C3。到 D 前必须先在本机实现相应生成代码和测试。

---

## 3. 第一次登录：选择仓库名、克隆并初始化目录

下面的命令在服务器登录节点执行。假设原来已经有 `PSLG-NILM` 文件夹，因此新仓库使用
`PSLG-NILM-c1`。仓库名可以再改，但确定后后续保持一致。

### 3.1 先设置两个需要人工填写的值

```bash
export PSLG_REPO_URL='把这里替换成真实Git仓库地址'
export PSLG_REPO_DIRNAME='PSLG-NILM-c1'
export PSLG_PROJECT_ROOT="$HOME/projects/$PSLG_REPO_DIRNAME"
```

检查即将使用的路径：

```bash
printf 'repo_url=%s\nproject_root=%s\n' \
  "$PSLG_REPO_URL" "$PSLG_PROJECT_ROOT"
```

如果输出中仍有“把这里替换”，先停止，不要执行 clone。

### 3.2 检查同名目录，禁止盲目覆盖

```bash
mkdir -p "$HOME/projects"

if [ -e "$PSLG_PROJECT_ROOT" ]; then
  echo "目标已经存在：$PSLG_PROJECT_ROOT"
  ls -la "$PSLG_PROJECT_ROOT"
else
  echo "目标不存在，可以克隆"
fi
```

根据结果选择一种情况：

#### 情况 A：目录不存在

```bash
git clone "$PSLG_REPO_URL" "$PSLG_PROJECT_ROOT"
```

#### 情况 B：目录已经是这个项目的 Git 仓库

```bash
git -C "$PSLG_PROJECT_ROOT" status
git -C "$PSLG_PROJECT_ROOT" remote -v
git -C "$PSLG_PROJECT_ROOT" fetch --all --tags
```

不要再次 clone，也不要删除目录。

#### 情况 C：目录存在，但不是这个项目

换一个目录名，例如：

```bash
export PSLG_REPO_DIRNAME='PSLG-NILM-c1-new'
export PSLG_PROJECT_ROOT="$HOME/projects/$PSLG_REPO_DIRNAME"
git clone "$PSLG_REPO_URL" "$PSLG_PROJECT_ROOT"
```

### 3.3 检出本轮服务器版本

本轮新增手册和辅助脚本位于 `phase-c0-freeze` 之后。先在本机提交并建立新标签，例如
`phase-c1-ops-ready`，推送后在服务器执行：

```bash
cd "$PSLG_PROJECT_ROOT"
git fetch --all --tags
git checkout phase-c1-ops-ready
git status --short
git rev-parse HEAD
```

`git status --short` 必须没有输出。

### 3.4 一次性初始化项目路径

仓库中的 `slurm/project_paths.sh` 会根据自身位置自动识别仓库根目录，因此仓库叫
`PSLG-NILM-c1` 或其他名字都可以：

```bash
cd "$PSLG_PROJECT_ROOT"
source slurm/project_paths.sh
```

它会定义：

```text
PSLG_PROJECT_ROOT   当前真实仓库目录
PSLG_DATA_ROOT      非 Git 数据或传输归档
PSLG_ARTIFACT_ROOT  checkpoint、预测、指标
PSLG_LOG_ROOT       Slurm 日志
PSLG_MANIFEST_ROOT  数据 manifest 和服务器矩阵副本
PSLG_REGISTRY_ROOT  Slurm job id 注册表
```

脚本会自动创建除仓库外的五个目录，不会删除或覆盖已有文件。

### 3.5 以后每次重新登录只执行这两行

```bash
cd "$HOME/projects/PSLG-NILM-c1"
source slurm/project_paths.sh
```

如果你选择的不是 `PSLG-NILM-c1`，第一行换成真实仓库目录。`source` 定义的变量只在当前
登录会话有效，所以重新登录需要再次执行；不用重复 clone，也不用重复创建环境。

### 3.6 在本机打包 C1 数据

下面命令在 Windows 本机 PowerShell 执行，不是在服务器执行：

```powershell
Set-Location 'D:\zhj\vscode\PSLG-NILM'

$pslgBundleDir = 'D:\zhj\vscode\PSLG-NILM\server_bundles'
New-Item -ItemType Directory -Force -Path $pslgBundleDir | Out-Null

.\.venv\Scripts\python.exe scripts\package_manifest_bundle.py `
  --manifest manifests\c1_state_discovery_trainonly_manifest.json `
  --repo-root . `
  --output "$pslgBundleDir\c1_state_discovery_trainonly.tar.gz"

.\.venv\Scripts\python.exe scripts\package_manifest_bundle.py `
  --manifest manifests\c1_nilm_b0_b2_trainval_manifest.json `
  --repo-root . `
  --output "$pslgBundleDir\c1_nilm_b0_b2_trainval.tar.gz"

Copy-Item manifests\c1_state_discovery_trainonly_manifest.json $pslgBundleDir
Copy-Item manifests\c1_nilm_b0_b2_trainval_manifest.json $pslgBundleDir
Get-ChildItem $pslgBundleDir
```

脚本会先按 manifest 重新验证每个源文件，再生成压缩包和 `.sha256` 文件。如果哈希不一致，
打包会直接停止。

### 3.7 通过 JumpServer SFTP 上传

在 JumpServer 文件管理/SFTP 页面，把 `server_bundles` 中以下6个文件上传到服务器：

```text
c1_state_discovery_trainonly.tar.gz
c1_state_discovery_trainonly.tar.gz.sha256
c1_state_discovery_trainonly_manifest.json
c1_nilm_b0_b2_trainval.tar.gz
c1_nilm_b0_b2_trainval.tar.gz.sha256
c1_nilm_b0_b2_trainval_manifest.json
```

服务器目标目录使用：

```text
$HOME/pslg_data/uploads/
```

上传完成后，在服务器执行：

```bash
source "$PSLG_PROJECT_ROOT/slurm/project_paths.sh"
export PSLG_UPLOAD_ROOT="$PSLG_DATA_ROOT/uploads"
mkdir -p "$PSLG_UPLOAD_ROOT"
ls -lh "$PSLG_UPLOAD_ROOT"
```

如果 SFTP 实际上传到了 `$HOME`，移动到专用目录：

```bash
mv "$HOME/c1_state_discovery_trainonly.tar.gz"* "$PSLG_UPLOAD_ROOT/"
mv "$HOME/c1_nilm_b0_b2_trainval.tar.gz"* "$PSLG_UPLOAD_ROOT/"
mv "$HOME/c1_state_discovery_trainonly_manifest.json" "$PSLG_UPLOAD_ROOT/"
mv "$HOME/c1_nilm_b0_b2_trainval_manifest.json" "$PSLG_UPLOAD_ROOT/"
```

只有确认文件确实位于 `$HOME` 时才能执行这组 `mv`；否则先用 `find "$HOME" -maxdepth 2
-name 'c1_*' -type f` 查找。

### 3.8 在服务器校验压缩包并安全解包

```bash
cd "$PSLG_UPLOAD_ROOT"
sha256sum -c c1_state_discovery_trainonly.tar.gz.sha256
sha256sum -c c1_nilm_b0_b2_trainval.tar.gz.sha256
```

两行都必须显示 `OK`。然后使用项目脚本解包；它不会覆盖内容不同的现有文件：

```bash
cd "$PSLG_PROJECT_ROOT"

python scripts/unpack_manifest_bundle.py \
  --archive "$PSLG_UPLOAD_ROOT/c1_state_discovery_trainonly.tar.gz" \
  --manifest "$PSLG_UPLOAD_ROOT/c1_state_discovery_trainonly_manifest.json" \
  --repo-root "$PSLG_PROJECT_ROOT"

python scripts/unpack_manifest_bundle.py \
  --archive "$PSLG_UPLOAD_ROOT/c1_nilm_b0_b2_trainval.tar.gz" \
  --manifest "$PSLG_UPLOAD_ROOT/c1_nilm_b0_b2_trainval_manifest.json" \
  --repo-root "$PSLG_PROJECT_ROOT"
```

复制 manifest 到后续固定位置：

```bash
cp "$PSLG_UPLOAD_ROOT/c1_state_discovery_trainonly_manifest.json" \
  "$PSLG_MANIFEST_ROOT/"
cp "$PSLG_UPLOAD_ROOT/c1_nilm_b0_b2_trainval_manifest.json" \
  "$PSLG_MANIFEST_ROOT/"
```

目录含义：

- `projects`：Git 代码；
- `pslg_data`：不进 Git 的输入数据；
- `pslg_artifacts`：checkpoint、预测、指标和环境记录；
- `pslg_logs`：Slurm stdout/stderr；
- `pslg_manifests`：数据清单和哈希；
- `pslg_registries`：批量提交后记录的 job id。

不要在代码目录里混放大型 checkpoint，也不要在不同 run 之间复用可覆盖的输出目录。
有一个例外：当前 C1 manifest 记录的是“项目相对路径”，所以解包后的 train/validation 输入
必须恢复到 `$PSLG_PROJECT_ROOT/reports/...` 的原相对位置，预检才能逐文件核验。`pslg_data`
用于保存传输归档或未来采用独立 data-root 协议的数据，不能把当前 bundle 随意挪过去。

---

## 4. 每次登录服务器都先做的五分钟检查

```bash
cd "$HOME/projects/PSLG-NILM-c1"
source slurm/project_paths.sh

hostname
date --iso-8601=seconds
sinfo
squeue -u "$USER"
module avail
df -h "$HOME"
cd "$PSLG_PROJECT_ROOT"
git status --short
git rev-parse HEAD
```

正常标准：

- Git 工作树为空；
- commit 是准备执行的冻结版本；
- home 还有足够空间；
- 分区名称和 module 名与 sbatch 一致；
- 登录节点只做查看、提交和小文件操作，不直接训练。

若服务器 module 版本与脚本不同，先在服务器本地副本中验证，再把确定修改同步回本机 Git。

---

## 5. 辅助脚本如何使用

### 5.1 自动生成服务器专用矩阵

不需要手工修改 YAML 里的用户名和仓库名。确认数据已经恢复到 manifest 指定的项目相对
路径后，执行：

```bash
cd "$PSLG_PROJECT_ROOT"

export PSLG_NILM_EXPERIMENT_DIR="$PSLG_PROJECT_ROOT/reports/core_validation/ukdale_b1_washing_machine/b2_policy_ablation_seed17_r0p5/k4_feat_duration_ratio_0p67_1p5/inputs"

python scripts/prepare_server_matrices.py \
  --project-root "$PSLG_PROJECT_ROOT" \
  --output-dir "$PSLG_MANIFEST_ROOT" \
  --manifest-root "$PSLG_MANIFEST_ROOT" \
  --artifact-root "$PSLG_ARTIFACT_ROOT" \
  --nilm-experiment-dir "$PSLG_NILM_EXPERIMENT_DIR" \
  --ratio 0p5
```

将生成：

```text
$PSLG_MANIFEST_ROOT/c1_gpu_smoke.server.yaml
$PSLG_MANIFEST_ROOT/c2_detsec.server.yaml
$PSLG_MANIFEST_ROOT/c3_seq2point_r0p5.server.yaml
```

这些服务器专用副本自动写入当前 Git commit 和真实仓库路径；不要加入 Git。

检查其中已经没有占位符：

```bash
grep -R "CHANGE_ME" "$PSLG_MANIFEST_ROOT"/*.server.yaml && \
  echo "仍有占位符，禁止提交" || echo "矩阵路径已填写"
```

### 5.2 矩阵提交脚本

`scripts/submit_slurm_matrix.py` 用一个 YAML 同时描述多组作业。默认只打印命令，不提交：

```bash
python scripts/submit_slurm_matrix.py \
  --matrix "$PSLG_MANIFEST_ROOT/c3_seq2point_r0p5.server.yaml" \
  --registry "$PSLG_REGISTRY_ROOT/c3_jobs.json"
```

确认九条命令、路径、seed 和 arm 都正确后，才增加：

```bash
python scripts/submit_slurm_matrix.py \
  --matrix "$PSLG_MANIFEST_ROOT/c3_seq2point_r0p5.server.yaml" \
  --registry "$PSLG_REGISTRY_ROOT/c3_jobs.json" \
  --submit
```

脚本不使用 shell 字符串执行；值中出现逗号、换行或非法变量名会被拒绝。含 `CHANGE_ME`
的矩阵不能真正提交。

### 5.3 作业状态脚本

```bash
python scripts/query_slurm_registry.py \
  --registry "$PSLG_REGISTRY_ROOT/c3_jobs.json" \
  --output "$PSLG_REGISTRY_ROOT/c3_status.json"
```

它同时读取 `squeue` 和 `sacct`。通俗地说，`squeue` 看现在还在排队/运行什么，`sacct`
看已经结束的作业是成功、失败、超时还是被取消。

### 5.4 作业产物收口

`scripts/finalize_server_run.py` 会复制数据 manifest、环境和日志，写 commit、run summary，
并给 run 内每个文件计算 SHA-256。C2/C3 模板已经自动调用。

### 5.5 无 test 审计

```bash
python scripts/audit_no_test_access.py \
  --manifest "$PSLG_MANIFEST_ROOT/c1_state_discovery_trainonly_manifest.json" \
  --manifest "$PSLG_MANIFEST_ROOT/c1_nilm_b0_b2_trainval_manifest.json" \
  --run-dir "$PSLG_ARTIFACT_ROOT" \
  --output "$PSLG_ARTIFACT_ROOT/no_test_audit.json"
```

在 G 冻结前，这个报告必须是 `all_clean: true`。

---

## 6. C1：服务器环境冻结

### 6.1 目标

证明代码、数据、TensorFlow GPU、PyTorch GPU、checkpoint 和 validation 推理在真实集群上
都能工作。C1 不追求最终指标。

### 6.2 确认代码和路径

```bash
cd "$PSLG_PROJECT_ROOT"
test -d .git
test -f scripts/server_preflight.py
test -f scripts/prepare_server_matrices.py
test -f slurm/project_paths.sh
git status --short
git rev-parse HEAD
git tag --points-at HEAD
```

前三个 `test` 命令应无错误，`git status --short` 应无输出，标签应是本轮新建的服务器操作
冻结标签，而不是旧的 `phase-c0-freeze`。

### 6.3 校验两个数据包

先确认 manifest 文件存在：

```bash
ls -lh \
  "$PSLG_MANIFEST_ROOT/c1_state_discovery_trainonly_manifest.json" \
  "$PSLG_MANIFEST_ROOT/c1_nilm_b0_b2_trainval_manifest.json"
```

```bash
python scripts/verify_server_manifest.py \
  --manifest "$PSLG_MANIFEST_ROOT/c1_state_discovery_trainonly_manifest.json" \
  --repo-root "$PSLG_PROJECT_ROOT" \
  --report "$PSLG_MANIFEST_ROOT/c1_state_verify.json"

python scripts/verify_server_manifest.py \
  --manifest "$PSLG_MANIFEST_ROOT/c1_nilm_b0_b2_trainval_manifest.json" \
  --repo-root "$PSLG_PROJECT_ROOT" \
  --report "$PSLG_MANIFEST_ROOT/c1_nilm_verify.json"
```

两次必须都是全部通过，而且 test 路径数为0。

### 6.4 创建环境

在短时计算节点执行，不在 slogin 长时间装包：

```bash
srun -p RTX3090 --gres=gpu:1 -c 8 --mem=32G -t 01:00:00 --pty /bin/bash
module purge
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
source "$(conda info --base)/bin/activate"
cd "$PSLG_PROJECT_ROOT"

conda env create -f environment_detsec_server.yml
conda env create -f environment_nilm_server.yml
```

版本以现场 `module avail` 为准。若调整环境文件，必须回传本机并重新提交，不能只在服务器
手改后忘记记录。

### 6.5 预览并提交 GPU 冒烟

如果还没有生成服务器矩阵，先执行第5.1节的 `prepare_server_matrices.py`。然后：

```bash
cd "$PSLG_PROJECT_ROOT"
python scripts/submit_slurm_matrix.py \
  --matrix "$PSLG_MANIFEST_ROOT/c1_gpu_smoke.server.yaml" \
  --registry "$PSLG_REGISTRY_ROOT/c1_jobs.json"

python scripts/submit_slurm_matrix.py \
  --matrix "$PSLG_MANIFEST_ROOT/c1_gpu_smoke.server.yaml" \
  --registry "$PSLG_REGISTRY_ROOT/c1_jobs.json" --submit
```

检查：

```bash
python scripts/query_slurm_registry.py \
  --registry "$PSLG_REGISTRY_ROOT/c1_jobs.json"
```

需要把作业钉到指定节点（例如某节点 CUDA 初始化故障需避开，2026-09-21 的
h103 即属此类）时，用 `--extra` 透传 sbatch 选项：

```bash
python scripts/submit_slurm_matrix.py \
  --matrix "$PSLG_MANIFEST_ROOT/c1_gpu_smoke.server.yaml" \
  --registry "$PSLG_REGISTRY_ROOT/c1_jobs.json" --submit \
  --extra='-w h104-slurm-a'
```

`--extra` 经 shlex 解析、以参数列表传给 sbatch（无 shell 注入面），并连同
`sbatch_extra` 字段写入 registry 备查。集群禁用 sacct 时查询脚本自动降级为
仅 squeue，并在 `accounting_note` 里说明原因。

### 6.6 C1 人工验收

GPU 框架冒烟成功后，再申请一次短时 GPU 交互会话：

```bash
srun -p RTX3090 --gres=gpu:1 -c 8 --mem=32G -t 01:00:00 --pty /bin/bash
cd "$PSLG_PROJECT_ROOT"
module purge
module load miniconda3/25.5.1-0
module load cuda-toolkit/12.1.1
source "$(conda info --base)/bin/activate"
```

运行8-cycle DETSEC-PC：

```bash
conda activate pslg-detsec
python scripts/train_state_discovery.py \
  --config config/experiments/core_wm_state_discovery_detsec_pc.yaml \
  --output-root "$PSLG_ARTIFACT_ROOT/c1_detsec_smoke" \
  --run-id c1_detsec_smoke_8cycles \
  --k 3,4,5 \
  --smoke-cycles 8 \
  --epochs-override 1 \
  --deterministic-tf
```

运行 Seq2Point 短训练、恢复和 validation 推理：

```bash
conda deactivate
conda activate pslg-nilm

# CUDA>=10.2 上确定性算法要求 CuBLAS 固定工作区；trainer 模块导入时已自动
# 设置（CUBLAS_WORKSPACE_CONFIG=:4096:8，不覆盖显式用户值），无需手工操作。
export PSLG_C1_NILM_SMOKE="$PSLG_ARTIFACT_ROOT/c1_nilm_smoke"
mkdir -p "$PSLG_C1_NILM_SMOKE"

python scripts/train_nilm.py \
  --experiment-dir "$PSLG_NILM_EXPERIMENT_DIR" \
  --arm B0 --seed 17 --batch-size 16 \
  --device cuda \
  --steps-per-epoch 2 --max-epochs 1 --patience 1 \
  --validation-count 256 \
  --output-dir "$PSLG_C1_NILM_SMOKE"

python scripts/train_nilm.py \
  --experiment-dir "$PSLG_NILM_EXPERIMENT_DIR" \
  --arm B0 --seed 17 --batch-size 16 \
  --device cuda \
  --steps-per-epoch 2 --max-epochs 2 --patience 1 \
  --validation-count 256 \
  --output-dir "$PSLG_C1_NILM_SMOKE" --resume

python scripts/predict_nilm.py \
  --experiment-dir "$PSLG_NILM_EXPERIMENT_DIR" \
  --checkpoint "$PSLG_C1_NILM_SMOKE/best_checkpoint.pt" \
  --device cuda \
  --arm B0 --partition validation --limit 256 \
  --output "$PSLG_C1_NILM_SMOKE/validation_predictions.npz"

python scripts/evaluate_nilm_predictions.py \
  --predictions "$PSLG_C1_NILM_SMOKE/validation_predictions.npz" \
  --output "$PSLG_C1_NILM_SMOKE/validation_metrics.json"
```

完成后退出计算节点：

```bash
exit
```

- TensorFlow `gpu_detected=true`；
- PyTorch `gpu_detected=true`；
- 两者都完成前向和反向；
- 环境 JSON 中 commit 正确、工作树干净；
- 再跑8-cycle DETSEC-PC 冒烟；
- 再跑 Seq2Point 短 epoch、保存、恢复、validation 推理；
- 记录实际 GPU、峰值显存和耗时；
- test 仍不在服务器。

全部通过才进入 C2。

---

## 7. C2：正式 DETSEC-PC 状态发现

### 7.1 提交前检查服务器本地矩阵

第5.1节已经自动生成矩阵。直接检查关键字段：

```bash
grep -E 'PSLG_PROJECT_ROOT|PSLG_FROZEN_COMMIT|PSLG_MANIFEST|PSLG_RUN_ROOT' \
  "$PSLG_MANIFEST_ROOT/c2_detsec.server.yaml"
```

仓库路径必须是你实际选择的 `PSLG-NILM-c1`（或自定义名），commit 必须等于：

```bash
git -C "$PSLG_PROJECT_ROOT" rev-parse HEAD
```

### 7.2 预览、提交、监控

```bash
python scripts/submit_slurm_matrix.py \
  --matrix "$PSLG_MANIFEST_ROOT/c2_detsec.server.yaml" \
  --registry "$PSLG_REGISTRY_ROOT/c2_jobs.json"

python scripts/submit_slurm_matrix.py \
  --matrix "$PSLG_MANIFEST_ROOT/c2_detsec.server.yaml" \
  --registry "$PSLG_REGISTRY_ROOT/c2_jobs.json" --submit

python scripts/query_slurm_registry.py \
  --registry "$PSLG_REGISTRY_ROOT/c2_jobs.json" \
  --output "$PSLG_REGISTRY_ROOT/c2_status.json"
```

### 7.3 C2 输出检查

检查 run 目录中至少存在：

- `state_discovery_summary.json`；
- k=3/4/5 的状态库；
- 每个状态的统计和 provenance；
- `environment.json`、`git_commit.txt`、`data_manifest.json`；
- `artifact_manifest.json` 和 `run_summary.md`；
- Slurm stdout/stderr。

人工比较 physical-stats 与 DETSEC-PC 的状态稳定性、可解释性、可交换性。validation 可以用来
选最终 k，但不能加入 DETSEC-PC 训练。

### 7.4 C2 决策

冻结：

- 正式状态库版本；
- k；
- 特征模型和 checkpoint；
- 合并规则；
- 状态标签语义；
- 输入 manifest、commit 和环境哈希。

冻结后重新生成正式 B2，不直接沿用 pilot B2。

---

## 8. C3：B0/B1/B2 第一轮正式验证

### 8.1 当前第一轮范围

```text
ratio = 0.5
seed = 17, 42, 73
arm = B0, B1, B2-matched
validation monitor = 同一组20,000个索引
```

B2-random 应使用单独冻结输入目录，再建一个矩阵，把该目录作为
`PSLG_EXPERIMENT_DIR`；在训练器内部仍可使用 `B2` arm，但 run id/注册表必须明确写
`B2random`，不能与 B2-matched 混淆。

### 8.2 检查并提交服务器矩阵

检查自动生成的 commit、manifest、experiment dir、artifact root：

```bash
grep -E 'PSLG_PROJECT_ROOT|PSLG_FROZEN_COMMIT|PSLG_MANIFEST|PSLG_EXPERIMENT_DIR|PSLG_RUN_ROOT|PSLG_RATIO' \
  "$PSLG_MANIFEST_ROOT/c3_seq2point_r0p5.server.yaml"
```

`experiment dir` 必须指向 manifest 中恢复出的项目相对目录，而不是另一个内容相似的副本。
然后先预览九个作业，再提交：

```bash
python scripts/submit_slurm_matrix.py \
  --matrix "$PSLG_MANIFEST_ROOT/c3_seq2point_r0p5.server.yaml" \
  --registry "$PSLG_REGISTRY_ROOT/c3_jobs.json"

python scripts/submit_slurm_matrix.py \
  --matrix "$PSLG_MANIFEST_ROOT/c3_seq2point_r0p5.server.yaml" \
  --registry "$PSLG_REGISTRY_ROOT/c3_jobs.json" --submit
```

### 8.3 每个作业的自动过程

`c3_seq2point.sbatch` 会依次：

1. 检查 commit、工作树、manifest、输出目录和 PyTorch；
2. 保存环境；
3. 训练一个 arm + seed；
4. 只对 validation 预测；
5. 计算 validation 指标；
6. 复制日志和 provenance；
7. 生成 artifact SHA-256 清单；
8. 检查正式 run 文件是否齐全。

### 8.4 C3 人工检查

- 九个作业是否都成功；
- prediction indices 是否完全一致；
- 每个 seed 的 MAE/F1/SAE；
- mean、std、paired bootstrap 95% CI；
- 是否有某个 seed 明显异常；
- 实际训练时间和显存是否公平；
- `test_accessed=false`。

失败作业先判断代码错误、OOM、超时还是数据错误。失败后恢复同一个 run，不把失败重跑偷换
成额外 seed。

---

## 9. C4：比例实验

C3 稳定后，再准备 ratio=1 和2的正式输入。不要在训练器里临时复制样本来伪造 ratio；每个
ratio 都要有自己的 dataset manifest、索引和哈希。

每个 ratio 复制一份矩阵，只改变：

- `PSLG_EXPERIMENT_DIR`；
- run id 中的 ratio；
- 对应 manifest。

矩阵中的 `PSLG_RATIO` 只负责 run id 标签，例如 `0p5`、`1p0`、`2p0`；真正的合成比例
仍由 `PSLG_EXPERIMENT_DIR` 指向的冻结输入决定。两者必须一致，不能只改名字不换数据。

ratio=1.0 的输入准备好后执行：

```bash
export PSLG_NILM_R1_DIR='把这里替换成ratio=1.0冻结输入的绝对路径'

python scripts/prepare_server_matrices.py \
  --project-root "$PSLG_PROJECT_ROOT" \
  --output-dir "$PSLG_MANIFEST_ROOT" \
  --manifest-root "$PSLG_MANIFEST_ROOT" \
  --artifact-root "$PSLG_ARTIFACT_ROOT" \
  --nilm-experiment-dir "$PSLG_NILM_R1_DIR" \
  --ratio 1p0

python scripts/submit_slurm_matrix.py \
  --matrix "$PSLG_MANIFEST_ROOT/c3_seq2point_r1p0.server.yaml" \
  --registry "$PSLG_REGISTRY_ROOT/c4_r1p0_jobs.json"
```

确认预览正确后，在最后一条命令增加 `--submit`。ratio=2.0 同理：

```bash
export PSLG_NILM_R2_DIR='把这里替换成ratio=2.0冻结输入的绝对路径'

python scripts/prepare_server_matrices.py \
  --project-root "$PSLG_PROJECT_ROOT" \
  --output-dir "$PSLG_MANIFEST_ROOT" \
  --manifest-root "$PSLG_MANIFEST_ROOT" \
  --artifact-root "$PSLG_ARTIFACT_ROOT" \
  --nilm-experiment-dir "$PSLG_NILM_R2_DIR" \
  --ratio 2p0

python scripts/submit_slurm_matrix.py \
  --matrix "$PSLG_MANIFEST_ROOT/c3_seq2point_r2p0.server.yaml" \
  --registry "$PSLG_REGISTRY_ROOT/c4_r2p0_jobs.json"
```

这两处带中文的路径必须替换；提交脚本不会自动理解中文说明。建议在真正提交前执行：

```bash
test -d "$PSLG_NILM_R1_DIR" && echo "ratio 1.0 data OK"
test -d "$PSLG_NILM_R2_DIR" && echo "ratio 2.0 data OK"
```

推荐先做 ratio=1；只有趋势稳定再做 ratio=2。若 ratio=2 退化，保留结果，不人为修改采样
直到它变好。

---

## 10. C5：决定是否进入生成阶段

C5 主要在本机完成。把服务器的 C2–C4 指标、预测和小型日志同步回来，然后回答：

先在服务器制作不含大型 checkpoint/预测的审阅包：

```bash
export PSLG_EXPORT_ROOT="$PSLG_DATA_ROOT/exports"
mkdir -p "$PSLG_EXPORT_ROOT"

tar --exclude='*.pt' --exclude='*.keras' --exclude='*predictions*.npz' \
  -czf "$PSLG_EXPORT_ROOT/c2_c4_review_metadata.tar.gz" \
  -C "$PSLG_ARTIFACT_ROOT" .

cd "$PSLG_EXPORT_ROOT"
sha256sum c2_c4_review_metadata.tar.gz > \
  c2_c4_review_metadata.tar.gz.sha256
```

通过 JumpServer SFTP 下载这两个文件。本机 PowerShell 校验：

```powershell
Get-FileHash -Algorithm SHA256 .\c2_c4_review_metadata.tar.gz
Get-Content .\c2_c4_review_metadata.tar.gz.sha256
tar -xzf .\c2_c4_review_metadata.tar.gz -C .\c2_c4_review_metadata
```

- B2-matched 是否超过 B2-random；
- B2-matched 是否接近或超过 B1；
- 改善是否跨 seed 稳定；
- 改善来自 MAE、F1还是 SAE；
- 更高 ratio 是否继续有效；
- 是否存在明显域偏移或误报增加。

如果 B2-matched 明显优于 random，即使尚未超过 B1，也说明“约束有效”，可以进入 D–F；
但论文措辞必须忠于结果。

---

## 11. Phase D：B3 四种完整周期生成

### 11.1 进入 D 前的代码门槛

本机必须先实现并测试：

```text
src/generation/full_cycle_transform.py
src/generation/full_cycle_cvae.py
src/generation/full_cycle_wgan.py
src/generation/full_cycle_diffusion.py
scripts/train_full_cycle_generator.py
scripts/generate_full_cycles.py
scripts/evaluate_synthetic_quality.py
scripts/audit_generation_memorization.py
scripts/build_shared_placement_schedule.py
```

每个入口先在8–20个 train cycle 上完成本机或短 GPU 冒烟，再写 sbatch。当前仓库尚未达到
这一门槛，因此不能直接提交 D。

可以立即执行以下门槛检查：

```bash
cd "$PSLG_PROJECT_ROOT"
for file in \
  src/generation/full_cycle_transform.py \
  src/generation/full_cycle_cvae.py \
  src/generation/full_cycle_wgan.py \
  src/generation/full_cycle_diffusion.py \
  scripts/train_full_cycle_generator.py \
  scripts/generate_full_cycles.py \
  scripts/evaluate_synthetic_quality.py \
  scripts/audit_generation_memorization.py \
  scripts/build_shared_placement_schedule.py
do
  test -f "$file" || echo "MISSING: $file"
done
```

只要出现 `MISSING`，就回本机实现和测试，不能在服务器创建空文件绕过检查。实现时建议冻结
统一命令接口，后续服务器命令采用：

```text
python scripts/train_full_cycle_generator.py --method cvae|wgan|diffusion --config <yaml> --output-dir <run>
python scripts/generate_full_cycles.py --method transform|cvae|wgan|diffusion --config <yaml> --output-dir <cycles>
python scripts/evaluate_synthetic_quality.py --input-dir <cycles> --output-dir <report>
python scripts/audit_generation_memorization.py --generated-dir <cycles> --train-library <train-only-library> --output <json>
```

这些命令是后续实现必须遵守的接口约定；在相应脚本进入仓库前不能执行。

### 11.2 服务器执行顺序

1. B3-T：真实周期变换，不训练网络；
2. B3-V：条件 CVAE；
3. B3-G：TraceGAN-style/SGAN-inspired 条件 WGAN；
4. B3-D：条件扩散。

每条路线都要：

- 只用 train；
- 使用同一目标周期数量和条件分布；
- 输出 donor/条件/seed/checkpoint provenance；
- 通过非负功率、峰值、能量、时长、复制、最近邻和多样性检查；
- 使用同一 placement schedule；
- 重新训练同预算 Seq2Point，并只看 validation。

B3-T 可在 CPU；CVAE/WGAN 先用 RTX3090；扩散优先 A6000。不要四条路线同时开发、同时
大跑，建议逐条通过门槛后再进入下一条。

---

## 12. Phase E：B4 基元生成

进入 E 前实现：

```text
src/generation/primitive_cvae.py
scripts/train_primitive_generator.py
scripts/generate_primitive_cycles.py
config/composition/b4_basic.yaml
```

可执行门槛检查：

```bash
cd "$PSLG_PROJECT_ROOT"
for file in \
  src/generation/primitive_cvae.py \
  scripts/train_primitive_generator.py \
  scripts/generate_primitive_cycles.py \
  config/composition/b4_basic.yaml
do
  test -f "$file" || echo "MISSING: $file"
done
```

计划统一接口：

```text
python scripts/train_primitive_generator.py --config config/generation/primitive_cvae.yaml --output-dir <run>
python scripts/generate_primitive_cycles.py --checkpoint <checkpoint> --config config/composition/b4_basic.yaml --output-dir <cycles>
```

第一版只做共享条件 CVAE。输入条件至少包含 state label 和持续时间；所有状态样本只来自
冻结 train 状态库。生成后先做基础拼接，不加入 HSMM 和复杂边界优化。

服务器过程：训练 → 生成基元 → 基础拼接 → 质量审计 → 放入共同背景 → Seq2Point 重训 →
validation。只有 B4-CVAE 有希望时，才考虑 B4-GAN/扩散扩展。

---

## 13. Phase F：B5 HSMM 受约束拼接

进入 F 前实现：

```text
src/composition/transition_model.py
src/composition/duration_model.py
src/composition/hsmm_sequence.py
src/composition/boundary_handler.py
src/composition/constrained_composer.py
scripts/fit_hsmm_composer.py
scripts/compose_generated_cycles.py
```

可执行门槛检查：

```bash
cd "$PSLG_PROJECT_ROOT"
for file in \
  src/composition/transition_model.py \
  src/composition/duration_model.py \
  src/composition/hsmm_sequence.py \
  src/composition/boundary_handler.py \
  src/composition/constrained_composer.py \
  scripts/fit_hsmm_composer.py \
  scripts/compose_generated_cycles.py
do
  test -f "$file" || echo "MISSING: $file"
done
```

计划统一接口：

```text
python scripts/fit_hsmm_composer.py --state-library <train-only-state-library> --config config/composition/b5_hsmm.yaml --output-dir <run>
python scripts/compose_generated_cycles.py --hsmm <frozen-hsmm> --primitive-generator <checkpoint> --config <ablation-yaml> --output-dir <cycles>
```

HSMM 拟合通常可用 CPU；大批量调用 B4 生成器时再用 GPU。正式消融顺序：

```text
B4-basic
B4 + Markov
B4 + HSMM
B4 + HSMM + endpoint
B5-final
```

每次只增加一个约束，并复用相同 seed、目标条件、背景和 placement schedule。B2 已发现通用
cross-fade 和端点偏移可能是负收益，所以不能默认打开；若重新使用，必须成为单独消融组。

每组仍要重训相同预算的 NILM，不能只比较波形“看起来像不像”。

---

## 14. Phase G：统一验证、协议冻结和最终 test

### 14.1 先完成 validation 总表

必须包含 B0、B1、B2、四种 B3、B4 和 B5；每组有冻结 seed、ratio、checkpoint、输入数据、
代码版本和指标。先在本机审阅异常和失败重跑记录。

### 14.2 运行最终无 test 审计

```bash
python scripts/audit_no_test_access.py \
  --manifest <all-development-manifests> \
  --run-dir <all-selected-run-roots> \
  --output reports/protocol_freeze/no_test_audit.json
```

实际使用时每个 manifest/run-dir 参数分别重复填写，不要把尖括号原样复制。

### 14.3 冻结协议

示例：

```bash
python scripts/freeze_protocol_before_test.py \
  --protocol-config config/experiments/b0_b5_protocol_v2.yaml \
  --protocol-config config/experiments/core_validation_b0_b5.yaml \
  --no-test-audit reports/protocol_freeze/no_test_audit.json \
  --run-dir <selected-B0-run> \
  --run-dir <selected-B1-run> \
  --run-dir <selected-B2-run> \
  --run-dir <selected-B3T-run> \
  --run-dir <selected-B3V-run> \
  --run-dir <selected-B3G-run> \
  --run-dir <selected-B3D-run> \
  --run-dir <selected-B4-run> \
  --run-dir <selected-B5-run> \
  --expected-group B0 --expected-group B1 --expected-group B2 \
  --expected-group B3-T --expected-group B3-V --expected-group B3-G \
  --expected-group B3-D --expected-group B4 --expected-group B5 \
  --output reports/protocol_freeze/protocol_freeze_before_test.json \
  --i-confirm-validation-complete
```

脚本会对配置、checkpoint、validation metrics 和 summary 计算 SHA-256，并要求所有 summary
明确记录 `test_accessed=false`。它不会读取 test。

### 14.4 人工签字式检查

打开生成的 JSON/MD，人工确认：

- 主要比较和指标已经固定；
- 所有组都选好了 checkpoint；
- 没有遗漏失败运行；
- 阈值20 W、bootstrap 方法和 seed 已固定；
- 没有根据 test 改任何超参数；
- 用户明确同意解锁最终 test。

### 14.5 最终 test

只有上述检查通过后，才把 test 包传到服务器或解除权限。每个冻结组只运行统一推理和统一
评价，不再训练、不重新选 checkpoint。

当前 `predict_nilm.py` 的 test 解锁形式为：

```bash
python scripts/predict_nilm.py \
  --experiment-dir <frozen-test-input> \
  --checkpoint <frozen-checkpoint> \
  --device cuda \
  --arm <arm> --partition test \
  --output <run-dir>/test_predictions.npz \
  --i-confirm-test-protocol-frozen
```

在 B3–B5 实现时必须把训练/推理入口扩展为通用 dataset arm；不能把 B3–B5 结果冒充成
现有 `B2` 而丢失组别身份。

如果最终 test 发现代码错误：记录错误，修复后让所有受影响组统一重跑；禁止只重跑对自己
有利的一组。

---

## 15. 日常监控与故障处理

### Pending 很久

```bash
scontrol show job <job_id>
```

看 `Reason`。资源繁忙就等待；不要重复提交相同作业。

### GPU OOM

先降低 batch size 或序列桶大小，再考虑 A6000。记录修改并让比较组使用同等规则。

### 系统内存 OOM

降低 data loader worker、检查是否整套数据复制到内存，并查看 `sacct` 的 MaxRSS。

### 超时

从 `last_checkpoint` 恢复。保留相同研究 run id，并记录续跑 job id。

### 环境/动态库错误

比较 `module list`、`nvidia-smi`、TensorFlow/PyTorch CUDA 版本和冻结环境。不要在作业脚本
中临时 `pip install`。

### 数据哈希失败

停止作业，重新传损坏文件并复验。不要从别的旧目录随手复制一个同名文件。

---

## 16. 每个阶段结束后的固定动作

1. 用 `query_slurm_registry.py` 保存最终 job 状态；
2. 检查 artifact manifest 和 SHA-256；
3. 运行无 test 审计；
4. 将小型指标、配置、日志、注册表同步回本机；
5. checkpoint 保存到可靠位置并记录哈希；
6. 在本机更新实验注册表和阶段总结；
7. Git 提交代码/配置/小报告，不提交大数据和 checkpoint；
8. 只有上一阶段 gate 通过才进入下一阶段。

---

## 17. 最简执行路线

```text
C1：双环境 + 双框架 GPU 冒烟
 ↓
C2：DETSEC-PC train-only，冻结正式状态库
 ↓
C3：B0/B1/B2，ratio 0.5，3 seeds
 ↓
C4：ratio 1、2
 ↓
C5：本机汇总并决定进入生成
 ↓
D：B3-T → CVAE → WGAN → Diffusion
 ↓
E：B4 基元 CVAE + 基础拼接
 ↓
F：Markov → HSMM → endpoint 消融，冻结 B5
 ↓
G：validation 总表 → 无 test 审计 → 协议冻结 → 一次最终 test
```

真正需要连续使用服务器的是训练和大批量生成；研究判断、代码开发、审计和协议冻结仍应回到
本机完成。这样既节省服务器资源，也能让每个结论有清楚、可追溯的人工确认。

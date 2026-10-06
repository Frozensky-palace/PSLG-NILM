# GPU 节点健康状态报告（提交管理员）

- 检测时间：2026-09-21 晚（北京时间）
- 检测人：scnu2024024563
- 方法：`sinfo -N` + `scontrol show node` + 逐节点 `srun` 极小探测（`nvidia-smi` 查询）
- 影响：当前仅 h104-slurm-a 可用；h103/h108 调度层显示 IDLE 但实际不可用，
  其他用户作业可能被调度上去后失败，建议管理员核查并考虑 `scontrol update` 隔离。

| 节点 | Slurm 状态 | 实际状况 | 证据 |
|---|---|---|---|
| h102-slurm-a (2×RTX3090) | DOWN+NOT_RESPONDING，Reason=Not responding [2026-01-02 22:40] | 长期失联 | `srun` 无法启动 |
| h103-slurm-a (2×RTX3090) | IDLE | **隐性故障：调度正常但用户进程 CUDA 初始化失败**。torch 2.5.1+cu121 与 2.14.0+cu130 两个构建均报 `CUDA initialization: CUDA unknown error`（典型 error 999）；TensorFlow 2.18.1 亦无法枚举 GPU。同驱动（580.65.06）同环境的 h104 一切正常。`nvidia-smi` 本身工作正常，怀疑 nvidia_uvm 内核模块或设备节点状态异常 | 2026-09-21 16:00–18:00 多次复现；同日 h104 全部通过 |
| h104-slurm-a (2×RTX3090) | IDLE | 健康。TF 2.18.1 与 torch 2.5.1+cu121 均通过 GPU 前向/反向冒烟 | gpu_smoke JSON 双框架 pass=true |
| h107-slurm-a (1×A6000) | DOWN+NOT_RESPONDING，Reason=Not responding [2026-04-27 16:40] | 长期失联（已 5 个月） | `srun` 无法启动 |
| h108-slurm-a (1×A6000) | IDLE | **隐性故障：`srun` 报 `Task launch failed: Unspecified error / Application launch failed`**（2026-09-21 21:10 与 21:40 两次复现，StepId 4109/4118） | 两次独立复现 |

## 建议

1. h103：优先检查 `nvidia_uvm` 内核模块状态（`lsmod | grep nvidia_uvm`），
   必要时重载模块或重启节点；该症状与驱动 580.65.06 用户态正常但
   CUDA 运行时初始化 error 999 的已知模式一致。
2. h108：检查 slurmd 与 cgroup 设备隔离配置（launch 阶段即失败，
   与 h103 的 CUDA 层故障不同）。
3. h102/h107：失联已 9 个月 / 5 个月，若计划修复请同步告知；
   若已退役建议从调度配置移除，避免误导排队的作业。
4. 修复前建议：`scontrol update nodename=h103,h108 state=drain reason="CUDA/launch failure, see user report 2026-09-21"`，
   防止其他用户作业调度上去后失败。

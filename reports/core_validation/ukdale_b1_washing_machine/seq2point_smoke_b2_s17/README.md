# Seq2Point 本机冒烟 run 说明（B2 arm，seed=17）

本目录是 Phase B1 的小数据冒烟（每 epoch 100 步、共 4 epoch），只验证
训练/恢复/推理/评价链路，不追求最终性能，也不得当成正式 Seq2Point 结论。

## 两个 MAE 的关系（2026-09-20 更正）

| 数字 | 来源 | 样本 |
|---|---|---|
| `training_summary.json` best_val_mae_w = 59.540 | 训练器在每 epoch 后计算 | 冻结 `validation_monitor_indices.npy` 的前 20,000 个窗口 |
| `validation_metrics.json` mae_w = 59.184 | 修复后的 `predict_nilm.py --sample-source monitor` + `evaluate_nilm_predictions.py` | 同一批前 20,000 个 monitor 窗口 |

两者一致；约 0.36 W 的差值仅来自推理输出侧的 `max(pred, 0)` 负值截断
（训练器在归一化空间直接算 MAE，不做截断）。

历史备注：修复前 `validation_metrics.json` 曾是 77.40 W，那是因为旧推理脚本取
validation 分区的前 20,000 个窗口（顺序索引），与训练选择样本不同，不可比较。
该旧产物已用修复后的脚本在原最佳 checkpoint（epoch 4）上重新生成，未重新训练。

## 复现命令

```powershell
.\.venv\Scripts\python.exe scripts\predict_nilm.py `
  --experiment-dir reports\core_validation\ukdale_b1_washing_machine\nilm_inputs_b2matched_k4_seed17_r0p5_v1 `
  --checkpoint reports\core_validation\ukdale_b1_washing_machine\seq2point_smoke_b2_s17\best_checkpoint.pt `
  --arm B2 --partition validation --limit 20000 `
  --output reports\core_validation\ukdale_b1_washing_machine\seq2point_smoke_b2_s17\validation_predictions.npz

.\.venv\Scripts\python.exe scripts\evaluate_nilm_predictions.py `
  --predictions reports\core_validation\ukdale_b1_washing_machine\seq2point_smoke_b2_s17\validation_predictions.npz `
  --output reports\core_validation\ukdale_b1_washing_machine\seq2point_smoke_b2_s17\validation_metrics.json
```

test_accessed = false。

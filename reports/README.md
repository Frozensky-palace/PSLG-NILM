# Reports 目录说明

`reports/` 保存研究运行产生的数据、清单、指标和人工总结。

Git 只追踪这里的 Markdown 结论报告。以下内容不进入 Git：

- `.npz`、`.npy`：对齐数据、状态波形、训练索引；
- `.csv`、`.json`：自动生成的 inventory、manifest 和指标；
- `.png`：可重新生成的检查图；
- 服务器传输包与模型输出。

不追踪不等于不重要。这些产物应通过服务器传输清单、SHA-256、备份盘或对象存储
保存。Markdown 报告必须记录生成它们所使用的脚本、配置、随机种子和关键结果。

正式数据只使用：

```text
core_validation/ukdale_b1_washing_machine/aligned_partitions_v2/
```

`aligned_partitions_v1/` 是修复分片边界前的旧产物，不得用于实验。

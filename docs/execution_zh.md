# 实际运行记录

## 已执行流程

|日期|过程|记录|
|---|---|---|
|2026-11-17|CUDA 与小样本试训练通过；开始三层五成员训练|`records/smoke/`、`records/training/baseline/`|
|2026-11-17|seed 101—105 完成，最佳轮分别为 6、7、3、11、3|`results/comparison/training_all_members.csv`|
|2026-11-18|三层 2000—2019 年预测和分析完成；二／四层配对训练|`records/baseline_evaluation/`、`records/training/ablation/`|
|2026-11-19|二／四层测试预测完成；CPU 分析遇到 OpenMP Error #15|`records/followup/` 中日志|
|2026-11-19|使用独立进程执行分析、热图计算和绘图，最终交接包完成|`records/followup/runtime_fix/`|
|2026-11-19|传回本机，检查来源指纹、汇总统计和图表，整理公开目录|`records/verification/`、`SOURCE_MANIFEST.json`|

二层最佳轮为 5、7、3、3、6；四层为 6、4、3、2、3。每个成员的完整轨迹包含提前停止前的非最佳轮。

|结构|五成员累计轮数|记录训练耗时|记录峰值分配显存|
|---|---:|---:|---:|
|二层|34|4.82 小时|402.5 MiB|
|三层|40|6.85 小时|949.8 MiB|
|四层|28|6.55 小时|1000.6 MiB|

训练耗时包括各轮训练、验证及保存等记录开销；不是整个项目墙钟时间，也不是通用设备性能基准。四层累计耗时少于三层由实际运行轮数不同造成，不能据此认为其每轮更快。

## 运行库故障及恢复

测试预测已保存后，分析进程同时初始化 `libomp.dll` 与 `libiomp5md.dll`，出现 OpenMP Error #15。恢复过程保留原权重、预测和协议；不重新训练，不使用 `KMP_DUPLICATE_LIB_OK=TRUE` 忽略错误。

`i20r20_resume_v1` 把核验、CPU 分析、比较、GPU Grad-CAM 数组计算、CPU 绘图、打包划分为新进程。CPU 绘图进程不导入 Torch，热图计算进程不导入 Matplotlib，子进程使用顺序 MKL 设置。最终日志记录 `RESUME: ALL COMPLETE`。

## 整理时的验证范围

原完整目录中的模型、输入、预测及交接包已核验，详细数值见 `records/verification/review_verification.json`。公开目录移除原始数据、权重和逐图预测后，提供的轻量复核重新检查：来源文件指纹、15 份训练最佳轮、混淆矩阵对应的 Accuracy/MCC、逐月收益对应的金融统计、Grad-CAM 配对数组和归一化。

CE/AUC 未在 Mac 上从逐图预测独立重算；新统一运行入口也未在整理期间完整重跑 GPU 实验。历史软件验证记录描述对应版本当时的检查，不能解释为所有旧检查脚本均可在不含原始数据和模型的公开目录直接运行。

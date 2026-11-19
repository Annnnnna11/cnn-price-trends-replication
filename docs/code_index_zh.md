# 代码索引

|位置|作用|
|---|---|
|`code/i20r20_formal/run.py`|正式数据准备、五成员训练、保存与恢复|
|`code/i20r20_formal/data_utils.py`|图像内存映射、样本筛选、划分、训练集标准化与统计|
|`code/i20r20_formal/cnn20.py`|三层 CNN 与初始化|
|`code/reproduce_i20r20/`|小样本试训练，不为正式训练提供权重|
|`code/i20r20_eval_v1/pipeline.py`|冻结三层协议、逐年测试预测、分析入口|
|`code/i20r20_eval_v1/evaluation_core.py`|分类指标、分组组合、金融统计、结果保存|
|`code/i20r20_ablation_v1/run.py`|二／四层配对准备、设备检查和训练|
|`code/i20r20_ablation_v1/models.py`|二／三／四层结构定义|
|`code/i20r20_ablation_v1/training_engine.py`|扩展训练循环、验证、早停和检查点|
|`code/i20r20_followup_v1/run.py`|扩展测试、统一分析与打包|
|`code/i20r20_followup_v1/guards.py`|来源、模型、数据和协议指纹检查|
|`code/i20r20_followup_v1/comparison.py`|统一分类、金融、训练成本表与图|
|`code/i20r20_followup_v1/gradcam.py`|原 Grad-CAM 工作流与样本选择|
|`code/i20r20_resume_v1/resume.py`|分析恢复入口，每阶段启动新进程|
|`code/i20r20_resume_v1/compute_cam.py`|加载固定模型，计算热图原始数组|
|`code/i20r20_resume_v1/render_cam.py`|只读取数组，绘制并排解释图|
|`scripts/reproduce.py`|整理时补充的从头复现统一入口，新输出写入 runs|
|`scripts/check_data.py`|按记录的 54 个输入文件大小／哈希检查数据|
|`scripts/review_recorded_results.py`|不加载神经网络，复核公开目录的汇总表及热图数组|
|`scripts/verify_submission.py`|检查公开包文件清单和来源副本指纹|
|`scripts/capture_environment.ps1`|可选：在实际 Windows 环境导出完整依赖及设备记录|
|`analysis/plot_years.py`|根据年度 CSV 重绘诊断图，不覆盖提交图片|
|`tests/`|整理入口及公共工具的轻量测试|

`code/` 保存实际执行代码，`scripts/` 为整理后增加的辅助入口。历史代码中的旧默认路径和实验身份保留；新实验通过 `scripts/reproduce.py` 显式传参并生成独立身份配置，避免覆盖历史结果。

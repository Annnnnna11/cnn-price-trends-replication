# 来源、使用范围与 AI 辅助

## 文献与数据

- Jingwen Jiang, Bryan Kelly, Dacheng Xiu. **(Re-)Imag(in)ing Price Trends**. *The Journal of Finance*, 78(6), 3193–3249, 2023。[SSRN 页面](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3756587)、[作者主页](https://dachxiu.chicagobooth.edu/)。
- 图像数据：[作者数据下载](https://dachxiu.chicagobooth.edu/download/img_data.zip)；[图像读取示例](https://dachxiu.chicagobooth.edu/download/img_demo.html)。课程提供的图像已包含价格、均线和成交量。
- Ramprasaath R. Selvaraju et al. **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization**. ICCV, 2017。[论文](https://arxiv.org/abs/1610.02391)。
- 课程任务书 `project2.pdf` 的 §5.2：20 日图像复现、数据处理、CNN、训练验证测试、性能评价与解释；结构敏感性及 Grad-CAM 为本次已执行扩展。

## 实现来源

结构与训练设置参考原论文，以及下载到本地的 `trend_submit/trend_code_submit/Model/cnn_model.py`、`Misc/config.py` 等参考实现。当前目录为课程实验的实现与整理，不是作者官方发布仓库。未提供能够确认的该参考代码的公开仓库或许可信息，因此不追加覆盖全部文件的开源许可证。公开可读不等同于已经授予全部上游代码的再许可权。

课程实现补充数据对齐与筛选、训练集标准化、种子划分、断点恢复、文件指纹、年度预测、指标审计、配对结构扩展、固定案例 Grad-CAM 和运行库故障隔离。参考数据本体、论文 PDF 和上游缓存不包含在提交目录。

原论文主实证涉及更多输入窗口、预测周期、组合策略与检验。本次只覆盖课程 I20/R20 及已声明扩展；不声称逐表完整复现，也不以不同训练条件下的结果判定超越原论文。

## AI 使用说明

AI 辅助用于论文与课程要求解释、实验流程规划、代码编写和调试、环境问题排查、指标口径检查、结果复核、可视化、文档草稿及提交目录整理。训练在 Windows CUDA 设备执行；结果解释依据保存的日志、表格和图像。

来源指纹、训练最佳轮、混淆矩阵统计、月度金融汇总和 Grad-CAM 数组进行了程序复核。CE/AUC 未在整理设备从逐图预测再次计算；新统一入口未重新执行完整 GPU 实验。热图解释和小幅指标差异没有作为因果发现或统计显著性证据。AI 生成说明不构成额外实验证据。

## 课程提交范围

任务书要求详细说明的源代码，可通过 GitHub 链接或文件提交；本目录对应代码与结果材料。技术报告、展示视频及可能使用的演示文稿是单独的课程交付，不包含在本次代码目录。报告中的 AI 使用说明应与实际使用一致。

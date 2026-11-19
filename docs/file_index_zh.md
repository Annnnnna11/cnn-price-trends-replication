# 结论分析文件索引

文件按正文、逐年分析、解释与附录用途标注；仓库采用英文文件名和相对链接。原始实验文件保留在本地。

## 最先看这六个入口

1. [【总览】完整实验复核与整体结论](results_review_zh.md)。
2. [【逐年】2000—2019 分类、金融与每年解读](yearly_analysis_zh.md)。
3. [【正文核心表】整体分类对比](../results/comparison/ensemble_comparison.csv)。
4. [【正文核心表】EW／VW 金融对比](../results/comparison/financial_HL_comparison.csv)。
5. [【结构扩展】计算成本](../results/comparison/training_cost_by_depth.csv)。
6. [【Grad-CAM 图像文件夹】12 张并排图](../results/gradcam)。

## 原始结果按用途标注

|用途标注|原文件入口|支持的分析|放在哪里|
|---|---|---|---|
|【主1：整体分类结论】|[ensemble_comparison.csv](../results/comparison/ensemble_comparison.csv)|二层 CE／Accuracy 略好，三层 AUC／MCC 较好；区分主区间与共同年份。|正文核心表|
|【主2：金融结论】|[financial_HL_comparison.csv](../results/comparison/financial_HL_comparison.csv)|三层 VW 较好、四层 EW Sharpe 略高；同时报告收益、波动、月份。|正文核心表|
|【主3：效率结论】|[training_cost_by_depth.csv](../results/comparison/training_cost_by_depth.csv)|参数、总轮数、记录时间与显存；解释四层总时间受早停影响。|正文简表／结构扩展|
|【主图1：结构对比】|[ensemble_metrics.png](../results/comparison/ensemble_metrics.png)|三结构 CE、Accuracy、AUC 的整体展示。|正文／PPT|
|【主图2：金融对比】|[financial_HL.png](../results/comparison/financial_HL.png)|三结构 EW／VW Sharpe。|正文／PPT|
|【补1：逐成员差异】|[classification_all_members.csv](../results/comparison/classification_all_members.csv)|检查五个成员、融合、固定基准与各年份，避免只看最佳 seed。|附录／详细分析|
|【补2：配对结构差值】|[paired_seed_comparison.csv](../results/comparison/paired_seed_comparison.csv)|同 seed 的结构差异；差值不等于显著性检验。|结构敏感性附录|
|【补3：训练过程】|[training_all_members.csv](../results/comparison/training_all_members.csv)|各模型最佳轮、验证 CE、总轮数与记录成本。|附录／训练过程|
|【补图1：学习曲线】|[learning_curves.png](../results/comparison/learning_curves.png)|展示验证 CE 的变化和成员差异。|附录／方法说明|
|【补图2：概率校准】|[calibration_comparison.png](../results/comparison/calibration_comparison.png)|预测概率与实际上涨频率的关系；不同于类别比例图。|附录／概率质量|
|【补4：完整十组组合】|[financial_all_deciles.csv](../results/comparison/financial_all_deciles.csv)|D1—D10 和 H-L 的收益、波动、Sharpe，研究排序梯度。|正文择要／附录全表|
|【补5：论文金融对照】|[paper_comparison_I20R20.csv](../results/depth_3/analysis/paper_comparison_I20R20.csv)|对应原论文 Table2 等权与 Table IA3 市值加权；两套年份均有。|复现对照表|
|【核1：样本对齐】|[paired_test_audit.csv](../results/comparison/paired_test_audit.csv)|逐年原始行、分类有效行和三结构一致性记录。|方法／复现核验|
|【核2：月份一致】|[financial_calendar_audit.csv](../results/comparison/financial_calendar_audit.csv)|三结构比较使用相同 239 个有效形成月。|复现核验|
|【核3：缺失与筛选】|[monthly_coverage.csv](../results/depth_3/analysis/monthly_coverage.csv)|披露未来月收益缺失、无效市值和省略月份。|局限性／附录|
|【核4：逐月原始比较】|[monthly_returns_all_depths.csv](../results/comparison/monthly_returns_all_depths.csv)|核验极端月、年度拆分和金融公式的直接来源。|详细分析／备查|
|【Grad1：样本来源】|[selected_samples.csv](../results/gradcam/selected_samples.csv)|样本日期、股票、标签、参考三层概率与 TP/TN/FP/FN。|解释方法／图注|
|【Grad2：解释数值】|[map_metadata.csv](../results/gradcam/map_metadata.csv)|每结构的 p(up)、目标类、目标层与权重指纹。|解释结果／图注|
|【Grad3：解释说明】|[report_zh.md](../results/gradcam/report_zh.md)|图像阅读方法与原生分辨率、单独归一化等限制。|解释方法／局限性|


## 新增的逐年分析资料

|用途标注|文件|内容|
|---|---|---|
|【年1：主阅读】|[yearly_analysis_zh.md](yearly_analysis_zh.md)|主表／补充表释义、两张年度表和每年解释。|
|【年2：完整分类数据】|[yearly_classification.csv](../analysis/yearly_classification.csv)|60 行结构×年份，包括 CE、准确率、AUC、MCC、类别比例、常数参考和差值。|
|【年3：完整金融数据】|[yearly_financial.csv](../analysis/yearly_financial.csv)|120 行结构×年份×权重，包括月数、年化均值、波动与正负月份。|
|【年图1：分类诊断】|[yearly_classification_diagnostics.png](../analysis/yearly_classification_diagnostics.png)|相对常涨准确率增益、逐年 AUC、真实与预测上涨类别比例。|
|【年图2：金融诊断】|[yearly_financial_comparison.png](../analysis/yearly_financial_comparison.png)|三个结构每年 EW／VW 的月均 H-L×12。|
|【核验：本机独立复核】|[review_verification.json](../records/verification/review_verification.json)|文件指纹、60 年度预测、15 历史、金融重算与热图检查范围。|

## Grad-CAM 全部图像入口

|参考组|固定案例|股票／日期|真实类别|三层 p(up)|图像入口|
|---|---|---|---:|---:|---|
|TP|54139|76096／1993-05-28|1|0.5261|[TP_54139.png](../results/gradcam/TP_54139.png)|
|TP|242922|78019／1995-10-31|1|0.5276|[TP_242922.png](../results/gradcam/TP_242922.png)|
|TP|410733|66130／1997-08-29|1|0.5099|[TP_410733.png](../results/gradcam/TP_410733.png)|
|TN|36983|60695／1993-09-30|0|0.4332|[TN_36983.png](../results/gradcam/TN_36983.png)|
|TN|273706|85172／1995-12-29|0|0.4355|[TN_273706.png](../results/gradcam/TN_273706.png)|
|TN|552914|81078／1998-03-31|0|0.4410|[TN_552914.png](../results/gradcam/TN_552914.png)|
|FP|249902|79226／1995-05-31|0|0.5482|[FP_249902.png](../results/gradcam/FP_249902.png)|
|FP|429581|77495／1997-12-31|0|0.5247|[FP_429581.png](../results/gradcam/FP_429581.png)|
|FP|627141|77313／1999-09-30|0|0.5146|[FP_627141.png](../results/gradcam/FP_627141.png)|
|FN|30820|50286／1993-09-30|1|0.4218|[FN_30820.png](../results/gradcam/FN_30820.png)|
|FN|497819|25961／1998-09-30|1|0.4444|[FN_497819.png](../results/gradcam/FN_497819.png)|
|FN|523795|76226／1998-02-27|1|0.4128|[FN_523795.png](../results/gradcam/FN_523795.png)|


每张图：左列原图，右侧依次二层／三层／四层；上行为非上涨 logit，下行为上涨 logit。
所有 TP/TN/FP/FN 名称均由三层 seed101 的验证判断定义；另外两个结构可能在同图作出不同判断。
图中颜色是各自图内归一化的重要性，不是上涨概率。目标类 0/1 指解释哪个 logit，不是实际标签。

正文可以展示每组固定抽样记录的第一个案例（TP_54139、TN_36983、FP_249902、FN_30820）；其余 8 图放附录。这一展示规则按抽样顺序确定，不按效果挑图。图注仍须说明只有 seed101、验证案例和原生分辨率差异。

## 不直接拿来写效果结论的文件

- `best.pt`、`last.pt`：权重与续训状态，用于复现，不是结果表。
- 原始 `images.dat`、标签 Feather：数据来源，方法部分需要，写结论优先看已核验统计。
- `runtime_fix/` 日志：用于说明兼容性修复与执行完成，不证明模型有效。
- `VALIDATION.json`、测试脚本：软件检查，不是模型测试集成绩。
- `manifest.json`、`followup.json`、SHA256 清单：记录实验身份与一致性，不替代统计证据。
- `gradcam/*.npz`：原始热图数组，重新绘图或核验时使用；常规展示使用 PNG 和元数据。

## 论文正文与附录的一种安排

- 正文 Table1：课程主区间的三结构分类指标及常数基准。
- 正文 Table2：课程主区间 EW／VW H-L 收益、波动与 Sharpe。
- 正文 Table3 或短段落：结构参数、轮数与成本。
- 正文 Figure1：逐年分类诊断；Figure2：四类固定 Grad-CAM 示例。
- 附录：共同年份补充、五成员全表、年度完整金融统计、D1—D10、其他 Grad-CAM 图及数据覆盖。

这是当前报告的排版安排，不等于原论文 Table2／IA3 的编号。原论文编号与当前“课程主区间／共同年份补充”是不同概念。

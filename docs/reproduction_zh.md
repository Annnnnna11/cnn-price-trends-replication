# 复现步骤

## 1. 安装环境与准备数据

环境安装见 [environment/README.md](../environment/README.md)，数据格式与下载位置见 [data/README.md](../data/README.md)。正式训练使用 CUDA GPU；CPU 可进行汇总复核和部分实现检查。

从仓库根目录执行。示例数据目录可替换为实际位置；每次独立实验使用新的 `--run-root`。

```powershell
conda activate cnn20_reproduction
python scripts/reproduce.py plan --data-dir "D:/data/monthly_20d" --run-root "runs/reproduction_01"
python scripts/reproduce.py check-data --data-dir "D:/data/monthly_20d" --run-root "runs/reproduction_01"
```

`check-data` 核对 54 个文件的大小和 SHA256。它针对本次使用的数据版本，不会修改数据；不同版本需先查明差异，不应删除校验来混用。

## 2. 划分与三层训练

```powershell
python scripts/reproduce.py prepare-baseline --data-dir "D:/data/monthly_20d" --run-root "runs/reproduction_01"
python scripts/reproduce.py train-baseline --run-root "runs/reproduction_01"
```

生成 1993—1999 年合格样本目录、五次独立 70/30 划分及训练集像素标准化参数。每个成员按最低验证 CE 保存最佳模型，连续两轮未改善时提前停止，最多 50 轮。终端出现一次 `SAVED` 仅表示该轮已保存，不表示全部成员完成。中断后同一训练命令可从最后完整轮继续。

## 3. 绑定新实验规则

```powershell
python scripts/reproduce.py configure-extensions --run-root "runs/reproduction_01"
```

历史代码检查原基线实验身份，不能直接用于不同身份的新实验。此步骤将扩展包复制到新运行目录，仅替换该次实验的基线身份、规则声明日期与关联指纹；训练及模型 Python 源码不变。随后执行测试或扩展时复用这些固定副本，并检查是否被修改。

原始规则声明保存在 `protocols/recorded_study_plan.json`；新运行规则位于 `runs/reproduction_01/code/`。二／四层、五种子、阈值、融合权重和 Grad-CAM 抽样规则在测试命令之前固定。

## 4. 三层测试与二／四层训练

```powershell
python scripts/reproduce.py evaluate-baseline --data-dir "D:/data/monthly_20d" --run-root "runs/reproduction_01"
python scripts/reproduce.py train-ablation --data-dir "D:/data/monthly_20d" --run-root "runs/reproduction_01"
```

三层使用五个最佳检查点预测 2000—2019 年，不再更新模型。二／四层使用与三层同 seed 完全相同的样本划分、标准化和训练规则，各训练五个成员。不能根据已查看的测试结果选择种子、阈值或重新设计结构后仍称其为未触及测试。

## 5. 扩展测试、统一分析与 Grad-CAM

```powershell
python scripts/reproduce.py evaluate-extensions --data-dir "D:/data/monthly_20d" --run-root "runs/reproduction_01"
```

顺序执行冻结检查、二／四层年度预测、三结构比较、Grad-CAM 计算、绘图与结果打包。三层复用原测试预测。分析、GPU 热图计算、CPU 绘图分别使用新进程，避免已出现的 OpenMP 运行库冲突。

新输出分布：

|目录|内容|
|---|---|
|`runs/reproduction_01/baseline`|三层权重、划分、标准化、训练日志|
|`runs/reproduction_01/baseline_evaluation`|三层逐年预测与分析|
|`runs/reproduction_01/ablation`|二／四层训练结果|
|`runs/reproduction_01/followup`|统一对比、Grad-CAM、交接包|
|`runs/reproduction_01/launcher_logs`|统一入口逐阶段命令与日志|

也可使用 README 中的 `all` 按上述顺序连续执行。该操作包含全部训练，不能与已运行任务同时写入相同目录。

## 6. 查看本次已完成结果

无需从头训练：

```powershell
python scripts/verify_submission.py
python scripts/review_recorded_results.py
python -m unittest discover -s tests -v
```

复核输出写入 `runs/review/`。需要重新绘制年度图时执行 `python analysis/plot_years.py`，图片写入 `runs/review/figures/`；提交的结果图保持原样。

历史执行包内部分旧验证脚本依赖原始模型、数据或原交付目录，属于执行过程存档。整理目录的轻量检查使用 `tests/`；不能将轻量检查通过等同于完整 GPU 重跑通过。

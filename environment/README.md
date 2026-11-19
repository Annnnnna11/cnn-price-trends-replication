# 环境说明

正式实验使用 Windows、Python 3.11.15、RTX 5060 Laptop GPU（8GB 显存），PyTorch 2.7.1+cu128，CUDA runtime 12.8。实际记录的核心版本见 [recorded_versions.json](recorded_versions.json)，原运行环境记录保存在 `records/`。

`environment.yml` 根据实际记录的核心版本重建，**不是完整 Conda 导出或全部传递依赖锁文件**。本地资料中没有完整 `conda list --explicit`，不能保证重建环境的每个底层包均与原设备相同。未使用原参考代码中过时的环境文件替代实际版本。

## Windows CUDA 运行

以下创建新的独立环境，不复用其他实验环境：

```powershell
conda env create -f environment/environment.yml
conda activate cnn20_reproduction
python -m pip install -r environment/requirements-torch-cu128.txt
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

核心数值和绘图库由 Conda 安装；Torch 使用官方 CUDA 12.8 wheel，与本次实际运行方式一致。[官方历史版本说明](https://pytorch.org/get-started/previous-versions/)列出 2.7.1 的 cu128 安装渠道。`requirements-core.txt` 仅列出相同核心版本，供检查或其他安装方式参考，完成 Conda 安装后无需重复安装。

Windows 新驱动显示的 CUDA 版本可能高于 Torch 自带 runtime；本实验不要求另装完整 CUDA Toolkit。

## 只查看和复核结果

文件清单校验只需 Python 标准库；结果汇总复核需 NumPy 和 pandas；重绘年度图另外需要 Matplotlib。它们不加载 Torch，不需要 GPU。

```powershell
python scripts/verify_submission.py
python scripts/review_recorded_results.py
```

## 可选完整环境补录

如仍可访问实际运行的 `cnn20` 环境，在该环境的 PowerShell 中执行：

```powershell
conda activate cnn20
powershell -ExecutionPolicy Bypass -File scripts/capture_environment.ps1
```

导出到 `runs/environment_capture/`，包含 Conda 完整导出、显式包清单、pip freeze、Python 和 GPU 信息。补录尚未执行，不能把本脚本当成已经存在的完整锁文件。

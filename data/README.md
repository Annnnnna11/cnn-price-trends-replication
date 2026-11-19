# 输入数据

原始数据不随公开代码提交。下载来源：[作者图像数据](https://dachxiu.chicagobooth.edu/download/img_data.zip)，读取示例见 [img_demo](https://dachxiu.chicagobooth.edu/download/img_demo.html)。解压后使用 `monthly_20d` 目录，不使用其他输入窗口。

1993—2019 年共 27 对文件：

```text
20d_month_has_vb_[20]_ma_1993_images.dat
20d_month_has_vb_[20]_ma_1993_labels_w_delay.feather
...
20d_month_has_vb_[20]_ma_2019_images.dat
20d_month_has_vb_[20]_ma_2019_labels_w_delay.feather
```

图像按 uint8、单张 1×64×60 读取，同年标签行与图像顺序对应。标签包含 Date、StockID、MarketCap、Ret_5d、Ret_20d、Ret_60d、Ret_month、EWMA_vol 等字段。输入共 2,196,994 张图；后续按训练／评价规则筛选。

`input_files.json` 保存本次所用 54 个文件的大小和 SHA256。文件名保持原样；不要单独排序或过滤标签后再按新行号匹配原图像。

```powershell
python scripts/check_data.py --data-dir "D:/data/monthly_20d" --sha256
```

可放在仓库内 `data/monthly_20d/`，也可保存在外部位置，通过 `--data-dir` 指定。数据目录已加入忽略规则。

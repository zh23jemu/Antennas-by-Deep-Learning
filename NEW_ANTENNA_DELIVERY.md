# 第二种天线最终交付说明

本交付版针对第二种天线数据，已经合并：

- `data1.xlsx`
- `data2.xlsx`
- `仿的数据` 文件夹中的新增 S 参数与增益 Excel

并完成：

- Excel 数据整理
- S11 与增益关键特征提取
- 新天线专用 MLP 模型训练
- `S11 + 增益` 联合优化
- 多权重方案比较

## 一、数据说明

- `data1.xlsx`：原始 S11 数据
- `data2.xlsx`：原始增益数据
- `仿的数据`：新增 HFSS 导出数据
- 当前共同尺寸样本数：`5314`
- 输入尺寸参数数：`8`

尺寸参数为：

- `cut_x [mm]`
- `cut_y [mm]`
- `fw [mm]`
- `gx [mm]`
- `gy [mm]`
- `h1 [mm]`
- `px [mm]`
- `py [mm]`

## 二、已生成的数据文件

整理后的训练数据位于 `outputs/new_antenna_dataset/`：

- `features.csv`
- `s11_curves.npy`
- `gain_patterns.npy`
- `s11_freqs_ghz.npy`
- `summary.json`

其中 `features.csv` 已经是“每组尺寸一行”的训练格式。

## 三、已训练模型

模型目录位于 `outputs/new_antenna_model/`：

- `new_antenna_mlp.joblib`
- `training_summary.json`
- `best_design.json`
- `prediction.json`
- `best_design_features.png`
- `prediction_features.png`
- `weight_comparison.json`

当前模型验证结果：

- 验证集 `MSE`：`0.80147`
- 验证集 `MAE`：`0.207635`

## 四、当前推荐结果

### 1. 均衡优化版

当前采用归一化后加权评分，兼顾 `S11` 与 `gain`。

推荐尺寸：

```text
2.57017, 16.8272, 2.97328, 157.277, 77.3126, 1.27523, 90.9747, 77.229
```

对应预测结果：

- `s11_min_db ≈ -29.1408`
- `gain_max ≈ 1.13158`

### 2. 多权重方案

详见 `outputs/new_antenna_model/weight_comparison.json`。

- `S11优先`
- `均衡`
- `增益优先`

可以根据需求选择对应尺寸方案。

当前三套方案的典型结果为：

- `S11优先`
  - `s11_min_db ≈ -29.1408`
  - `gain_max ≈ 1.13158`
- `均衡`
  - `s11_min_db ≈ -29.1408`
  - `gain_max ≈ 1.13158`
- `增益优先`
  - `s11_min_db ≈ -20.1034`
  - `gain_max ≈ 1.34519`

## 五、运行方式

### 1. 一键运行

```powershell
.\.venv\Scripts\python.exe run_new_antenna.py --s11-file data1.xlsx --s11-file "C:\Coding\Antennas-by-Deep-Learning\仿的数据\S Parameter Plot 1.xlsx" --s11-file "C:\Coding\Antennas-by-Deep-Learning\仿的数据\S Parameter Plot 2.xlsx" --s11-file "C:\Coding\Antennas-by-Deep-Learning\仿的数据\S Parameter Plot 3.xlsx" --s11-file "C:\Coding\Antennas-by-Deep-Learning\仿的数据\S Parameter Plot 4.xlsx" --gain-file data2.xlsx --gain-file "C:\Coding\Antennas-by-Deep-Learning\仿的数据\Gain Plot 01.xlsx" --gain-file "C:\Coding\Antennas-by-Deep-Learning\仿的数据\Gain Plot 02.xlsx" --gain-file "C:\Coding\Antennas-by-Deep-Learning\仿的数据\Gain Plot 03.xlsx" --gain-file "C:\Coding\Antennas-by-Deep-Learning\仿的数据\Gain Plot 4.xlsx"
```

### 2. 只比较不同权重方案

```powershell
.\.venv\Scripts\python.exe compare_new_antenna_weights.py
```

## 六、脚本说明

- `prepare_new_antenna_dataset.py`：整理 Excel 数据并提取特征
- `train_new_antenna.py`：训练新天线专用模型
- `optimize_new_antenna.py`：做联合优化
- `predict_new_antenna.py`：预测指定尺寸的关键特征
- `run_new_antenna.py`：一键完成整理、训练、优化、预测
- `compare_new_antenna_weights.py`：比较不同权重下的优化结果

## 七、建议

如果后面用户继续补这类新天线的数据，优先保持：

- 同一套 8 个尺寸参数定义不变
- S11 和增益数据一一对应
- 尽量补新的尺寸组合，而不是重复已有组合

这样后续模型效果会提升更明显。

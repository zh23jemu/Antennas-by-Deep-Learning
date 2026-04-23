# 三组结果预测代码说明

这个包用于预测第二种天线的三组尺寸结果。

## 1. 输入文件

默认输入文件是：

- `three_cases_input.csv`

表头必须包含这 8 个尺寸参数：

- `cut_x [mm]`
- `cut_y [mm]`
- `fw [mm]`
- `gx [mm]`
- `gy [mm]`
- `h1 [mm]`
- `px [mm]`
- `py [mm]`

可以额外保留 `label` 列方便区分不同方案。

## 2. 运行命令

```powershell
.\.venv\Scripts\python.exe predict_new_antenna_batch.py --model outputs\new_antenna_model\new_antenna_mlp.joblib --input-csv three_cases_input.csv
```

## 3. 输出文件

运行后会生成：

- `three_cases_prediction.csv`
- `three_cases_prediction.json`

## 4. 当前默认三组

- 联合优化版
- 真正 S 参数最优版
- 增益最优版

如果后面想换成别的三组，直接修改 `three_cases_input.csv` 即可，不需要改代码。

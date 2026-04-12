# 基于深度学习的小型天线快速设计与优化

本项目提供一个最小可用流程：读取天线尺寸和 S 参数样本，训练 MLP 代理模型，预测给定尺寸的 S 参数，并搜索预测 S 参数最优的天线尺寸。

## 环境

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

后续命令都使用项目本地解释器，不需要激活虚拟环境。

## 训练

```powershell
.\.venv\Scripts\python.exe train.py
```

训练后会生成：

- `outputs/antenna_mlp.joblib`
- `outputs/training_summary.json`

## 预测

`--dimensions` 需要输入 11 个逗号分隔的尺寸参数。

```powershell
.\.venv\Scripts\python.exe predict.py --dimensions "1,2,3,4,5,6,7,8,9,10,11"
```

结果保存到 `outputs/prediction.json`。

## 优化

```powershell
.\.venv\Scripts\python.exe optimize.py
```

优化脚本会在训练数据尺寸范围内随机搜索，目标是让预测 S 参数最小，结果保存到 `outputs/best_design.json`。

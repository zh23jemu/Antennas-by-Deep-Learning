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

结果保存到 `outputs/prediction.json`，曲线图保存到 `outputs/prediction_s_curve.png`。

## 优化

```powershell
.\.venv\Scripts\python.exe optimize.py
```

优化脚本会在训练数据尺寸范围内随机搜索，目标是让预测 S 参数最小，结果保存到 `outputs/best_design.json`。

默认还会生成 `outputs/best_design_s_curve.png`，可用于论文或报告中展示优化尺寸对应的预测 S 参数曲线。

## 后续扩展：增益和效率

当前优化目标只使用 S 参数。代码中已经预留 `antenna_ml/scoring.py`，后续增益和效率数据跑完后，可以把模型输出扩展为 S 参数、增益、效率三个目标，并使用综合评分：

```text
综合评分 = S参数目标 - 增益权重 * 增益 - 效率权重 * 效率
```

评分越低代表设计越好。这样可以保持“低 S 参数、高增益、高效率”的统一优化方向。

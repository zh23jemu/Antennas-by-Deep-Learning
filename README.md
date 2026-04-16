# 基于深度学习的小型天线快速设计与优化

本项目提供一个最小可用流程：读取天线尺寸和 S 参数样本，训练 MLP 代理模型，预测给定尺寸下的关键 S 参数特征，并搜索预测最小 S 参数最优的天线尺寸。

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

默认训练轮数为 `500`。如需更快试跑，可手动传入更小的 `--max-iter`。

训练后会生成：

- `outputs/antenna_mlp.joblib`
- `outputs/training_summary.json`
- `outputs/validation_plots/validation_compare_*.png`

训练阶段会默认从验证集中抽取若干样本，生成“真实特征 vs 预测特征”对比图，方便快速判断模型拟合效果。

## 预测

`--dimensions` 需要输入 11 个逗号分隔的尺寸参数。

```powershell
.\.venv\Scripts\python.exe predict.py --dimensions "1,2,3,4,5,6,7,8,9,10,11"
```

结果保存到 `outputs/prediction.json`，特征图保存到 `outputs/prediction_features.png`。

## 优化

```powershell
.\.venv\Scripts\python.exe optimize.py
```

优化脚本会在训练数据尺寸范围内随机搜索，目标是让预测 S 参数最小，结果保存到 `outputs/best_design.json`。

默认还会生成 `outputs/best_design_features.png`，用于展示优化尺寸对应的关键 S 参数特征。

## 一键运行

```powershell
.\.venv\Scripts\python.exe run_all.py
```

这个脚本会自动完成：

- 训练模型
- 生成验证集真实/预测对比图
- 搜索最优尺寸
- 生成优化特征图
- 使用最优尺寸再次预测并生成预测特征图

## 后续扩展：增益和效率

当前优化目标只使用 S 参数。代码中已经预留 `antenna_ml/scoring.py`，后续增益和效率数据跑完后，可以把模型输出扩展为 S 参数、增益、效率三个目标，并使用综合评分：

```text
综合评分 = S参数目标 - 增益权重 * 增益 - 效率权重 * 效率
```

评分越低代表设计越好。这样可以保持“低 S 参数、高增益、高效率”的统一优化方向。

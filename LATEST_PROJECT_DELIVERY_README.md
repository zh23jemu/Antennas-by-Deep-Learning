# 深度学习天线设计与优化项目说明

## 1. 项目用途

本项目用于基于深度学习模型完成天线尺寸参数到性能指标的快速预测，并进一步做尺寸优化。

当前最终整理版主要针对第二种天线数据，输入为 8 个尺寸参数，输出为 S11 与增益相关关键特征。

输入尺寸参数：

- `cut_x [mm]`
- `cut_y [mm]`
- `fw [mm]`
- `gx [mm]`
- `gy [mm]`
- `h1 [mm]`
- `px [mm]`
- `py [mm]`

预测目标特征：

- `s11_min_db`
- `s11_min_freq_ghz`
- `s11_mean_db`
- `s11_std_db`
- `s11_bandwidth_below_minus10_db_ghz`
- `gain_max`
- `gain_mean`
- `gain_std`

## 2. 当前最终版说明

本版本是在用户验证通过后的最终整理版。

核心改动是：在 MLP 模型预测基础上，针对增益预测加入了保守校准，尤其用于修正第一组联合优化结果中 `gain_max` 偏乐观的问题。

当前三组尺寸预测结果保存在：

- `three_cases_prediction.csv`
- `three_cases_prediction.json`

第一组联合优化版的最终修正后关键结果：

- `s11_min_db ≈ -29.1408`
- `s11_min_freq_ghz ≈ 0.9575 GHz`
- `gain_max ≈ 0.7311`
- `gain_mean ≈ 0.2829`
- `gain_std ≈ 0.1853`

## 3. 目录说明

主要代码：

- `prepare_new_antenna_dataset.py`：整理新天线 Excel 数据并提取特征
- `train_new_antenna.py`：训练新天线专用 MLP 模型
- `predict_new_antenna.py`：输入单组尺寸并预测 S11 + 增益特征
- `predict_new_antenna_batch.py`：批量预测多组尺寸
- `optimize_new_antenna.py`：根据模型搜索较优尺寸
- `run_new_antenna.py`：一键执行数据整理、训练、优化、预测流程
- `generate_paper_figures.py`：生成论文用模型结构图、Loss 曲线、预测散点图和相关性热力图

核心模块：

- `antenna_ml/new_antenna.py`：新天线字段、数据读取、评分函数
- `antenna_ml/new_antenna_calibration.py`：增益保守校准逻辑
- `antenna_ml/model.py`：MLP 模型训练、保存、读取
- `antenna_ml/io.py`：JSON 输出工具
- `antenna_ml/new_antenna_plotting.py`：新天线特征图绘制工具
- `antenna_ml/plotting.py`：通用绘图工具

当前结果：

- `outputs/new_antenna_dataset/features.csv`：整理后的训练特征数据
- `outputs/new_antenna_model/new_antenna_mlp.joblib`：最终模型文件
- `outputs/new_antenna_model/training_summary.json`：训练摘要
- `outputs/paper_figures/`：论文图

## 4. 环境安装

建议使用 Python 虚拟环境。

安装依赖：

```bash
pip install -r requirements.txt
```

如果使用当前项目本地虚拟环境，Windows 下运行方式类似：

```bash
.venv\Scripts\python.exe train_new_antenna.py
```

## 5. 常用命令

重新训练模型：

```bash
.venv\Scripts\python.exe train_new_antenna.py --features-csv outputs\new_antenna_dataset\features.csv --output-dir outputs\new_antenna_model --max-iter 1200
```

批量预测三组尺寸：

```bash
.venv\Scripts\python.exe predict_new_antenna_batch.py --model outputs\new_antenna_model\new_antenna_mlp.joblib --input-csv three_cases_input.csv --output-csv three_cases_prediction.csv --output-json three_cases_prediction.json
```

单组尺寸预测示例：

```bash
.venv\Scripts\python.exe predict_new_antenna.py --dimensions "2.57017,16.8272,2.97328,157.277,77.3126,1.27523,90.9747,77.229"
```

生成论文图：

```bash
.venv\Scripts\python.exe generate_paper_figures.py --features-csv outputs\new_antenna_dataset\features.csv --model outputs\new_antenna_model\new_antenna_mlp.joblib --output-dir outputs\paper_figures --loss-epochs 220
```

## 6. 论文图说明

论文图位于 `outputs/paper_figures/`：

- `figure_1_mlp_architecture.png`：MLP 神经网络结构示意图
- `figure_2_loss_curve.png`：模型训练 Loss 曲线，包含训练集和验证集
- `figure_3a_s11_min_db_scatter.png`：`S11_min_db` 真实值 vs 预测值散点图
- `figure_3b_gain_max_scatter.png`：`Gain_max` 真实值 vs 预测值散点图
- `figure_4_parameter_correlation_heatmap.png`：尺寸参数与性能指标相关性热力图
- `loss_history.csv`：Loss 曲线对应数据

## 7. 注意事项

1. 当前模型是第二种天线专用模型。若更换天线结构，需要重新准备该天线对应的数据并重新训练。
2. 增益预测已经根据用户仿真反馈做了保守校准，尤其适用于当前这批优化尺寸的验证场景。
3. S11 与增益的联合优化结果建议仍以 HFSS 或其他电磁仿真软件最终复核。
4. 如果后续补充更多增益数据，可以继续合并数据并重新训练，模型稳定性会进一步提升。

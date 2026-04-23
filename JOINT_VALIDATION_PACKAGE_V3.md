# 四组尺寸验证包说明（v3）

本次验证包是在上一轮基础上进一步修正后的版本，主要变化如下：

- 保留昨天效果较稳定的原联合优化版，作为基线方案
- 第 2 组不再使用旧的 `S 参数最优版`
- 第 2 组改为新的 `0.90 GHz` 目标频点下的 `综合 S 参数优化版`
- 保留增益优先版和联合优化局部修正版，用于观察不同侧重点下的真实表现

这次的核心目的是确认：

1. 用户真实关注频点是否更接近 `0.90 GHz`
2. 新的第 2 组是否比旧的 `S 参数最优版` 更符合真实 HFSS 排序

## 1. 基线联合优化版（建议优先验证）

这组就是用户反馈“昨天效果较好”的基线版本，建议继续作为对照组。

尺寸：

```text
2.57017, 16.8272, 2.97328, 157.277, 77.3126, 1.27523, 90.9747, 77.229
```

预测结果：

- `s11_min_db ≈ -29.1408`
- `s11_min_freq_ghz ≈ 0.9575`
- `gain_max ≈ 1.13158`

对应文件：

- `outputs/new_antenna_model/best_design_conservative.json`
- `outputs/new_antenna_model/best_design_conservative.png`

## 2. 综合 S 参数优化版（0.90 GHz 目标频点）

这组用于替代旧的 `S 参数最优版`。  
与旧版不同，这次不再只看 `s11_min_db` 最低点，而是综合考虑：

- 谐振深度
- 目标频点偏差
- `-10 dB` 带宽
- 整体平均反射水平

并且本次目标频点明确设为更贴近用户关注的 `0.90 GHz`。

尺寸：

```text
2.64656, 19.9136, 3.23610, 157.206, 152.997, 1.77862, 73.4583, 78.2912
```

预测结果：

- `s11_min_db ≈ -32.7856`
- `s11_min_freq_ghz ≈ 0.9029`
- `s11_mean_db ≈ -2.0468`
- `s11_bandwidth_below_minus10_db_ghz ≈ 0.01656`
- `gain_max ≈ 0.9486`

对应文件：

- `outputs/new_antenna_model/best_design_s11_composite_target090_v1.json`
- `outputs/new_antenna_model/best_design_s11_composite_target090_v1.png`

## 3. 增益优先版（保守修正）

这组用于在不完全放弃 S 参数的前提下，尽量把增益做高。

尺寸：

```text
3.775, 16.4032, 2.7003, 154.515, 75.0, 1.40849, 75.0019, 67.6363
```

预测结果：

- `s11_min_db ≈ -22.6774`
- `s11_min_freq_ghz ≈ 1.0191`
- `gain_max ≈ 1.19867`

对应文件：

- `outputs/new_antenna_model/best_design_gain_priority_v2.json`
- `outputs/new_antenna_model/best_design_gain_priority_v2.png`

## 4. 联合优化局部修正版

这组是基于修正后的局部训练流程重新生成的联合优化候选，用于观察它相对基线方案是否有真实提升。

尺寸：

```text
3.775, 15.6312, 1.84656, 153.898, 80.2709, 1.43235, 73.4751, 70.7027
```

预测结果：

- `s11_min_db ≈ -35.0028`
- `s11_min_freq_ghz ≈ 1.0205`
- `gain_max ≈ 1.09919`

对应文件：

- `outputs/new_antenna_model_joint_local_v2/best_design_joint_local_v2.json`
- `outputs/new_antenna_model_joint_local_v2/best_design_joint_local_v2.png`

## 建议的验证顺序

建议按下面顺序验证：

1. 先验证第 1 组基线联合优化版，用来确认昨天较优结果的稳定性
2. 再重点验证新的第 2 组，确认 `0.90 GHz` 目标频点下的综合 S 方案是否更合理
3. 如果用户更看重增益，再补验证第 3 组
4. 第 4 组作为联合优化修正版参考验证

## 本次修正重点

上一轮用户反馈说明，旧的第 2 组 `S 参数最优版` 在真实仿真中排序并不合理：

- 第 1 组和第 4 组的 S 参数实际表现反而更好
- 说明旧版 `S 最优` 目标定义过于单一，只看了最低点

因此本次第 2 组已经改为新的 `综合 S 参数优化版`，用于更贴近真实验证逻辑。

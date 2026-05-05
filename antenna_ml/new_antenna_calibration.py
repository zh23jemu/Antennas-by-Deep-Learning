from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

GAIN_MAX_INDEX = 5
GAIN_MEAN_INDEX = 6
GAIN_STD_INDEX = 7


@dataclass(frozen=True)
class GainCalibration:
    """增益后校准参数。

    采用非常轻量的线性缩放 + 偏置方式，优先把明显偏乐观的增益预测收回来。
    这样不会动 S11 预测链路，也方便后续继续用更多真实验证点微调。
    """

    gain_max_scale: float = 1.0
    gain_max_bias: float = 0.0
    gain_mean_scale: float = 1.0
    gain_mean_bias: float = 0.0
    gain_std_scale: float = 1.0
    gain_std_bias: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "gain_max_scale": float(self.gain_max_scale),
            "gain_max_bias": float(self.gain_max_bias),
            "gain_mean_scale": float(self.gain_mean_scale),
            "gain_mean_bias": float(self.gain_mean_bias),
            "gain_std_scale": float(self.gain_std_scale),
            "gain_std_bias": float(self.gain_std_bias),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GainCalibration":
        if not data:
            return cls()
        return cls(
            gain_max_scale=float(data.get("gain_max_scale", 1.0)),
            gain_max_bias=float(data.get("gain_max_bias", 0.0)),
            gain_mean_scale=float(data.get("gain_mean_scale", 1.0)),
            gain_mean_bias=float(data.get("gain_mean_bias", 0.0)),
            gain_std_scale=float(data.get("gain_std_scale", 1.0)),
            gain_std_bias=float(data.get("gain_std_bias", 0.0)),
        )


@dataclass(frozen=True)
class LocalBlendCalibration:
    """基于历史邻域样本的增益保守融合参数。"""

    enabled: bool = False
    k_neighbors: int = 12
    blend_weight: float = 0.65
    gain_max_quantile: float = 0.35
    gain_mean_quantile: float = 0.45
    gain_std_quantile: float = 0.45

    def to_dict(self) -> dict[str, float | bool | int]:
        return {
            "enabled": bool(self.enabled),
            "k_neighbors": int(self.k_neighbors),
            "blend_weight": float(self.blend_weight),
            "gain_max_quantile": float(self.gain_max_quantile),
            "gain_mean_quantile": float(self.gain_mean_quantile),
            "gain_std_quantile": float(self.gain_std_quantile),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LocalBlendCalibration":
        if not data:
            return cls()
        return cls(
            enabled=bool(data.get("enabled", False)),
            k_neighbors=int(data.get("k_neighbors", 12)),
            blend_weight=float(data.get("blend_weight", 0.65)),
            gain_max_quantile=float(data.get("gain_max_quantile", 0.35)),
            gain_mean_quantile=float(data.get("gain_mean_quantile", 0.45)),
            gain_std_quantile=float(data.get("gain_std_quantile", 0.45)),
        )


def fit_gain_calibration(y_true: np.ndarray, y_pred: np.ndarray, shrink: float = 0.6) -> GainCalibration:
    """基于验证集拟合增益校准器。

    说明：
    - 这里只校准 `gain_max / gain_mean / gain_std`
    - `shrink` 用来把校准强度收一点，避免被单次验证集拟合得过头
    - 若遇到数值不稳定，则自动回退为“只做比例缩放”
    """

    def fit_one(true_values: np.ndarray, pred_values: np.ndarray) -> tuple[float, float]:
        pred_values = np.asarray(pred_values, dtype=np.float64)
        true_values = np.asarray(true_values, dtype=np.float64)
        if pred_values.size == 0:
            return 1.0, 0.0

        pred_mean = float(np.mean(pred_values))
        true_mean = float(np.mean(true_values))
        pred_std = float(np.std(pred_values))

        if pred_std < 1e-12:
            scale = true_mean / max(pred_mean, 1e-12)
            bias = 0.0
        else:
            cov = float(np.mean((pred_values - pred_mean) * (true_values - true_mean)))
            var = float(np.var(pred_values))
            scale = cov / max(var, 1e-12)
            bias = true_mean - scale * pred_mean

        scale = (1.0 - shrink) + shrink * scale
        bias = shrink * bias
        return float(scale), float(bias)

    max_scale, max_bias = fit_one(y_true[:, GAIN_MAX_INDEX], y_pred[:, GAIN_MAX_INDEX])
    mean_scale, mean_bias = fit_one(y_true[:, GAIN_MEAN_INDEX], y_pred[:, GAIN_MEAN_INDEX])
    std_scale, std_bias = fit_one(y_true[:, GAIN_STD_INDEX], y_pred[:, GAIN_STD_INDEX])
    return GainCalibration(
        gain_max_scale=max_scale,
        gain_max_bias=max_bias,
        gain_mean_scale=mean_scale,
        gain_mean_bias=mean_bias,
        gain_std_scale=std_scale,
        gain_std_bias=std_bias,
    )


def apply_gain_calibration(predictions: np.ndarray, calibration: GainCalibration | None) -> np.ndarray:
    """对预测结果应用增益校准。"""
    calibrated = np.asarray(predictions, dtype=np.float64).copy()
    if calibration is None:
        return calibrated
    if calibrated.ndim == 1:
        calibrated = calibrated.reshape(1, -1)
        squeeze_back = True
    else:
        squeeze_back = False

    calibrated[:, GAIN_MAX_INDEX] = np.maximum(
        0.0,
        calibrated[:, GAIN_MAX_INDEX] * calibration.gain_max_scale + calibration.gain_max_bias,
    )
    calibrated[:, GAIN_MEAN_INDEX] = np.maximum(
        0.0,
        calibrated[:, GAIN_MEAN_INDEX] * calibration.gain_mean_scale + calibration.gain_mean_bias,
    )
    calibrated[:, GAIN_STD_INDEX] = np.maximum(
        0.0,
        calibrated[:, GAIN_STD_INDEX] * calibration.gain_std_scale + calibration.gain_std_bias,
    )
    if squeeze_back:
        return calibrated[0]
    return calibrated


def summarize_gain_calibration(y_true: np.ndarray, y_pred: np.ndarray, calibrated_pred: np.ndarray) -> dict[str, float]:
    """输出校准前后的增益误差摘要，便于写入训练总结。"""

    def mae(a: np.ndarray, b: np.ndarray, index: int) -> float:
        return float(np.mean(np.abs(a[:, index] - b[:, index])))

    return {
        "gain_max_mae_before": mae(y_true, y_pred, GAIN_MAX_INDEX),
        "gain_max_mae_after": mae(y_true, calibrated_pred, GAIN_MAX_INDEX),
        "gain_mean_mae_before": mae(y_true, y_pred, GAIN_MEAN_INDEX),
        "gain_mean_mae_after": mae(y_true, calibrated_pred, GAIN_MEAN_INDEX),
        "gain_std_mae_before": mae(y_true, y_pred, GAIN_STD_INDEX),
        "gain_std_mae_after": mae(y_true, calibrated_pred, GAIN_STD_INDEX),
    }


def apply_local_gain_blend(
    dimensions: np.ndarray,
    predictions: np.ndarray,
    reference_dimensions: np.ndarray,
    reference_targets: np.ndarray,
    calibration: LocalBlendCalibration | None,
) -> np.ndarray:
    """利用邻域真实样本对增益预测做保守融合。

    设计思路：
    - S11 仍然完全使用模型输出；
    - 增益仅在推理阶段参考“附近真实样本”的统计水平；
    - 对 `gain_max` 使用更保守的较低分位数，优先压掉虚高峰值；
    - 对 `gain_mean / gain_std` 使用中位附近分位数，尽量保留趋势。
    """
    adjusted = np.asarray(predictions, dtype=np.float64).copy()
    if calibration is None or not calibration.enabled:
        return adjusted
    if adjusted.ndim == 1:
        adjusted = adjusted.reshape(1, -1)
        query_dimensions = np.asarray(dimensions, dtype=np.float64).reshape(1, -1)
        squeeze_back = True
    else:
        query_dimensions = np.asarray(dimensions, dtype=np.float64)
        squeeze_back = False

    ref_x = np.asarray(reference_dimensions, dtype=np.float64)
    ref_y = np.asarray(reference_targets, dtype=np.float64)
    if ref_x.size == 0 or ref_y.size == 0:
        return adjusted[0] if squeeze_back else adjusted

    safe_span = np.maximum(ref_x.max(axis=0) - ref_x.min(axis=0), 1e-12)
    k = max(1, min(int(calibration.k_neighbors), ref_x.shape[0]))
    blend_weight = float(np.clip(calibration.blend_weight, 0.0, 1.0))

    for row_index, query in enumerate(query_dimensions):
        distances = np.sqrt((((ref_x - query) / safe_span) ** 2).sum(axis=1))
        neighbor_index = np.argsort(distances)[:k]
        local_targets = ref_y[neighbor_index]

        local_gain_max = float(np.quantile(local_targets[:, GAIN_MAX_INDEX], calibration.gain_max_quantile))
        local_gain_mean = float(np.quantile(local_targets[:, GAIN_MEAN_INDEX], calibration.gain_mean_quantile))
        local_gain_std = float(np.quantile(local_targets[:, GAIN_STD_INDEX], calibration.gain_std_quantile))

        adjusted[row_index, GAIN_MAX_INDEX] = (1.0 - blend_weight) * adjusted[row_index, GAIN_MAX_INDEX] + blend_weight * local_gain_max
        adjusted[row_index, GAIN_MEAN_INDEX] = (1.0 - blend_weight) * adjusted[row_index, GAIN_MEAN_INDEX] + blend_weight * local_gain_mean
        adjusted[row_index, GAIN_STD_INDEX] = (1.0 - blend_weight) * adjusted[row_index, GAIN_STD_INDEX] + blend_weight * local_gain_std

    adjusted[:, GAIN_MAX_INDEX] = np.maximum(0.0, adjusted[:, GAIN_MAX_INDEX])
    adjusted[:, GAIN_MEAN_INDEX] = np.maximum(0.0, adjusted[:, GAIN_MEAN_INDEX])
    adjusted[:, GAIN_STD_INDEX] = np.maximum(0.0, adjusted[:, GAIN_STD_INDEX])
    if squeeze_back:
        return adjusted[0]
    return adjusted

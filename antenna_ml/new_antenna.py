from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DIMENSION_COLUMNS = [
    "cut_x [mm]",
    "cut_y [mm]",
    "fw [mm]",
    "gx [mm]",
    "gy [mm]",
    "h1 [mm]",
    "px [mm]",
    "py [mm]",
]

TARGET_COLUMNS = [
    "s11_min_db",
    "s11_min_freq_ghz",
    "s11_mean_db",
    "s11_std_db",
    "s11_bandwidth_below_minus10_db_ghz",
    "gain_max",
    "gain_mean",
    "gain_std",
]


@dataclass(frozen=True)
class NewAntennaDataset:
    dataframe: pd.DataFrame
    dimensions: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class ObjectiveNormalizer:
    s11_min: float
    s11_max: float
    s11_mean_min: float
    s11_mean_max: float
    s11_bw_min: float
    s11_bw_max: float
    s11_freq_error_min: float
    s11_freq_error_max: float
    effective_gain_min: float
    effective_gain_max: float

    def normalize_s11(self, value: float) -> float:
        span = max(self.s11_max - self.s11_min, 1e-12)
        return (value - self.s11_min) / span

    def normalize_s11_mean(self, value: float) -> float:
        span = max(self.s11_mean_max - self.s11_mean_min, 1e-12)
        return (value - self.s11_mean_min) / span

    def normalize_s11_bandwidth(self, value: float) -> float:
        span = max(self.s11_bw_max - self.s11_bw_min, 1e-12)
        return (value - self.s11_bw_min) / span

    def normalize_s11_freq_error(self, value: float) -> float:
        span = max(self.s11_freq_error_max - self.s11_freq_error_min, 1e-12)
        return (value - self.s11_freq_error_min) / span

    def normalize_effective_gain(self, value: float) -> float:
        span = max(self.effective_gain_max - self.effective_gain_min, 1e-12)
        return (value - self.effective_gain_min) / span


def load_new_antenna_features(csv_path: Path) -> NewAntennaDataset:
    dataframe = pd.read_csv(csv_path)
    return NewAntennaDataset(
        dataframe=dataframe,
        dimensions=dataframe[DIMENSION_COLUMNS].to_numpy(dtype=np.float64),
        targets=dataframe[TARGET_COLUMNS].to_numpy(dtype=np.float64),
    )


def dimension_bounds(dimensions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return dimensions.min(axis=0), dimensions.max(axis=0)


def build_objective_normalizer(dataframe: pd.DataFrame, target_freq_ghz: float = 0.95) -> ObjectiveNormalizer:
    s11_low = float(dataframe["s11_min_db"].quantile(0.05))
    s11_high = float(dataframe["s11_min_db"].quantile(0.95))
    s11_mean_low = float(dataframe["s11_mean_db"].quantile(0.05))
    s11_mean_high = float(dataframe["s11_mean_db"].quantile(0.95))
    s11_bw_low = float(dataframe["s11_bandwidth_below_minus10_db_ghz"].quantile(0.05))
    s11_bw_high = float(dataframe["s11_bandwidth_below_minus10_db_ghz"].quantile(0.95))
    s11_freq_error = np.abs(dataframe["s11_min_freq_ghz"].to_numpy(dtype=np.float64) - target_freq_ghz)
    s11_freq_error_low = float(np.quantile(s11_freq_error, 0.05))
    s11_freq_error_high = float(np.quantile(s11_freq_error, 0.95))
    effective_gain = build_effective_gain(
        dataframe["gain_max"].to_numpy(dtype=np.float64),
        dataframe["gain_mean"].to_numpy(dtype=np.float64),
        alpha=0.5,
    )
    gain_low = float(np.quantile(effective_gain, 0.05))
    gain_high = float(np.quantile(effective_gain, 0.95))
    return ObjectiveNormalizer(
        s11_min=s11_low,
        s11_max=s11_high,
        s11_mean_min=s11_mean_low,
        s11_mean_max=s11_mean_high,
        s11_bw_min=s11_bw_low,
        s11_bw_max=s11_bw_high,
        s11_freq_error_min=s11_freq_error_low,
        s11_freq_error_max=s11_freq_error_high,
        effective_gain_min=gain_low,
        effective_gain_max=gain_high,
    )


def build_effective_gain(gain_max: np.ndarray | float, gain_mean: np.ndarray | float, alpha: float = 0.5) -> np.ndarray | float:
    """Conservative gain proxy.

    alpha=0 uses gain_mean only; alpha=1 uses gain_max only.
    """
    return np.asarray(gain_mean) + alpha * (np.asarray(gain_max) - np.asarray(gain_mean))


def score_prediction(
    target_vector: np.ndarray,
    normalizer: ObjectiveNormalizer,
    s11_weight: float,
    gain_weight: float,
    gain_alpha: float = 0.5,
) -> float:
    s11_min_db = float(target_vector[0])
    gain_max = float(target_vector[5])
    gain_mean = float(target_vector[6])
    normalized_s11 = normalizer.normalize_s11(s11_min_db)
    effective_gain = float(build_effective_gain(gain_max, gain_mean, alpha=gain_alpha))
    normalized_gain = normalizer.normalize_effective_gain(effective_gain)
    return s11_weight * normalized_s11 - gain_weight * normalized_gain


def score_s11_prediction(
    target_vector: np.ndarray,
    normalizer: ObjectiveNormalizer,
    target_freq_ghz: float = 0.95,
    s11_min_weight: float = 0.45,
    freq_weight: float = 0.30,
    bandwidth_weight: float = 0.15,
    mean_weight: float = 0.10,
) -> float:
    """计算综合 S 参数评分。

    评分越小越好。

    这里不再只看 `s11_min_db` 的最低点，而是同时考虑：
    - 谐振深度：最低点更深通常更好
    - 频点偏差：离目标频点越近越好
    - 带宽：`-10 dB` 带宽越宽越好
    - 整体反射水平：`s11_mean_db` 更低通常意味着整段曲线更稳
    """
    s11_min_db = float(target_vector[0])
    s11_min_freq_ghz = float(target_vector[1])
    s11_mean_db = float(target_vector[2])
    s11_bandwidth = float(target_vector[4])

    normalized_min = normalizer.normalize_s11(s11_min_db)
    normalized_mean = normalizer.normalize_s11_mean(s11_mean_db)
    normalized_bandwidth = normalizer.normalize_s11_bandwidth(s11_bandwidth)
    normalized_freq_error = normalizer.normalize_s11_freq_error(abs(s11_min_freq_ghz - target_freq_ghz))

    return (
        s11_min_weight * normalized_min
        + freq_weight * normalized_freq_error
        - bandwidth_weight * normalized_bandwidth
        + mean_weight * normalized_mean
    )

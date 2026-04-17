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
    gain_min: float
    gain_max: float

    def normalize_s11(self, value: float) -> float:
        span = max(self.s11_max - self.s11_min, 1e-12)
        return (value - self.s11_min) / span

    def normalize_gain(self, value: float) -> float:
        span = max(self.gain_max - self.gain_min, 1e-12)
        return (value - self.gain_min) / span


def load_new_antenna_features(csv_path: Path) -> NewAntennaDataset:
    dataframe = pd.read_csv(csv_path)
    return NewAntennaDataset(
        dataframe=dataframe,
        dimensions=dataframe[DIMENSION_COLUMNS].to_numpy(dtype=np.float64),
        targets=dataframe[TARGET_COLUMNS].to_numpy(dtype=np.float64),
    )


def dimension_bounds(dimensions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return dimensions.min(axis=0), dimensions.max(axis=0)


def build_objective_normalizer(dataframe: pd.DataFrame) -> ObjectiveNormalizer:
    s11_low = float(dataframe["s11_min_db"].quantile(0.05))
    s11_high = float(dataframe["s11_min_db"].quantile(0.95))
    gain_low = float(dataframe["gain_max"].quantile(0.05))
    gain_high = float(dataframe["gain_max"].quantile(0.95))
    return ObjectiveNormalizer(
        s11_min=s11_low,
        s11_max=s11_high,
        gain_min=gain_low,
        gain_max=gain_high,
    )


def score_prediction(
    target_vector: np.ndarray,
    normalizer: ObjectiveNormalizer,
    s11_weight: float,
    gain_weight: float,
) -> float:
    s11_min_db = float(target_vector[0])
    gain_max = float(target_vector[5])
    normalized_s11 = normalizer.normalize_s11(s11_min_db)
    normalized_gain = normalizer.normalize_gain(gain_max)
    return s11_weight * normalized_s11 - gain_weight * normalized_gain

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


PARAMETER_KEY = "generate_parameter"
S_VALUE_KEY = "s_value"


@dataclass(frozen=True)
class AntennaDataset:
    dimensions: np.ndarray
    s_values: np.ndarray
    source_files: list[str]


def extract_s_features(s_values: np.ndarray) -> np.ndarray:
    min_values = np.min(s_values, axis=1)
    max_index = max(1, s_values.shape[1] - 1)
    min_indices = np.argmin(s_values, axis=1).astype(np.float64) / max_index
    mean_values = np.mean(s_values, axis=1)
    std_values = np.std(s_values, axis=1)
    return np.column_stack((min_values, min_indices, mean_values, std_values))


def decode_s_features(features: np.ndarray, s_point_count: int) -> np.ndarray:
    decoded = np.asarray(features, dtype=np.float64).copy()
    max_index = max(1, s_point_count - 1)
    decoded[..., 1] = np.clip(np.rint(decoded[..., 1] * max_index), 0, max_index)
    return decoded


def _read_matrix(file_path: Path, key: str) -> np.ndarray:
    with h5py.File(file_path, "r") as handle:
        if key not in handle:
            available = ", ".join(handle.keys())
            raise KeyError(f"{file_path} 中没有字段 {key!r}。可用字段: {available}")
        return np.asarray(handle[key], dtype=np.float64)


def _as_samples_by_features(matrix: np.ndarray, expected_samples: int | None = None) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError(f"只支持二维数组，实际形状为 {matrix.shape}")
    if expected_samples is not None:
        if matrix.shape[0] == expected_samples:
            return matrix
        if matrix.shape[1] == expected_samples:
            return matrix.T
    return matrix.T if matrix.shape[0] < matrix.shape[1] else matrix


def load_dataset(data_paths: list[Path]) -> AntennaDataset:
    dimensions_list: list[np.ndarray] = []
    s_values_list: list[np.ndarray] = []

    for file_path in data_paths:
        raw_dimensions = _read_matrix(file_path, PARAMETER_KEY)
        raw_s_values = _read_matrix(file_path, S_VALUE_KEY)
        sample_count = min(raw_dimensions.shape)

        dimensions = _as_samples_by_features(raw_dimensions)
        s_values = _as_samples_by_features(raw_s_values, expected_samples=dimensions.shape[0])

        if dimensions.shape[0] != s_values.shape[0]:
            raise ValueError(
                f"{file_path} 的样本数不一致: 尺寸 {dimensions.shape}, S参数 {s_values.shape}"
            )
        if dimensions.shape[0] != sample_count and sample_count in raw_s_values.shape:
            s_values = _as_samples_by_features(raw_s_values, expected_samples=sample_count)

        dimensions_list.append(dimensions)
        s_values_list.append(s_values)

    return AntennaDataset(
        dimensions=np.vstack(dimensions_list),
        s_values=np.vstack(s_values_list),
        source_files=[str(path) for path in data_paths],
    )


def default_data_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("*.h5"))
    if not paths:
        raise FileNotFoundError(f"未在 {data_dir} 中找到 .h5 数据文件")
    return paths


def parameter_bounds(dimensions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return dimensions.min(axis=0), dimensions.max(axis=0)

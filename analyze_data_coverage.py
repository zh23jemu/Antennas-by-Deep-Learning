from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from antenna_ml.data import default_data_paths, load_dataset
from antenna_ml.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析样本重复情况和参数覆盖区间")
    parser.add_argument("--data-dir", type=Path, action="append", dest="data_dirs")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "data_coverage_summary.json")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def read_raw_dimensions(data_paths: list[Path]) -> np.ndarray:
    rows: list[np.ndarray] = []
    for file_path in data_paths:
        with h5py.File(file_path, "r") as handle:
            matrix = np.asarray(handle["generate_parameter"], dtype=np.float64)
            rows.append(matrix.T if matrix.shape[0] < matrix.shape[1] else matrix)
    return np.vstack(rows)


def build_duplicate_summary(raw_dimensions: np.ndarray, top_k: int) -> list[dict[str, object]]:
    rounded_rows = [tuple(np.round(row, 12)) for row in raw_dimensions]
    counter = Counter(rounded_rows)
    summary: list[dict[str, object]] = []
    for values, count in counter.most_common(top_k):
        summary.append(
            {
                "repeat_count": int(count),
                "dimensions": list(values),
            }
        )
    return summary


def build_parameter_coverage(unique_dimensions: np.ndarray) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for index in range(unique_dimensions.shape[1]):
        values = np.unique(np.round(unique_dimensions[:, index], 12))
        value_counts = Counter(np.round(unique_dimensions[:, index], 12))
        sorted_counts = sorted(value_counts.items(), key=lambda item: (item[1], item[0]))
        sparse_values = [{"value": float(value), "count": int(count)} for value, count in sorted_counts[:3]]
        diffs = np.diff(values)
        largest_gap = float(diffs.max()) if len(diffs) else 0.0
        gap_start = float(values[np.argmax(diffs)]) if len(diffs) else float(values[0])
        gap_end = float(values[np.argmax(diffs) + 1]) if len(diffs) else float(values[0])
        summary.append(
            {
                "parameter_index": index + 1,
                "min": float(values.min()),
                "max": float(values.max()),
                "unique_value_count": int(len(values)),
                "sparsest_values": sparse_values,
                "largest_gap": largest_gap,
                "largest_gap_range": [gap_start, gap_end],
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    data_dirs = args.data_dirs or [Path("样本数据")]
    data_paths = default_data_paths(data_dirs)
    dataset = load_dataset(data_paths)
    raw_dimensions = read_raw_dimensions(data_paths)

    duplicate_summary = build_duplicate_summary(raw_dimensions, args.top_k)
    parameter_coverage = build_parameter_coverage(dataset.dimensions)

    result = {
        "data_dirs": [str(path) for path in data_dirs],
        "source_files": [str(path) for path in data_paths],
        "raw_sample_count": dataset.raw_sample_count,
        "deduplicated_sample_count": dataset.deduplicated_sample_count,
        "removed_duplicate_count": dataset.removed_duplicate_count,
        "top_duplicate_parameter_sets": duplicate_summary,
        "parameter_coverage": parameter_coverage,
    }
    write_json(args.output, result)

    print("分析完成")
    print("数据目录: " + ", ".join(str(path) for path in data_dirs))
    print(f"原始样本数: {dataset.raw_sample_count}")
    print(f"去重后样本数: {dataset.deduplicated_sample_count}")
    print(f"去除重复样本数: {dataset.removed_duplicate_count}")
    print(f"重复最严重的参数组合已写入: {args.output}")


if __name__ == "__main__":
    main()

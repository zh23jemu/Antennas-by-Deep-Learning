from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from antenna_ml.io import write_json


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
S11_FREQ_COLUMN = "Freq [GHz]"
S11_VALUE_COLUMN = "dB(S(1,1)) []"
GAIN_FREQ_COLUMN = "Freq [GHz]"
GAIN_PHI_COLUMN = "Phi [deg]"
GAIN_THETA_COLUMN = "Theta [deg]"
GAIN_VALUE_COLUMN = "GainTotal []"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理新天线 Excel 数据并提取第一版 S11 + 增益特征")
    parser.add_argument("--s11-file", type=Path, default=Path("data1.xlsx"))
    parser.add_argument("--gain-file", type=Path, default=Path("data2.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "new_antenna_dataset")
    return parser.parse_args()


def build_s11_features(s11_group: pd.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    ordered = s11_group.sort_values(S11_FREQ_COLUMN)
    freqs = ordered[S11_FREQ_COLUMN].to_numpy(dtype=np.float64)
    s11_values = ordered[S11_VALUE_COLUMN].to_numpy(dtype=np.float64)
    min_index = int(np.argmin(s11_values))
    bandwidth_mask = s11_values <= -10.0
    bandwidth = float(freqs[bandwidth_mask].max() - freqs[bandwidth_mask].min()) if bandwidth_mask.any() else 0.0
    features = {
        "s11_min_db": float(s11_values[min_index]),
        "s11_min_freq_ghz": float(freqs[min_index]),
        "s11_mean_db": float(np.mean(s11_values)),
        "s11_std_db": float(np.std(s11_values)),
        "s11_bandwidth_below_minus10_db_ghz": bandwidth,
    }
    return features, s11_values


def build_gain_features(gain_group: pd.DataFrame) -> tuple[dict[str, float], np.ndarray]:
    ordered = gain_group.sort_values([GAIN_PHI_COLUMN, GAIN_THETA_COLUMN])
    gain_values = ordered[GAIN_VALUE_COLUMN].to_numpy(dtype=np.float64)
    max_index = int(np.argmax(gain_values))
    mean_index = int(np.argmin(np.abs(gain_values - np.mean(gain_values))))
    features = {
        "gain_max": float(np.max(gain_values)),
        "gain_mean": float(np.mean(gain_values)),
        "gain_std": float(np.std(gain_values)),
        "gain_phi_at_max_deg": float(ordered.iloc[max_index][GAIN_PHI_COLUMN]),
        "gain_theta_at_max_deg": float(ordered.iloc[max_index][GAIN_THETA_COLUMN]),
        "gain_phi_at_mean_deg": float(ordered.iloc[mean_index][GAIN_PHI_COLUMN]),
        "gain_theta_at_mean_deg": float(ordered.iloc[mean_index][GAIN_THETA_COLUMN]),
    }
    return features, gain_values


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    s11_df = pd.read_excel(args.s11_file)
    gain_df = pd.read_excel(args.gain_file)

    s11_groups = {key: group for key, group in s11_df.groupby(DIMENSION_COLUMNS, sort=True)}
    gain_groups = {key: group for key, group in gain_df.groupby(DIMENSION_COLUMNS, sort=True)}
    common_keys = sorted(set(s11_groups) & set(gain_groups))

    if not common_keys:
        raise ValueError("S11 和增益数据中没有找到共同的尺寸参数组合")

    rows: list[dict[str, float]] = []
    s11_curves: list[np.ndarray] = []
    gain_patterns: list[np.ndarray] = []

    reference_freqs = None
    for key in common_keys:
        s11_features, s11_curve = build_s11_features(s11_groups[key])
        gain_features, gain_pattern = build_gain_features(gain_groups[key])
        row = {column: float(value) for column, value in zip(DIMENSION_COLUMNS, key)}
        row.update(s11_features)
        row.update(gain_features)
        rows.append(row)
        s11_curves.append(s11_curve)
        gain_patterns.append(gain_pattern)

        if reference_freqs is None:
            reference_freqs = (
                s11_groups[key]
                .sort_values(S11_FREQ_COLUMN)[S11_FREQ_COLUMN]
                .to_numpy(dtype=np.float64)
            )

    feature_df = pd.DataFrame(rows)
    feature_df.to_csv(args.output_dir / "features.csv", index=False, encoding="utf-8-sig")
    np.save(args.output_dir / "s11_curves.npy", np.vstack(s11_curves))
    np.save(args.output_dir / "gain_patterns.npy", np.vstack(gain_patterns))
    np.save(args.output_dir / "s11_freqs_ghz.npy", reference_freqs)

    summary = {
        "s11_file": str(args.s11_file),
        "gain_file": str(args.gain_file),
        "sample_count": int(len(common_keys)),
        "dimension_columns": DIMENSION_COLUMNS,
        "s11_feature_columns": [
            "s11_min_db",
            "s11_min_freq_ghz",
            "s11_mean_db",
            "s11_std_db",
            "s11_bandwidth_below_minus10_db_ghz",
        ],
        "gain_feature_columns": [
            "gain_max",
            "gain_mean",
            "gain_std",
            "gain_phi_at_max_deg",
            "gain_theta_at_max_deg",
            "gain_phi_at_mean_deg",
            "gain_theta_at_mean_deg",
        ],
        "s11_curve_point_count": int(np.vstack(s11_curves).shape[1]),
        "gain_pattern_point_count": int(np.vstack(gain_patterns).shape[1]),
        "output_files": {
            "features_csv": str(args.output_dir / "features.csv"),
            "s11_curves_npy": str(args.output_dir / "s11_curves.npy"),
            "gain_patterns_npy": str(args.output_dir / "gain_patterns.npy"),
            "s11_freqs_ghz_npy": str(args.output_dir / "s11_freqs_ghz.npy"),
        },
    }
    write_json(args.output_dir / "summary.json", summary)

    print("整理完成")
    print(f"共同尺寸样本数: {len(common_keys)}")
    print(f"S11 曲线点数: {np.vstack(s11_curves).shape[1]}")
    print(f"增益方向图点数: {np.vstack(gain_patterns).shape[1]}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

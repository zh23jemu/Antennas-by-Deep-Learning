from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from antenna_ml.io import write_json
from antenna_ml.model import save_model, train_model
from antenna_ml.new_antenna import DIMENSION_COLUMNS, TARGET_COLUMNS
from antenna_ml.new_antenna_plotting import plot_feature_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为联合优化训练局部增强版新天线模型")
    parser.add_argument("--features-csv", type=Path, default=Path("outputs") / "new_antenna_dataset" / "features.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "new_antenna_model_joint_local")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--compare-count", type=int, default=3)
    parser.add_argument("--s11-threshold-db", type=float, default=-20.0)
    parser.add_argument("--frontier-quantile", type=float, default=0.85)
    parser.add_argument("--feasible-repeat", type=int, default=2, help="对满足 S11 门限的样本重复采样次数，用于加强可行区域学习")
    parser.add_argument("--frontier-repeat", type=int, default=6, help="对高增益前沿样本重复采样次数，用于强化联合最优区域")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features_csv)
    feasible = df[df["s11_min_db"] <= args.s11_threshold_db].copy()
    effective_gain = feasible["gain_mean"] + 0.2 * (feasible["gain_max"] - feasible["gain_mean"])
    cutoff = effective_gain.quantile(args.frontier_quantile)
    frontier = feasible.loc[effective_gain >= cutoff].copy()

    # 这里不再丢弃重复样本。
    # 原来的 `drop_duplicates()` 会把 frontier 的“强调作用”完全抹掉，
    # 实际训练集又退回到单纯的 feasible 子集，导致样本量骤减、泛化明显变差。
    # 改为保留全量样本作为全局基线，再对可行样本和前沿样本做重复采样，
    # 既保留全局几何-性能关系，又让模型更关注真正值得优化的局部高质量区域。
    training_parts: list[pd.DataFrame] = [df]
    if not feasible.empty and args.feasible_repeat > 1:
        training_parts.extend([feasible.copy() for _ in range(args.feasible_repeat - 1)])
    if not frontier.empty and args.frontier_repeat > 1:
        training_parts.extend([frontier.copy() for _ in range(args.frontier_repeat - 1)])
    training_df = pd.concat(training_parts, ignore_index=True)

    dimensions = training_df[DIMENSION_COLUMNS].to_numpy(dtype=np.float64)
    targets = training_df[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    result = train_model(
        dimensions=dimensions,
        targets=targets,
        random_state=args.random_state,
        max_iter=args.max_iter,
        test_size=args.test_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "new_antenna_joint_local_mlp.joblib"
    save_model(result.model, model_path)

    compare_dir = args.output_dir / "validation_plots"
    compare_count = max(0, min(args.compare_count, result.y_valid.shape[0]))
    sample_indices = np.linspace(0, result.y_valid.shape[0] - 1, num=compare_count, dtype=int) if compare_count else np.array([], dtype=int)
    compare_records: list[dict[str, object]] = []
    for plot_number, sample_index in enumerate(sample_indices, start=1):
        plot_path = compare_dir / f"validation_compare_{plot_number}.png"
        plot_feature_comparison(
            true_values=result.y_valid[sample_index],
            predicted_values=result.y_pred[sample_index],
            output_path=plot_path,
            title=f"Joint Local Validation Sample {plot_number}",
        )
        compare_records.append(
            {
                "sample_index": int(sample_index),
                "plot_path": str(plot_path),
                "dimensions": result.x_valid[sample_index],
                "true_targets": result.y_valid[sample_index],
                "predicted_targets": result.y_pred[sample_index],
            }
        )

    write_json(
        args.output_dir / "training_summary.json",
        {
            "features_csv": str(args.features_csv),
            "model_path": str(model_path),
            "base_sample_count": int(df.shape[0]),
            "feasible_sample_count": int(feasible.shape[0]),
            "frontier_sample_count": int(frontier.shape[0]),
            "feasible_repeat": int(args.feasible_repeat),
            "frontier_repeat": int(args.frontier_repeat),
            "local_training_sample_count": int(training_df.shape[0]),
            "dimension_columns": DIMENSION_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "metrics": result.metrics,
            "validation_compare_plots": compare_records,
        },
    )

    print("联合优化局部模型训练完成")
    print(f"原始样本数: {df.shape[0]}")
    print(f"可行样本数: {feasible.shape[0]}")
    print(f"前沿样本数: {frontier.shape[0]}")
    print(f"局部训练样本数: {training_df.shape[0]}")
    print(f"验证集 MSE: {result.metrics['valid_mse']:.6g}")
    print(f"验证集 MAE: {result.metrics['valid_mae']:.6g}")
    print(f"模型已保存: {model_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from antenna_ml.io import write_json
from antenna_ml.model import save_model, train_model
from antenna_ml.new_antenna import DIMENSION_COLUMNS, TARGET_COLUMNS, dimension_bounds, load_new_antenna_features
from antenna_ml.new_antenna_plotting import plot_feature_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练新天线专用的 S11 + 增益特征 MLP 模型")
    parser.add_argument("--features-csv", type=Path, default=Path("outputs") / "new_antenna_dataset" / "features.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "new_antenna_model")
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--compare-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_new_antenna_features(args.features_csv)
    result = train_model(
        dimensions=dataset.dimensions,
        targets=dataset.targets,
        random_state=args.random_state,
        max_iter=args.max_iter,
        test_size=args.test_size,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "new_antenna_mlp.joblib"
    save_model(result.model, model_path)
    lower_bounds, upper_bounds = dimension_bounds(dataset.dimensions)

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
            title=f"New Antenna Validation Sample {plot_number}",
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
            "sample_count": int(dataset.dimensions.shape[0]),
            "dimension_columns": DIMENSION_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "x_train_shape": result.x_train_shape,
            "y_train_shape": result.y_train_shape,
            "x_valid_shape": result.x_valid_shape,
            "y_valid_shape": result.y_valid_shape,
            "metrics": result.metrics,
            "parameter_lower_bounds": lower_bounds,
            "parameter_upper_bounds": upper_bounds,
            "validation_compare_plots": compare_records,
        },
    )

    print("新天线模型训练完成")
    print(f"样本数: {dataset.dimensions.shape[0]}")
    print(f"输入参数数: {dataset.dimensions.shape[1]}")
    print(f"输出特征数: {dataset.targets.shape[1]}")
    print(f"验证集 MSE: {result.metrics['valid_mse']:.6g}")
    print(f"验证集 MAE: {result.metrics['valid_mae']:.6g}")
    print(f"模型已保存: {model_path}")


if __name__ == "__main__":
    main()

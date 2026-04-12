from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from antenna_ml.data import default_data_paths, load_dataset, parameter_bounds
from antenna_ml.io import write_json
from antenna_ml.model import save_model, train_model
from antenna_ml.plotting import plot_true_vs_predicted_s_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练小型天线 S 参数 MLP 代理模型")
    parser.add_argument("--data-dir", type=Path, default=Path("样本数据"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--compare-count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_paths = default_data_paths(args.data_dir)
    dataset = load_dataset(data_paths)
    result = train_model(
        dimensions=dataset.dimensions,
        s_values=dataset.s_values,
        random_state=args.random_state,
        max_iter=args.max_iter,
        test_size=args.test_size,
    )

    model_path = args.output_dir / "antenna_mlp.joblib"
    save_model(result.model, model_path)
    lower_bounds, upper_bounds = parameter_bounds(dataset.dimensions)
    compare_dir = args.output_dir / "validation_plots"
    compare_count = max(0, min(args.compare_count, result.y_valid.shape[0]))
    sample_indices = np.linspace(0, result.y_valid.shape[0] - 1, num=compare_count, dtype=int) if compare_count else np.array([], dtype=int)

    compare_records: list[dict[str, object]] = []
    for plot_number, sample_index in enumerate(sample_indices, start=1):
        plot_path = compare_dir / f"validation_compare_{plot_number}.png"
        plot_true_vs_predicted_s_curve(
            true_s_values=result.y_valid[sample_index],
            predicted_s_values=result.y_pred[sample_index],
            output_path=plot_path,
            title=f"Validation Sample {plot_number}: True vs Predicted",
        )
        compare_records.append(
            {
                "sample_index": int(sample_index),
                "plot_path": str(plot_path),
                "dimensions": result.x_valid[sample_index],
                "true_min_s_value": float(result.y_valid[sample_index].min()),
                "predicted_min_s_value": float(result.y_pred[sample_index].min()),
            }
        )

    write_json(
        args.output_dir / "training_summary.json",
        {
            "source_files": dataset.source_files,
            "model_path": str(model_path),
            "sample_count": dataset.dimensions.shape[0],
            "dimension_count": dataset.dimensions.shape[1],
            "s_parameter_points": dataset.s_values.shape[1],
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

    print("训练完成")
    print(f"样本数: {dataset.dimensions.shape[0]}")
    print(f"尺寸参数维度: {dataset.dimensions.shape[1]}")
    print(f"S参数采样点: {dataset.s_values.shape[1]}")
    print(f"验证集 MSE: {result.metrics['valid_mse']:.6g}")
    print(f"验证集 MAE: {result.metrics['valid_mae']:.6g}")
    print(f"模型已保存: {model_path}")
    if compare_records:
        print(f"验证集对比图数量: {len(compare_records)}")
        print(f"对比图目录: {compare_dir}")


if __name__ == "__main__":
    main()

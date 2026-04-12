from __future__ import annotations

import argparse
from pathlib import Path

from antenna_ml.data import default_data_paths, load_dataset, parameter_bounds
from antenna_ml.io import write_json
from antenna_ml.model import save_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练小型天线 S 参数 MLP 代理模型")
    parser.add_argument("--data-dir", type=Path, default=Path("样本数据"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
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
        },
    )

    print("训练完成")
    print(f"样本数: {dataset.dimensions.shape[0]}")
    print(f"尺寸参数维度: {dataset.dimensions.shape[1]}")
    print(f"S参数采样点: {dataset.s_values.shape[1]}")
    print(f"验证集 MSE: {result.metrics['valid_mse']:.6g}")
    print(f"验证集 MAE: {result.metrics['valid_mae']:.6g}")
    print(f"模型已保存: {model_path}")


if __name__ == "__main__":
    main()

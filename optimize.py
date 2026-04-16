from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from antenna_ml.data import decode_s_features, default_data_paths, load_dataset, parameter_bounds
from antenna_ml.io import write_json
from antenna_ml.model import load_model
from antenna_ml.optimize import random_search
from antenna_ml.plotting import plot_predicted_feature_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 MLP 代理模型搜索预测关键 S 参数特征最优的天线尺寸")
    parser.add_argument("--model", type=Path, default=Path("outputs") / "antenna_mlp.joblib")
    parser.add_argument("--data-dir", type=Path, default=Path("样本数据"))
    parser.add_argument("--output", type=Path, default=Path("outputs") / "best_design.json")
    parser.add_argument("--plot", type=Path, default=Path("outputs") / "best_design_features.png")
    parser.add_argument("--n-candidates", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(default_data_paths(args.data_dir))
    lower_bounds, upper_bounds = parameter_bounds(dataset.dimensions)
    model = load_model(args.model)
    best_dimensions, best_features, best_score, best_point_index = random_search(
        model=model,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        n_candidates=args.n_candidates,
        random_state=args.random_state,
    )

    decoded_features = decode_s_features(best_features, dataset.s_values.shape[1])
    best_point_index = int(decoded_features[1])
    write_json(
        args.output,
        {
            "best_dimensions": best_dimensions,
            "predicted_features": {
                "min_s_value": float(decoded_features[0]),
                "min_point_index": best_point_index,
                "mean_s_value": float(decoded_features[2]),
                "std_s_value": float(decoded_features[3]),
            },
            "best_s_value": float(decoded_features[0]),
            "best_point_index": best_point_index,
            "search_candidates": args.n_candidates,
            "parameter_lower_bounds": lower_bounds,
            "parameter_upper_bounds": upper_bounds,
        },
    )
    plot_predicted_feature_summary(decoded_features, args.plot, "Optimized Design Predicted S Parameter Features")
    print("优化完成")
    print("推荐尺寸: " + ", ".join(f"{value:.6g}" for value in best_dimensions))
    print(f"预测最低 S 参数值: {float(decoded_features[0]):.6g}")
    print(f"最低点索引: {best_point_index}")
    print(f"结果已保存: {args.output}")
    print(f"曲线图已保存: {args.plot}")


if __name__ == "__main__":
    main()

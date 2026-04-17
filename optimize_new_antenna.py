from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from antenna_ml.io import write_json
from antenna_ml.model import load_model
from antenna_ml.new_antenna import (
    TARGET_COLUMNS,
    build_objective_normalizer,
    dimension_bounds,
    load_new_antenna_features,
    score_prediction,
)
from antenna_ml.new_antenna_plotting import plot_prediction_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于新天线专用模型搜索 S11 + 增益联合最优尺寸")
    parser.add_argument("--features-csv", type=Path, default=Path("outputs") / "new_antenna_dataset" / "features.csv")
    parser.add_argument("--model", type=Path, default=Path("outputs") / "new_antenna_model" / "new_antenna_mlp.joblib")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "new_antenna_model" / "best_design.json")
    parser.add_argument("--plot", type=Path, default=Path("outputs") / "new_antenna_model" / "best_design_features.png")
    parser.add_argument("--n-candidates", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--s11-weight", type=float, default=0.5)
    parser.add_argument("--gain-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_new_antenna_features(args.features_csv)
    lower_bounds, upper_bounds = dimension_bounds(dataset.dimensions)
    normalizer = build_objective_normalizer(dataset.dataframe)
    model = load_model(args.model)

    rng = np.random.default_rng(args.random_state)
    candidates = rng.uniform(lower_bounds, upper_bounds, size=(args.n_candidates, lower_bounds.size))
    predictions = np.asarray(model.predict(candidates), dtype=np.float64)
    scores = np.array(
        [score_prediction(row, normalizer, args.s11_weight, args.gain_weight) for row in predictions],
        dtype=np.float64,
    )
    best_index = int(np.argmin(scores))
    best_dimensions = candidates[best_index]
    best_prediction = predictions[best_index]

    write_json(
        args.output,
        {
            "best_dimensions": best_dimensions,
            "predicted_targets": {name: float(value) for name, value in zip(TARGET_COLUMNS, best_prediction)},
            "objective_score": float(scores[best_index]),
            "weights": {
                "s11_weight": args.s11_weight,
                "gain_weight": args.gain_weight,
            },
            "objective_normalizer": {
                "s11_min_db_p05": normalizer.s11_min,
                "s11_min_db_p95": normalizer.s11_max,
                "gain_max_p05": normalizer.gain_min,
                "gain_max_p95": normalizer.gain_max,
            },
            "search_candidates": args.n_candidates,
            "parameter_lower_bounds": lower_bounds,
            "parameter_upper_bounds": upper_bounds,
        },
    )
    plot_prediction_summary(best_prediction, args.plot, "Optimized New Antenna Feature Prediction")

    print("新天线优化完成")
    print("推荐尺寸: " + ", ".join(f"{value:.6g}" for value in best_dimensions))
    print(f"预测 s11_min_db: {best_prediction[0]:.6g}")
    print(f"预测 gain_max: {best_prediction[5]:.6g}")
    print(f"综合目标分数: {scores[best_index]:.6g}")
    print(f"结果已保存: {args.output}")
    print(f"特征图已保存: {args.plot}")


if __name__ == "__main__":
    main()

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


PRESETS = [
    ("s11_priority", 0.7, 0.3),
    ("balanced", 0.5, 0.5),
    ("gain_priority", 0.3, 0.7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较新天线在不同 S11/增益权重下的优化结果")
    parser.add_argument("--features-csv", type=Path, default=Path("outputs") / "new_antenna_dataset" / "features.csv")
    parser.add_argument("--model", type=Path, default=Path("outputs") / "new_antenna_model" / "new_antenna_mlp.joblib")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "new_antenna_model" / "weight_comparison.json")
    parser.add_argument("--n-candidates", type=int, default=8000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_new_antenna_features(args.features_csv)
    normalizer = build_objective_normalizer(dataset.dataframe)
    lower_bounds, upper_bounds = dimension_bounds(dataset.dimensions)
    model = load_model(args.model)

    rng = np.random.default_rng(args.random_state)
    candidates = rng.uniform(lower_bounds, upper_bounds, size=(args.n_candidates, lower_bounds.size))
    predictions = np.asarray(model.predict(candidates), dtype=np.float64)

    results: list[dict[str, object]] = []
    for name, s11_weight, gain_weight in PRESETS:
        scores = np.array(
            [score_prediction(row, normalizer, s11_weight, gain_weight) for row in predictions],
            dtype=np.float64,
        )
        best_index = int(np.argmin(scores))
        best_dimensions = candidates[best_index]
        best_prediction = predictions[best_index]
        results.append(
            {
                "preset": name,
                "weights": {
                    "s11_weight": s11_weight,
                    "gain_weight": gain_weight,
                },
                "best_dimensions": best_dimensions,
                "predicted_targets": {
                    key: float(value) for key, value in zip(TARGET_COLUMNS, best_prediction)
                },
                "objective_score": float(scores[best_index]),
            }
        )

    write_json(
        args.output,
        {
            "features_csv": str(args.features_csv),
            "model_path": str(args.model),
            "search_candidates": args.n_candidates,
            "objective_normalizer": {
                "s11_min_db_p05": normalizer.s11_min,
                "s11_min_db_p95": normalizer.s11_max,
                "gain_max_p05": normalizer.gain_min,
                "gain_max_p95": normalizer.gain_max,
            },
            "results": results,
        },
    )

    print("多权重优化完成")
    print(f"结果已保存: {args.output}")
    for item in results:
        print(
            f"{item['preset']}: s11_min_db={item['predicted_targets']['s11_min_db']:.6g}, "
            f"gain_max={item['predicted_targets']['gain_max']:.6g}, "
            f"score={item['objective_score']:.6g}"
        )


if __name__ == "__main__":
    main()

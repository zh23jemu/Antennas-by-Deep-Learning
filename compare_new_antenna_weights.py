from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from antenna_ml.io import write_json
from antenna_ml.model import load_model
from antenna_ml.new_antenna import (
    DIMENSION_COLUMNS,
    TARGET_COLUMNS,
    build_objective_normalizer,
    dimension_bounds,
    build_effective_gain,
    load_new_antenna_features,
    score_prediction,
    score_s11_prediction,
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
    parser.add_argument("--gain-alpha", type=float, default=0.5, help="增益保守系数，0只看gain_mean，1直接看gain_max")
    parser.add_argument("--target-freq-ghz", type=float, default=0.95, help="综合 S 参数评分使用的目标频点")
    return parser.parse_args()


def build_seed_candidates(dataset, s11_threshold_db: float, gain_alpha: float, n_select: int = 120) -> np.ndarray:
    """优先从历史上 S11 达标且有效增益较高的样本附近选点，减少比较时的无效随机性。"""
    dataframe = dataset.dataframe
    feasible = dataframe.loc[dataframe["s11_min_db"] <= s11_threshold_db].copy()
    if feasible.empty:
        feasible = dataframe.copy()
    effective_gain = build_effective_gain(
        feasible["gain_max"].to_numpy(dtype=np.float64),
        feasible["gain_mean"].to_numpy(dtype=np.float64),
        alpha=gain_alpha,
    )
    ranked = feasible.assign(_effective_gain=effective_gain).sort_values(
        by=["_effective_gain", "s11_min_db"],
        ascending=[False, True],
    )
    return ranked.head(min(max(1, n_select), ranked.shape[0]))[DIMENSION_COLUMNS].to_numpy(dtype=np.float64)


def main() -> None:
    args = parse_args()
    dataset = load_new_antenna_features(args.features_csv)
    normalizer = build_objective_normalizer(dataset.dataframe, target_freq_ghz=args.target_freq_ghz)
    lower_bounds, upper_bounds = dimension_bounds(dataset.dimensions)
    model = load_model(args.model)

    rng = np.random.default_rng(args.random_state)
    seeds = build_seed_candidates(dataset, s11_threshold_db=-20.0, gain_alpha=args.gain_alpha)
    local_count = int(round(args.n_candidates * 0.8))
    global_count = max(0, args.n_candidates - local_count)
    span = np.maximum(upper_bounds - lower_bounds, 1e-6)
    local_sigma = span * 0.06
    seed_indices = rng.integers(0, seeds.shape[0], size=max(1, local_count))
    local_candidates = np.clip(
        seeds[seed_indices] + rng.normal(0.0, local_sigma, size=(max(1, local_count), lower_bounds.size)),
        lower_bounds,
        upper_bounds,
    )
    global_candidates = rng.uniform(lower_bounds, upper_bounds, size=(global_count, lower_bounds.size)) if global_count else np.empty((0, lower_bounds.size))
    candidates = np.vstack([local_candidates, global_candidates])
    predictions = np.asarray(model.predict(candidates), dtype=np.float64)

    results: list[dict[str, object]] = []
    composite_s_scores = np.array(
        [
            score_s11_prediction(
                row,
                normalizer,
                target_freq_ghz=args.target_freq_ghz,
            )
            for row in predictions
        ],
        dtype=np.float64,
    )
    composite_best_index = int(np.argmin(composite_s_scores))
    results.append(
        {
            "preset": "s11_composite",
            "weights": {
                "s11_min_weight": 0.45,
                "freq_weight": 0.30,
                "bandwidth_weight": 0.15,
                "mean_weight": 0.10,
            },
            "best_dimensions": candidates[composite_best_index],
            "predicted_targets": {
                key: float(value) for key, value in zip(TARGET_COLUMNS, predictions[composite_best_index])
            },
            "objective_score": float(composite_s_scores[composite_best_index]),
        }
    )

    for name, s11_weight, gain_weight in PRESETS:
        scores = np.array(
            [score_prediction(row, normalizer, s11_weight, gain_weight, gain_alpha=args.gain_alpha) for row in predictions],
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
                "effective_gain_p05": normalizer.effective_gain_min,
                "effective_gain_p95": normalizer.effective_gain_max,
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

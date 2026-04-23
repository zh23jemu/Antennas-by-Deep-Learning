from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from antenna_ml.io import write_json
from antenna_ml.model import load_model
from antenna_ml.new_antenna import (
    DIMENSION_COLUMNS,
    TARGET_COLUMNS,
    build_objective_normalizer,
    dimension_bounds,
    load_new_antenna_features,
    score_prediction,
    score_s11_prediction,
)
from antenna_ml.new_antenna_plotting import S11_FEATURE_LABELS, plot_prediction_summary


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
    parser.add_argument("--gain-alpha", type=float, default=0.5, help="增益保守系数，0只看gain_mean，1直接看gain_max")
    parser.add_argument("--s11-threshold-db", type=float, default=-20.0, help="联合优化时优先满足的S11门限")
    parser.add_argument("--objective-mode", choices=("joint", "s11_composite"), default="joint", help="优化目标类型：联合优化或综合 S 参数优化")
    parser.add_argument("--target-freq-ghz", type=float, default=0.95, help="综合 S 参数优化时使用的目标谐振频点")
    parser.add_argument("--s11-min-weight", type=float, default=0.45, help="综合 S 参数评分中，谐振深度权重")
    parser.add_argument("--s11-freq-weight", type=float, default=0.30, help="综合 S 参数评分中，频点偏差权重")
    parser.add_argument("--s11-bandwidth-weight", type=float, default=0.15, help="综合 S 参数评分中，带宽权重")
    parser.add_argument("--s11-mean-weight", type=float, default=0.10, help="综合 S 参数评分中，整体平均反射权重")
    parser.add_argument("--baseline-dimensions", type=str, default=None, help="逗号分隔的基线尺寸；综合 S 优化时可要求新方案优于基线评分")
    parser.add_argument("--require-better-than-baseline", action="store_true", help="综合 S 优化时，要求候选评分优于给定基线")
    parser.add_argument("--seed-top-k", type=int, default=120, help="从历史样本中选择多少个优良种子作为局部搜索中心")
    parser.add_argument("--local-ratio", type=float, default=0.75, help="局部邻域候选占总候选的比例")
    parser.add_argument("--local-scale", type=float, default=0.1, help="局部扰动相对边界跨度的比例")
    parser.add_argument("--distance-penalty", type=float, default=0.2, help="距离历史样本越远惩罚越大，用于抑制外推过强的虚高解")
    return parser.parse_args()


def build_seed_candidates(
    dataframe: pd.DataFrame,
    n_select: int,
    s11_threshold_db: float,
    gain_alpha: float,
) -> np.ndarray:
    """从历史数据中挑选联合表现较好的尺寸，作为局部搜索的中心点。"""
    feasible = dataframe.loc[dataframe["s11_min_db"] <= s11_threshold_db].copy()
    if feasible.empty:
        feasible = dataframe.copy()

    effective_gain = feasible["gain_mean"] + gain_alpha * (feasible["gain_max"] - feasible["gain_mean"])
    ranked = feasible.assign(_effective_gain=effective_gain).sort_values(
        by=["_effective_gain", "s11_min_db"],
        ascending=[False, True],
    )
    selected = ranked.head(max(1, min(n_select, ranked.shape[0])))
    return selected[DIMENSION_COLUMNS].to_numpy(dtype=np.float64)


def build_s11_seed_candidates(
    dataframe: pd.DataFrame,
    n_select: int,
    target_freq_ghz: float,
) -> np.ndarray:
    """为综合 S 参数优化挑选更合适的历史种子。"""
    ranked = dataframe.copy()
    ranked["_freq_error"] = np.abs(ranked["s11_min_freq_ghz"] - target_freq_ghz)
    ranked = ranked.sort_values(
        by=["s11_min_db", "_freq_error", "s11_bandwidth_below_minus10_db_ghz", "s11_mean_db"],
        ascending=[True, True, False, True],
    )
    selected = ranked.head(max(1, min(n_select, ranked.shape[0])))
    return selected[DIMENSION_COLUMNS].to_numpy(dtype=np.float64)


def sample_candidates(
    rng: np.random.Generator,
    seed_dimensions: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    n_candidates: int,
    local_ratio: float,
    local_scale: float,
) -> tuple[np.ndarray, int, int]:
    """混合生成候选点。

    1. 局部搜索：围绕历史优良样本做高斯扰动，优先落在可信区域附近。
    2. 全局搜索：保留少量均匀随机点，避免只在局部打转。
    """
    total = max(1, n_candidates)
    local_count = int(round(total * np.clip(local_ratio, 0.0, 1.0)))
    global_count = max(0, total - local_count)

    span = upper_bounds - lower_bounds
    local_sigma = np.maximum(span * max(local_scale, 1e-6), 1e-6)

    local_candidates = np.empty((0, lower_bounds.size), dtype=np.float64)
    if local_count > 0 and seed_dimensions.size > 0:
        seed_indices = rng.integers(0, seed_dimensions.shape[0], size=local_count)
        seed_block = seed_dimensions[seed_indices]
        noise = rng.normal(loc=0.0, scale=local_sigma, size=seed_block.shape)
        local_candidates = np.clip(seed_block + noise, lower_bounds, upper_bounds)

    global_candidates = np.empty((0, lower_bounds.size), dtype=np.float64)
    if global_count > 0:
        global_candidates = rng.uniform(lower_bounds, upper_bounds, size=(global_count, lower_bounds.size))

    candidates = np.vstack([local_candidates, global_candidates])
    if candidates.shape[0] == 0:
        candidates = rng.uniform(lower_bounds, upper_bounds, size=(1, lower_bounds.size))
    return candidates, int(local_candidates.shape[0]), int(global_candidates.shape[0])


def nearest_seed_distance(candidates: np.ndarray, seeds: np.ndarray, span: np.ndarray) -> np.ndarray:
    """计算候选点到历史优良样本的最近归一化距离。"""
    if seeds.size == 0:
        return np.zeros(candidates.shape[0], dtype=np.float64)
    safe_span = np.maximum(span, 1e-12)
    normalized_candidates = candidates / safe_span
    normalized_seeds = seeds / safe_span
    distances = np.sqrt(((normalized_candidates[:, None, :] - normalized_seeds[None, :, :]) ** 2).sum(axis=2))
    return distances.min(axis=1)


def parse_dimensions(raw: str | None) -> np.ndarray | None:
    """解析逗号分隔的尺寸参数。"""
    if raw is None:
        return None
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != len(DIMENSION_COLUMNS):
        raise ValueError(f"需要输入 {len(DIMENSION_COLUMNS)} 个尺寸参数")
    return np.asarray(values, dtype=np.float64).reshape(1, -1)


def main() -> None:
    args = parse_args()
    dataset = load_new_antenna_features(args.features_csv)
    lower_bounds, upper_bounds = dimension_bounds(dataset.dimensions)
    normalizer = build_objective_normalizer(dataset.dataframe, target_freq_ghz=args.target_freq_ghz)
    model = load_model(args.model)

    rng = np.random.default_rng(args.random_state)
    if args.objective_mode == "s11_composite":
        seed_dimensions = build_s11_seed_candidates(
            dataframe=dataset.dataframe,
            n_select=args.seed_top_k,
            target_freq_ghz=args.target_freq_ghz,
        )
    else:
        seed_dimensions = build_seed_candidates(
            dataframe=dataset.dataframe,
            n_select=args.seed_top_k,
            s11_threshold_db=args.s11_threshold_db,
            gain_alpha=args.gain_alpha,
        )
    candidates, local_candidate_count, global_candidate_count = sample_candidates(
        rng=rng,
        seed_dimensions=seed_dimensions,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        n_candidates=args.n_candidates,
        local_ratio=args.local_ratio,
        local_scale=args.local_scale,
    )
    predictions = np.asarray(model.predict(candidates), dtype=np.float64)
    distance_penalties = nearest_seed_distance(candidates, seed_dimensions, upper_bounds - lower_bounds)

    if args.objective_mode == "s11_composite":
        baseline_dimensions = parse_dimensions(args.baseline_dimensions)
        baseline_prediction = None
        baseline_score = None
        if baseline_dimensions is not None:
            baseline_prediction = np.asarray(model.predict(baseline_dimensions)[0], dtype=np.float64)
            baseline_score = float(
                score_s11_prediction(
                    baseline_prediction,
                    normalizer,
                    target_freq_ghz=args.target_freq_ghz,
                    s11_min_weight=args.s11_min_weight,
                    freq_weight=args.s11_freq_weight,
                    bandwidth_weight=args.s11_bandwidth_weight,
                    mean_weight=args.s11_mean_weight,
                )
            )

        scores = np.array(
            [
                score_s11_prediction(
                    row,
                    normalizer,
                    target_freq_ghz=args.target_freq_ghz,
                    s11_min_weight=args.s11_min_weight,
                    freq_weight=args.s11_freq_weight,
                    bandwidth_weight=args.s11_bandwidth_weight,
                    mean_weight=args.s11_mean_weight,
                )
                + args.distance_penalty * penalty
                for row, penalty in zip(predictions, distance_penalties)
            ],
            dtype=np.float64,
        )
        if args.require_better_than_baseline and baseline_score is not None:
            better_mask = scores < baseline_score
            if np.any(better_mask):
                candidates = candidates[better_mask]
                predictions = predictions[better_mask]
                distance_penalties = distance_penalties[better_mask]
                scores = scores[better_mask]
        best_index = int(np.argmin(scores))
        best_dimensions = candidates[best_index]
        best_prediction = predictions[best_index]
        best_score = float(scores[best_index])
        feasible_count = int(np.sum(predictions[:, 0] <= args.s11_threshold_db))
        best_distance_penalty = float(distance_penalties[best_index])
    else:
        s11_values = predictions[:, 0]
        feasible_mask = s11_values <= args.s11_threshold_db
        if np.any(feasible_mask):
            filtered_candidates = candidates[feasible_mask]
            filtered_predictions = predictions[feasible_mask]
            filtered_penalties = distance_penalties[feasible_mask]
            scores = np.array(
                [
                    score_prediction(
                        row,
                        normalizer,
                        args.s11_weight,
                        args.gain_weight,
                        gain_alpha=args.gain_alpha,
                    )
                    + args.distance_penalty * penalty
                    for row, penalty in zip(filtered_predictions, filtered_penalties)
                ],
                dtype=np.float64,
            )
            best_index = int(np.argmin(scores))
            best_dimensions = filtered_candidates[best_index]
            best_prediction = filtered_predictions[best_index]
            best_score = float(scores[best_index])
            feasible_count = int(filtered_predictions.shape[0])
            best_distance_penalty = float(filtered_penalties[best_index])
        else:
            scores = np.array(
                [
                    score_prediction(
                        row,
                        normalizer,
                        args.s11_weight,
                        args.gain_weight,
                        gain_alpha=args.gain_alpha,
                    )
                    + args.distance_penalty * penalty
                    for row, penalty in zip(predictions, distance_penalties)
                ],
                dtype=np.float64,
            )
            best_index = int(np.argmin(scores))
            best_dimensions = candidates[best_index]
            best_prediction = predictions[best_index]
            best_score = float(scores[best_index])
            feasible_count = 0
            best_distance_penalty = float(distance_penalties[best_index])

    write_json(
        args.output,
        {
            "best_dimensions": best_dimensions,
            "predicted_targets": {name: float(value) for name, value in zip(TARGET_COLUMNS, best_prediction)},
            "objective_score": best_score,
            "weights": {
                "s11_weight": args.s11_weight,
                "gain_weight": args.gain_weight,
                "gain_alpha": args.gain_alpha,
            },
            "objective_mode": args.objective_mode,
            "target_freq_ghz": args.target_freq_ghz,
            "s11_composite_weights": {
                "s11_min_weight": args.s11_min_weight,
                "freq_weight": args.s11_freq_weight,
                "bandwidth_weight": args.s11_bandwidth_weight,
                "mean_weight": args.s11_mean_weight,
            },
            "baseline_dimensions": baseline_dimensions,
            "require_better_than_baseline": bool(args.require_better_than_baseline),
            "baseline_s11_composite_score": baseline_score if args.objective_mode == "s11_composite" else None,
            "s11_threshold_db": args.s11_threshold_db,
            "feasible_candidate_count": feasible_count,
            "seed_top_k": args.seed_top_k,
            "local_ratio": args.local_ratio,
            "local_scale": args.local_scale,
            "distance_penalty": args.distance_penalty,
            "best_distance_to_seed": best_distance_penalty,
            "local_candidate_count": local_candidate_count,
            "global_candidate_count": global_candidate_count,
            "objective_normalizer": {
                "s11_min_db_p05": normalizer.s11_min,
                "s11_min_db_p95": normalizer.s11_max,
                "s11_mean_db_p05": normalizer.s11_mean_min,
                "s11_mean_db_p95": normalizer.s11_mean_max,
                "s11_bandwidth_p05": normalizer.s11_bw_min,
                "s11_bandwidth_p95": normalizer.s11_bw_max,
                "s11_freq_error_p05": normalizer.s11_freq_error_min,
                "s11_freq_error_p95": normalizer.s11_freq_error_max,
                "effective_gain_p05": normalizer.effective_gain_min,
                "effective_gain_p95": normalizer.effective_gain_max,
            },
            "search_candidates": args.n_candidates,
            "parameter_lower_bounds": lower_bounds,
            "parameter_upper_bounds": upper_bounds,
        },
    )
    plot_labels = S11_FEATURE_LABELS if len(best_prediction) == len(S11_FEATURE_LABELS) else None
    plot_prediction_summary(
        best_prediction,
        args.plot,
        "Optimized New Antenna Feature Prediction",
        labels=plot_labels,
    )

    print("新天线优化完成")
    print("推荐尺寸: " + ", ".join(f"{value:.6g}" for value in best_dimensions))
    print(f"预测 s11_min_db: {best_prediction[0]:.6g}")
    if len(best_prediction) > 5:
        print(f"预测 gain_max: {best_prediction[5]:.6g}")
    print(f"满足S11门限的候选数: {feasible_count}")
    print(f"最优点到历史优良样本的归一化距离: {best_distance_penalty:.6g}")
    print(f"综合目标分数: {best_score:.6g}")
    print(f"结果已保存: {args.output}")
    print(f"特征图已保存: {args.plot}")


if __name__ == "__main__":
    main()

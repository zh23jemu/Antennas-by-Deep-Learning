from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from antenna_ml.data import default_data_paths, load_dataset, parameter_bounds
from antenna_ml.io import write_json
from antenna_ml.model import load_model
from antenna_ml.optimize import random_search
from antenna_ml.plotting import plot_s_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 MLP 代理模型搜索预测 S 参数最优的天线尺寸")
    parser.add_argument("--model", type=Path, default=Path("outputs") / "antenna_mlp.joblib")
    parser.add_argument("--data-dir", type=Path, default=Path("样本数据"))
    parser.add_argument("--output", type=Path, default=Path("outputs") / "best_design.json")
    parser.add_argument("--plot", type=Path, default=Path("outputs") / "best_design_s_curve.png")
    parser.add_argument("--n-candidates", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(default_data_paths(args.data_dir))
    lower_bounds, upper_bounds = parameter_bounds(dataset.dimensions)
    model = load_model(args.model)
    best_dimensions, best_curve, best_score, best_point_index = random_search(
        model=model,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        n_candidates=args.n_candidates,
        random_state=args.random_state,
    )

    write_json(
        args.output,
        {
            "best_dimensions": best_dimensions,
            "predicted_s_values": best_curve,
            "best_s_value": best_score,
            "best_point_index": best_point_index,
            "search_candidates": args.n_candidates,
            "parameter_lower_bounds": lower_bounds,
            "parameter_upper_bounds": upper_bounds,
        },
    )
    plot_s_curve(best_curve, args.plot, "Optimized Design Predicted S Parameter Curve")
    print("优化完成")
    print("推荐尺寸: " + ", ".join(f"{value:.6g}" for value in best_dimensions))
    print(f"预测最低 S 参数值: {best_score:.6g}")
    print(f"最低点索引: {best_point_index}")
    print(f"结果已保存: {args.output}")
    print(f"曲线图已保存: {args.plot}")


if __name__ == "__main__":
    main()

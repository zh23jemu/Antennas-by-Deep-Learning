from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键完成训练、优化、预测和画图")
    parser.add_argument("--data-dir", type=Path, action="append", dest="data_dirs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-candidates", type=int, default=2000)
    parser.add_argument("--compare-count", type=int, default=3)
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    print("执行命令:", " ".join(args))
    subprocess.run(args, check=True)


def main() -> None:
    args = parse_args()
    python_executable = sys.executable
    data_dirs = args.data_dirs or [Path("样本数据")]

    train_command = [
        python_executable,
        "train.py",
        "--output-dir",
        str(args.output_dir),
        "--max-iter",
        str(args.max_iter),
        "--test-size",
        str(args.test_size),
        "--random-state",
        str(args.random_state),
        "--compare-count",
        str(args.compare_count),
    ]
    for data_dir in data_dirs:
        train_command.extend(["--data-dir", str(data_dir)])
    run_command(train_command)

    optimize_command = [
        python_executable,
        "optimize.py",
        "--model",
        str(args.output_dir / "antenna_mlp.joblib"),
        "--output",
        str(args.output_dir / "best_design.json"),
        "--plot",
        str(args.output_dir / "best_design_features.png"),
        "--n-candidates",
        str(args.n_candidates),
        "--random-state",
        str(args.random_state),
    ]
    for data_dir in data_dirs:
        optimize_command.extend(["--data-dir", str(data_dir)])
    run_command(optimize_command)

    best_design = json.loads((args.output_dir / "best_design.json").read_text(encoding="utf-8"))
    dimensions = ",".join(str(value) for value in best_design["best_dimensions"])
    run_command(
        [
            python_executable,
            "predict.py",
            "--model",
            str(args.output_dir / "antenna_mlp.joblib"),
            "--dimensions",
            dimensions,
            "--output",
            str(args.output_dir / "prediction.json"),
            "--plot",
            str(args.output_dir / "prediction_features.png"),
        ]
    )

    print("一键流程完成")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键完成新天线数据整理、训练、优化和预测")
    parser.add_argument("--s11-file", type=Path, action="append", dest="s11_files")
    parser.add_argument("--gain-file", type=Path, action="append", dest="gain_files")
    parser.add_argument("--dataset-dir", type=Path, default=Path("outputs") / "new_antenna_dataset")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs") / "new_antenna_model")
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--compare-count", type=int, default=3)
    parser.add_argument("--n-candidates", type=int, default=5000)
    parser.add_argument("--s11-weight", type=float, default=0.5)
    parser.add_argument("--gain-weight", type=float, default=0.5)
    return parser.parse_args()


def run_command(args: list[str]) -> None:
    print("执行命令:", " ".join(args))
    subprocess.run(args, check=True)


def main() -> None:
    args = parse_args()
    python_executable = sys.executable
    s11_files = args.s11_files or [Path("data1.xlsx")]
    gain_files = args.gain_files or [Path("data2.xlsx")]

    prepare_command = [
        python_executable,
        "prepare_new_antenna_dataset.py",
        "--output-dir",
        str(args.dataset_dir),
    ]
    for path in s11_files:
        prepare_command.extend(["--s11-file", str(path)])
    for path in gain_files:
        prepare_command.extend(["--gain-file", str(path)])
    run_command(prepare_command)

    features_csv = args.dataset_dir / "features.csv"
    run_command(
        [
            python_executable,
            "train_new_antenna.py",
            "--features-csv",
            str(features_csv),
            "--output-dir",
            str(args.model_dir),
            "--max-iter",
            str(args.max_iter),
            "--test-size",
            str(args.test_size),
            "--random-state",
            str(args.random_state),
            "--compare-count",
            str(args.compare_count),
        ]
    )

    run_command(
        [
            python_executable,
            "optimize_new_antenna.py",
            "--features-csv",
            str(features_csv),
            "--model",
            str(args.model_dir / "new_antenna_mlp.joblib"),
            "--output",
            str(args.model_dir / "best_design.json"),
            "--plot",
            str(args.model_dir / "best_design_features.png"),
            "--n-candidates",
            str(args.n_candidates),
            "--random-state",
            str(args.random_state),
            "--s11-weight",
            str(args.s11_weight),
            "--gain-weight",
            str(args.gain_weight),
        ]
    )

    best_design = json.loads((args.model_dir / "best_design.json").read_text(encoding="utf-8"))
    dimensions = ",".join(str(value) for value in best_design["best_dimensions"])
    run_command(
        [
            python_executable,
            "predict_new_antenna.py",
            "--model",
            str(args.model_dir / "new_antenna_mlp.joblib"),
            "--dimensions",
            dimensions,
            "--output",
            str(args.model_dir / "prediction.json"),
            "--plot",
            str(args.model_dir / "prediction_features.png"),
        ]
    )

    print("新天线一键流程完成")
    print(f"数据目录: {args.dataset_dir}")
    print(f"模型目录: {args.model_dir}")


if __name__ == "__main__":
    main()

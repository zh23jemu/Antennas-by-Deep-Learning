from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from antenna_ml.io import write_json


PARAM_COLUMNS = ["cut_x", "cut_y", "fw", "gx", "gy", "h1", "px", "py"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="围绕两组 S 参数候选生成局部扫参方案")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "s11_local_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # A: 模型给出的 900 MHz 附近方案
    model_best = {
        "cut_x": 3.135907839176984,
        "cut_y": 19.06896927590043,
        "fw": 3.331516387794181,
        "gx": 153.58596440258293,
        "gy": 150.70194281473826,
        "h1": 1.709344946196049,
        "px": 77.67893745269373,
        "py": 78.46067471115761,
    }

    # B: 历史真实最深 S11 样本
    historical_best = {
        "cut_x": 2.25,
        "cut_y": 20.0,
        "fw": 2.25,
        "gx": 75.0,
        "gy": 75.0,
        "h1": 1.2,
        "px": 80.0,
        "py": 80.0,
    }

    candidates: list[dict[str, float | str]] = []

    # 围绕历史最深 S11 样本：优先微调影响谐振频率的参数
    historical_variants = [
        ("hist_base", historical_best),
        ("hist_cutx_up", {**historical_best, "cut_x": 2.35}),
        ("hist_cutx_down", {**historical_best, "cut_x": 2.15}),
        ("hist_fw_up", {**historical_best, "fw": 2.35}),
        ("hist_fw_down", {**historical_best, "fw": 2.15}),
        ("hist_h1_up", {**historical_best, "h1": 1.3}),
        ("hist_h1_down", {**historical_best, "h1": 1.1}),
        ("hist_cuty_up", {**historical_best, "cut_y": 20.5}),
        ("hist_cuty_down", {**historical_best, "cut_y": 19.5}),
    ]

    # 围绕 900 MHz 邻近解：优先尝试把 S11 再压深
    model_variants = [
        ("model_base", model_best),
        ("model_cuty_up", {**model_best, "cut_y": 19.57}),
        ("model_cuty_down", {**model_best, "cut_y": 18.57}),
        ("model_fw_up", {**model_best, "fw": 3.43}),
        ("model_fw_down", {**model_best, "fw": 3.23}),
        ("model_h1_up", {**model_best, "h1": 1.81}),
        ("model_h1_down", {**model_best, "h1": 1.61}),
        ("model_px_up", {**model_best, "px": 79.68}),
        ("model_px_down", {**model_best, "px": 75.68}),
        ("model_py_up", {**model_best, "py": 80.46}),
        ("model_py_down", {**model_best, "py": 76.46}),
    ]

    for label, values in historical_variants + model_variants:
        row = {"label": label}
        row.update(values)
        candidates.append(row)

    df = pd.DataFrame(candidates)
    df.to_csv(args.output_dir / "s11_local_sweep_candidates.csv", index=False, encoding="utf-8-sig")
    write_json(
        args.output_dir / "s11_local_sweep_candidates.json",
        {
            "count": int(df.shape[0]),
            "columns": ["label"] + PARAM_COLUMNS,
            "candidates": df.to_dict(orient="records"),
        },
    )

    print("S 参数局部扫参方案已生成")
    print(f"候选数量: {df.shape[0]}")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()

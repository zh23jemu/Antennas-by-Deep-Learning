from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from antenna_ml.io import write_json
from antenna_ml.model import load_model
from antenna_ml.new_antenna import DIMENSION_COLUMNS, TARGET_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量预测第二种天线的关键特征")
    parser.add_argument("--model", type=Path, default=Path("outputs") / "new_antenna_model" / "new_antenna_mlp.joblib")
    parser.add_argument("--input-csv", type=Path, default=Path("three_cases_input.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("three_cases_prediction.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("three_cases_prediction.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    missing = [column for column in DIMENSION_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"输入表缺少这些列: {missing}")

    model = load_model(args.model)
    dimensions = df[DIMENSION_COLUMNS].to_numpy(dtype=np.float64)
    predictions = np.asarray(model.predict(dimensions), dtype=np.float64)

    result_df = df.copy()
    for index, column in enumerate(TARGET_COLUMNS):
        result_df[column] = predictions[:, index]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    write_json(
        args.output_json,
        {
            "model_path": str(args.model),
            "input_csv": str(args.input_csv),
            "count": int(result_df.shape[0]),
            "records": result_df.to_dict(orient="records"),
        },
    )

    print("批量预测完成")
    print(f"输入数量: {result_df.shape[0]}")
    print(f"CSV结果: {args.output_csv}")
    print(f"JSON结果: {args.output_json}")


if __name__ == "__main__":
    main()

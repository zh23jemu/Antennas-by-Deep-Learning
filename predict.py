from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from antenna_ml.io import write_json
from antenna_ml.model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用已训练模型预测小型天线 S 参数")
    parser.add_argument("--model", type=Path, default=Path("outputs") / "antenna_mlp.joblib")
    parser.add_argument("--dimensions", required=True, help="逗号分隔的尺寸参数，例如 1,2,3")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "prediction.json")
    return parser.parse_args()


def parse_dimensions(raw: str) -> np.ndarray:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("请通过 --dimensions 输入至少一个尺寸参数")
    return np.asarray(values, dtype=np.float64).reshape(1, -1)


def main() -> None:
    args = parse_args()
    dimensions = parse_dimensions(args.dimensions)
    model = load_model(args.model)
    prediction = np.asarray(model.predict(dimensions)[0], dtype=np.float64)
    best_index = int(np.argmin(prediction))
    best_value = float(prediction[best_index])

    write_json(
        args.output,
        {
            "dimensions": dimensions[0],
            "predicted_s_values": prediction,
            "best_s_value": best_value,
            "best_point_index": best_index,
        },
    )
    print(f"预测完成，最低 S 参数值: {best_value:.6g}，采样点索引: {best_index}")
    print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()

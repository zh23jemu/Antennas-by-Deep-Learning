from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from antenna_ml.data import decode_s_features
from antenna_ml.io import write_json
from antenna_ml.model import load_model
from antenna_ml.plotting import plot_predicted_feature_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用已训练模型预测小型天线关键 S 参数特征")
    parser.add_argument("--model", type=Path, default=Path("outputs") / "antenna_mlp.joblib")
    parser.add_argument("--dimensions", required=True, help="逗号分隔的尺寸参数，例如 1,2,3")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "prediction.json")
    parser.add_argument("--plot", type=Path, default=Path("outputs") / "prediction_features.png")
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
    decoded_prediction = decode_s_features(prediction, s_point_count=500)
    best_value = float(decoded_prediction[0])
    best_index = int(decoded_prediction[1])

    write_json(
        args.output,
        {
            "dimensions": dimensions[0],
            "predicted_features": {
                "min_s_value": best_value,
                "min_point_index": best_index,
                "mean_s_value": float(decoded_prediction[2]),
                "std_s_value": float(decoded_prediction[3]),
            },
            "best_s_value": best_value,
            "best_point_index": best_index,
        },
    )
    plot_predicted_feature_summary(decoded_prediction, args.plot, "Predicted S Parameter Features")
    print(f"预测完成，最低 S 参数值: {best_value:.6g}，最低点索引: {best_index}")
    print(f"结果已保存: {args.output}")
    print(f"曲线图已保存: {args.plot}")


if __name__ == "__main__":
    main()

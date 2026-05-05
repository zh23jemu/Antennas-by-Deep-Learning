from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from antenna_ml.io import write_json
from antenna_ml.model import load_model_artifact
from antenna_ml.new_antenna import DIMENSION_COLUMNS, TARGET_COLUMNS
from antenna_ml.new_antenna_calibration import (
    GainCalibration,
    LocalBlendCalibration,
    apply_gain_calibration,
    apply_local_gain_blend,
)
from antenna_ml.new_antenna_plotting import plot_prediction_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预测新天线的 S11 + 增益关键特征")
    parser.add_argument("--model", type=Path, default=Path("outputs") / "new_antenna_model" / "new_antenna_mlp.joblib")
    parser.add_argument("--dimensions", required=True, help="逗号分隔的 8 个尺寸参数")
    parser.add_argument("--output", type=Path, default=Path("outputs") / "new_antenna_model" / "prediction.json")
    parser.add_argument("--plot", type=Path, default=Path("outputs") / "new_antenna_model" / "prediction_features.png")
    return parser.parse_args()


def parse_dimensions(raw: str) -> np.ndarray:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != len(DIMENSION_COLUMNS):
        raise ValueError(f"需要输入 {len(DIMENSION_COLUMNS)} 个尺寸参数")
    return np.asarray(values, dtype=np.float64).reshape(1, -1)


def main() -> None:
    args = parse_args()
    dimensions = parse_dimensions(args.dimensions)
    model_artifact = load_model_artifact(args.model)
    model = model_artifact.model
    gain_calibration = GainCalibration.from_dict(model_artifact.metadata.get("gain_calibration"))
    local_blend_calibration = LocalBlendCalibration.from_dict(model_artifact.metadata.get("local_blend_calibration"))
    reference_dimensions = np.asarray(model_artifact.metadata.get("reference_dimensions", []), dtype=np.float64)
    reference_targets = np.asarray(model_artifact.metadata.get("reference_targets", []), dtype=np.float64)
    raw_prediction = np.asarray(model.predict(dimensions), dtype=np.float64)
    prediction = apply_gain_calibration(raw_prediction, gain_calibration)[0]
    prediction = apply_local_gain_blend(
        dimensions=dimensions,
        predictions=prediction,
        reference_dimensions=reference_dimensions,
        reference_targets=reference_targets,
        calibration=local_blend_calibration,
    )

    write_json(
        args.output,
        {
            "dimensions": {name: float(value) for name, value in zip(DIMENSION_COLUMNS, dimensions[0])},
            "raw_predicted_targets": {name: float(value) for name, value in zip(TARGET_COLUMNS, raw_prediction[0])},
            "predicted_targets": {name: float(value) for name, value in zip(TARGET_COLUMNS, prediction)},
            "gain_calibration": gain_calibration.to_dict(),
            "local_blend_calibration": local_blend_calibration.to_dict(),
        },
    )
    plot_prediction_summary(prediction, args.plot, "Predicted New Antenna Features")
    print("新天线预测完成")
    print(f"结果已保存: {args.output}")
    print(f"特征图已保存: {args.plot}")


if __name__ == "__main__":
    main()

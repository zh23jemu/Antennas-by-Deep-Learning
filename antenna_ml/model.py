from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor


@dataclass(frozen=True)
class TrainingResult:
    model: Any
    metrics: dict[str, float]
    x_train_shape: tuple[int, int]
    y_train_shape: tuple[int, int]
    x_valid_shape: tuple[int, int]
    y_valid_shape: tuple[int, int]
    x_valid: np.ndarray
    y_valid: np.ndarray
    y_pred: np.ndarray


def build_model(random_state: int, max_iter: int, sample_count: int) -> TransformedTargetRegressor:
    hidden_layer_sizes = (128, 128, 64)
    learning_rate_init = 1e-3
    alpha = 1e-4

    if sample_count >= 5000:
        hidden_layer_sizes = (256, 256, 128)
        learning_rate_init = 5e-4
        alpha = 5e-5

    base_model = Pipeline(
        steps=[
            ("x_scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    random_state=random_state,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=40,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=base_model,
        transformer=StandardScaler(),
    )


def train_model(
    dimensions: np.ndarray,
    targets: np.ndarray,
    random_state: int,
    max_iter: int,
    test_size: float,
) -> TrainingResult:
    x_train, x_valid, y_train, y_valid = train_test_split(
        dimensions,
        targets,
        test_size=test_size,
        random_state=random_state,
    )

    model = build_model(
        random_state=random_state,
        max_iter=max_iter,
        sample_count=dimensions.shape[0],
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_valid)
    true_min = y_valid[:, 0]
    pred_min = y_pred[:, 0]

    metrics = {
        "valid_mse": float(mean_squared_error(y_valid, y_pred)),
        "valid_mae": float(mean_absolute_error(y_valid, y_pred)),
        "min_s_error_mae": float(mean_absolute_error(true_min, pred_min)),
    }
    return TrainingResult(
        model=model,
        metrics=metrics,
        x_train_shape=x_train.shape,
        y_train_shape=y_train.shape,
        x_valid_shape=x_valid.shape,
        y_valid_shape=y_valid.shape,
        x_valid=x_valid,
        y_valid=y_valid,
        y_pred=y_pred,
    )


def save_model(model: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(model_path: Path) -> Any:
    return joblib.load(model_path)

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from antenna_ml.new_antenna import DIMENSION_COLUMNS, TARGET_COLUMNS, load_new_antenna_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成论文用的模型结构、Loss、预测散点和相关性热力图")
    parser.add_argument("--features-csv", type=Path, default=Path("outputs") / "new_antenna_dataset" / "features.csv")
    parser.add_argument("--model", type=Path, default=Path("outputs") / "new_antenna_model" / "new_antenna_mlp.joblib")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "paper_figures")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--loss-epochs", type=int, default=220)
    return parser.parse_args()


def setup_style() -> None:
    """统一论文图风格，避免每张图视觉风格不一致。"""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def unwrap_model(model_path: Path):
    """兼容当前项目中“模型 + metadata”的保存格式。"""
    payload = joblib.load(model_path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def get_mlp_hidden_layers(model) -> tuple[int, ...]:
    """从已保存的 Pipeline 中读取 MLP 隐藏层结构。"""
    regressor = getattr(model, "regressor_", None) or getattr(model, "regressor", None)
    if regressor is None:
        return (256, 256, 128)
    if isinstance(regressor, Pipeline):
        mlp = regressor.named_steps.get("mlp")
        if mlp is not None:
            return tuple(int(v) for v in mlp.hidden_layer_sizes)
    return (256, 256, 128)


def draw_network_structure(output_path: Path, input_count: int, hidden_layers: tuple[int, ...], output_count: int) -> None:
    """绘制 MLP 网络结构示意图。

    真实隐藏层神经元很多，图中只展示每层若干代表节点，并在层标题中标出完整节点数。
    """
    layer_sizes = (input_count, *hidden_layers, output_count)
    layer_names = ["Input\n8 parameters", "Hidden 1", "Hidden 2", "Hidden 3", "Output\n8 features"]
    max_visible = 7
    colors = ["#315C7A", "#4F8A8B", "#5F9E6E", "#C98B3B", "#A64B4B"]

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_axis_off()
    x_positions = np.linspace(0.08, 0.92, len(layer_sizes))

    for layer_index, (x_value, size) in enumerate(zip(x_positions, layer_sizes)):
        visible_count = min(size, max_visible)
        y_positions = np.linspace(0.18, 0.82, visible_count)
        for y_value in y_positions:
            circle = Circle((x_value, y_value), 0.026, facecolor=colors[layer_index], edgecolor="white", linewidth=1.0)
            ax.add_patch(circle)

        if size > visible_count:
            ax.text(x_value, 0.50, "...", ha="center", va="center", fontsize=15, color="#333333")

        ax.text(x_value, 0.94, layer_names[layer_index], ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(x_value, 0.07, f"{size} nodes", ha="center", va="center", fontsize=10, color="#444444")

        if layer_index < len(layer_sizes) - 1:
            next_x = x_positions[layer_index + 1]
            for start_y in y_positions[:: max(1, visible_count // 4)]:
                arrow = FancyArrowPatch(
                    (x_value + 0.032, start_y),
                    (next_x - 0.032, 0.50),
                    arrowstyle="-",
                    mutation_scale=8,
                    linewidth=0.7,
                    alpha=0.28,
                    color="#555555",
                )
                ax.add_patch(arrow)

    ax.text(0.5, 0.995, "MLP Neural Network Architecture", ha="center", va="top", fontsize=15, fontweight="bold")
    ax.text(0.5, 0.015, "Input dimensions -> hidden nonlinear mapping -> S11 and gain feature prediction", ha="center", va="bottom", fontsize=9.5, color="#555555")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def train_loss_history(
    x_train: np.ndarray,
    x_valid: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    hidden_layers: tuple[int, ...],
    epochs: int,
    random_state: int,
) -> pd.DataFrame:
    """逐轮训练同结构 MLP，记录训练集和验证集 MSE。

    当前生产模型使用 sklearn 的 MLPRegressor，默认不会直接保存验证集 loss 曲线。
    这里用同样的标准化方式和相同隐藏层结构重新跑一条可复现实验曲线，专门用于论文展示。
    """
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_valid_scaled = x_scaler.transform(x_valid)
    y_train_scaled = y_scaler.fit_transform(y_train)

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=5e-5,
        learning_rate_init=5e-4,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        random_state=random_state,
    )

    records: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        # 这里 intentionally 每轮只训练 1 次迭代，用 warm_start 逐轮记录 loss；
        # sklearn 会提示“单轮未收敛”，这是预期行为，因此生成论文图时将其静音。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mlp.fit(x_train_scaled, y_train_scaled)
        train_pred = y_scaler.inverse_transform(mlp.predict(x_train_scaled))
        valid_pred = y_scaler.inverse_transform(mlp.predict(x_valid_scaled))
        records.append(
            {
                "epoch": float(epoch),
                "train_mse": float(mean_squared_error(y_train, train_pred)),
                "valid_mse": float(mean_squared_error(y_valid, valid_pred)),
            }
        )
    return pd.DataFrame(records)


def plot_loss_curve(history: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(history["epoch"], history["train_mse"], label="Training MSE", color="#315C7A", linewidth=2.0)
    ax.plot(history["epoch"], history["valid_mse"], label="Validation MSE", color="#C96F32", linewidth=2.0)
    ax.set_title("Model Training Loss Curve", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.legend(frameon=False)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(true_values: np.ndarray, predicted_values: np.ndarray, label: str, output_path: Path) -> None:
    r2 = r2_score(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    lower = float(min(np.min(true_values), np.min(predicted_values)))
    upper = float(max(np.max(true_values), np.max(predicted_values)))
    padding = (upper - lower) * 0.06 if upper > lower else 1.0
    lower -= padding
    upper += padding

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    ax.scatter(true_values, predicted_values, s=20, alpha=0.55, color="#315C7A", edgecolors="none")
    ax.plot([lower, upper], [lower, upper], color="#B94E48", linewidth=2.0, label="Ideal prediction")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_title(f"True vs Predicted {label}", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.text(
        0.04,
        0.96,
        f"R2 = {r2:.3f}\nMSE = {mse:.4g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.95},
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(dataframe: pd.DataFrame, output_path: Path) -> None:
    columns = DIMENSION_COLUMNS + ["s11_min_db", "s11_min_freq_ghz", "gain_max", "gain_mean"]
    corr = dataframe[columns].corr()
    labels = [column.replace(" [mm]", "").replace("_", "\n") for column in columns]

    fig, ax = plt.subplots(figsize=(10.8, 8.8))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            value = corr.iloc[row, col]
            color = "white" if abs(value) > 0.55 else "#222222"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color=color)

    ax.set_title("Parameter and Performance Correlation Heatmap", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_new_antenna_features(args.features_csv)
    model = unwrap_model(args.model)
    hidden_layers = get_mlp_hidden_layers(model)

    x_train, x_valid, y_train, y_valid = train_test_split(
        dataset.dimensions,
        dataset.targets,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    draw_network_structure(
        output_path=args.output_dir / "figure_1_mlp_architecture.png",
        input_count=len(DIMENSION_COLUMNS),
        hidden_layers=hidden_layers,
        output_count=len(TARGET_COLUMNS),
    )

    loss_history = train_loss_history(
        x_train=x_train,
        x_valid=x_valid,
        y_train=y_train,
        y_valid=y_valid,
        hidden_layers=hidden_layers,
        epochs=args.loss_epochs,
        random_state=args.random_state,
    )
    loss_history.to_csv(args.output_dir / "loss_history.csv", index=False, encoding="utf-8-sig")
    plot_loss_curve(loss_history, args.output_dir / "figure_2_loss_curve.png")

    valid_pred = np.asarray(model.predict(x_valid), dtype=np.float64)
    plot_prediction_scatter(
        true_values=y_valid[:, TARGET_COLUMNS.index("s11_min_db")],
        predicted_values=valid_pred[:, TARGET_COLUMNS.index("s11_min_db")],
        label="S11_min_db",
        output_path=args.output_dir / "figure_3a_s11_min_db_scatter.png",
    )
    plot_prediction_scatter(
        true_values=y_valid[:, TARGET_COLUMNS.index("gain_max")],
        predicted_values=valid_pred[:, TARGET_COLUMNS.index("gain_max")],
        label="Gain_max",
        output_path=args.output_dir / "figure_3b_gain_max_scatter.png",
    )

    plot_correlation_heatmap(dataset.dataframe, args.output_dir / "figure_4_parameter_correlation_heatmap.png")

    print("论文图已生成:")
    for path in sorted(args.output_dir.glob("*.png")):
        print(path)
    print(args.output_dir / "loss_history.csv")


if __name__ == "__main__":
    main()

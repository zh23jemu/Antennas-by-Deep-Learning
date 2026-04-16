from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_true_vs_predicted_feature_curves(
    true_features: np.ndarray,
    predicted_features: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["min_s_value", "min_point_index", "mean_s_value", "std_s_value"]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5), dpi=160)
    plt.bar(x - width / 2, true_features, width=width, color="#1f5f8b", label="True")
    plt.bar(x + width / 2, predicted_features, width=width, color="#d1495b", label="Predicted")
    plt.xticks(x, labels, rotation=15)
    plt.title(title)
    plt.ylabel("Feature value")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_predicted_feature_summary(
    predicted_features: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["min_s_value", "min_point_index", "mean_s_value", "std_s_value"]
    x = np.arange(len(labels))

    plt.figure(figsize=(9, 5), dpi=160)
    plt.bar(x, predicted_features, color=["#d1495b", "#edae49", "#1f5f8b", "#66a182"])
    plt.xticks(x, labels, rotation=15)
    plt.title(title)
    plt.ylabel("Feature value")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

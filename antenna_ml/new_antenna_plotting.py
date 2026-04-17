from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_LABELS = [
    "s11_min_db",
    "s11_min_freq_ghz",
    "s11_mean_db",
    "s11_std_db",
    "s11_bw<-10dB",
    "gain_max",
    "gain_mean",
    "gain_std",
]


def plot_feature_comparison(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(FEATURE_LABELS))
    width = 0.35

    plt.figure(figsize=(11, 5), dpi=160)
    plt.bar(x - width / 2, true_values, width=width, color="#1f5f8b", label="True")
    plt.bar(x + width / 2, predicted_values, width=width, color="#d1495b", label="Predicted")
    plt.xticks(x, FEATURE_LABELS, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel("Feature value")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_prediction_summary(predicted_values: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(FEATURE_LABELS))

    plt.figure(figsize=(11, 5), dpi=160)
    plt.bar(x, predicted_values, color=["#d1495b", "#edae49", "#1f5f8b", "#66a182", "#8c6f56", "#00798c", "#ff9f1c", "#6a4c93"])
    plt.xticks(x, FEATURE_LABELS, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel("Feature value")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

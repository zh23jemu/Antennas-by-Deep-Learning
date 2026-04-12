from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_s_curve(s_values: np.ndarray, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    point_indices = np.arange(s_values.size)
    best_index = int(np.argmin(s_values))
    best_value = float(s_values[best_index])

    plt.figure(figsize=(9, 5), dpi=160)
    plt.plot(point_indices, s_values, color="#1f5f8b", linewidth=2.0, label="Predicted S parameter")
    plt.scatter([best_index], [best_value], color="#d1495b", zorder=3, label="Minimum point")
    plt.axvline(best_index, color="#d1495b", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.title(title)
    plt.xlabel("Sample point index")
    plt.ylabel("S parameter value")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

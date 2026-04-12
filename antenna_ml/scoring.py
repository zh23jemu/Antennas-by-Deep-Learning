from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObjectiveWeights:
    s_parameter: float = 1.0
    gain: float = 0.0
    efficiency: float = 0.0


def s_parameter_objective(s_values: np.ndarray) -> float:
    return float(np.min(s_values))


def composite_objective(
    s_values: np.ndarray,
    gain: np.ndarray | None = None,
    efficiency: np.ndarray | None = None,
    weights: ObjectiveWeights = ObjectiveWeights(),
) -> float:
    """Lower score is better.

    Gain and efficiency are negated because larger values should improve the design.
    """
    score = weights.s_parameter * s_parameter_objective(s_values)
    if gain is not None and weights.gain:
        score -= weights.gain * float(np.max(gain))
    if efficiency is not None and weights.efficiency:
        score -= weights.efficiency * float(np.max(efficiency))
    return float(score)

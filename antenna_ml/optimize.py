from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline

from antenna_ml.scoring import s_parameter_objective


def score_s_curve(s_curve: np.ndarray) -> float:
    return s_parameter_objective(s_curve)


def random_search(
    model: Pipeline,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    n_candidates: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    rng = np.random.default_rng(random_state)
    candidates = rng.uniform(lower_bounds, upper_bounds, size=(n_candidates, lower_bounds.size))
    predictions = model.predict(candidates)
    scores = predictions.min(axis=1)
    best_index = int(np.argmin(scores))
    best_dimensions = candidates[best_index]
    best_curve = predictions[best_index]
    best_point_index = int(np.argmin(best_curve))
    return best_dimensions, best_curve, float(scores[best_index]), best_point_index

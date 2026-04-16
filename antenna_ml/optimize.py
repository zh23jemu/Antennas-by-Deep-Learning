from __future__ import annotations

import numpy as np
from typing import Any

def score_features(features: np.ndarray) -> float:
    return float(features[0])


def random_search(
    model: Any,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    n_candidates: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    rng = np.random.default_rng(random_state)
    candidates = rng.uniform(lower_bounds, upper_bounds, size=(n_candidates, lower_bounds.size))
    predictions = model.predict(candidates)
    scores = predictions[:, 0]
    best_index = int(np.argmin(scores))
    best_dimensions = candidates[best_index]
    best_features = predictions[best_index]
    best_point_index = int(round(best_features[1]))
    return best_dimensions, best_features, float(scores[best_index]), best_point_index

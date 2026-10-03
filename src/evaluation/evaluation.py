"""Evaluación sobre predicciones ya calculadas (RPS por partido y vectorizado).

Este módulo NO entrena. El entrenamiento y la CV temporal viven en
`src.training.training`; las métricas agregadas, en `src.metrics.metrics`.
"""

from typing import List, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Re-export por conveniencia: histórcamente `evaluate_model` se importaba desde
# este módulo. La implementación real es la de `src.metrics.metrics`.
from src.metrics.metrics import evaluate_model  # noqa: F401


def calculate_rps_single(
    probs: Union[NDArray[np.float64], List[float]], outcome_idx: int
) -> float:
    """Calculates Ranked Probability Score (RPS) for a single match (3 classes:

    Away, Draw, Home).
    """
    cdf_pred: NDArray[np.float64] = np.cumsum(probs)
    cdf_obs: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
    cdf_obs[outcome_idx:] = 1.0

    # Sum over r-1 (first two outcomes for a 3-class system)
    diff: NDArray[np.float64] = cdf_pred[:2] - cdf_obs[:2]
    return float(0.5 * np.sum(diff**2))


def get_vectorized_rps(
    probs_array: NDArray[np.float64],
    y_true: Union[pd.Series, NDArray[np.int_], List[int]],
) -> NDArray[np.float64]:
    """Calculates RPS for an array of predictions and ground truth labels."""
    n_samples: int = len(y_true)
    rps_values: List[float] = []

    # Convert y_true values explicitly for indexing
    y_true_array: NDArray[np.int_] = np.asarray(y_true, dtype=np.int_)

    i: int
    for i in range(n_samples):
        rps_val: float = calculate_rps_single(
            probs_array[i], int(y_true_array[i])
        )
        rps_values.append(rps_val)

    return np.array(rps_values, dtype=np.float64)




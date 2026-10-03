import pandas as pd 
import numpy as np 
from sklearn.metrics import accuracy_score
from typing import Dict

def calculate_rps(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
        Ranked Probability Score - The ONE metric that matters for W-D-L.

        Args:
            y_true: Shape (n,) with values [0=Home, 1=Draw, 2=Away]
            y_prob: Shape (n, 3) with [P(Home), P(Draw), P(Away)]

        Returns:
            Average RPS (lower is better, theoretical min=0)
    """

    n_samples, n_classes = y_prob.shape
    assert n_classes == 3

    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1

    # Cumulative distributions
    cdf_pred = np.cumsum(y_prob, axis=1)
    cdf_true = np.cumsum(y_true_onehot, axis=1)

    # RPS formula
    squared_diff = (cdf_pred[:, :-1] - cdf_true[:, :-1]) ** 2
    rps_per_sample = np.sum(squared_diff, axis=1) / (n_classes - 1)

    return np.mean(rps_per_sample)

def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Standard log loss for reference"""
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1 - eps)
    true_class_probs = y_prob[np.arange(len(y_true)), y_true]
    return -np.mean(np.log(true_class_probs))



def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error multiclass.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
    y_prob : array-like, shape (n_samples, n_classes)
    n_bins : int

    Returns
    -------
    float
        Macro-average ECE across classes.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n_classes = y_prob.shape[1]
    ece_values = []

    for c in range(n_classes):
        y_binary = (y_true == c).astype(int)
        probs_c = y_prob[:, c]

        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(probs_c, bin_edges[1:-1], right=True)

        ece_c = 0.0
        n = len(probs_c)

        for b in range(n_bins):
            mask = bin_ids == b
            if not np.any(mask):
                continue

            accuracy_bin = y_binary[mask].mean()
            confidence_bin = probs_c[mask].mean()
            weight = mask.sum() / n

            ece_c += weight * abs(accuracy_bin - confidence_bin)

        ece_values.append(ece_c)

    return np.mean(ece_values)



def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Complete evaluation metrics"""

    if len(y_true) == 0:
        return {'rps': np.nan, 'log_loss': np.nan, 'accuracy': np.nan}

    y_pred = np.argmax(y_prob, axis=1)

    return {
        'rps': calculate_rps(y_true, y_prob),
        'log_loss': log_loss(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'ece': calculate_ece(y_true, y_prob)
    }


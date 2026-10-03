"""Metricas para modelos de distribucion de conteos (Poisson / NegBin).

Agnostico al target: las funciones reciben los parametros predichos y la grilla
discreta, sin asumir que se trata de goles. Los umbrales Over/Under y el maximo
de la grilla vienen de la configuracion YAML del target.

Convencion NB2
--------------
    Var(Y) = mu + alpha * mu^2
    Relacion con (n, p) de scipy.stats.nbinom:
        n = 1 / alpha
        p = n / (n + mu)
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import nbinom, poisson

DistFamily = Literal["poisson", "negbin"]
TotalMethod = Literal["convolution", "poisson_approx"]

# Claves esperadas en el dict `params` segun la familia.
_PARAM_KEYS: Dict[str, Tuple[str, ...]] = {
    "poisson": ("lambda",),
    "negbin": ("mu", "alpha"),
}


# ---------------------------------------------------------------------------
# Validacion y normalizacion de parametros
# ---------------------------------------------------------------------------

def validate_params(params: Dict[str, NDArray], family: DistFamily) -> None:
    """Falla temprano y con mensaje legible si faltan claves en `params`."""
    if family not in _PARAM_KEYS:
        raise ValueError(
            f"Familia no soportada: '{family}'. Opciones: {sorted(_PARAM_KEYS)}"
        )

    required: Tuple[str, ...] = _PARAM_KEYS[family]
    missing: List[str] = [k for k in required if k not in params]
    if missing:
        raise KeyError(
            f"Faltan parametros para family='{family}': {missing}. "
            f"Se esperaba {list(required)}, recibi {list(params)}."
        )


def get_mean(params: Dict[str, NDArray], family: DistFamily) -> NDArray[np.float64]:
    """Media predicha, sea cual sea la familia (lambda en Poisson, mu en NegBin)."""
    validate_params(params, family)
    key: str = "lambda" if family == "poisson" else "mu"
    return np.asarray(params[key], dtype=float).ravel()


def _nb_n_p(params: Dict[str, NDArray]) -> Tuple[NDArray, NDArray]:
    """Convierte (mu, alpha) de NB2 a los (n, p) que espera scipy."""
    mu: NDArray = np.asarray(params["mu"], dtype=float)
    alpha: NDArray = np.asarray(params["alpha"], dtype=float)
    n: NDArray = 1.0 / alpha
    p: NDArray = n / (n + mu)
    return n, p


# ---------------------------------------------------------------------------
# Helpers de distribucion: PMF, CDF y logPMF agnosticos a la familia
# ---------------------------------------------------------------------------

def get_pmf(
    k_grid: NDArray[np.int_],
    params: Dict[str, NDArray],
    family: DistFamily,
) -> NDArray[np.float64]:
    """PMF sobre la grilla, shape (N, K).

    Necesaria para la convolucion exacta del total.
    """
    validate_params(params, family)

    if family == "poisson":
        return poisson.pmf(k_grid[:, None], params["lambda"]).T

    n, p = _nb_n_p(params)
    return nbinom.pmf(k_grid[:, None], n, p).T


def get_cdf(
    k_grid: NDArray[np.int_],
    params: Dict[str, NDArray],
    family: DistFamily,
) -> NDArray[np.float64]:
    """CDF predictiva por observacion y punto de la grilla, shape (N, K)."""
    validate_params(params, family)

    if family == "poisson":
        # poisson.cdf(k, mu): shape (K,) x (N,) -> transponer a (N, K)
        return poisson.cdf(k_grid[:, None], params["lambda"]).T

    n, p = _nb_n_p(params)
    return nbinom.cdf(k_grid[:, None], n, p).T


def get_logpmf(
    y_true: NDArray,
    params: Dict[str, NDArray],
    family: DistFamily,
) -> NDArray[np.float64]:
    """log-PMF evaluado en los valores observados, shape (N,)."""
    validate_params(params, family)

    if family == "poisson":
        return poisson.logpmf(y_true, params["lambda"])

    n, p = _nb_n_p(params)
    return nbinom.logpmf(y_true, n, p)


def get_survival(
    floor_t: int,
    params: Dict[str, NDArray],
    family: DistFamily,
) -> NDArray[np.float64]:
    """P(X > floor_t) = 1 - CDF(floor_t), shape (N,)."""
    validate_params(params, family)

    if family == "poisson":
        return 1.0 - poisson.cdf(floor_t, params["lambda"])

    n, p = _nb_n_p(params)
    return 1.0 - nbinom.cdf(floor_t, n, p)


# ---------------------------------------------------------------------------
# Distribucion del total (suma de los dos componentes)
# ---------------------------------------------------------------------------

def total_pmf(
    params_a: Dict[str, NDArray],
    params_b: Dict[str, NDArray],
    family: DistFamily,
    max_k: int,
    method: TotalMethod = "convolution",
    renormalize: bool = True,
) -> NDArray[np.float64]:
    """PMF del total (componente A + componente B), shape (N, 2*max_k + 1).

    Parameters
    ----------
    params_a, params_b : parametros predichos de cada componente.
    family : familia de ambos componentes.
    max_k  : maximo de la grilla de cada componente. El total llega a 2*max_k.
    method :
        'convolution'    -> convolucion exacta de las PMF, valida para cualquier
                            familia. Asume independencia entre componentes.
        'poisson_approx' -> Poisson(mu_a + mu_b). Exacto solo si family es
                            Poisson; para NegBin subestima la cola alta.
    renormalize : reescala la PMF para que sume 1. La grilla finita [0, max_k]
        trunca masa en la cola (~1e-3 con max_k=8 y lambda ~1.5); sin
        renormalizar ese faltante se descuenta de P(total > threshold) y sesga
        el ECE a la baja de forma sistematica.

    Notes
    -----
    La convolucion es exacta bajo independencia. Bajo NegBin la suma no es
    cerrada, por eso 'poisson_approx' introduce sesgo: se conserva unicamente
    para reproducir resultados calculados con la version anterior.
    """
    if method not in ("convolution", "poisson_approx"):
        raise ValueError(
            f"total_method invalido: '{method}'. "
            f"Opciones: 'convolution', 'poisson_approx'."
        )

    k_grid: NDArray[np.int_] = np.arange(max_k + 1)
    n_total: int = 2 * max_k + 1

    if method == "poisson_approx":
        mean_total: NDArray = get_mean(params_a, family) + get_mean(params_b, family)
        grid_total: NDArray[np.int_] = np.arange(n_total)
        pmf_approx: NDArray = poisson.pmf(grid_total[:, None], mean_total).T
        if renormalize:
            pmf_approx = pmf_approx / pmf_approx.sum(axis=1, keepdims=True)
        return pmf_approx

    pmf_a: NDArray = get_pmf(k_grid, params_a, family)  # (N, K)
    pmf_b: NDArray = get_pmf(k_grid, params_b, family)  # (N, K)

    # Convolucion discreta fila a fila:
    #   P(T = t) = sum_i P(A = i) * P(B = t - i)
    # El producto exterior (N, K, K) indexado por i+j se acumula en t.
    outer: NDArray = pmf_a[:, :, None] * pmf_b[:, None, :]      # (N, K, K)
    idx_sum: NDArray[np.int_] = (
        k_grid[:, None] + k_grid[None, :]
    ).ravel()                                                    # (K*K,)

    n_obs: int = outer.shape[0]
    flat: NDArray = outer.reshape(n_obs, -1)                     # (N, K*K)

    pmf_total: NDArray = np.zeros((n_obs, n_total), dtype=float)
    np.add.at(pmf_total.T, idx_sum, flat.T)

    if renormalize:
        pmf_total = pmf_total / pmf_total.sum(axis=1, keepdims=True)

    return pmf_total


def total_survival(
    threshold: float,
    params_a: Dict[str, NDArray],
    params_b: Dict[str, NDArray],
    family: DistFamily,
    max_k: int,
    method: TotalMethod = "convolution",
) -> NDArray[np.float64]:
    """P(total > threshold) por observacion, shape (N,)."""
    pmf: NDArray = total_pmf(params_a, params_b, family, max_k, method)
    grid: NDArray[np.int_] = np.arange(pmf.shape[1])
    # threshold es semientero (2.5): 'total > 2.5' equivale a 'total >= 3'.
    mask: NDArray[np.bool_] = grid > threshold
    return pmf[:, mask].sum(axis=1)


# ---------------------------------------------------------------------------
# Metricas individuales
# ---------------------------------------------------------------------------

def rps_per_sample(
    y_true: NDArray,
    params: Dict[str, NDArray],
    family: DistFamily = "poisson",
    max_k: int = 8,
) -> NDArray[np.float64]:
    """RPS observacion a observacion sobre la grilla [0, max_k], shape (N,)."""
    y_arr: NDArray = np.asarray(y_true).ravel()
    k_grid: NDArray[np.int_] = np.arange(max_k + 1)

    cdf_pred: NDArray = get_cdf(k_grid, params, family)               # (N, K)
    cdf_obs: NDArray = (k_grid >= y_arr[:, None]).astype(float)       # (N, K)

    return np.sum((cdf_pred - cdf_obs) ** 2, axis=1)


def calculate_rps(
    y_true: NDArray,
    params: Dict[str, NDArray],
    family: DistFamily = "poisson",
    max_k: int = 8,
    label: str = "Model",
    verbose: bool = False,
) -> Dict[str, object]:
    """Ranked Probability Score sobre la distribucion predictiva completa.

    Mide la calidad de toda la distribucion sobre la grilla discreta
    [0, ..., max_k]. Metrica estandar en prediccion deportiva (menor es mejor).

    Parameters
    ----------
    y_true  : valores observados.
    params  : Poisson -> {'lambda': ...}; NegBin -> {'mu': ..., 'alpha': ...}.
    family  : 'poisson' | 'negbin'.
    max_k   : maximo de la grilla de la CDF.
    label   : nombre del modelo para el reporte.
    verbose : imprime el reporte por consola.

    Returns
    -------
    dict con rps_mean, rps_per_sample, rps_std, rps_median, rps_p95, n_samples.
    """
    y_arr: NDArray = np.asarray(y_true).ravel()
    per_sample: NDArray = rps_per_sample(y_arr, params, family, max_k)

    result: Dict[str, object] = {
        "rps_mean": float(np.mean(per_sample)),
        "rps_per_sample": per_sample,
        "rps_std": float(np.std(per_sample)),
        "rps_median": float(np.median(per_sample)),
        "rps_p95": float(np.percentile(per_sample, 95)),
        "n_samples": int(len(y_arr)),
    }

    if verbose:
        print(f"\n{'-' * 50}")
        print(f"  RPS Report - {label} [{family}]")
        print(f"{'-' * 50}")
        print(f"  N observaciones : {result['n_samples']}")
        print(f"  RPS mean        : {result['rps_mean']:.5f}   <- principal")
        print(f"  RPS std         : {result['rps_std']:.5f}")
        print(f"  RPS median      : {result['rps_median']:.5f}")
        print(f"  RPS p95         : {result['rps_p95']:.5f}   <- cola")
        print(f"{'-' * 50}")

    return result


def calculate_nll(
    y_true: NDArray,
    params: Dict[str, NDArray],
    family: DistFamily = "poisson",
    label: str = "Model",
    verbose: bool = False,
) -> float:
    """Negative Log-Likelihood promedio (menor es mejor).

    Penaliza asignar baja probabilidad al valor efectivamente observado. Es la
    metrica principal para seleccionar familia de distribucion y comparar modelos.
    """
    y_arr: NDArray = np.asarray(y_true).ravel()
    log_probs: NDArray = get_logpmf(y_arr, params, family)
    nll: float = float(-np.mean(log_probs))

    if verbose:
        print(f"\n{'-' * 50}")
        print(f"  NLL Report - {label} [{family}]")
        print(f"{'-' * 50}")
        print(f"  N observaciones : {len(y_arr)}")
        print(f"  NLL mean        : {nll:.5f}   <- (menor es mejor)")
        print(f"{'-' * 50}")

    return nll


def calculate_ece_by_threshold(
    y_true_a: NDArray,
    y_true_b: NDArray,
    params_a: Dict[str, NDArray],
    params_b: Dict[str, NDArray],
    family: DistFamily = "poisson",
    thresholds: Optional[Sequence[float]] = None,
    max_k: int = 8,
    total_method: TotalMethod = "convolution",
    label: str = "Model",
    verbose: bool = False,
) -> pd.DataFrame:
    """Expected Calibration Error del mercado Over/Under sobre el total.

    Para cada umbral X:
        p_pred = P(componente_a + componente_b > X)
        p_real = frecuencia observada
        error  = |mean(p_pred) - p_real|

    Un modelo bien calibrado tiene ECE cercano a 0 en cada umbral.

    Parameters
    ----------
    y_true_a, y_true_b : observados de cada componente.
    params_a, params_b : parametros predichos de cada componente.
    thresholds   : grilla Over/Under. Sin valor, se deriva de max_k.
    total_method : 'convolution' (exacto) | 'poisson_approx' (compatibilidad).

    Returns
    -------
    DataFrame con threshold, p_pred_mean, p_real, abs_error, n_over.
    """
    if thresholds is None:
        thresholds = [k + 0.5 for k in range(max_k)]

    arr_a: NDArray = np.asarray(y_true_a).ravel()
    arr_b: NDArray = np.asarray(y_true_b).ravel()
    y_total: NDArray = arr_a + arr_b

    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        p_pred: NDArray = total_survival(
            threshold, params_a, params_b, family, max_k, total_method
        )
        p_real: float = float(np.mean(y_total > threshold))

        rows.append({
            "threshold": float(threshold),
            "p_pred_mean": float(np.mean(p_pred)),
            "p_real": p_real,
            "abs_error": float(abs(np.mean(p_pred) - p_real)),
            "n_over": int(np.sum(y_total > threshold)),
        })

    results_df: pd.DataFrame = pd.DataFrame(rows)

    if verbose:
        ece_global: float = float(results_df["abs_error"].mean())
        print(f"\n{'-' * 65}")
        print(f"  ECE by Threshold - {label} [{family}, total={total_method}]")
        print(f"{'-' * 65}")
        print(results_df.to_string(index=False,
                                   float_format=lambda x: f"{x:.4f}"))
        print(f"  ECE global (mean abs error): {ece_global:.5f}")
        print(f"{'-' * 65}")

    return results_df


# ---------------------------------------------------------------------------
# Wrapper principal
# ---------------------------------------------------------------------------

def evaluate_distribution_model(
    y_true: Dict[str, NDArray],
    params: Dict[str, Dict[str, NDArray]],
    keys: Sequence[str],
    family: DistFamily = "poisson",
    label: str = "Model",
    max_k: int = 8,
    thresholds: Optional[Sequence[float]] = None,
    total_method: TotalMethod = "convolution",
    verbose: bool = False,
) -> Dict[str, object]:
    """Ejecuta RPS, NLL y ECE para los dos componentes en una sola llamada.

    Agnostico al target: `keys` define los nombres de los componentes
    (['home', 'away'] para goles, pero podria ser cualquier par).

    Parameters
    ----------
    y_true : {key: observados} para cada componente.
    params : {key: parametros predichos} para cada componente.
    keys   : orden de los componentes (el primero es 'a' en el ECE).
    family : 'poisson' | 'negbin'.

    Returns
    -------
    dict con:
        rps        : {key: dict de metricas RPS}
        nll        : {key: float}
        ece_df     : DataFrame del ECE por umbral
        ece_global : float
        family, label, total_method

    Examples
    --------
    >>> evaluate_distribution_model(
    ...     y_true={'home': y_h, 'away': y_a},
    ...     params={'home': {'lambda': lh}, 'away': {'lambda': la}},
    ...     keys=['home', 'away'], family='poisson',
    ... )
    """
    if len(keys) != 2:
        raise ValueError(
            f"Se esperan exactamente 2 componentes para el mercado Over/Under; "
            f"recibi {list(keys)}."
        )

    missing: List[str] = [k for k in keys if k not in y_true or k not in params]
    if missing:
        raise KeyError(
            f"Faltan datos para los componentes {missing} en y_true o params."
        )

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"  FULL EVALUATION - {label} [{family}]")
        print(f"{'=' * 50}")

    key_a, key_b = keys[0], keys[1]

    rps: Dict[str, Dict[str, object]] = {
        key: calculate_rps(
            y_true[key], params[key], family=family, max_k=max_k,
            label=f"{label} | {key}", verbose=verbose,
        )
        for key in keys
    }

    nll: Dict[str, float] = {
        key: calculate_nll(
            y_true[key], params[key], family=family,
            label=f"{label} | {key}", verbose=verbose,
        )
        for key in keys
    }

    ece_df: pd.DataFrame = calculate_ece_by_threshold(
        y_true[key_a], y_true[key_b],
        params[key_a], params[key_b],
        family=family,
        thresholds=thresholds,
        max_k=max_k,
        total_method=total_method,
        label=label,
        verbose=verbose,
    )

    return {
        "rps": rps,
        "nll": nll,
        "ece_df": ece_df,
        "ece_global": float(ece_df["abs_error"].mean()),
        "family": family,
        "label": label,
        "total_method": total_method,
    }


def evaluate_from_config(
    y_true: Dict[str, NDArray],
    params: Dict[str, Dict[str, NDArray]],
    config,
    family: DistFamily = "poisson",
    label: str = "Model",
    verbose: bool = False,
) -> Dict[str, object]:
    """`evaluate_distribution_model` tomando la grilla y umbrales del YAML.

    Parameters
    ----------
    config : `src.config.ModelConfig` ya cargado.
    """
    target = config.target
    return evaluate_distribution_model(
        y_true=y_true,
        params=params,
        keys=target.keys,
        family=family,
        label=label,
        max_k=target.max_k,
        thresholds=target.thresholds,
        total_method=target.total_method,
        verbose=verbose,
    )

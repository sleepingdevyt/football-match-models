"""Helpers de seleccion de features compartidos entre objetivos.

Este modulo contiene la parte del pipeline que NO depende del objective de
LightGBM: muestreo temporal, permutacion por bloques, correlacion de Spearman,
inferencia sobre la matriz nula y el paralelismo de los modelos nulos.

Los pipelines concretos (`feature_selection.py` para clasificacion,
`poisson_feature_selection.py` para conteos) importan de aca y solo aportan lo
que si es especifico de su objective: los parametros de LightGBM, el estimador
de mutual information y la metrica de validacion.

Antes de este modulo, ambos pipelines llevaban su propia copia de
`_block_permute`, `_spearman_abs_matrix`, `_disjoint_block_indices`,
`_temporal_stratified_index`, `_parametric_pvalue`, `preflight_permutation_power`
y `reanalyze_significance`. Las copias ya habian divergido entre si.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Final, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Aliases de tipo
# ---------------------------------------------------------------------------
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

EPS: Final[float] = 1e-12


# ===========================================================================
# Muestreo temporal
# ===========================================================================
def temporal_stratified_index(
    n_rows: int, sample_size: int, n_strata: int
) -> IntArray:
    """Indices estratificados a lo largo del eje temporal.

    Tomar `.iloc[:sample_size]` usaria solo el tramo MAS ANTIGUO del dataset: el
    ranking de importancia reflejaria un unico regimen viejo, mientras que
    produccion corre sobre el regimen mas nuevo.
    """
    if sample_size >= n_rows:
        return np.arange(n_rows, dtype=np.int64)

    bounds: IntArray = np.linspace(0, n_rows, n_strata + 1).astype(np.int64)
    per_stratum: int = max(1, sample_size // n_strata)
    picked: List[IntArray] = []

    s: int
    for s in range(n_strata):
        lo: int = int(bounds[s])
        hi: int = int(bounds[s + 1])
        if hi <= lo:
            continue
        take: int = min(per_stratum, hi - lo)
        picked.append(np.linspace(lo, hi - 1, take).astype(np.int64))

    idx: IntArray = np.unique(np.concatenate(picked))
    return idx[:sample_size].astype(np.int64)


def disjoint_block_indices(n_rows: int, n_blocks: int) -> List[Tuple[int, int]]:
    """Bloques temporales DISJUNTOS y contiguos.

    `TimeSeriesSplit(...).split()` devuelve ventanas ANIDADAS (fold1 subset de
    fold2 ...): la mediana entre folds queda dominada por los datos iniciales
    repetidos, y no son regimenes independientes.
    """
    bounds: IntArray = np.linspace(0, n_rows, n_blocks + 1).astype(np.int64)
    i: int
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_blocks)]


# ===========================================================================
# Permutacion por bloques
# ===========================================================================
def estimate_block_size_runlength(
    y: IntArray, min_block: int = 5, max_block: int = 250
) -> int:
    """Longitud de bloque por persistencia del label (targets CATEGORICOS).

    Usa la longitud media de racha. Si el target cambia en cada fila, el bloque
    colapsa a `min_block` y la permutacion degenera hacia la i.i.d.

    Para un conteo no tiene sentido (no hay "rachas" de un entero): usar
    `estimate_block_size_acf`.
    """
    n: int = int(y.shape[0])
    if n < 2:
        return min_block

    changes: int = int(np.sum(y[1:] != y[:-1]))
    mean_run: float = float(n) / float(max(1, changes + 1))
    block: int = int(np.ceil(mean_run * 3.0))
    return int(np.clip(block, min_block, max(min_block, min(max_block, n // 10))))


def estimate_block_size_acf(
    y: FloatArray,
    max_lag: int = 60,
    min_block: int = 1,
    max_block: int = 250,
) -> Tuple[int, float, int]:
    """Longitud de bloque desde la AUTOCORRELACION del target (targets CONTINUOS).

    Busca el primer lag cuya |ACF| cae por debajo de la banda de ruido
    2/sqrt(n) y toma 3x ese lag como bloque.

    Si el target no tiene autocorrelacion significativa (caso tipico de goles o
    corners por partido: filas consecutivas son equipos distintos), el bloque
    colapsa a 1 y la permutacion por bloques degenera correctamente a i.i.d.
    Eso NO es un fallo del estimador: es la respuesta correcta para ese dato.

    Returns
    -------
    (block_size, acf_lag1, primer_lag_no_significativo)
    """
    n: int = int(y.shape[0])
    if n < 20:
        return min_block, 0.0, 1

    y_c: FloatArray = y.astype(np.float64) - float(np.mean(y))
    denom: float = float(np.dot(y_c, y_c))
    if denom <= EPS:
        return min_block, 0.0, 1

    band: float = 2.0 / np.sqrt(float(n))
    lag_cut: int = 1
    acf_lag1: float = 0.0

    lag: int
    for lag in range(1, min(max_lag, n // 4) + 1):
        acf: float = float(np.dot(y_c[:-lag], y_c[lag:]) / denom)
        if lag == 1:
            acf_lag1 = acf
        if abs(acf) < band:
            lag_cut = lag
            break
        lag_cut = lag + 1

    block: int = int(
        np.clip(3 * (lag_cut - 1) if lag_cut > 1 else 1, min_block, max_block)
    )
    return int(min(block, max(1, n // 10))), acf_lag1, lag_cut


def block_permute(
    y: npt.NDArray, block_size: int, rng: np.random.Generator
) -> npt.NDArray:
    """Permutacion por bloques contiguos (moving-block sin reemplazo).

    `rng.permutation` i.i.d. destruye la autocorrelacion del target y produce un
    modelo nulo ARTIFICIALMENTE DEBIL: cualquier feature con tendencia lenta
    (deriva por temporada, cambios de reglas, mix de ligas) bate a ese nulo
    aunque no tenga poder predictivo real. Rotar bloques preserva la estructura
    local y sube la barra.

    Con `block_size <= 1` esto es exactamente la permutacion i.i.d.
    """
    n: int = int(y.shape[0])
    if block_size <= 1:
        return rng.permutation(y)

    n_blocks: int = int(np.ceil(n / block_size))
    starts: IntArray = np.arange(n_blocks, dtype=np.int64) * block_size
    order: IntArray = rng.permutation(n_blocks).astype(np.int64)

    pieces: List[npt.NDArray] = [y[int(s): int(s) + block_size] for s in starts[order]]
    return np.concatenate(pieces)[:n]


def safe_block_permute_variance(
    y: FloatArray,
    block_size: int,
    rng: np.random.Generator,
    max_tries: int = 10,
) -> FloatArray:
    """Permutacion por bloques que garantiza varianza > 0 (targets de CONTEO).

    Un conteo permutado no necesita preservar clases (a diferencia del caso
    multiclase), pero si debe conservar varianza: si colapsara a un valor
    constante, LightGBM devolveria el intercepto y la importancia nula seria 0
    para todas las features, inflando artificialmente la significancia.
    """
    attempt: int
    for attempt in range(max_tries):
        candidate: FloatArray = np.asarray(
            block_permute(y, block_size, rng), dtype=np.float64
        )
        if float(np.std(candidate)) > EPS:
            return candidate
    return np.asarray(rng.permutation(y), dtype=np.float64)


def safe_block_permute_classes(
    y: IntArray,
    block_size: int,
    rng: np.random.Generator,
    n_classes: int,
    max_tries: int = 20,
) -> IntArray:
    """Permutacion por bloques que preserva las N clases (targets CATEGORICOS).

    LightGBM en modo multiclase exige que el target contenga las `n_classes`
    etiquetas; una rotacion desafortunada puede perder una clase rara.
    """
    attempt: int
    for attempt in range(max_tries):
        candidate: IntArray = np.asarray(
            block_permute(y, block_size, rng), dtype=np.int64
        )
        if int(np.unique(candidate).size) == n_classes:
            return candidate
    return np.asarray(rng.permutation(y), dtype=np.int64)


# ===========================================================================
# Correlacion e imputacion
# ===========================================================================
def spearman_abs_matrix(X_block: FloatArray) -> FloatArray:
    """|Spearman| feature-feature via rank-transform + Pearson. Vectorizado."""
    ranks: FloatArray = np.apply_along_axis(rankdata, 0, X_block).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr: FloatArray = np.corrcoef(ranks, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(corr).astype(np.float64)


def median_impute(X: pd.DataFrame, cols: Sequence[str]) -> FloatArray:
    """Imputacion por mediana para MI y Spearman.

    Un sentinel tipo -999 rompe dos cosas: (a) el estimador kNN de MI ve un
    cluster outlier lejano que distorsiona las vecindades; (b) dos features con
    el MISMO patron de missingness (equipos recien ascendidos, principio de
    temporada) correlacionan alto SOLO por co-ausencia, y el filtro dropea por
    la razon equivocada. LightGBM sigue recibiendo NaN nativo aparte.
    """
    arr: FloatArray = X[list(cols)].to_numpy(dtype=np.float64, copy=True)
    med: FloatArray = np.asarray(np.nanmedian(arr, axis=0), dtype=np.float64)
    med = np.nan_to_num(med, nan=0.0)
    nan_rows, nan_cols = np.where(np.isnan(arr))
    arr[nan_rows, nan_cols] = np.take(med, nan_cols)
    return arr


# ===========================================================================
# Importancia
# ===========================================================================
def mean_abs_contrib(
    booster: lgb.Booster, X: pd.DataFrame, n_classes: int = 1
) -> FloatArray:
    """Importancia SHAP media (|contribucion|) via `pred_contrib` nativo.

    Es el mismo TreeSHAP que `shap.TreeExplainer` pero sin la capa intermedia de
    Python (~5-10x mas rapido) y sin los problemas de dtype 'category' de
    algunas versiones de shap.

    Maneja los dos layouts:
      - multiclase: (n, (n_features + 1) * n_classes), class-major.
      - regresion / binario: (n, n_features + 1).
    La ultima columna de cada bloque es el base value y se descarta.
    """
    n_rows: int = int(X.shape[0])
    n_features: int = int(X.shape[1])

    raw: FloatArray = np.asarray(
        booster.predict(X, pred_contrib=True, raw_score=True), dtype=np.float64
    )

    expected_multiclass: int = (n_features + 1) * int(n_classes)
    if int(n_classes) > 1 and raw.shape[1] == expected_multiclass:
        contrib: FloatArray = raw.reshape(n_rows, int(n_classes), n_features + 1)
        mean_abs: FloatArray = np.abs(contrib[:, :, :n_features]).mean(axis=(0, 1))
    else:
        mean_abs = np.abs(raw[:, :n_features]).mean(axis=0)

    return np.asarray(mean_abs, dtype=np.float64)


# ===========================================================================
# Paralelismo de los modelos nulos
# ===========================================================================
def resolve_n_jobs(n_jobs: int, threads_per_model: int) -> int:
    """Workers efectivos: explicito si se pidio, si no cores // threads_per_model."""
    if int(n_jobs) > 0:
        return int(n_jobs)
    n_cores: int = int(os.cpu_count() or 1)
    return max(1, n_cores // max(1, int(threads_per_model)))


def _train_null_batch(
    X: pd.DataFrame,
    y_nulls: List[npt.NDArray],
    cats: List[str],
    params: Dict[str, object],
    n_rounds: int,
    X_imp: pd.DataFrame,
    n_classes: int,
) -> FloatArray:
    """Worker: entrena un LOTE de modelos nulos y devuelve sus importancias.

    Se despacha por lotes y no por corrida individual a proposito: con joblib
    cada tarea pickle-a el DataFrame completo. Un lote por worker significa
    pickle-arlo `n_jobs` veces en lugar de `n_runs` veces.
    """
    n_features: int = int(X_imp.shape[1])
    out: FloatArray = np.zeros((len(y_nulls), n_features), dtype=np.float64)

    i: int
    for i in range(len(y_nulls)):
        ds: lgb.Dataset = lgb.Dataset(
            X, label=y_nulls[i], categorical_feature=cats, free_raw_data=False
        )
        booster: lgb.Booster = lgb.train(params, ds, num_boost_round=n_rounds)
        out[i, :] = mean_abs_contrib(booster, X_imp, n_classes)

    return out


def build_null_matrix(
    X_cur: pd.DataFrame,
    y_nulls: List[npt.NDArray],
    cats: List[str],
    params: Dict[str, object],
    n_rounds: int,
    X_imp: pd.DataFrame,
    n_jobs: int,
    n_classes: int = 1,
    verbose: bool = True,
) -> FloatArray:
    """Construye la matriz nula `(n_runs, n_features)`.

    Los targets permutados se reciben YA generados desde el proceso principal:
    el resultado es IDENTICO corra en 1 worker o en 8. La reproducibilidad no
    depende del orden en que terminen los workers.
    """
    n_runs: int = len(y_nulls)
    batches: List[List[npt.NDArray]] = [y_nulls[i::n_jobs] for i in range(n_jobs)]
    batches = [b for b in batches if len(b) > 0]

    t0: float = time.time()
    results: List[FloatArray]
    if len(batches) == 1:
        results = [
            _train_null_batch(X_cur, batches[0], cats, params, n_rounds, X_imp, n_classes)
        ]
    else:
        computed: Any = Parallel(n_jobs=len(batches), backend="loky", verbose=0)(
            delayed(_train_null_batch)(
                X_cur, b, cats, params, n_rounds, X_imp, n_classes
            )
            for b in batches
        )
        results = [np.asarray(r, dtype=np.float64) for r in computed]

    if verbose:
        elapsed: float = time.time() - t0
        print(
            f"    {n_runs} corridas en {elapsed / 60.0:.1f} min "
            f"({elapsed / max(1, n_runs):.2f} s/run, {len(batches)} workers)"
        )

    return np.vstack(results).astype(np.float64)


# ===========================================================================
# Inferencia
# ===========================================================================
def parametric_pvalue(
    null_samples: FloatArray, real_value: float, dist: str
) -> float:
    """p-value parametrico: ajusta una distribucion a las muestras nulas de la
    feature y evalua la cola derecha en el valor real.

    Rompe el piso de resolucion del test empirico (1/(n_runs+1)) sin necesidad
    de decenas de miles de permutaciones. Todas las importancias son >= 0, asi
    que se usan familias de soporte positivo. Lognormal por defecto: gamma
    extrapola demasiado lejos en la cola con pocas muestras.
    """
    clean: FloatArray = null_samples[np.isfinite(null_samples)]
    if clean.size < 5 or float(np.std(clean)) <= 0.0:
        return 1.0 if real_value <= float(np.max(clean, initial=0.0)) else 1e-12
    if real_value <= 0.0:
        return 1.0

    try:
        if dist == "gamma":
            a: float
            loc: float
            scale: float
            a, loc, scale = stats.gamma.fit(clean, floc=0.0)
            p: float = float(stats.gamma.sf(real_value, a, loc=loc, scale=scale))
        else:
            log_null: FloatArray = np.log(clean + EPS)
            mu: float = float(np.mean(log_null))
            sigma: float = float(np.std(log_null, ddof=1))
            if sigma <= 0.0:
                return 1e-12
            p = float(stats.norm.sf(np.log(real_value + EPS), loc=mu, scale=sigma))
    except Exception:  # noqa: BLE001
        mu_raw: float = float(np.mean(clean))
        sd_raw: float = float(np.std(clean, ddof=1))
        p = float(stats.norm.sf(real_value, loc=mu_raw, scale=max(sd_raw, EPS)))

    return float(np.clip(p, 1e-300, 1.0))


def preflight_permutation_power(
    n_features: int, n_runs: int, alpha: float, method: str
) -> bool:
    """Verifica ANTES de gastar el computo si la configuracion puede rechazar
    alguna hipotesis.

    Con `n_runs` permutaciones el p empirico minimo es 1/(n_runs+1). BH sobre m
    tests exige, para la feature de rango 1, p <= alpha/m. La regla operativa:

        BH rechaza  <=>  (features en el piso) / m  >=  1 / ((n_runs+1) * alpha)

    Con m=688 y n_runs=100 el piso es 0.0099 y BH pide 7.3e-5: el test no puede
    rechazar nada, por construccion.
    """
    floor: float = 1.0 / (float(n_runs) + 1.0)
    required_frac: float = floor / alpha
    rank1_needed: float = alpha / float(n_features)
    n_runs_rank1: int = int(np.ceil(1.0 / rank1_needed)) - 1

    print(f"\n  [PREFLIGHT] m (tests simultaneos)         : {n_features}")
    print(f"  [PREFLIGHT] p_raw minimo alcanzable       : {floor:.5f}")
    print(f"  [PREFLIGHT] p requerido por BH (rango 1)  : {rank1_needed:.3e}")
    print(f"  [PREFLIGHT] n_runs para rango 1 empirico  : {n_runs_rank1:,}")
    print(
        f"  [PREFLIGHT] fraccion en el piso requerida : "
        f"{required_frac:.1%} de {n_features}"
    )

    viable: bool = True
    if method == "empirical_bh" and required_frac > 0.30:
        print("  [PREFLIGHT] >> ADVERTENCIA: es improbable que esa fraccion de features")
        print("               este en el piso. Subi n_runs, baja m (corr_threshold),")
        print("               o usa significance_method='parametric_bh' / 'zscore'.")
        viable = False
    elif method == "empirical_bh":
        print("  [PREFLIGHT] >> Viable si al menos esa fraccion supera a TODAS las nulas.")
    elif method == "parametric_bh":
        print("  [PREFLIGHT] >> method='parametric_bh': el piso no aplica. Viable.")
    else:
        print("  [PREFLIGHT] >> method='zscore': sin correccion de multiplicidad formal.")

    return viable


def significance_from_null(
    null_matrix: FloatArray,
    real_imp: FloatArray,
    alpha: float,
    method: str,
    dist: str = "lognormal",
    require_floor: bool = True,
    z_threshold: float = 4.0,
) -> Dict[str, npt.NDArray]:
    """Calcula todos los criterios de significancia sobre una matriz nula.

    Devuelve las columnas crudas para que cada pipeline arme su propia tabla:
    `p_emp`, `p_bh_emp`, `p_param`, `p_bh_param`, `z_score`, `at_floor`,
    `null_mean`, `null_max`, `rej_emp`, `rej_par` y `significant`.

    p = (1 + #{null >= real}) / (n_runs + 1). Nunca 0.

    `require_floor`: el p parametrico solo puede ORDENAR features que ya
    superaron a TODAS las nulas observadas. Sin esa guarda se estaria
    extrapolando la cola desde n_runs puntos y rescatando features que nunca
    ganaron de forma no parametrica.
    """
    n_runs: int = int(null_matrix.shape[0])
    m: int = int(null_matrix.shape[1])

    n_exceed: IntArray = (
        (null_matrix >= real_imp[None, :]).sum(axis=0).astype(np.int64)
    )
    p_emp: FloatArray = (1.0 + n_exceed.astype(np.float64)) / (float(n_runs) + 1.0)
    at_floor: BoolArray = n_exceed == 0

    null_mean: FloatArray = null_matrix.mean(axis=0)
    null_std: FloatArray = null_matrix.std(axis=0, ddof=1) + EPS
    z_score: FloatArray = (real_imp - null_mean) / null_std

    p_param: FloatArray = np.array(
        [parametric_pvalue(null_matrix[:, j], float(real_imp[j]), dist) for j in range(m)],
        dtype=np.float64,
    )

    rej_emp: BoolArray
    p_bh_emp: FloatArray
    rej_emp, p_bh_emp, _, _ = multipletests(p_emp, alpha=alpha, method="fdr_bh")

    rej_par: BoolArray
    p_bh_par: FloatArray
    rej_par, p_bh_par, _, _ = multipletests(p_param, alpha=alpha, method="fdr_bh")

    significant: BoolArray
    if method == "empirical_bh":
        significant = rej_emp
    elif method == "zscore":
        significant = z_score >= float(z_threshold)
    else:
        significant = (rej_par & at_floor) if require_floor else rej_par

    return {
        "p_emp": p_emp,
        "p_bh_emp": p_bh_emp,
        "p_param": p_param,
        "p_bh_param": p_bh_par,
        "z_score": z_score,
        "at_floor": at_floor,
        "null_mean": null_mean,
        "null_max": null_matrix.max(axis=0),
        "rej_emp": rej_emp,
        "rej_par": rej_par,
        "significant": significant,
    }


def reanalyze_significance(
    null_matrix: FloatArray,
    real_imp: FloatArray,
    feat_list: Sequence[str],
    alpha: float = 0.05,
    method: str = "empirical_bh",
    dist: str = "lognormal",
    require_floor: bool = True,
    z_threshold: float = 4.0,
) -> pd.DataFrame:
    """Recalcula la significancia con otros criterios REUTILIZANDO la matriz nula.

    Cambiar alpha o de criterio ya no cuesta media hora de computo.

    `feat_list` debe ser el orden CANONICO de columnas de `null_matrix`
    (`report.shuffle_feature_order`), no el de la tabla ordenada por z_score.
    """
    stats_cols: Dict[str, npt.NDArray] = significance_from_null(
        null_matrix, real_imp, alpha, method, dist, require_floor, z_threshold
    )

    return (
        pd.DataFrame(
            {
                "feature": list(feat_list),
                "imp_real": real_imp,
                "z_score": stats_cols["z_score"],
                "at_floor": stats_cols["at_floor"],
                "p_emp": stats_cols["p_emp"],
                "p_bh_emp": stats_cols["p_bh_emp"],
                "p_bh_param": stats_cols["p_bh_param"],
                "significant": stats_cols["significant"],
            }
        )
        .sort_values("z_score", ascending=False)
        .reset_index(drop=True)
    )

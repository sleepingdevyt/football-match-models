"""
================================================================================
  TEMPORAL FEATURE SELECTION PIPELINE — TIME SERIES CLASSIFICATION  (v2.1)
  Multiclass | Diseñado para datasets anchos (~1.4k variables) y pocas filas.

  Flujo:
    [0] Sparsity Filter
    [1] Mutual Information Pre-Filter
    [2] Temporal Correlation Filter (clustering, bloques disjuntos)
    [3] Block-Permutation Importance + BH Correction
    [4] Feature Stability Filter (dispersión de percentiles)
    [5] Validación en VAL: baseline vs selected

  Notas de diseño relevantes para quien reutilice el módulo:

  [F-CAT] LightGBM compara la CANTIDAD de columnas con dtype 'category' del
      dataset de validación contra las del de entrenamiento. Cualquier columna
      que venga como 'category' en X_val/X_test pero no en X_train dispara
      "train and valid dataset categorical_feature do not match" recién en el
      STEP 5. `align_frame_to_reference` fuerza el mismo esquema en los tres
      frames y `assert_same_categorical_schema` falla temprano con un mensaje
      legible.
  [C1] p-value empírico -> (b + 1) / (m + 1). Nunca puede ser 0.
  [C2] Permutación por BLOQUES: preserva la autocorrelación del target.
  [C4] STEP 4 mide DISPERSIÓN de la importancia, no la media.
  [C5] Muestra SHAP estratificada temporalmente (no las primeras N filas).
  [C6] bagging_freq=1: sin esto `bagging_fraction` es un no-op en LightGBM.
  [C7] pred_contrib nativo de LightGBM en vez de shap.TreeExplainer (~5-10x).
  [C8] STEP 2: clustering jerárquico sobre |rho| mediano en bloques DISJUNTOS.
  [C10] Validación obligatoria en VAL: baseline (todas) vs seleccionadas.
  [C14] Walk-forward de re-selección para medir estabilidad del propio pipeline.
================================================================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional, Sequence, Set, Tuple

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, log_loss
from statsmodels.stats.multitest import multipletests

from src.utils.utils import (
    SEP,
    SUB,
    align_frame_to_reference,
    assert_same_categorical_schema,
    categorical_columns,
    print_header,
)

# ================================================================================
# ALIASES DE TIPO
# ================================================================================
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

DEFAULT_RANDOM_STATE: Final[int] = 67


# ================================================================================
# CONFIGURACIÓN
# ================================================================================
@dataclass(frozen=True)
class PipelineConfig:
    """Todos los hiperparámetros del pipeline, tipados e inmutables."""

    auto_detect_string_categoricals: bool = True
    corr_n_blocks: int = 5
    corr_threshold: float = 0.85
    explicit_categorical: Optional[List[str]] = None
    mi_n_neighbors: int = 3
    mi_threshold: float = 1e-4
    parametric_dist: str = "lognormal"           # "lognormal" (conservador) | "gamma"
    preflight_abort: bool = False                # True -> aborta si es imposible
    random_state: int = DEFAULT_RANDOM_STATE
    require_empirical_floor: bool = True         # [F2] guarda anti-extrapolación
    run_stability_filter: bool = True
    run_validation: bool = True
    shuffle_block_size: int = 0                  # 0 -> auto (run-length del target)
    shuffle_boost_rounds: int = 150
    shuffle_fdr_alpha: float = 0.05
    shuffle_n_runs: int = 60
    shuffle_n_strata: int = 10
    shuffle_sample_size: int = 2000
    shuffle_use_two_pass: bool = False
    significance_method: str = "parametric_bh"   # "empirical_bh"|"parametric_bh"|"zscore"
    sparsity_threshold: float = 0.98
    stability_boost_rounds: int = 100
    stability_min_rows_per_window: int = 300
    stability_min_score: float = 0.35
    stability_n_windows: int = 5
    stability_penalty_k: float = 1.0
    strict_schema_alignment: bool = True         # [F-CAT] abortar si no se puede alinear
    target_is_ordinal: bool = False
    validation_boost_rounds: int = 500
    validation_early_stopping: int = 50
    zscore_threshold: float = 4.0


@dataclass
class PipelineReport:
    """Trazabilidad completa de cada paso, para auditoría."""

    n_start: int = 0
    n_after_sparsity: int = 0
    n_after_mi: int = 0
    n_after_corr: int = 0
    n_after_shuffle: int = 0
    n_after_stability: int = 0

    dropped_sparse: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    mi_scores: Dict[str, float] = field(default_factory=dict)
    corr_clusters: Dict[int, List[str]] = field(default_factory=dict)
    stability_table: Optional[pd.DataFrame] = None
    validation_table: Optional[pd.DataFrame] = None
    categorical_audit: Optional[pd.DataFrame] = None

    shuffle_table: Optional[pd.DataFrame] = None
    shuffle_null_matrix: Optional[np.ndarray] = None
    shuffle_real_imp: Optional[np.ndarray] = None

    # [F-CAT] traza del alineamiento de esquemas
    schema_alignment: Optional[pd.DataFrame] = None

    final_features: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ================================================================================
# HELPERS DE BAJO NIVEL
# ================================================================================
def preflight_permutation_power(
    n_features: int,
    n_runs: int,
    alpha: float,
    method: str,
) -> bool:
    """
    Verifica si la configuración puede rechazar alguna hipótesis.
    Retorna True si el test es viable.
    """
    floor: float = 1.0 / (float(n_runs) + 1.0)
    required_frac: float = floor / alpha
    rank1_needed: float = alpha / float(n_features)
    n_runs_for_rank1: int = int(np.ceil(1.0 / rank1_needed)) - 1

    print(f"\n  [PREFLIGHT] p_raw mínimo alcanzable      : {floor:.5f}")
    print(f"  [PREFLIGHT] p requerido por BH (rango 1)  : {rank1_needed:.3e}")
    print(f"  [PREFLIGHT] n_runs para rango 1 empírico  : {n_runs_for_rank1:,}")
    print(f"  [PREFLIGHT] fracción en el piso requerida : {required_frac:.1%} de {n_features}")

    viable: bool = required_frac <= 1.0
    if method == "empirical_bh" and required_frac > 0.30:
        print(f"  [PREFLIGHT] >> ADVERTENCIA: con method='empirical_bh' necesitás que")
        print(f"               más del {required_frac:.0%} de las features estén en el piso.")
        print(f"               Es improbable. Usá significance_method='parametric_bh'")
        print(f"               o subí n_runs a ~{int(np.ceil(1.0/(0.046*alpha)))-1:,}.")
        viable = False
    elif method == "parametric_bh":
        print(f"  [PREFLIGHT] >> method='parametric_bh': el piso no aplica. Viable.")

    return viable


def _parametric_pvalue(
    null_samples: FloatArray,
    real_value: float,
    dist: str,
) -> float:
    """
    [F1] Ajusta una distribución a las muestras nulas y evalúa la cola derecha
    en `real_value`. Todas las importancias son >= 0, así que se usan familias
    con soporte positivo.

    Retorna un p-value acotado por abajo en 1e-300 para evitar underflow.
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
        else:  # lognormal — conservador
            log_null: FloatArray = np.log(clean + 1e-12)
            mu: float = float(np.mean(log_null))
            sigma: float = float(np.std(log_null, ddof=1))
            if sigma <= 0.0:
                return 1e-12
            p = float(stats.norm.sf(np.log(real_value + 1e-12), loc=mu, scale=sigma))
    except Exception:
        # Fallback: normal sobre los valores crudos
        mu_raw: float = float(np.mean(clean))
        sd_raw: float = float(np.std(clean, ddof=1))
        p = float(stats.norm.sf(real_value, loc=mu_raw, scale=max(sd_raw, 1e-12)))

    return float(np.clip(p, 1e-300, 1.0))


def _lgb_base_params(
    n_classes: int, random_state: int, strong: bool = False
) -> Dict[str, object]:
    """
    Parámetros base de LightGBM.

    [C6] bagging_freq=1 es OBLIGATORIO: sin él, bagging_fraction se ignora
    silenciosamente y no hay subsampling en ningún modelo del pipeline.
    """
    params: Dict[str, object] = {
        "objective": "multiclass",
        "num_class": int(n_classes),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 4,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 50,
        "lambda_l2": 1.0,
        "verbose": -1,
        "seed": int(random_state),
        "bagging_seed": int(random_state),
        "feature_fraction_seed": int(random_state),
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 0,
    }
    if strong:
        params["num_leaves"] = 63
        params["max_depth"] = 6
        params["min_data_in_leaf"] = 30
    return params


def _mean_abs_contrib(booster: lgb.Booster, X: pd.DataFrame, n_classes: int) -> FloatArray:
    """
    [C7] Importancia SHAP media (|contribución|) usando pred_contrib nativo.

    LightGBM devuelve un array (n_samples, (n_features + 1) * n_classes) con
    layout class-major; la última columna de cada bloque es el base value.
    Es el mismo TreeSHAP que usa shap.TreeExplainer pero sin la capa de Python
    intermedia, y sin los problemas de dtype 'category' de algunas versiones.

    Retorna un vector float64 de longitud n_features.
    """
    n_rows: int = int(X.shape[0])
    n_features: int = int(X.shape[1])

    raw: FloatArray = np.asarray(
        booster.predict(X, pred_contrib=True, raw_score=True), dtype=np.float64
    )

    expected_multiclass: int = (n_features + 1) * n_classes
    if raw.shape[1] == expected_multiclass:
        contrib: FloatArray = raw.reshape(n_rows, n_classes, n_features + 1)
        mean_abs: FloatArray = np.abs(contrib[:, :, :n_features]).mean(axis=(0, 1))
    else:
        # Binario / regresión: (n_samples, n_features + 1)
        mean_abs = np.abs(raw[:, :n_features]).mean(axis=0)

    return np.asarray(mean_abs, dtype=np.float64)


def _temporal_stratified_index(n_rows: int, sample_size: int, n_strata: int) -> IntArray:
    """
    [C5] Índices estratificados a lo largo del eje temporal.

    Tomar `.iloc[:sample_size]` usaría solo el tramo MÁS ANTIGUO del dataset: el
    ranking de importancia reflejaría un único régimen viejo, mientras producción
    corre sobre el régimen más nuevo.
    """
    if sample_size >= n_rows:
        return np.arange(n_rows, dtype=np.int64)

    strata_bounds: IntArray = np.linspace(0, n_rows, n_strata + 1).astype(np.int64)
    per_stratum: int = max(1, sample_size // n_strata)
    picked: List[IntArray] = []

    s: int
    for s in range(n_strata):
        lo: int = int(strata_bounds[s])
        hi: int = int(strata_bounds[s + 1])
        if hi <= lo:
            continue
        take: int = min(per_stratum, hi - lo)
        picked.append(np.linspace(lo, hi - 1, take).astype(np.int64))

    idx: IntArray = np.unique(np.concatenate(picked))
    return idx[:sample_size]


def _estimate_block_size(y: IntArray, min_block: int = 5, max_block: int = 250) -> int:
    """
    [C2] Estima la longitud de bloque a partir de la persistencia del target.

    Usa la longitud media de racha (run-length) del label. Si el target cambia
    en cada fila (sin persistencia), el bloque colapsa a `min_block` y la
    permutación por bloques degenera correctamente hacia la i.i.d.
    """
    n: int = int(y.shape[0])
    if n < 2:
        return min_block

    changes: int = int(np.sum(y[1:] != y[:-1]))
    mean_run: float = float(n) / float(max(1, changes + 1))
    block: int = int(np.ceil(mean_run * 3.0))
    return int(np.clip(block, min_block, max(min_block, min(max_block, n // 10))))


def _block_permute(y: IntArray, block_size: int, rng: np.random.Generator) -> IntArray:
    """
    [C2] Permutación por bloques contiguos (moving-block bootstrap sin reemplazo).

    `np.random.permutation` i.i.d. destruye la autocorrelación del target y
    produce un modelo nulo ARTIFICIALMENTE DÉBIL: cualquier feature con
    tendencia lenta bate a ese nulo aunque no tenga poder predictivo real.
    Rotar bloques preserva la estructura local y sube la barra del nulo.
    """
    n: int = int(y.shape[0])
    n_blocks: int = int(np.ceil(n / block_size))
    starts: IntArray = np.arange(n_blocks, dtype=np.int64) * block_size
    order: IntArray = rng.permutation(n_blocks).astype(np.int64)

    pieces: List[IntArray] = [y[int(s): int(s) + block_size] for s in starts[order]]
    out: IntArray = np.concatenate(pieces)[:n]
    return np.asarray(out, dtype=np.int64)


def _safe_block_permute(
    y: IntArray,
    block_size: int,
    rng: np.random.Generator,
    n_classes: int,
    max_tries: int = 20,
) -> IntArray:
    """Garantiza que el target permutado conserve las N clases (LightGBM lo exige)."""
    attempt: int
    for attempt in range(max_tries):
        candidate: IntArray = _block_permute(y, block_size, rng)
        if int(np.unique(candidate).size) == n_classes:
            return candidate
    return np.asarray(rng.permutation(y), dtype=np.int64)


def _spearman_abs_matrix(X_block: FloatArray) -> FloatArray:
    """|Spearman| feature-feature vía rank-transform + Pearson. Vectorizado."""
    ranks: FloatArray = np.apply_along_axis(rankdata, 0, X_block).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr: FloatArray = np.corrcoef(ranks, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.abs(corr)


def _disjoint_block_indices(n_rows: int, n_blocks: int) -> List[Tuple[int, int]]:
    """
    [C8] Bloques temporales DISJUNTOS y contiguos.

    `TimeSeriesSplit(...).split()` devuelve ventanas ANIDADAS (fold1 ⊂ fold2 ⊂ ...):
    la mediana entre folds queda dominada por los datos iniciales repetidos, y no
    son regímenes independientes.
    """
    bounds: IntArray = np.linspace(0, n_rows, n_blocks + 1).astype(np.int64)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_blocks)]


def encode_target(y: pd.Series) -> Tuple[IntArray, Dict[object, int]]:
    """Codifica el target a enteros 0..K-1 y devuelve el mapeo para producción."""
    classes: npt.NDArray = np.unique(y.to_numpy())
    mapping: Dict[object, int] = {cls: idx for idx, cls in enumerate(classes)}
    encoded: IntArray = y.map(mapping).to_numpy().astype(np.int64)
    return encoded, mapping


def encode_target_with_map(
    y: pd.Series, class_map: Dict[object, int], name: str
) -> IntArray:
    """
    [F-YVAL] Aplica un mapeo de clases ya fijado en train.

    `y.map(class_map).astype(np.int64)` sobre clases no vistas produce NaN y,
    tras el cast, enteros basura que LightGBM acepta sin chistar. Acá falla.
    """
    mapped: pd.Series = y.map(class_map)
    if mapped.isna().any():
        unseen = sorted({str(v) for v in y[mapped.isna()].unique()})
        raise ValueError(
            f"'{name}' contiene clases que no existen en y_train: {unseen[:10]}. "
            f"Clases conocidas: {list(class_map.keys())}"
        )
    return mapped.to_numpy().astype(np.int64)


# ================================================================================
# STEP 0 — Sparsity Filter
# ================================================================================
def drop_sparse_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    sparsity_threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Elimina columnas cuyo valor más frecuente domina más de `sparsity_threshold`
    de las filas de X_train. Se evalúa SOLO sobre train (sin leakage).

    El default es 0.98 y no 0.75: con pocas miles de filas y clases
    desbalanceadas, 0.75 descarta features raras pero informativas.
    """
    print_header(f"STEP 0 — Sparsity Filter  (threshold = {sparsity_threshold:.0%})")

    cols_to_drop: List[str] = []
    col: str
    for col in X_train.columns:
        counts: pd.Series = X_train[col].value_counts(normalize=True, dropna=False)
        if counts.empty:
            cols_to_drop.append(col)
            continue
        top_share: float = float(counts.iloc[0])
        if top_share > sparsity_threshold:
            cols_to_drop.append(col)

    n_in: int = int(X_train.shape[1])
    print(f"  Features entrada    : {n_in}")
    print(f"  Features esparsas   : {len(cols_to_drop)}")
    print(f"  Features restantes  : {n_in - len(cols_to_drop)}")

    X_train_f: pd.DataFrame = X_train.drop(columns=cols_to_drop)
    X_val_f: pd.DataFrame = X_val.drop(columns=cols_to_drop, errors="ignore")
    X_test_f: pd.DataFrame = X_test.drop(columns=cols_to_drop, errors="ignore")

    return X_train_f, X_val_f, X_test_f, cols_to_drop


# ================================================================================
# HELPER — Formato de categóricas
# ================================================================================
def format_categorical_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: PipelineConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    """
    [C9] Solo se convierten a 'category' las columnas object/string y las
    declaradas explícitamente.

    [F-CAT] Después de declarar las categóricas de TRAIN, val y test se ALINEAN
    completos contra el esquema de train, columna por columna. Tocar solo las
    columnas candidatas detectadas en train deja pasar cualquier columna que ya
    viniera como 'category' en val/test, y `len(cat_cols)` deja de coincidir
    cuando LightGBM construye el valid set.
    """
    X_train_c: pd.DataFrame = X_train.copy()

    candidates: Set[str] = set()
    if cfg.auto_detect_string_categoricals:
        candidates.update(
            X_train_c.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        )
    if cfg.explicit_categorical is not None:
        candidates.update([c for c in cfg.explicit_categorical if c in X_train_c.columns])

    col: str
    for col in sorted(candidates):
        X_train_c[col] = X_train_c[col].astype("category")

    cat_cols: List[str] = categorical_columns(X_train_c)

    # [F-CAT] alineación total de val/test contra el esquema de train
    X_val_c: pd.DataFrame
    X_test_c: pd.DataFrame
    val_changes: pd.DataFrame
    test_changes: pd.DataFrame
    X_val_c, val_changes = align_frame_to_reference(X_train_c, X_val, "val")
    X_test_c, test_changes = align_frame_to_reference(X_train_c, X_test, "test")

    assert_same_categorical_schema(X_train_c, X_val_c, "train", "val")
    assert_same_categorical_schema(X_train_c, X_test_c, "train", "test")

    changes: pd.DataFrame = pd.concat([val_changes, test_changes], ignore_index=True)

    print(f"\n  Categóricas declaradas: {len(cat_cols)}")
    print(f"  Esquema train/val/test: alineado y verificado.")
    return X_train_c, X_val_c, X_test_c, cat_cols, changes


def audit_categorical_coverage(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: Sequence[str],
) -> pd.DataFrame:
    """
    [C11] Cuantifica el fallo silencioso: niveles presentes en val/test que no
    existen en train se convierten a NaN sin error ni warning, y LightGBM
    devuelve una predicción perfectamente plausible.

    En producción esta misma métrica debe monitorearse con alerta.
    """
    rows: List[Dict[str, object]] = []
    col: str
    for col in cat_cols:
        train_levels: Set[object] = set(X_train[col].dropna().unique().tolist())
        train_nan: float = float(X_train[col].isna().mean())

        val_unseen: float = float(X_val[col].isna().mean()) - train_nan
        test_unseen: float = float(X_test[col].isna().mean()) - train_nan

        rows.append(
            {
                "feature": col,
                "n_levels_train": len(train_levels),
                "val_extra_nan_rate": round(max(0.0, val_unseen), 4),
                "test_extra_nan_rate": round(max(0.0, test_unseen), 4),
            }
        )

    audit: pd.DataFrame = pd.DataFrame(rows)
    if not audit.empty:
        flagged: pd.DataFrame = audit[
            (audit["val_extra_nan_rate"] > 0.01) | (audit["test_extra_nan_rate"] > 0.01)
        ]
        if not flagged.empty:
            print(f"\n  [AUDIT] {len(flagged)} categóricas con niveles no vistos (>1%):")
            print(flagged.to_string(index=False))
    return audit


# ================================================================================
# STEP 1 — Mutual Information Pre-Filter
# ================================================================================
def mutual_information_prefilter(
    X_train: pd.DataFrame,
    y_train: IntArray,
    cat_cols: Sequence[str],
    cfg: PipelineConfig,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Filtro barato O(n*p). MI ~ 0 implica independencia estadística del target.

    Los scores se RETORNAN y se reutilizan en STEP 2 como criterio de relevancia
    para elegir el representante de cada cluster de features correlacionadas
    (evita usar Spearman sobre un target nominal).
    """
    print_header(
        f"STEP 1 — Mutual Information Pre-Filter  (threshold = {cfg.mi_threshold:.0e})"
    )

    all_cols: List[str] = X_train.columns.tolist()
    cat_set: Set[str] = set(cat_cols)
    num_cols: List[str] = [c for c in all_cols if c not in cat_set]

    X_enc: pd.DataFrame = X_train.copy()
    if len(num_cols) > 0:
        medians: pd.Series = X_enc[num_cols].median(numeric_only=True)
        X_enc[num_cols] = X_enc[num_cols].fillna(medians).fillna(0.0)

    col: str
    for col in cat_cols:
        X_enc[col] = X_enc[col].cat.codes.astype(np.int64)   # -1 = NaN, categoría propia

    discrete_mask: BoolArray = np.array([c in cat_set for c in all_cols], dtype=bool)

    mi_values: FloatArray = mutual_info_classif(
        X_enc.to_numpy(dtype=np.float64),
        y_train,
        discrete_features=discrete_mask,
        n_neighbors=cfg.mi_n_neighbors,
        random_state=cfg.random_state,
    ).astype(np.float64)

    mi_scores: Dict[str, float] = {c: float(v) for c, v in zip(all_cols, mi_values)}
    selected: List[str] = [c for c in all_cols if mi_scores[c] >= cfg.mi_threshold]

    print(f"  Features entrada    : {len(all_cols)}")
    print(f"  Eliminadas (MI ~ 0) : {len(all_cols) - len(selected)}")
    print(f"  Features restantes  : {len(selected)}")

    return selected, mi_scores


# ================================================================================
# STEP 2 — Temporal Correlation Filter
# ================================================================================
def temporal_correlation_filter(
    X_train: pd.DataFrame,
    y_train: IntArray,
    features: Sequence[str],
    cat_cols: Sequence[str],
    mi_scores: Dict[str, float],
    cfg: PipelineConfig,
) -> Tuple[List[str], Dict[int, List[str]]]:
    """
    [C8] Clustering jerárquico sobre la distancia (1 - |rho| mediano).

    Detalles:
      - |rho| se calcula en bloques temporales DISJUNTOS y se toma la mediana:
        una pareja solo se considera redundante si lo es en la mayoría de los
        regímenes (esto es lo que ataca la no-estacionariedad).
      - Clustering jerárquico en vez de recorrido greedy sobre la matriz
        triangular: el resultado no depende del orden de las columnas y maneja
        transitividad (A~B, B~C => un solo cluster).
      - El representante se elige por MI con el target si el target es nominal,
        o por |Spearman| mediano si es ordinal.
    """
    print_header(
        f"STEP 2 — Temporal Correlation Filter  "
        f"(|rho| = {cfg.corr_threshold}, bloques disjuntos = {cfg.corr_n_blocks})"
    )

    cat_set: Set[str] = set(cat_cols)
    numeric_feats: List[str] = [f for f in features if f not in cat_set]
    passthrough: List[str] = [f for f in features if f in cat_set]

    if len(numeric_feats) < 2:
        print("  Menos de 2 features numéricas; se omite el filtro.")
        return list(features), {}

    n_rows: int = int(X_train.shape[0])
    # [F-COW] copia explícita: con copy-on-write (pandas >= 2.0) `to_numpy()`
    # puede devolver un buffer de solo lectura y la imputación in-place
    # revienta con "assignment destination is read-only".
    X_num: FloatArray = np.array(
        X_train[numeric_feats].astype(np.float64).to_numpy(), dtype=np.float64, copy=True
    )
    col_medians: FloatArray = np.nanmedian(X_num, axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)
    inds: Tuple[IntArray, IntArray] = np.where(np.isnan(X_num))
    X_num[inds] = np.take(col_medians, inds[1])

    blocks: List[Tuple[int, int]] = _disjoint_block_indices(n_rows, cfg.corr_n_blocks)
    n_feat: int = len(numeric_feats)
    corr_stack: FloatArray = np.zeros((len(blocks), n_feat, n_feat), dtype=np.float64)

    b_idx: int
    lo: int
    hi: int
    for b_idx, (lo, hi) in enumerate(blocks):
        corr_stack[b_idx] = _spearman_abs_matrix(X_num[lo:hi, :])

    median_corr: FloatArray = np.median(corr_stack, axis=0)
    np.fill_diagonal(median_corr, 1.0)

    distance: FloatArray = 1.0 - median_corr
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, 2.0)
    distance = (distance + distance.T) / 2.0

    try:
        clusterer: AgglomerativeClustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(1.0 - cfg.corr_threshold),
            metric="precomputed",
            linkage="average",
        )
    except TypeError:  # sklearn < 1.2
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(1.0 - cfg.corr_threshold),
            affinity="precomputed",
            linkage="average",
        )

    labels: IntArray = clusterer.fit_predict(distance).astype(np.int64)

    # --- Criterio de relevancia para elegir representante ---
    relevance: Dict[str, float] = {}
    if cfg.target_is_ordinal:
        f_idx: int
        feat: str
        for f_idx, feat in enumerate(numeric_feats):
            per_block: List[float] = []
            for lo, hi in blocks:
                rho: float
                rho, _ = spearmanr(X_num[lo:hi, f_idx], y_train[lo:hi])
                per_block.append(0.0 if np.isnan(rho) else abs(float(rho)))
            relevance[feat] = float(np.median(per_block))
    else:
        # Target nominal: Spearman(feature, y) no tiene interpretación. Se usa MI.
        relevance = {f: float(mi_scores.get(f, 0.0)) for f in numeric_feats}

    clusters: Dict[int, List[str]] = {}
    lab: int
    for f_idx, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(numeric_feats[f_idx])

    kept_numeric: List[str] = []
    members: List[str]
    for lab, members in clusters.items():
        best: str = max(members, key=lambda f: relevance.get(f, 0.0))
        kept_numeric.append(best)

    kept_set: Set[str] = set(kept_numeric)
    final_features: List[str] = [f for f in features if f in kept_set or f in cat_set]
    n_clusters_multi: int = int(sum(1 for m in clusters.values() if len(m) > 1))

    print(f"  Features entrada    : {len(features)}")
    print(f"  Clusters formados   : {len(clusters)}  ({n_clusters_multi} con >1 miembro)")
    print(f"  Eliminadas (redund.): {len(numeric_feats) - len(kept_numeric)}")
    print(f"  Features restantes  : {len(final_features)}  (+{len(passthrough)} categóricas)")

    return final_features, clusters


# ================================================================================
# STEP 3 — Block-Permutation Importance + Benjamini-Hochberg
# ================================================================================
def block_permutation_importance_bh(
    X_train: pd.DataFrame,
    y_train: IntArray,
    features: Sequence[str],
    cat_cols: Sequence[str],
    n_classes: int,
    cfg: PipelineConfig,
) -> Tuple[List[str], pd.DataFrame, FloatArray]:
    """
    Contrasta la importancia real de cada feature contra su distribución nula.

    Retorna:
        selected     : List[str]
        table        : pd.DataFrame con todos los criterios (empírico, paramétrico, z)
        null_matrix  : FloatArray (n_runs, n_features)  [F4] para re-análisis
    """
    method: str = str(cfg.significance_method)
    dist: str = str(cfg.parametric_dist)
    require_floor: bool = bool(cfg.require_empirical_floor)
    z_thr: float = float(cfg.zscore_threshold)
    abort_on_fail: bool = bool(cfg.preflight_abort)

    n_runs: int = int(cfg.shuffle_n_runs)
    alpha: float = float(cfg.shuffle_fdr_alpha)

    print_header(
        f"STEP 3 — Block-Permutation Importance  "
        f"(n_runs = {n_runs}, alpha = {alpha}, method = {method})"
    )

    feat_list: List[str] = list(features)
    n_feat: int = len(feat_list)

    viable: bool = preflight_permutation_power(n_feat, n_runs, alpha, method)
    if not viable and abort_on_fail:
        raise RuntimeError(
            "Configuración incapaz de rechazar. Subí n_runs o usá 'parametric_bh'."
        )

    rng: np.random.Generator = np.random.default_rng(int(cfg.random_state))
    cat_set: Set[str] = set(cat_cols)
    cats_pass: List[str] = [c for c in feat_list if c in cat_set]
    n_rows: int = int(X_train.shape[0])

    block_size: int = (
        int(cfg.shuffle_block_size)
        if int(cfg.shuffle_block_size) > 0
        else _estimate_block_size(y_train)
    )
    shap_idx: IntArray = _temporal_stratified_index(
        n_rows, int(cfg.shuffle_sample_size), int(cfg.shuffle_n_strata)
    )
    mean_run: float = float(n_rows) / float(max(1, int(np.sum(y_train[1:] != y_train[:-1])) + 1))
    print(f"\n  Longitud de bloque  : {block_size} filas "
          f"(racha media del target = {mean_run:.1f} filas)")
    if block_size <= 5:
        print("                        -> el target no tiene persistencia temporal;")
        print("                           la permutación por bloques ~ i.i.d. Correcto.")
    print(f"  Muestra importancia : {len(shap_idx)} filas estratificadas")

    params: Dict[str, object] = _lgb_base_params(n_classes, int(cfg.random_state), strong=False)

    X_cur: pd.DataFrame = X_train[feat_list]
    X_imp: pd.DataFrame = X_cur.iloc[shap_idx]

    print(f"\n  Modelo real...")
    real_ds: lgb.Dataset = lgb.Dataset(
        X_cur, label=y_train, categorical_feature=cats_pass, free_raw_data=False
    )
    real_model: lgb.Booster = lgb.train(
        params, real_ds, num_boost_round=int(cfg.shuffle_boost_rounds)
    )
    real_imp: FloatArray = _mean_abs_contrib(real_model, X_imp, n_classes)

    print(f"  Distribución nula ({n_runs} corridas)...")
    null_matrix: FloatArray = np.zeros((n_runs, n_feat), dtype=np.float64)
    t0: float = time.time()

    run_i: int
    for run_i in range(n_runs):
        y_null: IntArray = _safe_block_permute(y_train, block_size, rng, n_classes)
        null_ds: lgb.Dataset = lgb.Dataset(
            X_cur, label=y_null, categorical_feature=cats_pass, free_raw_data=False
        )
        null_model: lgb.Booster = lgb.train(
            params, null_ds, num_boost_round=int(cfg.shuffle_boost_rounds)
        )
        null_matrix[run_i, :] = _mean_abs_contrib(null_model, X_imp, n_classes)

        if (run_i + 1) % 10 == 0:
            el: float = time.time() - t0
            eta: float = el / (run_i + 1) * (n_runs - run_i - 1)
            print(f"    {run_i + 1}/{n_runs}  ({el / 60.0:.1f} min, ETA {eta / 60.0:.1f} min)")

    # ── Criterios ───────────────────────────────────────────────────────────
    n_exceed: IntArray = (null_matrix >= real_imp[None, :]).sum(axis=0).astype(np.int64)
    p_emp: FloatArray = (1.0 + n_exceed.astype(np.float64)) / (float(n_runs) + 1.0)
    at_floor: BoolArray = n_exceed == 0

    null_mean: FloatArray = null_matrix.mean(axis=0)
    null_std: FloatArray = null_matrix.std(axis=0, ddof=1) + 1e-12
    z_score: FloatArray = (real_imp - null_mean) / null_std

    p_param: FloatArray = np.array(
        [_parametric_pvalue(null_matrix[:, j], float(real_imp[j]), dist)
         for j in range(n_feat)],
        dtype=np.float64,
    )

    rej_emp: BoolArray
    p_bh_emp: FloatArray
    rej_emp, p_bh_emp, _, _ = multipletests(p_emp, alpha=alpha, method="fdr_bh")

    rej_par: BoolArray
    p_bh_par: FloatArray
    rej_par, p_bh_par, _, _ = multipletests(p_param, alpha=alpha, method="fdr_bh")

    if method == "empirical_bh":
        significant: BoolArray = rej_emp
    elif method == "zscore":
        significant = z_score >= z_thr
    else:  # parametric_bh
        significant = rej_par
        if require_floor:
            # [F2] guarda: el paramétrico no puede rescatar features que no
            # superaron a todas las nulas observadas
            significant = significant & at_floor

    table: pd.DataFrame = pd.DataFrame(
        {
            "feature": feat_list,
            "imp_real": real_imp,
            "imp_null_mean": null_mean,
            "imp_null_max": null_matrix.max(axis=0),
            "z_score": z_score,
            "at_floor": at_floor,
            "p_emp": p_emp,
            "p_bh_emp": p_bh_emp,
            "p_param": p_param,
            "p_bh_param": p_bh_par,
            "significant": significant,
        }
    ).sort_values("z_score", ascending=False).reset_index(drop=True)

    n_sig: int = int(significant.sum())
    print(f"\n  Features en el piso empírico (real > todas las nulas): "
          f"{int(at_floor.sum())} / {n_feat}  ({at_floor.mean():.1%})")
    print(f"  Significativas [{method}]: {n_sig} / {n_feat}   (eliminadas: {n_feat - n_sig})")
    print(f"  Comparación de criterios:")
    print(f"    empirical_bh          : {int(rej_emp.sum()):>4}")
    print(f"    parametric_bh ({dist:<9}): {int(rej_par.sum()):>4}")
    print(f"    parametric_bh + floor : {int((rej_par & at_floor).sum()):>4}")
    print(f"    z_score >= {z_thr:<4}       : {int((z_score >= z_thr).sum()):>4}")

    if n_sig == 0:
        print("\n  >> 0 features. Revisá el PREFLIGHT de arriba antes de tocar los datos.")

    selected: List[str] = table.loc[table["significant"], "feature"].tolist()
    return selected, table, null_matrix


def reanalyze_significance(
    null_matrix: FloatArray,
    real_imp: FloatArray,
    feat_list: Sequence[str],
    alpha: float = 0.05,
    dist: str = "lognormal",
    require_floor: bool = True,
) -> pd.DataFrame:
    """
    Recalcula la significancia con otros criterios reutilizando la matriz nula.
    Cambiar alpha o el criterio ya no cuesta 20 minutos.
    """
    n_feat: int = int(null_matrix.shape[1])

    n_exceed: IntArray = (null_matrix >= real_imp[None, :]).sum(axis=0).astype(np.int64)
    at_floor: BoolArray = n_exceed == 0
    p_param: FloatArray = np.array(
        [_parametric_pvalue(null_matrix[:, j], float(real_imp[j]), dist)
         for j in range(n_feat)],
        dtype=np.float64,
    )
    rej: BoolArray
    p_bh: FloatArray
    rej, p_bh, _, _ = multipletests(p_param, alpha=alpha, method="fdr_bh")
    sig: BoolArray = (rej & at_floor) if require_floor else rej

    return pd.DataFrame(
        {
            "feature": list(feat_list),
            "imp_real": real_imp,
            "at_floor": at_floor,
            "p_param": p_param,
            "p_bh_param": p_bh,
            "significant": sig,
        }
    ).sort_values("imp_real", ascending=False).reset_index(drop=True)


# ================================================================================
# STEP 4 — Feature Stability Filter
# ================================================================================
def feature_stability_filter(
    X_train: pd.DataFrame,
    y_train: IntArray,
    features: Sequence[str],
    cat_cols: Sequence[str],
    n_classes: int,
    cfg: PipelineConfig,
) -> Tuple[List[str], pd.DataFrame]:
    """
    [C4] Estabilidad = importancia alta Y CONSISTENTE a lo largo del tiempo.

    Usar `mean(pct_vector)` (el percentil PROMEDIO de importancia) no penaliza
    la dispersión en absoluto: una feature con percentiles
    [0.95, 0.10, 0.95, 0.10, 0.95] promedia 0.61 y sobrevive, mientras una
    consistente en 0.35 se elimina. Ese criterio selecciona ACTIVAMENTE a favor
    de la dependencia de régimen — lo contrario de su propósito.

    Métrica usada:
        score = mean(pct) - k * std(pct)

    donde `pct` es el percentil de la feature dentro de cada ventana temporal.
    """
    print_header(
        f"STEP 4 — Feature Stability Filter  "
        f"(ventanas = {cfg.stability_n_windows}, score min = {cfg.stability_min_score}, "
        f"k = {cfg.stability_penalty_k})"
    )

    feat_list: List[str] = list(features)
    cat_set: Set[str] = set(cat_cols)
    cats_valid: List[str] = [c for c in feat_list if c in cat_set]
    n_rows: int = int(X_train.shape[0])
    n_feat: int = len(feat_list)

    params: Dict[str, object] = _lgb_base_params(n_classes, cfg.random_state, strong=False)
    params["min_data_in_leaf"] = 20

    windows: List[Tuple[int, int]] = _disjoint_block_indices(n_rows, cfg.stability_n_windows)
    importances: List[FloatArray] = []

    w_idx: int
    lo: int
    hi: int
    for w_idx, (lo, hi) in enumerate(windows):
        X_w: pd.DataFrame = X_train[feat_list].iloc[lo:hi]
        y_w: IntArray = y_train[lo:hi]

        if int(X_w.shape[0]) < cfg.stability_min_rows_per_window:
            print(f"  Ventana {w_idx + 1}: {X_w.shape[0]} filas (< "
                  f"{cfg.stability_min_rows_per_window}), omitida.")
            continue
        if int(np.unique(y_w).size) < n_classes:
            print(f"  Ventana {w_idx + 1}: faltan clases, omitida.")
            continue

        ds: lgb.Dataset = lgb.Dataset(
            X_w, label=y_w, categorical_feature=cats_valid, free_raw_data=False
        )
        model: lgb.Booster = lgb.train(
            params, ds, num_boost_round=cfg.stability_boost_rounds
        )
        # [C5] importancia medida sobre TODA la ventana, no sobre su inicio
        imp_w: FloatArray = _mean_abs_contrib(model, X_w, n_classes)
        importances.append(imp_w)
        print(f"  Ventana {w_idx + 1}/{cfg.stability_n_windows} procesada "
              f"({X_w.shape[0]} filas).")

    if len(importances) < 2:
        print("  ADVERTENCIA: menos de 2 ventanas válidas. Filtro omitido.")
        return feat_list, pd.DataFrame()

    n_valid: int = len(importances)
    pct_matrix: FloatArray = np.zeros((n_valid, n_feat), dtype=np.float64)

    w: int
    for w in range(n_valid):
        pct_matrix[w, :] = rankdata(importances[w]) / float(n_feat)

    mean_pct: FloatArray = pct_matrix.mean(axis=0)
    std_pct: FloatArray = pct_matrix.std(axis=0)
    score: FloatArray = mean_pct - cfg.stability_penalty_k * std_pct

    # Diagnóstico agregado: IC global entre ventanas consecutivas
    global_ic: List[float] = []
    for w in range(n_valid - 1):
        rho: float
        rho, _ = spearmanr(importances[w], importances[w + 1])
        global_ic.append(0.0 if np.isnan(rho) else float(rho))
    print(f"\n  IC global entre ventanas consecutivas: "
          f"{[round(v, 3) for v in global_ic]}  (media = {np.mean(global_ic):.3f})")
    if float(np.mean(global_ic)) < 0.3:
        print("  ADVERTENCIA: el ranking de importancias es inestable a nivel global.")
        print("               El conjunto seleccionado probablemente no generalice")
        print("               fuera del régimen de entrenamiento.")

    table: pd.DataFrame = pd.DataFrame(
        {
            "feature": feat_list,
            "mean_pct": mean_pct,
            "std_pct": std_pct,
            "stability_score": score,
            "stable": score >= cfg.stability_min_score,
        }
    ).sort_values("stability_score", ascending=False).reset_index(drop=True)

    stable_features: List[str] = table.loc[table["stable"], "feature"].tolist()

    print(f"\n  Features estables   : {len(stable_features)}")
    print(f"  Features inestables : {n_feat - len(stable_features)}")

    return stable_features, table


# ================================================================================
# STEP 5 — Validación: baseline vs seleccionadas
# ================================================================================
def validate_selection(
    X_train: pd.DataFrame,
    y_train: IntArray,
    X_val: pd.DataFrame,
    y_val: IntArray,
    all_features: Sequence[str],
    selected_features: Sequence[str],
    cat_cols: Sequence[str],
    n_classes: int,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """
    [C10] Sin este paso el pipeline puede quemar horas sin verificar NUNCA que la
    selección mejore algo. Si el modelo con las features seleccionadas no le gana
    al baseline con todas + regularización, la selección no está aportando.

    [F-CAT] Éste es el único paso que construye un par (train, valid) de
    LightGBM, y por eso es donde explota el desalineamiento de categóricas. El
    slice de val se realinea contra el slice de train justo antes de construir
    los Dataset, y se verifica con un assert explícito.
    """
    print_header("STEP 5 — Validación en VAL: baseline vs seleccionadas")

    params: Dict[str, object] = _lgb_base_params(n_classes, cfg.random_state, strong=False)
    cat_set: Set[str] = set(cat_cols)
    rows: List[Dict[str, object]] = []

    name: str
    feats: Sequence[str]
    for name, feats in [("baseline_all", all_features), ("selected", selected_features)]:
        feat_list: List[str] = list(dict.fromkeys(feats))  # sin duplicados, orden estable
        feat_list = [f for f in feat_list if f in X_train.columns]
        if len(feat_list) == 0:
            print(f"  [{name}] 0 features utilizables; se omite.")
            continue

        cats_valid: List[str] = [c for c in feat_list if c in cat_set]

        X_tr_use: pd.DataFrame = X_train[feat_list]
        # [F-CAT] realineación defensiva del slice de validación
        X_vl_use: pd.DataFrame
        X_vl_use, _ = align_frame_to_reference(
            X_tr_use, X_val.reindex(columns=feat_list), f"val[{name}]", verbose=False
        )
        assert_same_categorical_schema(X_tr_use, X_vl_use, "train", f"val[{name}]")

        train_ds: lgb.Dataset = lgb.Dataset(
            X_tr_use, label=y_train,
            categorical_feature=cats_valid, free_raw_data=False,
        )
        val_ds: lgb.Dataset = lgb.Dataset(
            X_vl_use, label=y_val, reference=train_ds,
            categorical_feature=cats_valid, free_raw_data=False,
        )

        model: lgb.Booster = lgb.train(
            params,
            train_ds,
            num_boost_round=cfg.validation_boost_rounds,
            valid_sets=[val_ds],
            callbacks=[lgb.early_stopping(cfg.validation_early_stopping, verbose=False)],
        )

        # [F-ITER] best_iteration <= 0 -> usar todos los árboles
        best_iter: Optional[int] = (
            int(model.best_iteration) if int(model.best_iteration or 0) > 0 else None
        )
        proba: FloatArray = np.asarray(
            model.predict(X_vl_use, num_iteration=best_iter), dtype=np.float64
        )
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        pred: IntArray = proba.argmax(axis=1).astype(np.int64)

        rows.append(
            {
                "model": name,
                "n_features": len(feat_list),
                "best_iter": int(best_iter or cfg.validation_boost_rounds),
                "val_logloss": round(
                    float(log_loss(y_val, proba, labels=list(range(n_classes)))), 5
                ),
                "val_accuracy": round(float(accuracy_score(y_val, pred)), 5),
                "val_macro_f1": round(float(f1_score(y_val, pred, average="macro")), 5),
            }
        )

    table: pd.DataFrame = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))

    if len(table) == 2:
        delta_ll: float = float(
            table.loc[table["model"] == "baseline_all", "val_logloss"].iloc[0]
            - table.loc[table["model"] == "selected", "val_logloss"].iloc[0]
        )
        print(f"\n  Delta logloss (baseline - selected) = {delta_ll:+.5f}")
        if delta_ll <= 0.0:
            print("  >> La selección NO mejora sobre el baseline.")
            print("     No poner en producción sin revisar los umbrales.")
        else:
            print("  >> La selección mejora el baseline en val.")

    return table


# ================================================================================
# MASTER PIPELINE
# ================================================================================
def run_feature_selection_pipeline(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: Optional[pd.Series] = None,
    cfg: Optional[PipelineConfig] = None,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, PipelineReport]:
    """
    Pipeline completo de selección de features para clasificación en series
    temporales.

    Retorna:
        final_features : List[str]        features seleccionadas
        X_train_sel    : pd.DataFrame     train filtrado y alineado
        X_val_sel      : pd.DataFrame     val   filtrado y alineado   [C13]
        X_test_sel     : pd.DataFrame     test  filtrado y alineado   [C13]
        report         : PipelineReport   trazabilidad de cada paso
    """
    config: PipelineConfig = cfg if cfg is not None else PipelineConfig()
    report: PipelineReport = PipelineReport()
    t_start: float = time.time()

    np.random.seed(config.random_state)

    y_tr_enc: IntArray
    class_map: Dict[object, int]
    y_tr_enc, class_map = encode_target(y_train)
    n_classes: int = len(class_map)

    report.n_start = int(X_train.shape[1])

    print(SEP)
    print("  TEMPORAL FEATURE SELECTION PIPELINE v2.1")
    print(SEP)
    print(f"  Train: {X_train.shape}   Val: {X_val.shape}   Test: {X_test.shape}")
    print(f"  Clases: {n_classes}  ->  {class_map}")
    print(f"  Ratio filas/features: {X_train.shape[0] / max(1, X_train.shape[1]):.1f}:1")
    print(f"  Semilla: {config.random_state}")

    # ── STEP 0 ──────────────────────────────────────────────────────────────
    X_tr: pd.DataFrame
    X_vl: pd.DataFrame
    X_te: pd.DataFrame
    dropped: List[str]
    X_tr, X_vl, X_te, dropped = drop_sparse_features(
        X_train, X_val, X_test, config.sparsity_threshold
    )
    report.dropped_sparse = dropped
    report.n_after_sparsity = int(X_tr.shape[1])

    # ── Categóricas + alineación de esquema [F-CAT] ─────────────────────────
    cat_cols: List[str]
    schema_changes: pd.DataFrame
    X_tr, X_vl, X_te, cat_cols, schema_changes = format_categorical_features(
        X_tr, X_vl, X_te, config
    )
    report.categorical_cols = cat_cols
    report.schema_alignment = schema_changes
    if len(cat_cols) > 0:
        report.categorical_audit = audit_categorical_coverage(X_tr, X_vl, X_te, cat_cols)

    baseline_features: List[str] = X_tr.columns.tolist()

    # ── STEP 1 ──────────────────────────────────────────────────────────────
    mi_features: List[str]
    mi_scores: Dict[str, float]
    mi_features, mi_scores = mutual_information_prefilter(X_tr, y_tr_enc, cat_cols, config)
    report.mi_scores = mi_scores
    report.n_after_mi = len(mi_features)

    # ── STEP 2 ──────────────────────────────────────────────────────────────
    corr_features: List[str]
    clusters: Dict[int, List[str]]
    corr_features, clusters = temporal_correlation_filter(
        X_tr, y_tr_enc, mi_features, cat_cols, mi_scores, config
    )
    report.corr_clusters = clusters
    report.n_after_corr = len(corr_features)

    # ── STEP 3 ──────────────────────────────────────────────────────────────
    shuffle_features: List[str]
    shuffle_table: pd.DataFrame
    shuffle_null: FloatArray
    shuffle_features, shuffle_table, shuffle_null = block_permutation_importance_bh(
        X_tr, y_tr_enc, corr_features, cat_cols, n_classes, config
    )
    report.shuffle_table = shuffle_table
    report.shuffle_null_matrix = shuffle_null
    report.shuffle_real_imp = shuffle_table.set_index("feature").loc[
        list(corr_features), "imp_real"
    ].to_numpy()
    report.n_after_shuffle = len(shuffle_features)

    # ── STEP 4 ──────────────────────────────────────────────────────────────
    final_features: List[str] = shuffle_features
    if config.run_stability_filter and len(shuffle_features) > 1:
        stability_table: pd.DataFrame
        final_features, stability_table = feature_stability_filter(
            X_tr, y_tr_enc, shuffle_features, cat_cols, n_classes, config
        )
        report.stability_table = stability_table
    report.n_after_stability = len(final_features)
    report.final_features = final_features

    # ── STEP 5 ──────────────────────────────────────────────────────────────
    if config.run_validation and y_val is not None and len(final_features) > 0:
        y_vl_enc: IntArray = encode_target_with_map(y_val, class_map, "y_val")  # [F-YVAL]
        report.validation_table = validate_selection(
            X_tr, y_tr_enc, X_vl, y_vl_enc,
            baseline_features, final_features, cat_cols, n_classes, config,
        )
    elif config.run_validation and len(final_features) == 0:
        print("\n  [AVISO] 0 features seleccionadas: se omite la validación del paso 5.")
    elif config.run_validation:
        print("\n  [AVISO] y_val no provisto: se omite la validación del paso 5.")
        print("          No pongas esto en producción sin comparar contra el baseline.")

    # ── Salidas alineadas ───────────────────────────────────────────────────
    X_train_sel: pd.DataFrame = X_tr[final_features]
    X_val_sel: pd.DataFrame = X_vl[final_features]
    X_test_sel: pd.DataFrame = X_te[final_features]

    report.elapsed_seconds = time.time() - t_start

    # ── Resumen ─────────────────────────────────────────────────────────────
    print_header("RESUMEN DEL PIPELINE")
    print(f"  [Start ] : {report.n_start}")
    print(f"  [Step 0] : {report.n_after_sparsity}  (-{len(report.dropped_sparse)} esparsas)")
    print(f"  [Step 1] : {report.n_after_mi}  (-MI~0)")
    print(f"  [Step 2] : {report.n_after_corr}  (-redundantes)")
    print(f"  [Step 3] : {report.n_after_shuffle}  (-ruido, BH-corrected)")
    print(f"  [Step 4] : {report.n_after_stability}  (-inestables)")
    print(f"  {SUB}")
    print(f"  FEATURES FINALES : {len(final_features)}")
    print(f"  Tiempo total     : {report.elapsed_seconds / 60.0:.1f} min")
    print(SEP)

    return final_features, X_train_sel, X_val_sel, X_test_sel, report


# ================================================================================
# [C14] WALK-FORWARD: ¿es estable el propio pipeline de selección?
# ================================================================================
def walk_forward_selection_stability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: PipelineConfig,
    n_folds: int = 3,
) -> pd.DataFrame:
    """
    Re-ejecuta la selección sobre ventanas expandidas y mide el solapamiento
    (Jaccard) entre los conjuntos elegidos.

    Es la única prueba real de que la selección no está sobreajustada a un
    régimen. Si cada fold devuelve un conjunto muy distinto, el pipeline no
    está descubriendo señal estable — está muestreando ruido.

    COSTO: n_folds veces el pipeline completo. Correr con shuffle_n_runs bajo.
    """
    print_header(f"WALK-FORWARD SELECTION STABILITY  ({n_folds} folds)")

    n_rows: int = int(X_train.shape[0])
    bounds: IntArray = np.linspace(n_rows // 2, n_rows, n_folds + 1).astype(np.int64)
    selections: List[Set[str]] = []

    fold: int
    for fold in range(n_folds):
        end: int = int(bounds[fold + 1])
        print(f"\n{SUB}\n  FOLD {fold + 1}/{n_folds} — train[0:{end}]\n{SUB}")

        X_fold: pd.DataFrame = X_train.iloc[:end]
        y_fold: pd.Series = y_train.iloc[:end]

        feats: List[str]
        feats, _, _, _, _ = run_feature_selection_pipeline(
            X_train=X_fold,
            X_val=X_fold.iloc[:10],
            X_test=X_fold.iloc[:10],
            y_train=y_fold,
            y_val=None,
            cfg=cfg,
        )
        selections.append(set(feats))

    rows: List[Dict[str, object]] = []
    i: int
    j: int
    for i in range(len(selections)):
        for j in range(i + 1, len(selections)):
            inter: int = len(selections[i] & selections[j])
            union: int = len(selections[i] | selections[j])
            rows.append(
                {
                    "fold_a": i + 1,
                    "fold_b": j + 1,
                    "n_a": len(selections[i]),
                    "n_b": len(selections[j]),
                    "overlap": inter,
                    "jaccard": round(inter / max(1, union), 4),
                }
            )

    table: pd.DataFrame = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))

    mean_j: float = float(table["jaccard"].mean())
    print(f"\n  Jaccard medio: {mean_j:.3f}")
    if mean_j < 0.5:
        print("  >> El pipeline NO es estable entre folds. Las features elegidas")
        print("     dependen del período. No usar en producción.")

    core: Set[str] = set.intersection(*selections) if selections else set()
    print(f"  Núcleo presente en TODOS los folds: {len(core)} features")

    return table


# ================================================================================
#  DIAGNÓSTICO RÁPIDO — correr ANTES del pipeline si aparece el ValueError
# ================================================================================
def diagnose_schema(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reporta las diferencias de esquema entre los tres frames SIN modificarlos.
    Útil para entender por qué se rompe el STEP 5.
    """
    print_header("DIAGNÓSTICO DE ESQUEMA train / val / test")

    frames: Dict[str, pd.DataFrame] = {"val": X_val, "test": X_test}
    cats_tr: Set[str] = set(categorical_columns(X_train))
    rows: List[Dict[str, object]] = []

    print(f"  Columnas    train/val/test : {X_train.shape[1]} / "
          f"{X_val.shape[1]} / {X_test.shape[1]}")
    print(f"  'category'  train/val/test : {len(cats_tr)} / "
          f"{len(categorical_columns(X_val))} / {len(categorical_columns(X_test))}")

    name: str
    frame: pd.DataFrame
    for name, frame in frames.items():
        cats_other: Set[str] = set(categorical_columns(frame))
        missing = [c for c in X_train.columns if c not in frame.columns]
        extra = [c for c in frame.columns if c not in X_train.columns]
        cat_only_other = sorted(cats_other - cats_tr)
        cat_only_train = sorted(cats_tr - cats_other)
        dtype_diff = [
            c for c in X_train.columns
            if c in frame.columns and X_train[c].dtype != frame[c].dtype
        ]
        rows.append(
            {
                "frame": name,
                "cols_faltantes": len(missing),
                "cols_extra": len(extra),
                "category_solo_en_este": len(cat_only_other),
                "category_solo_en_train": len(cat_only_train),
                "dtypes_distintos": len(dtype_diff),
                "ejemplos": ", ".join((cat_only_other + cat_only_train + dtype_diff)[:5]),
            }
        )

    table: pd.DataFrame = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))
    print("\n  Cualquier valor > 0 en 'category_solo_en_*' produce el ValueError")
    print("  'train and valid dataset categorical_feature do not match'.")
    print("  El pipeline lo corrige automáticamente al formatear categóricas.")
    return table

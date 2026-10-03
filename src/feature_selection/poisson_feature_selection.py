"""Seleccion de features temporal para targets de CONTEO (Poisson).

Pipeline paralelo al de clasificacion (`feature_selection.py`), con LightGBM
`objective="poisson"`. Los helpers agnosticos del objective viven en
`common.py`; aca solo esta lo especifico del caso de conteo.

QUE CAMBIA RESPECTO DEL PIPELINE DE CLASIFICACION
-------------------------------------------------
[R1] `objective="poisson"` en TODOS los modelos (screening, nulos, ventanas,
     validacion). La importancia se mide en el espacio del log-rate.
[R2] `mutual_info_regression` en vez de `mutual_info_classif`.
[R3] `pred_contrib` de salida unica -> (n, n_features + 1). Sin bloque por clase.
[R4] Bloque de permutacion estimado por AUTOCORRELACION del target, no por
     run-length de labels (que no existe para un conteo).
[R5] El nulo NO necesita preservar clases; si que el target permutado no sea
     degenerado (varianza > 0) para que LightGBM no colapse al intercepto.
[R6] Representante de cluster por MI o por |Spearman| con el target: para un
     conteo Spearman SI tiene interpretacion (el target es ordinal).
[R7] Validacion con RPS (`src.metrics.distribution_metrics`) + desvianza de
     Poisson como metrica interna.

FLUJO
-----
  [0]   Sparsity filter + formato de categoricas + auditoria de cobertura
  [0.5] Auditoria de leakage temporal (screening de features sospechosas)
  [1]   Mutual Information pre-filter
  [2]   Temporal correlation filter (clustering, bloques disjuntos)
  [3]   Block-permutation importance + BH
  [4]   Feature stability filter (dispersion de percentiles entre ventanas)
  [5]   Validacion en VAL: baseline_all vs selected vs random_k
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import mutual_info_regression

from src.config.config import ModelConfig
from src.feature_selection.common import (
    EPS,
    BoolArray,
    FloatArray,
    IntArray,
    build_null_matrix,
    disjoint_block_indices,
    estimate_block_size_acf,
    mean_abs_contrib,
    median_impute,
    parametric_pvalue,  # noqa: F401  (re-export por compatibilidad)
    preflight_permutation_power,
    reanalyze_significance,  # noqa: F401  (re-export por compatibilidad)
    resolve_n_jobs,
    safe_block_permute_variance,
    significance_from_null,
    spearman_abs_matrix,
    temporal_stratified_index,
)
from src.utils.utils import (
    SEP,
    SUB,
    align_frame_to_reference,
    assert_same_categorical_schema,
    categorical_columns,
    print_header,
)

DEFAULT_RANDOM_STATE: int = 67


# ===========================================================================
# CONFIGURACION
# ===========================================================================
@dataclass(frozen=True)
class PoissonSelectionConfig:
    """Hiperparametros del pipeline de seleccion para conteos.

    Se construye desde el YAML del target con `from_model_config`, de modo que
    ningun valor quede hardcodeado en el notebook.
    """

    # --- Reproducibilidad ---
    random_state: int = DEFAULT_RANDOM_STATE
    assert_temporal_order: bool = True

    # --- Paralelismo ---
    n_jobs: int = 0                        # 0 -> auto (cores // threads_per_model)
    threads_per_model: int = 2             # pocos threads, muchos modelos

    # --- Modelo base (Poisson) ---
    learning_rate: float = 0.05
    num_leaves: int = 15
    max_depth: int = 3
    min_data_in_leaf: int = 45             # nombre canonico unico
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8          # requiere bagging_freq > 0
    lambda_l2: float = 1.0
    poisson_max_delta_step: float = 0.7

    # --- STEP 0: Sparsity ---
    sparsity_threshold: float = 0.98
    nan_threshold: float = 0.95

    # --- Categoricas ---
    explicit_categorical: Optional[List[str]] = None
    auto_detect_string_categoricals: bool = True
    strict_schema_alignment: bool = True

    # --- STEP 0.5: Auditoria de leakage ---
    run_leakage_audit: bool = True
    leakage_top_n: int = 40
    leakage_dev_reduction_flag: float = 0.15
    leakage_boost_rounds: int = 60

    # --- STEP 1: Mutual Information ---
    mi_threshold: float = 1e-4
    mi_n_neighbors: int = 3
    mi_n_seeds: int = 3                    # MI es estocastica: se promedia

    # --- STEP 2: Correlacion temporal ---
    corr_threshold: float = 0.80
    corr_n_blocks: int = 5
    cluster_representative: str = "spearman"   # "spearman" | "mi"

    # --- STEP 3: Block-Permutation Importance ---
    shuffle_n_runs: int = 300
    shuffle_alpha: float = 0.01
    shuffle_block_size: int = 0            # 0 -> auto por ACF
    shuffle_sample_size: int = 2000
    shuffle_n_strata: int = 10
    shuffle_boost_rounds: int = 150
    shuffle_n_real_seeds: int = 5          # el real se promedia
    significance_method: str = "parametric_bh"
    parametric_dist: str = "lognormal"
    require_empirical_floor: bool = True
    zscore_threshold: float = 4.0
    preflight_abort: bool = False
    robustness_pass: bool = True           # chequeo, NO test

    # --- STEP 4: Estabilidad ---
    run_stability_filter: bool = True
    stability_n_windows: int = 5
    stability_min_score: float = 0.35
    stability_penalty_k: float = 1.25
    stability_boost_rounds: int = 120
    stability_min_rows_per_window: int = 300

    # --- STEP 5: Validacion ---
    run_validation: bool = True
    validation_boost_rounds: int = 1500
    validation_early_stopping: int = 75
    validation_random_seeds: int = 3

    @classmethod
    def from_model_config(
        cls, config: ModelConfig, **overrides: Any
    ) -> "PoissonSelectionConfig":
        """Construye la config desde el bloque `feature_selection` del YAML.

        Lee `config.raw["feature_selection"]` y toma `cv.random_state` como
        semilla por defecto. Las claves desconocidas se reportan como error en
        vez de ignorarse en silencio: un typo en el YAML no debe degradar
        calladamente a los defaults.

        Parameters
        ----------
        config    : ModelConfig ya cargado (`load_config`).
        overrides : valores que pisan al YAML, para iterar desde el notebook.
        """
        block: Dict[str, Any] = dict(config.raw.get("feature_selection", {}))
        block.setdefault("random_state", config.random_state)
        block.update(overrides)

        valid: Set[str] = {f.name for f in cls.__dataclass_fields__.values()}
        unknown: List[str] = sorted(set(block) - valid)
        if unknown:
            raise ValueError(
                f"Claves desconocidas en 'feature_selection' del YAML: {unknown}. "
                f"Validas: {sorted(valid)}"
            )

        return cls(**block)

    def with_overrides(self, **overrides: Any) -> "PoissonSelectionConfig":
        """Copia con campos reemplazados (la dataclass es frozen)."""
        return replace(self, **overrides)


@dataclass
class SelectionReport:
    """Trazabilidad completa de cada paso, para auditoria."""

    target_name: str = ""

    n_start: int = 0
    n_after_sparsity: int = 0
    n_after_mi: int = 0
    n_after_corr: int = 0
    n_after_shuffle: int = 0
    n_after_stability: int = 0

    dropped_sparse: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    categorical_audit: Optional[pd.DataFrame] = None
    schema_alignment: Optional[pd.DataFrame] = None
    leakage_audit: Optional[pd.DataFrame] = None
    mi_scores: Dict[str, float] = field(default_factory=dict)
    corr_clusters: Dict[int, List[str]] = field(default_factory=dict)
    shuffle_table: Optional[pd.DataFrame] = None
    shuffle_null_matrix: Optional[FloatArray] = None
    shuffle_real_imp: Optional[FloatArray] = None
    shuffle_feature_order: List[str] = field(default_factory=list)
    stability_table: Optional[pd.DataFrame] = None
    validation_table: Optional[pd.DataFrame] = None

    block_size_used: int = 0
    final_features: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ===========================================================================
# HELPERS especificos de Poisson
# ===========================================================================
def lgb_poisson_params(
    cfg: PoissonSelectionConfig, weak: bool = False
) -> Dict[str, object]:
    """Parametros base de LightGBM para conteos.

    `bagging_freq=1` es OBLIGATORIO: sin el, `bagging_fraction` se ignora
    SILENCIOSAMENTE y no hay subsampling en ningun modelo del pipeline.

    Se usa `min_data_in_leaf` unicamente. `min_child_samples` es su alias:
    declarar los dos genera un conflicto que LightGBM resuelve sin avisar.
    """
    params: Dict[str, object] = {
        "objective": "poisson",
        "metric": "poisson",
        "poisson_max_delta_step": float(cfg.poisson_max_delta_step),
        "learning_rate": float(cfg.learning_rate),
        "num_leaves": int(cfg.num_leaves),
        "max_depth": int(cfg.max_depth),
        "min_data_in_leaf": int(cfg.min_data_in_leaf),
        "feature_fraction": float(cfg.feature_fraction),
        "bagging_fraction": float(cfg.bagging_fraction),
        "bagging_freq": 1,
        "lambda_l2": float(cfg.lambda_l2),
        "verbose": -1,
        "seed": int(cfg.random_state),
        "bagging_seed": int(cfg.random_state),
        "feature_fraction_seed": int(cfg.random_state),
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": int(cfg.threads_per_model),
    }
    if weak:
        params["num_leaves"] = 15
        params["max_depth"] = 4
        params["min_data_in_leaf"] = int(cfg.min_data_in_leaf) * 2
    return params


def poisson_deviance(y_true: FloatArray, mu: FloatArray) -> float:
    """Desvianza media de Poisson. Metrica INTERNA de diagnostico.

        D = 2 * mean( y*log(y/mu) - (y - mu) ),   con y*log(y/mu) := 0 si y = 0
    """
    mu_safe: FloatArray = np.clip(mu.astype(np.float64), 1e-9, None)
    y_safe: FloatArray = y_true.astype(np.float64)
    term: FloatArray = np.where(y_safe > 0.0, y_safe * np.log(y_safe / mu_safe), 0.0)
    return float(2.0 * np.mean(term - (y_safe - mu_safe)))


def _rps_poisson(
    y_true: FloatArray, lambdas: FloatArray, label: str, max_k: int
) -> Optional[float]:
    """RPS via `src.metrics.distribution_metrics.calculate_rps`.

    Se importa adentro para que el pipeline siga siendo utilizable si el modulo
    de metricas no esta disponible; en ese caso se cae a la desvianza.
    """
    try:
        from src.metrics.distribution_metrics import calculate_rps
    except ImportError:
        return None

    try:
        result: Any = calculate_rps(
            y_true,
            params={"lambda": lambdas},
            family="poisson",
            max_k=max_k,
            label=label,
        )
        if isinstance(result, dict) and "rps_mean" in result:
            return float(cast(float, result["rps_mean"]))
        return float(result)
    except Exception as exc:  # noqa: BLE001
        print(f"    [AVISO] calculate_rps fallo ({type(exc).__name__}); se usa desvianza.")
        return None


# ===========================================================================
# STEP 0 — Sparsity Filter
# ===========================================================================
def drop_sparse_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: PoissonSelectionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Elimina columnas cuasi-constantes o casi enteramente NaN.

    Se evalua SOLO sobre X_train: val y test nunca participan (sin leakage).
    """
    print_header(
        f"STEP 0 — Sparsity Filter  (moda > {cfg.sparsity_threshold:.0%}, "
        f"NaN > {cfg.nan_threshold:.0%})"
    )

    cols_to_drop: List[str] = []
    col: str
    for col in X_train.columns:
        nan_rate: float = float(X_train[col].isna().mean())
        if nan_rate > cfg.nan_threshold:
            cols_to_drop.append(col)
            continue
        non_nan: pd.Series = X_train[col].dropna()
        if len(non_nan) == 0:
            cols_to_drop.append(col)
            continue
        top_share: float = float(non_nan.value_counts(normalize=True).iloc[0])
        if top_share > cfg.sparsity_threshold:
            cols_to_drop.append(col)

    n_in: int = int(X_train.shape[1])
    print(f"  Features entrada    : {n_in}")
    print(f"  Cuasi-constantes    : {len(cols_to_drop)}")
    print(f"  Features restantes  : {n_in - len(cols_to_drop)}")

    return (
        X_train.drop(columns=cols_to_drop),
        X_val.drop(columns=cols_to_drop, errors="ignore"),
        X_test.drop(columns=cols_to_drop, errors="ignore"),
        cols_to_drop,
    )


def format_categorical_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cfg: PoissonSelectionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    """Declara las categoricas de train y ALINEA val/test contra ese esquema.

    Solo se convierten a 'category' las columnas object/string y las declaradas
    explicitamente. Las numericas de baja cardinalidad SIGUEN siendo numericas:
    si se convierten, quedan excluidas del filtro de correlacion y sujetas al
    fallo silencioso de categorias no vistas en produccion.

    La alineacion es total (no solo sobre las candidatas detectadas en train):
    cualquier columna que ya viniera como 'category' en val/test rompe el conteo
    de categoricas cuando LightGBM construye el valid set.
    """
    X_tr: pd.DataFrame = X_train.copy()

    candidates: Set[str] = set()
    if cfg.auto_detect_string_categoricals:
        candidates.update(
            X_tr.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        )
    if cfg.explicit_categorical is not None:
        candidates.update([c for c in cfg.explicit_categorical if c in X_tr.columns])

    col: str
    for col in sorted(candidates):
        X_tr[col] = X_tr[col].astype("category")

    cat_cols: List[str] = categorical_columns(X_tr)

    X_vl: pd.DataFrame
    X_te: pd.DataFrame
    val_changes: pd.DataFrame
    test_changes: pd.DataFrame
    X_vl, val_changes = align_frame_to_reference(X_tr, X_val, "val")
    X_te, test_changes = align_frame_to_reference(X_tr, X_test, "test")

    if cfg.strict_schema_alignment:
        assert_same_categorical_schema(X_tr, X_vl, "train", "val")
        assert_same_categorical_schema(X_tr, X_te, "train", "test")

    changes: pd.DataFrame = pd.concat([val_changes, test_changes], ignore_index=True)

    print(f"\n  Categoricas declaradas: {len(cat_cols)}")
    print("  Esquema train/val/test: alineado y verificado.")
    return X_tr, X_vl, X_te, cat_cols, changes


def audit_categorical_coverage(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: Sequence[str],
) -> pd.DataFrame:
    """Cuantifica el fallo silencioso de categorias no vistas.

    Niveles presentes en val/test que no existen en train se convierten a NaN
    sin error ni warning, y LightGBM devuelve una prediccion perfectamente
    plausible por su rama de missing.

    En produccion esta misma metrica debe monitorearse CON ALERTA.
    """
    rows: List[Dict[str, object]] = []
    col: str
    for col in cat_cols:
        base_nan: float = float(X_train[col].isna().mean())
        val_extra: float = float(X_val[col].isna().mean()) - base_nan
        test_extra: float = float(X_test[col].isna().mean()) - base_nan
        rows.append(
            {
                "feature": col,
                "n_levels_train": int(X_train[col].cat.categories.size),
                "val_extra_nan_rate": round(max(0.0, val_extra), 4),
                "test_extra_nan_rate": round(max(0.0, test_extra), 4),
            }
        )

    audit: pd.DataFrame = pd.DataFrame(rows)
    if not audit.empty:
        flagged: pd.DataFrame = audit[
            (audit["val_extra_nan_rate"] > 0.01) | (audit["test_extra_nan_rate"] > 0.01)
        ]
        if not flagged.empty:
            print(f"\n  [AUDIT] {len(flagged)} categoricas con niveles no vistos (>1%):")
            print(flagged.to_string(index=False))
    return audit


# ===========================================================================
# STEP 0.5 — Auditoria de leakage temporal
# ===========================================================================
def audit_temporal_leakage(
    X_train: pd.DataFrame,
    y_train: FloatArray,
    cat_cols: Sequence[str],
    cfg: PoissonSelectionConfig,
) -> pd.DataFrame:
    """Detecta features con poder predictivo IMPLAUSIBLE para el dominio.

    Por que existe este paso: en datos de futbol el error que arruina un modelo
    no es el FDR — es un rolling mal shifteado que incluye el partido actual, o
    cierre de cuotas mezclado con apertura. Un test de permutacion NO lo
    detecta: la feature ES genuinamente predictiva del target. Ese es el
    problema.

    Metodo: se toman las `leakage_top_n` features por |Spearman| con el target,
    se entrena un modelo Poisson de UNA sola feature sobre el 70% inicial y se
    mide la reduccion de desvianza sobre el 30% final. Una sola feature honesta
    (incluso el mercado) rara vez baja la desvianza mas de ~8-10%.

    Este paso NO elimina nada. Reporta. La decision es del usuario.
    """
    print_header(
        f"STEP 0.5 — Auditoria de leakage temporal  "
        f"(top {cfg.leakage_top_n} por |rho|, flag > {cfg.leakage_dev_reduction_flag:.0%})"
    )

    cat_set: Set[str] = set(cat_cols)
    num_cols: List[str] = [c for c in X_train.columns if c not in cat_set]
    if len(num_cols) == 0:
        print("  Sin features numericas para auditar.")
        return pd.DataFrame()

    X_num: FloatArray = median_impute(X_train, num_cols)
    n_rows: int = int(X_num.shape[0])

    rho_abs: FloatArray = np.zeros(len(num_cols), dtype=np.float64)
    j: int
    for j in range(len(num_cols)):
        rho: float
        rho, _ = spearmanr(X_num[:, j], y_train)
        rho_abs[j] = 0.0 if np.isnan(rho) else abs(float(rho))

    top_n: int = int(min(cfg.leakage_top_n, len(num_cols)))
    top_idx: IntArray = np.argsort(-rho_abs)[:top_n].astype(np.int64)

    split: int = int(n_rows * 0.7)
    y_tr: FloatArray = y_train[:split]
    y_te: FloatArray = y_train[split:]
    if len(y_te) < 50 or float(np.std(y_tr)) <= EPS:
        print("  Train demasiado corto para auditar. Paso omitido.")
        return pd.DataFrame()

    mu_base: FloatArray = np.full(len(y_te), float(np.mean(y_tr)), dtype=np.float64)
    dev_base: float = poisson_deviance(y_te, mu_base)
    print(f"  Desvianza del intercepto (holdout interno 30%): {dev_base:.5f}")

    params: Dict[str, object] = lgb_poisson_params(cfg, weak=True)
    params["feature_fraction"] = 1.0
    params["num_threads"] = 1

    rows: List[Dict[str, object]] = []
    k: int
    for k in top_idx:
        name: str = num_cols[int(k)]
        x_col: FloatArray = X_num[:, int(k)].reshape(-1, 1)

        ds: lgb.Dataset = lgb.Dataset(x_col[:split], label=y_tr, free_raw_data=False)
        booster: lgb.Booster = lgb.train(
            params, ds, num_boost_round=int(cfg.leakage_boost_rounds)
        )
        mu_hat: FloatArray = np.asarray(booster.predict(x_col[split:]), dtype=np.float64)
        dev: float = poisson_deviance(y_te, mu_hat)
        reduction: float = (dev_base - dev) / abs(dev_base) if abs(dev_base) > EPS else 0.0

        rows.append(
            {
                "feature": name,
                "abs_spearman": round(float(rho_abs[int(k)]), 4),
                "dev_single": round(dev, 5),
                "dev_reduction": round(float(reduction), 4),
                "suspect": bool(reduction > cfg.leakage_dev_reduction_flag),
            }
        )

    table: pd.DataFrame = (
        pd.DataFrame(rows).sort_values("dev_reduction", ascending=False).reset_index(drop=True)
    )

    n_suspect: int = int(table["suspect"].sum())
    print("\n  Top 10 por reduccion de desvianza con UNA sola feature:")
    print(table.head(10).to_string(index=False))

    if n_suspect > 0:
        print(f"\n  >> {n_suspect} feature(s) SOSPECHOSA(S) de leakage temporal.")
        print("     Verifica su definicion: se puede calcular ANTES del kickoff?")
        print("     Un rolling sin shift(1) o cuotas de cierre entran aca.")
        print("     Ninguna metrica posterior es creible hasta resolver esto.")
    else:
        print("\n  >> Sin senales de leakage evidente en el screening.")
        print("     No es una garantia: solo se auditaron las de mayor |rho|.")

    return table


# ===========================================================================
# STEP 1 — Mutual Information Pre-Filter
# ===========================================================================
def mutual_information_prefilter(
    X_train: pd.DataFrame,
    y_train: FloatArray,
    cat_cols: Sequence[str],
    cfg: PoissonSelectionConfig,
) -> Tuple[List[str], Dict[str, float]]:
    """Filtro barato O(n*p). MI ~ 0 implica independencia estadistica del target.

    Los scores se RETORNAN y quedan disponibles para STEP 2 como criterio de
    relevancia alternativo al Spearman.
    """
    print_header(
        f"STEP 1 — Mutual Information Pre-Filter  "
        f"(threshold = {cfg.mi_threshold:.0e}, seeds = {cfg.mi_n_seeds})"
    )

    all_cols: List[str] = X_train.columns.tolist()
    cat_set: Set[str] = set(cat_cols)
    num_cols: List[str] = [c for c in all_cols if c not in cat_set]

    X_enc: pd.DataFrame = X_train.copy()
    if len(num_cols) > 0:
        X_enc[num_cols] = median_impute(X_train, num_cols)

    col: str
    for col in cat_cols:
        X_enc[col] = X_enc[col].cat.codes.astype(np.int64)  # -1 = NaN, categoria propia

    discrete_mask: BoolArray = np.array([c in cat_set for c in all_cols], dtype=bool)
    X_mat: FloatArray = X_enc[all_cols].to_numpy(dtype=np.float64)

    mi_runs: FloatArray = np.zeros((int(cfg.mi_n_seeds), len(all_cols)), dtype=np.float64)
    seed_i: int
    for seed_i in range(int(cfg.mi_n_seeds)):
        mi_runs[seed_i, :] = mutual_info_regression(
            X_mat,
            y_train,
            discrete_features=discrete_mask,
            n_neighbors=int(cfg.mi_n_neighbors),
            random_state=int(cfg.random_state) + seed_i,
        ).astype(np.float64)

    mi_mean: FloatArray = mi_runs.mean(axis=0)
    mi_scores: Dict[str, float] = {c: float(v) for c, v in zip(all_cols, mi_mean)}
    selected: List[str] = [c for c in all_cols if mi_scores[c] >= cfg.mi_threshold]

    print(f"  Features entrada    : {len(all_cols)}")
    print(f"  Eliminadas (MI ~ 0) : {len(all_cols) - len(selected)}")
    print(f"  Features restantes  : {len(selected)}")

    return selected, mi_scores


# ===========================================================================
# STEP 2 — Temporal Correlation Filter
# ===========================================================================
def temporal_correlation_filter(
    X_train: pd.DataFrame,
    y_train: FloatArray,
    features: Sequence[str],
    cat_cols: Sequence[str],
    mi_scores: Dict[str, float],
    cfg: PoissonSelectionConfig,
) -> Tuple[List[str], Dict[int, List[str]]]:
    """Clustering jerarquico sobre la distancia (1 - |rho| mediano entre bloques).

    - |rho| se calcula en bloques temporales DISJUNTOS y se toma la mediana: una
      pareja es redundante solo si lo es en la mayoria de los regimenes.
    - Clustering jerarquico en vez de recorrido greedy sobre la matriz
      triangular: el resultado no depende del orden de las columnas y maneja
      transitividad (A~B, B~C => un solo cluster).
    - El representante se elige por |Spearman| mediano con el target (valido: un
      conteo es ordinal) o por MI, segun `cluster_representative`.

    Sobre el umbral: 0.95 deja pasar clones al 0.94. El |SHAP| se reparte entre
    ellos y el STEP 3 pierde poder sobre los tres a la vez. 0.80 es un punto de
    partida mas razonable, y ademas baja m (afloja BH).
    """
    print_header(
        f"STEP 2 — Temporal Correlation Filter  "
        f"(|rho| = {cfg.corr_threshold}, bloques disjuntos = {cfg.corr_n_blocks})"
    )

    cat_set: Set[str] = set(cat_cols)
    numeric_feats: List[str] = [f for f in features if f not in cat_set]
    passthrough: List[str] = [f for f in features if f in cat_set]

    if len(numeric_feats) < 2:
        print("  Menos de 2 features numericas; filtro omitido.")
        return list(features), {}

    n_rows: int = int(X_train.shape[0])
    X_num: FloatArray = median_impute(X_train, numeric_feats)

    blocks: List[Tuple[int, int]] = disjoint_block_indices(n_rows, int(cfg.corr_n_blocks))
    n_feat: int = len(numeric_feats)
    corr_stack: FloatArray = np.zeros((len(blocks), n_feat, n_feat), dtype=np.float64)

    b_idx: int
    lo: int
    hi: int
    for b_idx, (lo, hi) in enumerate(blocks):
        corr_stack[b_idx] = spearman_abs_matrix(X_num[lo:hi, :])

    median_corr: FloatArray = np.median(corr_stack, axis=0)
    np.fill_diagonal(median_corr, 1.0)

    distance: FloatArray = 1.0 - median_corr
    np.fill_diagonal(distance, 0.0)
    distance = np.clip((distance + distance.T) / 2.0, 0.0, 2.0)

    try:
        clusterer: AgglomerativeClustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(1.0 - cfg.corr_threshold),
            metric="precomputed",
            linkage="average",
        )
    except TypeError:  # sklearn < 1.2 usaba `affinity` en vez de `metric`
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(1.0 - cfg.corr_threshold),
            linkage="average",
            **{"affinity": "precomputed"},  # type: ignore[arg-type]
        )

    labels: IntArray = clusterer.fit_predict(distance).astype(np.int64)

    # --- Relevancia para elegir representante ---
    relevance: Dict[str, float] = {}
    if cfg.cluster_representative == "spearman":
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
        relevance = {f: float(mi_scores.get(f, 0.0)) for f in numeric_feats}

    clusters: Dict[int, List[str]] = {}
    lab: int
    for f_idx, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(numeric_feats[f_idx])

    kept_numeric: List[str] = []
    members: List[str]
    for lab, members in clusters.items():
        kept_numeric.append(max(members, key=lambda f: relevance.get(f, 0.0)))

    kept_set: Set[str] = set(kept_numeric)
    final_features: List[str] = [f for f in features if f in kept_set or f in cat_set]
    n_multi: int = int(sum(1 for m in clusters.values() if len(m) > 1))

    print(f"  Features entrada     : {len(features)}")
    print(f"  Clusters formados    : {len(clusters)}  ({n_multi} con >1 miembro)")
    print(f"  Eliminadas (redund.) : {len(numeric_feats) - len(kept_numeric)}")
    print(f"  Features restantes   : {len(final_features)}  ({len(passthrough)} categoricas)")
    print(f"  Criterio representante: {cfg.cluster_representative}")

    return final_features, clusters


# ===========================================================================
# STEP 3 — Block-Permutation Importance
# ===========================================================================
def block_permutation_importance(
    X_train: pd.DataFrame,
    y_train: FloatArray,
    features: Sequence[str],
    cat_cols: Sequence[str],
    cfg: PoissonSelectionConfig,
) -> Tuple[List[str], pd.DataFrame, FloatArray, FloatArray, int]:
    """Contrasta la importancia REAL de cada feature contra su distribucion nula.

    - p = (1 + #{null >= real}) / (n_runs + 1). Nunca 0.
    - Target nulo por rotacion de bloques (tamano por ACF).
    - La importancia real se promedia sobre `shuffle_n_real_seeds` semillas: con
      `feature_fraction=0.8` un unico fit es un draw ruidoso comparado contra un
      nulo promediado sobre cientos. La significancia de una feature borderline
      no deberia depender de la suerte de un solo fit.
    - `require_empirical_floor`: el p parametrico solo puede ORDENAR features que
      ya superaron a TODAS las nulas observadas.

    Returns
    -------
    (seleccionadas, tabla, matriz_nula, importancia_real, block_size)
    """
    feat_list: List[str] = list(features)
    m: int = len(feat_list)
    n_runs: int = int(cfg.shuffle_n_runs)
    alpha: float = float(cfg.shuffle_alpha)
    method: str = str(cfg.significance_method)

    print_header(
        f"STEP 3 — Block-Permutation Importance  "
        f"(n_runs = {n_runs}, alpha = {alpha}, method = {method})"
    )

    viable: bool = preflight_permutation_power(m, n_runs, alpha, method)
    if not viable and cfg.preflight_abort:
        raise RuntimeError(
            "Configuracion incapaz de rechazar. Subi n_runs, baja m, "
            "o usa significance_method='parametric_bh'."
        )

    n_rows: int = int(X_train.shape[0])
    cat_set: Set[str] = set(cat_cols)
    cats: List[str] = [c for c in feat_list if c in cat_set]

    block_size: int
    acf1: float
    lag_cut: int
    if int(cfg.shuffle_block_size) > 0:
        block_size, acf1, lag_cut = int(cfg.shuffle_block_size), float("nan"), -1
    else:
        block_size, acf1, lag_cut = estimate_block_size_acf(y_train)

    print(
        f"\n  ACF(1) del target   : {acf1:.4f}   "
        f"(banda de ruido = {2.0 / np.sqrt(n_rows):.4f})"
    )
    print(
        f"  Longitud de bloque  : {block_size} filas "
        f"({'manual' if cfg.shuffle_block_size > 0 else 'auto por ACF'})"
    )
    if block_size <= 1:
        print("                        -> el target no tiene autocorrelacion")
        print("                           significativa; la permutacion degenera")
        print("                           a i.i.d., que aca es lo CORRECTO.")

    shap_idx: IntArray = temporal_stratified_index(
        n_rows, int(cfg.shuffle_sample_size), int(cfg.shuffle_n_strata)
    )
    print(
        f"  Muestra importancia : {len(shap_idx)} filas estratificadas en "
        f"{cfg.shuffle_n_strata} bloques temporales "
        f"(indices {int(shap_idx.min())}..{int(shap_idx.max())} de {n_rows})"
    )

    n_jobs: int = resolve_n_jobs(cfg.n_jobs, cfg.threads_per_model)
    params: Dict[str, object] = lgb_poisson_params(cfg, weak=False)

    X_cur: pd.DataFrame = X_train[feat_list]
    X_imp: pd.DataFrame = X_cur.iloc[shap_idx]

    # --- Importancia real promediada sobre semillas ---
    print(f"\n  Modelo real ({int(cfg.shuffle_n_real_seeds)} semillas, {m} features)...")
    real_runs: FloatArray = np.zeros((int(cfg.shuffle_n_real_seeds), m), dtype=np.float64)
    s: int
    for s in range(int(cfg.shuffle_n_real_seeds)):
        params_s: Dict[str, object] = dict(params)
        params_s["seed"] = int(cfg.random_state) + s
        params_s["bagging_seed"] = int(cfg.random_state) + s
        params_s["feature_fraction_seed"] = int(cfg.random_state) + s

        real_ds: lgb.Dataset = lgb.Dataset(
            X_cur, label=y_train, categorical_feature=cats, free_raw_data=False
        )
        real_model: lgb.Booster = lgb.train(
            params_s, real_ds, num_boost_round=int(cfg.shuffle_boost_rounds)
        )
        real_runs[s, :] = mean_abs_contrib(real_model, X_imp)

    real_imp: FloatArray = real_runs.mean(axis=0)
    real_cv: FloatArray = real_runs.std(axis=0) / np.maximum(real_imp, EPS)
    print(f"    CV entre semillas del modelo real: mediana {np.median(real_cv):.3f}")

    # --- Distribucion nula ---
    print(f"\n  Distribucion nula ({n_runs} corridas)...")
    rng: np.random.Generator = np.random.default_rng(int(cfg.random_state) + 10_000)
    y_nulls: List[npt.NDArray] = [
        safe_block_permute_variance(y_train, block_size, rng) for _ in range(n_runs)
    ]
    null_matrix: FloatArray = build_null_matrix(
        X_cur, y_nulls, cats, params, int(cfg.shuffle_boost_rounds), X_imp, n_jobs
    )

    # --- Criterios de significancia ---
    sig: Dict[str, npt.NDArray] = significance_from_null(
        null_matrix,
        real_imp,
        alpha=alpha,
        method=method,
        dist=str(cfg.parametric_dist),
        require_floor=bool(cfg.require_empirical_floor),
        z_threshold=float(cfg.zscore_threshold),
    )
    at_floor: BoolArray = sig["at_floor"]
    significant: BoolArray = sig["significant"]

    table: pd.DataFrame = (
        pd.DataFrame(
            {
                "feature": feat_list,
                "imp_real": real_imp,
                "imp_real_cv_seeds": real_cv,
                "imp_null_mean": sig["null_mean"],
                "imp_null_max": sig["null_max"],
                "z_score": sig["z_score"],
                "at_floor": at_floor,
                "p_emp": sig["p_emp"],
                "p_bh_emp": sig["p_bh_emp"],
                "p_param": sig["p_param"],
                "p_bh_param": sig["p_bh_param"],
                "significant": significant,
            }
        )
        .sort_values("z_score", ascending=False)
        .reset_index(drop=True)
    )

    print(
        f"\n  En el piso empirico (real > TODAS las nulas): "
        f"{int(at_floor.sum())} / {m}  ({at_floor.mean():.1%})"
    )
    print("  Comparacion de criterios:")
    print(f"    empirical_bh                : {int(sig['rej_emp'].sum()):>5}")
    print(
        f"    parametric_bh ({str(cfg.parametric_dist):<9})   : "
        f"{int(sig['rej_par'].sum()):>5}"
    )
    print(
        f"    parametric_bh + floor guard : "
        f"{int((sig['rej_par'] & at_floor).sum()):>5}"
    )
    print(
        f"    z_score >= {float(cfg.zscore_threshold):<5}            : "
        f"{int((sig['z_score'] >= float(cfg.zscore_threshold)).sum()):>5}"
    )
    print(f"\n  SELECCIONADAS [{method}]: {int(significant.sum())} / {m}")

    if int(significant.sum()) == 0:
        print("\n  >> 0 features. Revisa el PREFLIGHT de arriba ANTES de tocar los datos:")
        print("     casi siempre es falta de resolucion del test, no ausencia de senal.")
        print("     Mira la columna z_score en la tabla — si hay z de 20+, la senal esta.")

    selected: List[str] = table.loc[table["significant"], "feature"].tolist()

    # --- Chequeo de robustez, declarado como tal ---
    if cfg.robustness_pass and len(selected) > 1:
        print(f"\n{SUB}")
        print("  ROBUSTNESS CHECK (modelo debil sobre las seleccionadas)")
        print("  AVISO: esto NO es un segundo test. Reusa los mismos datos sobre")
        print("         los ganadores del primer paso — inferencia selectiva sin")
        print("         corregir. Sus p-values no son p-values. Es un chequeo de")
        print("         sensibilidad: que sobrevive con un learner mas pobre.")
        print(f"{SUB}")

        params_weak: Dict[str, object] = lgb_poisson_params(cfg, weak=True)
        cats_w: List[str] = [c for c in selected if c in cat_set]
        X_w: pd.DataFrame = X_train[selected]
        X_imp_w: pd.DataFrame = X_w.iloc[shap_idx]

        ds_w: lgb.Dataset = lgb.Dataset(
            X_w, label=y_train, categorical_feature=cats_w, free_raw_data=False
        )
        model_w: lgb.Booster = lgb.train(
            params_weak, ds_w, num_boost_round=int(cfg.shuffle_boost_rounds)
        )
        real_w: FloatArray = mean_abs_contrib(model_w, X_imp_w)

        rng_w: np.random.Generator = np.random.default_rng(int(cfg.random_state) + 20_000)
        n_runs_w: int = max(30, n_runs // 4)
        y_nulls_w: List[npt.NDArray] = [
            safe_block_permute_variance(y_train, block_size, rng_w) for _ in range(n_runs_w)
        ]
        null_w: FloatArray = build_null_matrix(
            X_w, y_nulls_w, cats_w, params_weak, int(cfg.shuffle_boost_rounds),
            X_imp_w, n_jobs,
        )
        survives: BoolArray = (null_w >= real_w[None, :]).sum(axis=0) == 0
        print(f"  Sobreviven al learner debil: {int(survives.sum())} / {len(selected)}")
        survives_map: Dict[str, bool] = {
            f: bool(v) for f, v in zip(selected, survives)
        }
        table["robust_weak"] = table["feature"].map(survives_map.get)

    return selected, table, null_matrix, real_imp, block_size


# ===========================================================================
# STEP 4 — Feature Stability Filter
# ===========================================================================
def feature_stability_filter(
    X_train: pd.DataFrame,
    y_train: FloatArray,
    features: Sequence[str],
    cat_cols: Sequence[str],
    cfg: PoissonSelectionConfig,
) -> Tuple[List[str], pd.DataFrame]:
    """Estabilidad = importancia alta Y CONSISTENTE a lo largo del tiempo.

    Metrica:

        score = mean(percentil) - k * std(percentil)

    donde `percentil` es el rango relativo de la feature DENTRO de cada ventana.
    Normalizar por ventana es esencial: si una ventana tiene el SHAP globalmente
    mas alto por el fit, contamina la dispersion de todas las features.

    (Un filtro por CV = std/mean no sirve: con `n` ventanas y std poblacional el
    CV maximo posible es sqrt(n-1), alcanzable solo en el caso degenerado de una
    ventana con todo y el resto en cero. Ninguna feature con media > 0 podria
    ser filtrada nunca.)

    Se reporta ademas el IC global (Spearman del ranking entre ventanas
    consecutivas) como diagnostico agregado, no como filtro.
    """
    print_header(
        f"STEP 4 — Feature Stability Filter  "
        f"(ventanas = {cfg.stability_n_windows}, score min = {cfg.stability_min_score}, "
        f"k = {cfg.stability_penalty_k})"
    )

    feat_list: List[str] = list(features)
    n_feat: int = len(feat_list)
    cat_set: Set[str] = set(cat_cols)
    cats: List[str] = [c for c in feat_list if c in cat_set]
    n_rows: int = int(X_train.shape[0])

    params: Dict[str, object] = lgb_poisson_params(cfg, weak=False)
    params["min_data_in_leaf"] = max(10, int(cfg.min_data_in_leaf) // 2)

    windows: List[Tuple[int, int]] = disjoint_block_indices(
        n_rows, int(cfg.stability_n_windows)
    )
    importances: List[FloatArray] = []

    w_idx: int
    lo: int
    hi: int
    for w_idx, (lo, hi) in enumerate(windows):
        X_w: pd.DataFrame = X_train[feat_list].iloc[lo:hi]
        y_w: FloatArray = y_train[lo:hi]

        if int(X_w.shape[0]) < int(cfg.stability_min_rows_per_window):
            print(
                f"  Ventana {w_idx + 1}: {X_w.shape[0]} filas "
                f"(< {cfg.stability_min_rows_per_window}), omitida."
            )
            continue
        if float(np.std(y_w)) <= EPS:
            print(f"  Ventana {w_idx + 1}: target sin varianza, omitida.")
            continue

        ds: lgb.Dataset = lgb.Dataset(
            X_w, label=y_w, categorical_feature=cats, free_raw_data=False
        )
        model: lgb.Booster = lgb.train(
            params, ds, num_boost_round=int(cfg.stability_boost_rounds)
        )
        # La importancia se mide sobre TODA la ventana, no sobre su inicio
        importances.append(mean_abs_contrib(model, X_w))
        print(
            f"  Ventana {w_idx + 1}/{cfg.stability_n_windows} procesada "
            f"({X_w.shape[0]} filas, filas {lo}-{hi})."
        )

    if len(importances) < 2:
        print("  ADVERTENCIA: menos de 2 ventanas validas. Filtro omitido.")
        return feat_list, pd.DataFrame()

    n_valid: int = len(importances)
    pct_matrix: FloatArray = np.zeros((n_valid, n_feat), dtype=np.float64)
    w: int
    for w in range(n_valid):
        pct_matrix[w, :] = rankdata(importances[w]) / float(n_feat)

    mean_pct: FloatArray = pct_matrix.mean(axis=0)
    std_pct: FloatArray = pct_matrix.std(axis=0)
    score: FloatArray = mean_pct - float(cfg.stability_penalty_k) * std_pct

    global_ic: List[float] = []
    for w in range(n_valid - 1):
        rho: float
        rho, _ = spearmanr(importances[w], importances[w + 1])
        global_ic.append(0.0 if np.isnan(rho) else float(rho))

    mean_ic: float = float(np.mean(global_ic)) if global_ic else float("nan")
    print(
        f"\n  IC global entre ventanas consecutivas: "
        f"{[round(v, 3) for v in global_ic]}  (media = {mean_ic:.3f})"
    )
    if mean_ic < 0.30:
        print("  ADVERTENCIA: el ranking de importancias es inestable a nivel global.")
        print("               El conjunto seleccionado probablemente NO generalice")
        print("               fuera del regimen de entrenamiento.")

    table: pd.DataFrame = (
        pd.DataFrame(
            {
                "feature": feat_list,
                "mean_pct": mean_pct,
                "std_pct": std_pct,
                "stability_score": score,
                "stable": score >= float(cfg.stability_min_score),
            }
        )
        .sort_values("stability_score", ascending=False)
        .reset_index(drop=True)
    )

    stable_features: List[str] = table.loc[table["stable"], "feature"].tolist()

    print(f"\n  Features estables   : {len(stable_features)}")
    print(f"  Features inestables : {n_feat - len(stable_features)}")
    if len(stable_features) == 0:
        print("  >> 0 estables. Baja stability_min_score o stability_penalty_k.")

    return stable_features, table


# ===========================================================================
# STEP 5 — Validacion
# ===========================================================================
def validate_selection(
    X_train: pd.DataFrame,
    y_train: FloatArray,
    X_val: pd.DataFrame,
    y_val: FloatArray,
    all_features: Sequence[str],
    selected_features: Sequence[str],
    cat_cols: Sequence[str],
    cfg: PoissonSelectionConfig,
    max_k: int = 8,
) -> pd.DataFrame:
    """Compara tres modelos en VAL.

      1. baseline_all : todas las features post-STEP-0 + regularizacion.
      2. selected     : las que sobrevivieron el pipeline.
      3. random_k     : k features AL AZAR del mismo tamano que la seleccion,
                        promediando `validation_random_seeds` sorteos.

    El control aleatorio no es decorativo. Sin el no se puede distinguir "la
    seleccion encontro senal" de "reducir la dimension ayuda por si solo". Si
    `selected` no le gana claramente a `random_k`, el pipeline no aporta nada
    mas que regularizacion por reduccion de dimension.

    Metricas: RPS (`calculate_rps`, family="poisson") y desvianza de Poisson.
    """
    print_header("STEP 5 — Validacion en VAL: baseline vs selected vs random_k")

    cat_set: Set[str] = set(cat_cols)
    params: Dict[str, object] = lgb_poisson_params(cfg, weak=False)
    params["num_threads"] = 0

    def _fit_eval(feats: Sequence[str], tag: str) -> Dict[str, object]:
        """Entrena con early stopping en VAL y evalua."""
        fl: List[str] = list(feats)
        cats_v: List[str] = [c for c in fl if c in cat_set]

        dtrain: lgb.Dataset = lgb.Dataset(
            X_train[fl], label=y_train, categorical_feature=cats_v, free_raw_data=False
        )
        dval: lgb.Dataset = lgb.Dataset(
            X_val[fl], label=y_val, reference=dtrain,
            categorical_feature=cats_v, free_raw_data=False,
        )
        booster: lgb.Booster = lgb.train(
            params,
            dtrain,
            num_boost_round=int(cfg.validation_boost_rounds),
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(int(cfg.validation_early_stopping), verbose=False)
            ],
        )
        lam: FloatArray = np.asarray(
            booster.predict(X_val[fl], num_iteration=booster.best_iteration),
            dtype=np.float64,
        )
        rps: Optional[float] = _rps_poisson(y_val, lam, tag, max_k)
        return {
            "model": tag,
            "n_features": len(fl),
            "best_iter": int(booster.best_iteration),
            "val_poisson_dev": round(poisson_deviance(y_val, lam), 5),
            "val_rps": round(rps, 5) if rps is not None else np.nan,
            "mean_lambda": round(float(np.mean(lam)), 4),
        }

    rows: List[Dict[str, object]] = [_fit_eval(all_features, "baseline_all")]

    if len(selected_features) > 0:
        rows.append(_fit_eval(selected_features, "selected"))

        k: int = len(selected_features)
        pool: List[str] = list(all_features)
        rng: np.random.Generator = np.random.default_rng(int(cfg.random_state) + 777)
        rnd_dev: List[float] = []
        rnd_rps: List[float] = []

        seed_i: int
        for seed_i in range(int(cfg.validation_random_seeds)):
            pick: List[str] = [
                pool[int(i)]
                for i in rng.choice(len(pool), size=min(k, len(pool)), replace=False)
            ]
            res: Dict[str, object] = _fit_eval(pick, f"random_k_{seed_i}")
            rnd_dev.append(float(res["val_poisson_dev"]))
            if not np.isnan(float(res["val_rps"])):
                rnd_rps.append(float(res["val_rps"]))

        rows.append(
            {
                "model": f"random_k (media de {cfg.validation_random_seeds})",
                "n_features": k,
                "best_iter": -1,
                "val_poisson_dev": round(float(np.mean(rnd_dev)), 5),
                "val_rps": round(float(np.mean(rnd_rps)), 5) if rnd_rps else np.nan,
                "mean_lambda": np.nan,
            }
        )

    table: pd.DataFrame = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))

    if len(table) >= 3:
        dev_base: float = float(
            table.loc[table["model"] == "baseline_all", "val_poisson_dev"].iloc[0]
        )
        dev_sel: float = float(
            table.loc[table["model"] == "selected", "val_poisson_dev"].iloc[0]
        )
        dev_rnd: float = float(table["val_poisson_dev"].iloc[-1])

        print(f"\n  Delta desvianza (baseline - selected) = {dev_base - dev_sel:+.5f}")
        print(f"  Delta desvianza (random_k - selected) = {dev_rnd - dev_sel:+.5f}")

        if dev_sel > dev_base:
            print("  >> La seleccion NO mejora al baseline. Esta pruneando de mas:")
            print("     relaja STEP 3 (alpha/method) o STEP 4 (stability_min_score).")
        elif dev_sel >= dev_rnd:
            print("  >> La seleccion NO le gana a un subconjunto ALEATORIO del mismo")
            print("     tamano. Lo que ayuda es reducir dimension, no ESTAS features.")
            print("     El pipeline no esta aportando senal.")
        else:
            print("  >> La seleccion gana al baseline Y al control aleatorio.")
            print("     Es el unico caso en que el pipeline se justifica.")

    return table


# ===========================================================================
# CALIBRADOR DE RUNTIME
# ===========================================================================
def calibrate_runtime(
    X_train: pd.DataFrame,
    y_train: FloatArray,
    features: Sequence[str],
    cat_cols: Sequence[str],
    cfg: PoissonSelectionConfig,
    n_probe: int = 4,
) -> pd.DataFrame:
    """Cronometra `n_probe` corridas bajo varias configuraciones y extrapola a
    `shuffle_n_runs`.

    Corre en pocos minutos y da el numero real antes de lanzar la corrida larga.
    """
    import os

    print_header(f"CALIBRACION DE RUNTIME  ({n_probe} corridas por configuracion)")

    n_cores: int = int(os.cpu_count() or 1)
    print(f"  Cores detectados: {n_cores}")

    feat_list: List[str] = list(features)
    cats: List[str] = [c for c in feat_list if c in set(cat_cols)]
    X_cur: pd.DataFrame = X_train[feat_list]

    shap_idx: IntArray = temporal_stratified_index(
        int(X_train.shape[0]), int(cfg.shuffle_sample_size), int(cfg.shuffle_n_strata)
    )
    X_imp: pd.DataFrame = X_cur.iloc[shap_idx]
    block_size: int = estimate_block_size_acf(y_train)[0]

    configs: List[Tuple[str, int, int, int]] = [
        ("all threads, serie", 0, 1, int(cfg.shuffle_boost_rounds)),
        ("2 threads x N workers", 2, max(1, n_cores // 2), int(cfg.shuffle_boost_rounds)),
        ("2 threads x N + 100 rounds", 2, max(1, n_cores // 2), 100),
        ("1 thread x N + 100 rounds", 1, n_cores, 100),
    ]

    rows: List[Dict[str, object]] = []
    name: str
    n_thr: int
    n_jb: int
    n_rd: int
    for name, n_thr, n_jb, n_rd in configs:
        params: Dict[str, object] = lgb_poisson_params(cfg)
        params["num_threads"] = n_thr

        rng: np.random.Generator = np.random.default_rng(int(cfg.random_state))
        y_nulls: List[npt.NDArray] = [
            safe_block_permute_variance(y_train, block_size, rng) for _ in range(n_probe)
        ]

        t0: float = time.time()
        _ = build_null_matrix(
            X_cur, y_nulls, cats, params, n_rd, X_imp, n_jb, verbose=False
        )
        per_run: float = (time.time() - t0) / float(n_probe)

        rows.append(
            {
                "config": name,
                "threads": n_thr if n_thr > 0 else "all",
                "n_jobs": n_jb,
                "rounds": n_rd,
                "s_por_run": round(per_run, 2),
                f"min_{cfg.shuffle_n_runs}_runs": round(
                    per_run * int(cfg.shuffle_n_runs) / 60.0, 1
                ),
            }
        )

    table: pd.DataFrame = (
        pd.DataFrame(rows).sort_values("s_por_run").reset_index(drop=True)
    )
    print()
    print(table.to_string(index=False))
    print(
        f"\n  Mejor: {table.iloc[0]['config']}  ->  "
        f"{table.iloc[0][f'min_{cfg.shuffle_n_runs}_runs']} min para "
        f"{cfg.shuffle_n_runs} corridas"
    )
    return table


# ===========================================================================
# MASTER PIPELINE (un target)
# ===========================================================================
def run_poisson_feature_selection(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: Optional[pd.Series] = None,
    target_name: str = "target",
    cfg: Optional[PoissonSelectionConfig] = None,
    max_k: int = 8,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, SelectionReport]:
    """Pipeline completo de seleccion de features para un target de CONTEO.

    Returns
    -------
    final_features : features seleccionadas
    X_train_sel    : train filtrado y alineado
    X_val_sel      : val   filtrado y alineado
    X_test_sel     : test  filtrado y alineado
    report         : trazabilidad completa

    ADVERTENCIA METODOLOGICA: todos los filtros se ajustan sobre X_train
    completo. Si despues se evalua con TimeSeriesSplit sobre ESE MISMO X_train,
    el fold de validacion ya participo en decidir que features sobreviven y la
    metrica OOF queda sesgada hacia arriba. La evaluacion honesta es X_val /
    X_test (que aca nunca participan) o `walk_forward_selection_stability` para
    medir cuanto varia la seleccion.
    """
    config: PoissonSelectionConfig = cfg if cfg is not None else PoissonSelectionConfig()
    report: SelectionReport = SelectionReport(target_name=target_name)
    t_start: float = time.time()

    np.random.seed(int(config.random_state))

    # Verificacion de orden cronologico
    if config.assert_temporal_order:
        if isinstance(X_train.index, pd.DatetimeIndex):
            assert X_train.index.is_monotonic_increasing, (
                "X_train NO esta ordenado cronologicamente. TODO el pipeline "
                "(bloques, ventanas, permutacion, estratificacion) lo asume."
            )
            print("  [CHECK] Indice temporal monotono creciente: OK")
        else:
            print("  [CHECK] El indice no es DatetimeIndex; no se puede verificar el")
            print("          orden cronologico automaticamente. Confirma a mano que")
            print("          X_train esta ordenado por fecha ASCENDENTE.")

    y_tr: FloatArray = y_train.to_numpy(dtype=np.float64)
    assert np.all(y_tr >= 0.0), "El target Poisson tiene valores negativos."
    assert np.all(np.isfinite(y_tr)), "El target Poisson tiene NaN o infinitos."

    report.n_start = int(X_train.shape[1])

    print(SEP)
    print(f"  POISSON TEMPORAL FEATURE SELECTION — target = {target_name}")
    print(SEP)
    print(f"  Train: {X_train.shape}   Val: {X_val.shape}   Test: {X_test.shape}")
    print(
        f"  Target: media = {y_tr.mean():.4f}, var = {y_tr.var():.4f}, "
        f"var/media = {y_tr.var() / max(y_tr.mean(), EPS):.3f}"
    )
    if y_tr.var() / max(y_tr.mean(), EPS) > 1.25:
        print("          >> Sobredispersion notable: considera Binomial Negativa")
        print("             para el modelo final. La seleccion sigue siendo valida.")
    print(f"  Ratio filas/features: {X_train.shape[0] / max(1, X_train.shape[1]):.1f}:1")
    print(
        f"  Semilla: {config.random_state}   "
        f"Workers: {resolve_n_jobs(config.n_jobs, config.threads_per_model)}"
    )

    # ── STEP 0 ──────────────────────────────────────────────────────────
    X_tr: pd.DataFrame
    X_vl: pd.DataFrame
    X_te: pd.DataFrame
    dropped: List[str]
    X_tr, X_vl, X_te, dropped = drop_sparse_features(X_train, X_val, X_test, config)
    report.dropped_sparse = dropped
    report.n_after_sparsity = int(X_tr.shape[1])

    cat_cols: List[str]
    changes: pd.DataFrame
    X_tr, X_vl, X_te, cat_cols, changes = format_categorical_features(
        X_tr, X_vl, X_te, config
    )
    report.categorical_cols = cat_cols
    report.schema_alignment = changes
    if len(cat_cols) > 0:
        report.categorical_audit = audit_categorical_coverage(X_tr, X_vl, X_te, cat_cols)

    baseline_features: List[str] = X_tr.columns.tolist()

    # ── STEP 0.5 ────────────────────────────────────────────────────────
    if config.run_leakage_audit:
        report.leakage_audit = audit_temporal_leakage(X_tr, y_tr, cat_cols, config)

    # ── STEP 1 ──────────────────────────────────────────────────────────
    mi_features: List[str]
    mi_scores: Dict[str, float]
    mi_features, mi_scores = mutual_information_prefilter(X_tr, y_tr, cat_cols, config)
    report.mi_scores = mi_scores
    report.n_after_mi = len(mi_features)

    # ── STEP 2 ──────────────────────────────────────────────────────────
    corr_features: List[str]
    clusters: Dict[int, List[str]]
    corr_features, clusters = temporal_correlation_filter(
        X_tr, y_tr, mi_features, cat_cols, mi_scores, config
    )
    report.corr_clusters = clusters
    report.n_after_corr = len(corr_features)

    # ── STEP 3 ──────────────────────────────────────────────────────────
    shuffle_features: List[str]
    shuffle_table: pd.DataFrame
    null_matrix: FloatArray
    real_imp: FloatArray
    block_size: int
    shuffle_features, shuffle_table, null_matrix, real_imp, block_size = (
        block_permutation_importance(X_tr, y_tr, corr_features, cat_cols, config)
    )
    report.shuffle_table = shuffle_table
    report.shuffle_null_matrix = null_matrix
    report.shuffle_real_imp = real_imp
    # Orden CANONICO de columnas de null_matrix / real_imp. La tabla viene
    # ordenada por z_score, asi que NO sirve para alinear: usar esta lista.
    report.shuffle_feature_order = list(corr_features)
    report.block_size_used = block_size
    report.n_after_shuffle = len(shuffle_features)

    # ── STEP 4 ──────────────────────────────────────────────────────────
    final_features: List[str] = shuffle_features
    if config.run_stability_filter and len(shuffle_features) > 1:
        stability_table: pd.DataFrame
        final_features, stability_table = feature_stability_filter(
            X_tr, y_tr, shuffle_features, cat_cols, config
        )
        report.stability_table = stability_table
    report.n_after_stability = len(final_features)

    # Fallback explicito: nunca devolver un conjunto vacio en silencio
    if len(final_features) == 0:
        print("\n  [FALLBACK] La seleccion quedo VACIA. Se devuelven las features")
        print("             post-STEP-2 para que el flujo no se corte. NO uses esto")
        print("             como seleccion: revisa el PREFLIGHT y los umbrales.")
        final_features = corr_features

    report.final_features = final_features

    # ── STEP 5 ──────────────────────────────────────────────────────────
    if config.run_validation and y_val is not None:
        y_vl: FloatArray = y_val.to_numpy(dtype=np.float64)
        report.validation_table = validate_selection(
            X_tr, y_tr, X_vl, y_vl, baseline_features, final_features,
            cat_cols, config, max_k=max_k,
        )
    elif config.run_validation:
        print("\n  [AVISO] y_val no provisto: se omite STEP 5.")
        print("          Sin el baseline y el control aleatorio no sabes si la")
        print("          seleccion aporta algo. No lo saltees.")

    X_train_sel: pd.DataFrame = X_tr[final_features]
    X_val_sel: pd.DataFrame = X_vl[final_features]
    X_test_sel: pd.DataFrame = X_te[final_features]

    report.elapsed_seconds = time.time() - t_start

    print_header(f"RESUMEN — {target_name}")
    print(f"  [Start   ] : {report.n_start}")
    print(f"  [Step 0  ] : {report.n_after_sparsity}  (-{len(report.dropped_sparse)} esparsas)")
    print(f"  [Step 1  ] : {report.n_after_mi}  (-MI~0)")
    print(f"  [Step 2  ] : {report.n_after_corr}  (-redundantes)")
    print(f"  [Step 3  ] : {report.n_after_shuffle}  (-ruido)")
    print(f"  [Step 4  ] : {report.n_after_stability}  (-inestables)")
    print(f"  {SUB}")
    print(f"  FEATURES FINALES : {len(final_features)}")
    print(f"  Tiempo total     : {report.elapsed_seconds / 60.0:.1f} min")
    print(SEP)

    return final_features, X_train_sel, X_val_sel, X_test_sel, report


# ===========================================================================
# WRAPPER — los dos componentes del target
# ===========================================================================
def run_dual_target_selection(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    config: ModelConfig,
    y_val: Optional[pd.DataFrame] = None,
    cfg: Optional[PoissonSelectionConfig] = None,
) -> Dict[str, object]:
    """Corre el pipeline para los DOS componentes del target y los compara.

    Las columnas salen de `config.target.components`, no de argumentos
    hardcodeados: cambiando el YAML se reutiliza para goals, corners, tarjetas.

    El solapamiento entre componentes es un diagnostico gratis: si los dos
    conjuntos son casi identicos, la seleccion esta capturando "calidad general
    del partido" mas que asimetria local/visitante.
    """
    selection_cfg: PoissonSelectionConfig = (
        cfg if cfg is not None else PoissonSelectionConfig.from_model_config(config)
    )

    results: Dict[str, object] = {}
    key: str
    for key in config.target.keys:
        column: str = config.target.component(key).column
        print(f"\n\n{'#' * 76}")
        print(f"#  COMPONENTE = {key}   (columna = {column})")
        print(f"{'#' * 76}")

        feats: List[str]
        Xtr_s: pd.DataFrame
        Xvl_s: pd.DataFrame
        Xte_s: pd.DataFrame
        rep: SelectionReport
        feats, Xtr_s, Xvl_s, Xte_s, rep = run_poisson_feature_selection(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train[column],
            y_val=y_val[column] if y_val is not None else None,
            target_name=column,
            cfg=selection_cfg,
            max_k=config.target.max_k,
        )
        results[key] = {
            "column": column,
            "features": feats,
            "X_train_sel": Xtr_s,
            "X_val_sel": Xvl_s,
            "X_test_sel": Xte_s,
            "report": rep,
        }

    keys: List[str] = config.target.keys
    set_a: Set[str] = set(results[keys[0]]["features"])   # type: ignore[index]
    set_b: Set[str] = set(results[keys[1]]["features"])   # type: ignore[index]
    inter: Set[str] = set_a & set_b
    union: Set[str] = set_a | set_b

    print_header(f"COMPARACION {keys[0].upper()} vs {keys[1].upper()}")
    print(f"  {keys[0]:<20}: {len(set_a)} features")
    print(f"  {keys[1]:<20}: {len(set_b)} features")
    print(f"  Interseccion        : {len(inter)}")
    print(f"  Union               : {len(union)}")
    print(f"  Jaccard             : {len(inter) / max(1, len(union)):.3f}")
    if len(union) > 0 and len(inter) / len(union) > 0.85:
        print("  >> Conjuntos casi identicos: la seleccion captura calidad general")
        print("     del partido, no asimetria entre componentes. Es plausible, pero")
        print("     verifica que las features de asimetria no se hayan perdido en el")
        print("     filtro de correlacion.")

    results["intersection"] = sorted(inter)
    results["union"] = sorted(union)
    return results


# ===========================================================================
# WALK-FORWARD
# ===========================================================================
def walk_forward_selection_stability(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: PoissonSelectionConfig,
    n_folds: int = 3,
) -> pd.DataFrame:
    """Re-ejecuta la seleccion sobre ventanas EXPANDIDAS y mide el Jaccard.

    Es la unica prueba real de que la seleccion no esta sobreajustada a un
    regimen. Si cada fold devuelve un conjunto muy distinto, el pipeline no esta
    descubriendo senal estable — esta muestreando ruido.

    COSTO: n_folds veces el pipeline completo. Correr con `shuffle_n_runs` bajo,
    `run_validation=False` y `run_leakage_audit=False`.
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
        feats, _, _, _, _ = run_poisson_feature_selection(
            X_train=X_fold,
            X_val=X_fold.iloc[:20],
            X_test=X_fold.iloc[:20],
            y_train=y_fold,
            y_val=None,
            target_name=f"wf_fold_{fold + 1}",
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

    mean_j: float = float(table["jaccard"].mean()) if not table.empty else 0.0
    print(f"\n  Jaccard medio: {mean_j:.3f}")
    if mean_j < 0.5:
        print("  >> El pipeline NO es estable entre folds. Las features elegidas")
        print("     dependen del periodo. No usar en produccion.")

    core: Set[str] = set.intersection(*selections) if selections else set()
    print(f"  Nucleo presente en TODOS los folds: {len(core)} features")

    return table

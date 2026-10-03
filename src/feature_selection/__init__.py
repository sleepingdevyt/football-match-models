"""Pipelines de seleccion de features para series temporales.

Dos variantes que comparten los helpers de `common.py`:

- `feature_selection`          -> clasificacion (LightGBM multiclase).
- `poisson_feature_selection`  -> conteos (LightGBM objective="poisson"),
                                  usado por todos los notebooks de distribucion.

Los nombres de los pasos coinciden entre las dos variantes, asi que conviene
importar del submodulo explicito en vez de confiar en el nombre corto:

    from src.feature_selection.poisson_feature_selection import (
        PoissonSelectionConfig, run_poisson_feature_selection,
    )
"""

from src.feature_selection.common import (
    block_permute,
    build_null_matrix,
    disjoint_block_indices,
    estimate_block_size_acf,
    estimate_block_size_runlength,
    mean_abs_contrib,
    median_impute,
    parametric_pvalue,
    preflight_permutation_power,
    reanalyze_significance,
    resolve_n_jobs,
    safe_block_permute_classes,
    safe_block_permute_variance,
    significance_from_null,
    spearman_abs_matrix,
    temporal_stratified_index,
)
from src.feature_selection.feature_selection import (
    DEFAULT_RANDOM_STATE,
    PipelineConfig,
    PipelineReport,
    diagnose_schema,
    encode_target,
    encode_target_with_map,
    run_feature_selection_pipeline,
)
from src.feature_selection.poisson_feature_selection import (
    PoissonSelectionConfig,
    SelectionReport,
    audit_temporal_leakage,
    calibrate_runtime,
    lgb_poisson_params,
    poisson_deviance,
    run_dual_target_selection,
    run_poisson_feature_selection,
    walk_forward_selection_stability,
)

__all__ = [
    # --- comunes ---
    "block_permute",
    "build_null_matrix",
    "disjoint_block_indices",
    "estimate_block_size_acf",
    "estimate_block_size_runlength",
    "mean_abs_contrib",
    "median_impute",
    "parametric_pvalue",
    "preflight_permutation_power",
    "reanalyze_significance",
    "resolve_n_jobs",
    "safe_block_permute_classes",
    "safe_block_permute_variance",
    "significance_from_null",
    "spearman_abs_matrix",
    "temporal_stratified_index",
    # --- clasificacion ---
    "DEFAULT_RANDOM_STATE",
    "PipelineConfig",
    "PipelineReport",
    "diagnose_schema",
    "encode_target",
    "encode_target_with_map",
    "run_feature_selection_pipeline",
    # --- conteos / Poisson ---
    "PoissonSelectionConfig",
    "SelectionReport",
    "audit_temporal_leakage",
    "calibrate_runtime",
    "lgb_poisson_params",
    "poisson_deviance",
    "run_dual_target_selection",
    "run_poisson_feature_selection",
    "walk_forward_selection_stability",
]

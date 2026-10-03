"""Tuning de hiperparametros con Optuna para modelos de distribucion.

El espacio de busqueda vive en `configs/<target>/optuna_search_spaces.yaml`,
no en el codigo: agregar o mover un rango no requiere tocar Python. La metrica
objetivo (NLL o RPS) y los parametros del pruner tambien salen del YAML.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import optuna
import pandas as pd
from numpy.typing import NDArray
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit

from src.config.config import ModelConfig, load_yaml
from src.metrics.distribution_metrics import (
    DistFamily,
    calculate_nll,
    calculate_rps,
)
from src.training.distribution_models import (
    clip_params,
    impute_frames,
    needs_imputation,
    predict_params,
    train_model,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Espacio de busqueda desde YAML
# =============================================================================

def load_search_spaces(config: ModelConfig) -> Dict[str, Dict[str, Any]]:
    """Carga el YAML de espacios de busqueda del target."""
    tuning: Dict[str, Any] = config.raw.get("tuning", {})
    filename: str = tuning.get("search_spaces_file", "optuna_search_spaces.yaml")
    return load_yaml(config.configs_dir / filename)


def suggest_params(
    trial: optuna.Trial,
    space: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Traduce el espacio declarado en YAML a llamadas `trial.suggest_*`.

    Formato esperado por hiperparametro:
        {type: int,   low: .., high: .., step: ..}
        {type: float, low: .., high: .., log: true}
        {type: categorical, choices: [..]}
    """
    params: Dict[str, Any] = {}

    for name, spec in space.items():
        if not isinstance(spec, dict) or "type" not in spec:
            raise ValueError(
                f"Espacio invalido para '{name}': se esperaba un dict con "
                f"clave 'type'; recibi {spec!r}."
            )

        kind: str = str(spec["type"])

        if kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]),
                step=int(spec.get("step", 1)),
            )
        elif kind == "float":
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(
                f"Tipo no soportado en '{name}': '{kind}'. "
                f"Opciones: int, float, categorical."
            )

    return params


# =============================================================================
# Objective
# =============================================================================

def _build_objective(
    model_name: str,
    family: DistFamily,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: Sequence[str],
    config: ModelConfig,
    space: Dict[str, Dict[str, Any]],
    n_splits: int,
    metric: str,
) -> Callable[[optuna.Trial], float]:
    """Fabrica la funcion objetivo con TSCV.

    Garantias anti-leakage
    ----------------------
    [1] El subset de features se aplica DENTRO del fold.
    [2] El imputer se fitea SOLO con el train del fold.
    [3] El early stopping usa el valid del propio fold.
    [4] Nunca se predice sobre el train del fold.
    """
    feature_list: List[str] = list(features)
    guards: Dict[str, float] = config.numeric_guards
    negbin_kwargs: Dict[str, Any] = config.raw.get("negbin_kwargs", {})
    base_params: Dict[str, Any] = config.model_params(model_name)
    fit_params: Dict[str, Any] = config.fit_params(model_name)

    def objective(trial: optuna.Trial) -> float:
        trial_params: Dict[str, Any] = dict(base_params)
        trial_params.update(suggest_params(trial, space))

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_scores: List[float] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            # [1] features aplicadas despues de partir
            X_tr: pd.DataFrame = X_train.iloc[train_idx][feature_list]
            y_tr: pd.Series = y_train.iloc[train_idx]
            X_va: pd.DataFrame = X_train.iloc[val_idx][feature_list]
            y_va: pd.Series = y_train.iloc[val_idx]

            # [2] imputer fiteado solo con el train del fold
            if needs_imputation(model_name):
                X_tr, X_va, _ = impute_frames(X_tr, X_va)

            # [3] early stopping contra el valid del fold
            model = train_model(
                model_name, family, X_tr, y_tr, X_va, y_va,
                params=trial_params,
                fit_params=fit_params,
                negbin_kwargs=negbin_kwargs,
            )

            # [4] se predice unicamente sobre el valid
            raw: Dict[str, NDArray] = predict_params(
                model, X_va, model_name, family,
            )
            params: Dict[str, NDArray] = clip_params(raw, family, guards)
            y_va_arr: NDArray = np.asarray(y_va).ravel()

            if metric == "nll":
                score: float = calculate_nll(y_va_arr, params, family=family)
            elif metric == "rps":
                rps_result: Dict[str, Any] = calculate_rps(
                    y_va_arr, params, family=family,
                    max_k=config.target.max_k,
                )
                score = float(rps_result["rps_mean"])
            else:
                raise ValueError(
                    f"Metrica no soportada: '{metric}'. Opciones: 'nll', 'rps'."
                )

            fold_scores.append(score)

            # El pruner recibe la media acumulada, consistente con el objetivo.
            trial.report(float(np.mean(fold_scores)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    return objective


# =============================================================================
# Runner
# =============================================================================

def run_tuning(
    model_name: str,
    family: DistFamily,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: Sequence[str],
    config: ModelConfig,
    component_key: str = "",
    n_trials: Optional[int] = None,
    n_splits: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Corre el estudio Optuna para una combinacion modelo/familia/componente.

    Returns
    -------
    dict con study, best_params, best_value, trials_df y metadatos.
    Nota: NO reentrena el modelo final; eso lo hace `evaluate_on_validation`
    pasandole `best_params`, evitando entrenar dos veces.
    """
    tuning_cfg: Dict[str, Any] = config.raw.get("tuning", {})
    trials: int = n_trials if n_trials is not None else int(
        tuning_cfg.get("n_trials", 100)
    )
    splits: int = n_splits if n_splits is not None else int(
        tuning_cfg.get("n_splits", config.n_splits)
    )
    metric: str = str(tuning_cfg.get("metric", "nll"))
    seed: int = int(tuning_cfg.get("random_state", config.random_state))
    pruner_cfg: Dict[str, Any] = tuning_cfg.get("pruner", {})

    spaces: Dict[str, Dict[str, Any]] = load_search_spaces(config)
    if model_name not in spaces:
        raise KeyError(
            f"No hay espacio de busqueda para '{model_name}' en "
            f"{config.configs_dir}. Definido: {list(spaces)}"
        )

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"  OPTUNA TUNING - {model_name} | {family} | {component_key}")
        print(f"  Metrica objetivo : {metric} (menor es mejor)")
        print(f"  Trials : {trials} | TSCV folds : {splits}")
        print(f"  Features : {len(list(features))}")
        print(f"{'=' * 55}")

    study: optuna.Study = optuna.create_study(
        direction=str(tuning_cfg.get("direction", "minimize")),
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 10)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 2)),
            interval_steps=int(pruner_cfg.get("interval_steps", 1)),
        ),
    )

    objective = _build_objective(
        model_name, family, X_train, y_train, features,
        config, spaces[model_name], splits, metric,
    )

    study.optimize(objective, n_trials=trials, show_progress_bar=verbose)

    trials_df: pd.DataFrame = study.trials_dataframe()
    n_complete: int = int((trials_df["state"] == "COMPLETE").sum()) if not trials_df.empty else 0
    n_pruned: int = int((trials_df["state"] == "PRUNED").sum()) if not trials_df.empty else 0

    if verbose:
        print(f"\n  Best {metric}      : {study.best_value:.5f}")
        print(f"  Completed trials : {n_complete}")
        print(f"  Pruned trials    : {n_pruned}")
        print(f"  Best params:")
        for key, value in study.best_params.items():
            print(f"    {key:<22}: {value}")

    return {
        "study": study,
        "best_params": dict(study.best_params),
        "best_value": float(study.best_value),
        "metric": metric,
        "model_name": model_name,
        "family": family,
        "component": component_key,
        "n_trials": trials,
        "n_splits": splits,
        "n_complete": n_complete,
        "n_pruned": n_pruned,
        "trials_df": trials_df,
        "tuning_date": datetime.now().isoformat(timespec="seconds"),
    }


def tune_and_save(
    model_name: str,
    family: DistFamily,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: Sequence[str],
    config: ModelConfig,
    component_key: str = "",
    n_trials: Optional[int] = None,
    n_splits: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """`run_tuning` + persistencia de best params (YAML) y metricas (JSON)."""
    from src.evaluation.persistence import save_best_params, save_metrics

    result: Dict[str, Any] = run_tuning(
        model_name, family, X_train, y_train, features, config,
        component_key=component_key, n_trials=n_trials, n_splits=n_splits,
        verbose=verbose,
    )

    suffix: str = f"_{component_key}" if component_key else ""
    stem: str = f"{model_name.lower()}_{family}{suffix}"

    save_best_params(
        {"params": result["best_params"]},
        model_name,
        config,
        family=family,
        component=component_key or None,
        extra={
            "model": model_name,
            "family": family,
            "component": component_key,
            "metric": result["metric"],
            "best_value": result["best_value"],
            "tuning_date": result["tuning_date"],
        },
    )

    save_metrics(
        {
            "best_value": result["best_value"],
            "metric": result["metric"],
            "best_params": result["best_params"],
            "n_trials": result["n_trials"],
            "n_complete": result["n_complete"],
            "n_pruned": result["n_pruned"],
        },
        f"tuning_{stem}",
        config,
        metadata={
            "model": model_name,
            "family": family,
            "component": component_key,
        },
    )

    return result

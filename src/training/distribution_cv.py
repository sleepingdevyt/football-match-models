"""Comparacion de modelos de distribucion: TSCV, LR test y evaluacion en validacion.

Todo el flujo se parametriza desde el `ModelConfig`: componentes del target,
familias a comparar, hiperparametros y guardas numericas. No hay referencias
hardcodeadas a goles ni a home/away.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2
from sklearn.model_selection import TimeSeriesSplit

from src.config.config import ModelConfig
from src.metrics.distribution_metrics import (
    DistFamily,
    calculate_nll,
    calculate_rps,
    evaluate_distribution_model,
)
from src.training.distribution_models import (
    MODEL_NAMES,
    clip_params,
    impute_frames,
    needs_imputation,
    predict_params,
    train_model,
)


# =============================================================================
# Estructura de datos por componente
# =============================================================================

def build_component_data(
    config: ModelConfig,
    X: Dict[str, pd.DataFrame],
    y: Dict[str, pd.Series],
    features: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Arma la lista de componentes a modelar a partir de la config.

    Parameters
    ----------
    config   : ModelConfig con el bloque `target`.
    X        : {key: DataFrame} de predictores por componente.
    y        : {key: Series} del target por componente.
    features : {key: lista de features} por componente.

    Returns
    -------
    Lista de dicts con key, column, X, y, features.
    """
    out: List[Dict[str, Any]] = []

    for component in config.target.components:
        key: str = component.key
        missing: List[str] = [
            name for name, src in (("X", X), ("y", y), ("features", features))
            if key not in src
        ]
        if missing:
            raise KeyError(
                f"Falta el componente '{key}' en: {missing}. "
                f"Componentes esperados: {config.target.keys}"
            )

        out.append({
            "key": key,
            "column": component.column,
            "X": X[key],
            "y": y[key],
            "features": features[key],
        })

    return out


def _params_to_arrays(
    family: DistFamily,
    mu: NDArray,
    alpha: Optional[NDArray] = None,
) -> Dict[str, NDArray]:
    """Empaqueta (mu, alpha) en el dict que esperan las metricas."""
    if family == "poisson":
        return {"lambda": mu}
    if alpha is None:
        raise ValueError("family='negbin' requiere alpha.")
    return {"mu": mu, "alpha": alpha}


# =============================================================================
# TSCV para un (modelo, familia, componente)
# =============================================================================

def run_tscv(
    model_name: str,
    family: DistFamily,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: Sequence[str],
    config: ModelConfig,
    n_splits: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Validacion cruzada temporal para una combinacion modelo/familia.

    Garantias anti-leakage
    ----------------------
    [1] El subset de features se aplica DENTRO del fold, sobre slices ya separados.
    [2] El imputer se fitea SOLO con el train del fold.
    [3] El early stopping usa el valid del propio fold.

    Returns
    -------
    dict con metricas por fold, predicciones OOF y metricas OOF globales.
    """
    splits: int = n_splits if n_splits is not None else config.n_splits
    tscv = TimeSeriesSplit(n_splits=splits)

    feature_list: List[str] = list(features)
    guards: Dict[str, float] = config.numeric_guards
    negbin_kwargs: Dict[str, Any] = config.raw.get("negbin_kwargs", {})

    # NaN distingue "no predicho" de "predicho como 0".
    oof_mu: NDArray = np.full(len(X_train), np.nan)
    oof_alpha: NDArray = np.full(len(X_train), np.nan)
    fold_rps: List[float] = []
    fold_nll: List[float] = []

    if verbose:
        print(f"\n  [{model_name} | {family}] TSCV (n_splits={splits})...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        # [1] features aplicadas despues de partir
        X_tr: pd.DataFrame = X_train.iloc[train_idx][feature_list]
        y_tr: pd.Series = y_train.iloc[train_idx]
        X_va: pd.DataFrame = X_train.iloc[val_idx][feature_list]
        y_va: pd.Series = y_train.iloc[val_idx]

        # [2] imputer fiteado solo en el train del fold
        if needs_imputation(model_name):
            X_tr, X_va, _ = impute_frames(X_tr, X_va)

        # [3] early stopping contra el valid del fold
        model = train_model(
            model_name, family, X_tr, y_tr, X_va, y_va,
            params=config.model_params(model_name),
            fit_params=config.fit_params(model_name),
            negbin_kwargs=negbin_kwargs,
        )

        raw: Dict[str, NDArray] = predict_params(model, X_va, model_name, family)
        params: Dict[str, NDArray] = clip_params(raw, family, guards)

        oof_mu[val_idx] = params["lambda"] if family == "poisson" else params["mu"]
        if family == "negbin":
            oof_alpha[val_idx] = params["alpha"]

        y_va_arr: NDArray = np.asarray(y_va).ravel()
        rps_fold: Dict[str, Any] = calculate_rps(
            y_va_arr, params, family=family, max_k=config.target.max_k,
        )
        nll_value: float = calculate_nll(y_va_arr, params, family=family)

        fold_rps.append(float(rps_fold["rps_mean"]))
        fold_nll.append(nll_value)

        if verbose:
            print(f"    Fold {fold + 1} - RPS: {fold_rps[-1]:.5f} | "
                  f"NLL: {nll_value:.5f}")

    # ── Metricas OOF globales ────────────────────────────────────────────────
    valid_idx: NDArray = np.where(~np.isnan(oof_mu))[0]
    params_oof: Dict[str, NDArray] = _params_to_arrays(
        family, oof_mu[valid_idx],
        oof_alpha[valid_idx] if family == "negbin" else None,
    )
    y_oof: NDArray = np.asarray(y_train.iloc[valid_idx]).ravel()

    rps_oof: Dict[str, Any] = calculate_rps(
        y_oof, params_oof, family=family, max_k=config.target.max_k,
    )
    nll_oof: float = calculate_nll(y_oof, params_oof, family=family)

    return {
        "model_name": model_name,
        "family": family,
        "avg_fold_rps": float(np.mean(fold_rps)),
        "avg_fold_nll": float(np.mean(fold_nll)),
        "fold_rps": fold_rps,
        "fold_nll": fold_nll,
        "oof_rps": float(rps_oof["rps_mean"]),
        "oof_rps_std": float(rps_oof["rps_std"]),
        "oof_nll": nll_oof,
        "oof_mu": oof_mu,
        "oof_alpha": oof_alpha,
        "n_valid": int(len(valid_idx)),
    }


# =============================================================================
# LR test - Poisson vs NegBin (modelos anidados)
# =============================================================================

def likelihood_ratio_test(
    nll_poisson: float,
    nll_negbin: float,
    n: int,
    alpha: float = 0.01,
    label: str = "",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Test de razon de verosimilitud Poisson vs NegBin.

    H0: alpha = 0 (la Poisson es suficiente).
    Bajo H0, LR = 2 * n * (NLL_poisson - NLL_negbin) ~ chi2(1).

    El test es one-sided: alpha esta en el borde del espacio parametrico
    (alpha >= 0), asi que el p-value correcto es la mitad del two-sided.
    """
    lr_stat: float = 2.0 * (nll_poisson - nll_negbin) * n
    p_two: float = float(1.0 - chi2.cdf(max(lr_stat, 0.0), df=1))
    p_one: float = p_two / 2.0
    significant: bool = p_one < alpha

    if verbose:
        print(f"\n  LR Test {label}:")
        print(f"    NLL Poisson : {nll_poisson:.5f}")
        print(f"    NLL NegBin  : {nll_negbin:.5f}")
        print(f"    LR stat     : {lr_stat:.3f}")
        print(f"    p (1-sided) : {p_one:.5f}")
        print(f"    Decision    : "
              f"{'usar NegBin' if significant else 'quedarse con Poisson'}")

    return {
        "lr_stat": lr_stat,
        "p_value": p_one,
        "alpha": alpha,
        "significant": significant,
        "recommended_family": "negbin" if significant else "poisson",
    }


# =============================================================================
# Runner: comparacion completa
# =============================================================================

def compare_models(
    config: ModelConfig,
    X: Dict[str, pd.DataFrame],
    y: Dict[str, pd.Series],
    features: Dict[str, List[str]],
    models: Sequence[str] = MODEL_NAMES,
    distributions: Optional[Sequence[str]] = None,
    n_splits: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Corre TSCV para cada (componente x modelo x familia) y el LR test.

    Returns
    -------
    dict con:
        summary   : DataFrame ordenado por (componente, oof_nll)
        results   : lista de dicts crudos por combinacion
        lr_tests  : DataFrame con el LR test por (componente, modelo)
        failures  : lista de combinaciones que fallaron
    """
    families: List[str] = list(distributions or config.distributions)
    components: List[Dict[str, Any]] = build_component_data(config, X, y, features)

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for comp in components:
        for model_name in models:
            for family in families:
                try:
                    result: Dict[str, Any] = run_tscv(
                        model_name=model_name,
                        family=family,  # type: ignore[arg-type]
                        X_train=comp["X"],
                        y_train=comp["y"],
                        features=comp["features"],
                        config=config,
                        n_splits=n_splits,
                        verbose=verbose,
                    )
                    result["component"] = comp["key"]
                    result["column"] = comp["column"]
                    results.append(result)
                except Exception as exc:
                    print(f"\n  [FALLO] {model_name} | {family} | "
                          f"{comp['key']}: {exc}")
                    failures.append({
                        "component": comp["key"],
                        "model": model_name,
                        "family": family,
                        "error": str(exc),
                    })

    summary: pd.DataFrame = pd.DataFrame([
        {
            "component": r["component"],
            "model": r["model_name"],
            "family": r["family"],
            "avg_fold_rps": r["avg_fold_rps"],
            "avg_fold_nll": r["avg_fold_nll"],
            "oof_rps": r["oof_rps"],
            "oof_nll": r["oof_nll"],
            "n_valid": r["n_valid"],
        }
        for r in results
    ])

    if not summary.empty:
        summary = (summary
                   .sort_values(["component", "oof_nll"])
                   .reset_index(drop=True))

    lr_df: pd.DataFrame = _run_lr_tests(
        results, components, models, config, verbose=verbose,
    )

    return {
        "summary": summary,
        "results": results,
        "lr_tests": lr_df,
        "failures": failures,
    }


def _run_lr_tests(
    results: List[Dict[str, Any]],
    components: List[Dict[str, Any]],
    models: Sequence[str],
    config: ModelConfig,
    verbose: bool = True,
) -> pd.DataFrame:
    """LR test Poisson vs NegBin para cada (componente, modelo) disponible."""
    lr_alpha: float = float(
        config.raw.get("likelihood_ratio_test", {}).get("alpha", 0.01)
    )

    if verbose:
        print(f"\n{'=' * 65}")
        print("  LIKELIHOOD RATIO TESTS - Poisson vs NegBin")
        print(f"{'=' * 65}")

    rows: List[Dict[str, Any]] = []

    for comp in components:
        for model_name in models:
            def _find(family: str) -> Optional[Dict[str, Any]]:
                return next(
                    (r for r in results
                     if r["component"] == comp["key"]
                     and r["model_name"] == model_name
                     and r["family"] == family),
                    None,
                )

            r_poisson: Optional[Dict[str, Any]] = _find("poisson")
            r_negbin: Optional[Dict[str, Any]] = _find("negbin")

            if r_poisson is None or r_negbin is None:
                continue

            n: int = min(r_poisson["n_valid"], r_negbin["n_valid"])
            lr: Dict[str, Any] = likelihood_ratio_test(
                nll_poisson=r_poisson["oof_nll"],
                nll_negbin=r_negbin["oof_nll"],
                n=n,
                alpha=lr_alpha,
                label=f"{comp['key']} | {model_name}",
                verbose=verbose,
            )

            rows.append({
                "component": comp["key"],
                "model": model_name,
                "nll_poisson": r_poisson["oof_nll"],
                "nll_negbin": r_negbin["oof_nll"],
                "lr_stat": lr["lr_stat"],
                "p_value": lr["p_value"],
                "significant": lr["significant"],
                "recommended_family": lr["recommended_family"],
            })

    return pd.DataFrame(rows)


# =============================================================================
# Evaluacion sobre el conjunto de validacion
# =============================================================================

def _resolve_best_params(
    best_params: Optional[Dict[str, Dict[str, Any]]],
    model_name: str,
    component_key: str,
    component_keys: Sequence[str],
) -> Dict[str, Any]:
    """Extrae los best params de un (modelo, componente).

    Acepta dos formatos:
        plano   : {modelo: {param: valor}}          -> mismos params para todos
        anidado : {modelo: {componente: {param: v}}} -> uno por componente

    El anidado es el caso normal cuando se tunea cada componente por separado.
    Se distingue mirando si las claves del dict son exactamente componentes.
    """
    if not best_params or model_name not in best_params:
        return {}

    entry: Dict[str, Any] = best_params[model_name]
    if not entry:
        return {}

    # Anidado solo si TODAS las claves son componentes conocidos. Un dict
    # plano de hiperparametros ('max_depth', 'learning_rate') no lo cumple.
    if set(entry).issubset(set(component_keys)):
        if component_key not in entry:
            raise KeyError(
                f"best_params['{model_name}'] esta anidado por componente pero "
                f"no trae '{component_key}'. Presentes: {sorted(entry)}"
            )
        return dict(entry[component_key])

    return dict(entry)


def evaluate_on_validation(
    config: ModelConfig,
    X_train: Dict[str, pd.DataFrame],
    y_train: Dict[str, pd.Series],
    X_valid: Dict[str, pd.DataFrame],
    y_valid: Dict[str, pd.Series],
    features: Dict[str, List[str]],
    models: Sequence[str] = MODEL_NAMES,
    family: DistFamily = "poisson",
    best_params: Optional[Dict[str, Dict[str, Any]]] = None,
    X_eval: Optional[Dict[str, pd.DataFrame]] = None,
    y_eval: Optional[Dict[str, pd.Series]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Entrena cada modelo sobre train y reporta metricas out-of-sample.

    Parameters
    ----------
    X_valid, y_valid : conjunto para el early stopping. Si no se pasa
        `X_eval`, tambien es sobre el que se calculan las metricas.
    best_params : params para pisar los del YAML (ej. tras el tuning). Admite
        {modelo: {param: valor}} para usar los mismos en todos los componentes,
        o {modelo: {componente: {param: valor}}} cuando se tuneo cada uno por
        separado (lo habitual). Ver `_resolve_best_params`.
    X_eval, y_eval : conjunto sobre el que se reportan las metricas, separado
        del de early stopping. Usarlo para el test: si el test hace de valid,
        el numero de arboles se elige mirandolo y deja de ser una estimacion
        limpia de generalizacion.

    Returns
    -------
    (summary_df ordenado por NLL del primer componente, resultados por modelo).
    Cada entrada de resultados incluye 'params' con los parametros predichos
    por componente, listos para `DistributionErrorAnalyzer`.
    """
    if (X_eval is None) != (y_eval is None):
        raise ValueError("X_eval y y_eval deben pasarse juntos.")
    guards: Dict[str, float] = config.numeric_guards
    negbin_kwargs: Dict[str, Any] = config.raw.get("negbin_kwargs", {})
    keys: List[str] = config.target.keys

    params_store: Dict[str, Dict[str, Dict[str, NDArray]]] = {
        m: {} for m in models
    }
    y_true_store: Dict[str, NDArray] = {}

    for component in config.target.components:
        key: str = component.key
        feature_list: List[str] = list(features[key])

        X_tr: pd.DataFrame = X_train[key].loc[:, feature_list]
        y_tr: pd.Series = y_train[key]
        X_va: pd.DataFrame = X_valid[key].loc[:, feature_list]
        y_va: pd.Series = y_valid[key]

        # Conjunto sobre el que se reportan las metricas. Coincide con el de
        # early stopping salvo que se pase X_eval/y_eval explicitamente.
        X_ev: pd.DataFrame = (
            X_eval[key].loc[:, feature_list] if X_eval is not None else X_va
        )
        y_ev: pd.Series = y_eval[key] if y_eval is not None else y_va

        y_true_store[key] = np.asarray(y_ev).ravel()

        for model_name in models:
            if verbose:
                print(f"--- Entrenando {model_name} | {key} [{family}] ---")

            X_tr_fit, X_va_fit, X_ev_fit = X_tr, X_va, X_ev
            if needs_imputation(model_name):
                # El imputer se fitea SOLO en train y se aplica al resto.
                X_tr_fit, X_va_fit, imputer = impute_frames(X_tr, X_va)
                if X_eval is not None:
                    # `transform` devuelve DataFrame con columnas en orden
                    # num->cat: se reindexa al orden del train, no se reetiqueta.
                    X_ev_out = cast(pd.DataFrame, imputer.transform(X_ev))
                    X_ev_fit = cast(
                        pd.DataFrame, X_ev_out[list(X_tr.columns)]
                    )
                else:
                    X_ev_fit = X_va_fit

            model_params: Dict[str, Any] = dict(config.model_params(model_name))
            model_params.update(
                _resolve_best_params(best_params, model_name, key, keys)
            )

            model = train_model(
                model_name, family, X_tr_fit, y_tr, X_va_fit, y_va,
                params=model_params,
                fit_params=config.fit_params(model_name),
                negbin_kwargs=negbin_kwargs,
            )

            raw: Dict[str, NDArray] = predict_params(
                model, X_ev_fit, model_name, family,
            )
            params_store[model_name][key] = clip_params(raw, family, guards)

    # ── Evaluacion conjunta por modelo ───────────────────────────────────────
    all_results: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    for model_name in models:
        results: Dict[str, Any] = evaluate_distribution_model(
            y_true=y_true_store,
            params=params_store[model_name],
            keys=keys,
            family=family,
            label=f"{model_name} | {family}",
            max_k=config.target.max_k,
            thresholds=config.target.thresholds,
            total_method=config.target.total_method,  # type: ignore[arg-type]
            verbose=verbose,
        )
        # Los parametros predichos alimentan DistributionErrorAnalyzer sin
        # tener que reentrenar el modelo solo para inspeccionar el error.
        results["params"] = params_store[model_name]
        all_results[model_name] = results

        row: Dict[str, Any] = {"model": model_name, "family": family}
        for key in keys:
            row[f"nll_{key}"] = results["nll"][key]
        for key in keys:
            row[f"rps_{key}"] = results["rps"][key]["rps_mean"]
        row["ece_global"] = results["ece_global"]
        rows.append(row)

    summary_df: pd.DataFrame = pd.DataFrame(rows)
    sort_col: str = f"nll_{keys[0]}"
    if sort_col in summary_df.columns:
        summary_df = summary_df.sort_values(sort_col).reset_index(drop=True)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  VALIDATION SUMMARY [{family}] - ordenado por {sort_col}")
        print(f"{'=' * 70}")
        print(summary_df.to_string(index=False,
                                   float_format=lambda x: f"{x:.5f}"))
        print(f"{'=' * 70}")

    return summary_df, all_results

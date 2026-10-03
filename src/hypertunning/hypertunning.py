import yaml
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union
import catboost as cb
import lightgbm as lgb
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.training.training import timeseries_cv_evaluate



def load_optuna_spaces(
    yaml_path: Union[str, Path],
) -> Dict[str, Dict[str, Any]]:
    """Carga los espacios de búsqueda de Optuna desde un archivo YAML."""
    path: Path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(
            f"El archivo de configuración no existe en: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        spaces: Dict[str, Dict[str, Any]] = yaml.safe_load(f)

    return spaces


# Nombres canonicos de modelo, indexados por su forma en minusculas. Los YAML se
# escriben como best_params_xgboost.yaml, pero train_and_evaluate_hardcoded
# indexa por "XGBoost", asi que hay que reconstruir la capitalizacion.
_CANONICAL_MODEL_NAMES: Dict[str, str] = {
    "catboost": "CatBoost",
    "extratrees": "ExtraTrees",
    "lightgbm": "LightGBM",
    "logisticregression": "LogisticRegression",
    "xgboost": "XGBoost",
}

# Parametros que train_and_evaluate_hardcoded ya inyecta al construir cada
# modelo. Si tambien vinieran del YAML, Python lanzaria
# "got multiple values for keyword argument".
_INJECTED_PARAMS: Dict[str, List[str]] = {
    "CatBoost": ["loss_function", "random_seed", "verbose"],
    "ExtraTrees": ["random_state", "n_jobs"],
    "LightGBM": ["objective", "num_class", "random_state", "n_jobs"],
    "LogisticRegression": [],
    "XGBoost": [
        "objective", "num_class", "tree_method", "random_state", "n_jobs",
    ],
}


def load_best_params(
    config_dir: Union[str, Path],
    models: Optional[List[str]] = None,
    strict: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Carga los best_params_*.yaml exportados por export_optuna_tuning_results.

    Devuelve un dict listo para pasar como `best_params_dict` a
    `train_and_evaluate_hardcoded`, con las claves ya en su forma canonica
    ("XGBoost", "LightGBM", ...) y sin los parametros que esa funcion inyecta
    por su cuenta.

    Args:
        config_dir: carpeta donde viven los best_params_*.yaml.
        models: nombres canonicos a cargar. Si es None, carga todo lo que haya.
        strict: si True, falla cuando un modelo pedido no tiene YAML.
    """
    directory: Path = Path(config_dir)
    if not directory.exists():
        raise FileNotFoundError(f"El directorio no existe: {directory}")

    best_params: Dict[str, Dict[str, Any]] = {}

    path: Path
    for path in sorted(directory.glob("best_params_*.yaml")):
        stem: str = path.stem.replace("best_params_", "")
        model_name: str = _CANONICAL_MODEL_NAMES.get(stem, stem)

        if models is not None and model_name not in models:
            continue

        with open(path, "r", encoding="utf-8") as f:
            params: Optional[Dict[str, Any]] = yaml.safe_load(f)

        if not params:
            print(f" [WARN] '{path.name}' esta vacio; se omite.")
            continue

        # Quitar lo que la funcion de entrenamiento ya pasa explicitamente.
        injected: List[str] = _INJECTED_PARAMS.get(model_name, [])
        dropped: List[str] = [k for k in params if k in injected]
        clean: Dict[str, Any] = {
            k: v for k, v in params.items() if k not in injected
        }

        if dropped:
            print(f" [INFO] {model_name}: ignorados {dropped} (los fija el entrenador).")

        best_params[model_name] = clean
        print(f" [OK] {model_name}: {len(clean)} parametros <- {path.name}")

    if models is not None:
        faltantes: List[str] = [m for m in models if m not in best_params]
        if faltantes:
            msg: str = f"Sin YAML de best params para: {faltantes} en {directory}"
            if strict:
                raise FileNotFoundError(msg)
            print(f" [WARN] {msg}")

    if not best_params:
        raise FileNotFoundError(
            f"No se encontro ningun 'best_params_*.yaml' en {directory}. "
            "Corre primero el tuning y export_optuna_tuning_results()."
        )

    return best_params


def create_optuna_study(
    direction: str = "minimize",
    random_state: int = 67,
) -> optuna.Study:
    """Crea e inicializa un objeto Study de Optuna con TPE Sampler y Median

    Pruner.
    """
    sampler: TPESampler = TPESampler(seed=random_state, multivariate=True, group=True)

    pruner: MedianPruner = MedianPruner(
        n_startup_trials=20, n_warmup_steps=2, interval_steps=1
    )

    study: optuna.Study = optuna.create_study(
        direction=direction, sampler=sampler, pruner=pruner
    )

    return study

def create_optuna_objective(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    optuna_spaces: Dict[str, Dict[str, Any]],
    n_splits: int = 5,
    cat_cols: Optional[List[str]] = None,
    random_state: int = 67
) -> Callable[[optuna.Trial], float]:
    """Fabrica la función objetivo de Optuna para un modelo específico usando el

    espacio de búsqueda del YAML.
    """
    actual_cat_cols: List[str] = cat_cols if cat_cols is not None else []

    def objective(trial: optuna.Trial) -> float:
        model_cls: Any
        params: Dict[str, Any] = {}
        fit_params: Dict[str, Any] = {}

        # ===============================
        # CATBOOST
        # ===============================
        if model_name == "CatBoost":
            spaces: Dict[str, Any] = optuna_spaces["CatBoost"]

            params = {
                "iterations": trial.suggest_int(
                    "iterations", *spaces["iterations"]
                ),
                "depth": trial.suggest_int("depth", *spaces["depth"]),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *spaces["learning_rate"], log=True
                ),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", *spaces["l2_leaf_reg"], log=True
                ),
                "random_strength": trial.suggest_float(
                    "random_strength", *spaces["random_strength"], log=True
                ),
                "bagging_temperature": trial.suggest_float(
                    "bagging_temperature", *spaces["bagging_temperature"]
                ),
                "border_count": trial.suggest_int(
                    "border_count", *spaces["border_count"]
                ),
                "leaf_estimation_iterations": trial.suggest_int(
                    "leaf_estimation_iterations",
                    *spaces["leaf_estimation_iterations"],
                ),
                "min_data_in_leaf": trial.suggest_int(
                    "min_data_in_leaf", *spaces["min_data_in_leaf"]
                ),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", spaces["grow_policy"]
                ),
                "loss_function": "MultiClass",
                "classes_count": 3,
                "verbose": 0,
                "random_state": random_state,
            }

            model_cls = cb.CatBoostClassifier
            cat_indices: List[int] = [
                X.columns.get_loc(c) for c in actual_cat_cols if c in X.columns
            ]
            fit_params = {"cat_features": cat_indices} if cat_indices else {}

        # ===============================
        # XGBOOST
        # ===============================
        elif model_name == "XGBoost":
            spaces: Dict[str, Any] = optuna_spaces["XGBoost"]

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", *spaces["n_estimators"]
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", *spaces["max_depth"]
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *spaces["learning_rate"], log=True
                ),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", *spaces["min_child_weight"]
                ),
                "subsample": trial.suggest_float(
                    "subsample", *spaces["subsample"]
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *spaces["colsample_bytree"]
                ),
                "colsample_bylevel": trial.suggest_float(
                    "colsample_bylevel", *spaces["colsample_bylevel"]
                ),
                "colsample_bynode": trial.suggest_float(
                    "colsample_bynode", *spaces["colsample_bynode"]
                ),
                "gamma": trial.suggest_float(
                    "gamma", *spaces["gamma"], log=True
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", *spaces["reg_alpha"], log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", *spaces["reg_lambda"], log=True
                ),
                "max_delta_step": trial.suggest_int(
                    "max_delta_step", *spaces["max_delta_step"]
                ),
                "max_bin": trial.suggest_int("max_bin", *spaces["max_bin"]),
                "grow_policy": trial.suggest_categorical(
                    "grow_policy", spaces["grow_policy"]
                ),
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": 3,
                "tree_method": "hist",
                "verbosity": 0,
                "random_state": random_state,
            }

            model_cls = xgb.XGBClassifier
            fit_params = {}

        # ===============================
        # LIGHTGBM
        # ===============================
        elif model_name == "LightGBM":
            spaces: Dict[str, Any] = optuna_spaces["LightGBM"]

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", *spaces["n_estimators"]
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", *spaces["learning_rate"], log=True
                ),
                "num_leaves": trial.suggest_int(
                    "num_leaves", *spaces["num_leaves"]
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", *spaces["max_depth"]
                ),
                "min_child_samples": trial.suggest_int(
                    "min_child_samples", *spaces["min_child_samples"]
                ),
                "subsample": trial.suggest_float(
                    "subsample", *spaces["subsample"]
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", *spaces["colsample_bytree"]
                ),
                "feature_fraction_bynode": trial.suggest_float(
                    "feature_fraction_bynode",
                    *spaces["feature_fraction_bynode"],
                ),
                "reg_alpha": trial.suggest_float(
                    "reg_alpha", *spaces["reg_alpha"], log=True
                ),
                "reg_lambda": trial.suggest_float(
                    "reg_lambda", *spaces["reg_lambda"], log=True
                ),
                "min_split_gain": trial.suggest_float(
                    "min_split_gain", *spaces["min_split_gain"], log=True
                ),
                "path_smooth": trial.suggest_float(
                    "path_smooth", *spaces["path_smooth"]
                ),
                "max_bin": trial.suggest_int("max_bin", *spaces["max_bin"]),
                "extra_trees": trial.suggest_categorical(
                    "extra_trees", spaces["extra_trees"]
                ),
                "objective": "multiclass",
                "num_class": 3,
                "verbosity": -1,
                "random_state": random_state,
                "n_jobs": -1,
            }

            model_cls = lgb.LGBMClassifier
            fit_params = (
                {"categorical_feature": actual_cat_cols}
                if actual_cat_cols
                else {}
            )

        # ===============================
        # EXTRATREES
        # ===============================
        elif model_name == "extra" or model_name == "ExtraTrees":
            spaces: Dict[str, Any] = optuna_spaces["ExtraTrees"]

            bootstrap: bool = trial.suggest_categorical(
                "bootstrap", spaces["bootstrap"]
            )

            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", *spaces["n_estimators"]
                ),
                "max_depth": trial.suggest_int(
                    "max_depth", *spaces["max_depth"]
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", *spaces["min_samples_split"]
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf", *spaces["min_samples_leaf"]
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", spaces["max_features"]
                ),
                "bootstrap": bootstrap,
                "random_state": random_state,
                "n_jobs": -1,
            }

            if bootstrap:
                params["max_samples"] = trial.suggest_float(
                    "max_samples", *spaces["max_samples"]
                )

            model_cls = ExtraTreesClassifier
            fit_params = {}

        # ===============================
        # LOGISTIC REGRESSION
        # ===============================
        elif model_name == "Logistic":
            penalty: str = trial.suggest_categorical(
                "penalty", ["l1", "l2", "elasticnet"]
            )
            C: float = float(trial.suggest_float("C", 1e-4, 10.0, log=True))

            l1_ratio: Optional[float] = None
            if penalty == "elasticnet":
                solver: str = "saga"
                l1_ratio = float(trial.suggest_float("l1_ratio", 0.0, 1.0))
            elif penalty == "l1":
                solver = "saga"
            else:
                solver = "lbfgs"

            params = {
                "penalty": penalty,
                "C": C,
                "solver": solver,
                "max_iter": 5000,
                "n_jobs": -1,
                "random_state": random_state,
            }
            if l1_ratio is not None:
                params["l1_ratio"] = l1_ratio

            def model_pipeline_factory(**kwargs: Any) -> Pipeline:
                return Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="mean")),
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(**kwargs)),
                    ]
                )

            model_cls = model_pipeline_factory
            fit_params = {}

        else:
            raise ValueError(f"Modelo no soportado: {model_name}")

        # ===============================
        # EVALUACIÓN
        # ===============================
        try:
            metrics: pd.DataFrame
            _: pd.DataFrame
            metrics, _ = timeseries_cv_evaluate(
                model_cls=model_cls,
                X=X,
                y=y,
                n_splits=n_splits,
                model_params=params,
                fit_params=fit_params,
                verbose=False,
            )

            mean_rps: float = float(
                metrics[metrics["fold"] == "Mean"]["rps"].values[0]
            )

            if not np.isfinite(mean_rps):
                return 0.99

            return mean_rps

        except Exception as e:
            print(f"[DEBUG] Error en trial ({model_name}): {e}")
            return 0.99

    return objective

def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte tipos de NumPy/Optuna a tipos nativos de Python (int, float,

    bool) para asegurar compatibilidad total con JSON y YAML.
    """
    clean_params: Dict[str, Any] = {}
    key: str
    val: Any
    for key, val in params.items():
        # bool va primero: isinstance(True, int) es True en Python, asi que el
        # orden inverso degradaria los booleanos a 1/0 y sklearn los rechaza
        # ("The 'bootstrap' parameter ... must be an instance of 'bool'").
        if isinstance(val, (np.bool_, bool)):
            clean_params[key] = bool(val)
        elif isinstance(val, (np.integer, int)):
            clean_params[key] = int(val)
        elif isinstance(val, (np.floating, float)):
            clean_params[key] = float(val)
        else:
            clean_params[key] = val
    return clean_params


def export_optuna_tuning_results(
    studies_map: Dict[str, optuna.Study],
    CONFIG_DIR: Union[str, Path],
    METRICS_DIR: Union[str, Path]
) -> None:
    """Exporta los resultados del tuning de Optuna:

    1. Un archivo JSON único con el resumen de métricas (best_rps) y parámetros.
    2. Archivos YAML individuales de configuración con los mejores
    hiperparámetros de cada modelo.
    """

    json_results: Dict[str, Dict[str, Any]] = {}

    print("\n" + "=" * 60)
    print(" EXPORTANDO RESULTADOS DE OPTUNA Y CONFIGURACIONES YAML")
    print("=" * 60)

    model_name: str
    study_obj: optuna.Study
    for model_name, study_obj in studies_map.items():
        best_value: float = float(study_obj.best_value)
        best_params: Dict[str, Any] = _sanitize_params(study_obj.best_params)

        # 1. Estructurar la entrada para el archivo JSON unificado
        json_results[model_name] = {
            "best_rps": round(best_value, 6),
            "best_params": best_params,
        }

        # 2. Exportar archivo YAML individual con los mejores parámetros
        yaml_filename: str = f"best_params_{model_name.lower()}.yaml"
        yaml_path: Path = CONFIG_DIR / yaml_filename

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                best_params, f, default_flow_style=False, sort_keys=False
            )

        print(
            f" [YAML] Parámetros guardados para '{model_name}' -> {yaml_path.name}"
        )

    # 3. Exportar archivo JSON unificado con las métricas de validación
    json_path: Path = METRICS_DIR / "tunned_models_validation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=4)

    print("-" * 60)
    print(
        f"[EXITO] Resumen de validación guardado en JSON:\n -> {json_path}\n"
    )
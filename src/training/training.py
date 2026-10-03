"""Entrenamiento de modelos: CV temporal, target encoding sin leakage y
entrenamiento con hiperparámetros fijos.

Este módulo ENTRENA. Las métricas que operan sobre predicciones ya calculadas
viven en `src.metrics.metrics` y `src.evaluation.evaluation`.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.evaluation import get_vectorized_rps
from src.metrics.metrics import evaluate_model


def train_and_evaluate_lgbm(
    features: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "Model",
    RANDOM_STATE: int = 67,
) -> Dict[str, float | int]:

    # 1. Filtrar features datetime
    valid_features: List[str] = [
        f for f in features
        if not pd.api.types.is_datetime64_any_dtype(X_train[f])
    ]

    print(f"\n--- Entrenando {model_name} ---")
    print(f"Features válidas: {len(valid_features)}")

    X_tr: pd.DataFrame = X_train[valid_features].copy()
    X_vl: pd.DataFrame = X_val[valid_features].copy()

    # 2. Detectar columnas categóricas
    cat_cols: List[str] = X_tr.select_dtypes(
        include=["category", "object", "string"]
    ).columns.tolist()

    col: str
    # 3. Convertir categóricas a códigos enteros (Alineación manual)
    for col in cat_cols:
        # Unificar categorías usando solo las del train
        categories = X_tr[col].astype("category").cat.categories
        X_tr[col] = pd.Categorical(X_tr[col], categories=categories).codes
        X_vl[col] = pd.Categorical(X_vl[col], categories=categories).codes
        # Forzar a int8 para eliminar metadatos de 'category'
        X_tr[col] = X_tr[col].astype(np.int8)
        X_vl[col] = X_vl[col].astype(np.int8)

    # Asegurar que todas las columnas sean numéricas para evitar detección automática
    X_tr = X_tr.apply(pd.to_numeric, errors="coerce").fillna(0)
    X_vl = X_vl.apply(pd.to_numeric, errors="coerce").fillna(0)

    # 4. LightGBM datasets — Forzamos categorical_feature=[] para evitar la
    #    validación interna de pandas_categorical
    trn_data: lgb.Dataset = lgb.Dataset(
        X_tr,
        label=y_train,
        categorical_feature=[],
        free_raw_data=False
    )
    val_data: lgb.Dataset = lgb.Dataset(
        X_vl,
        label=y_val,
        reference=trn_data,
        categorical_feature=[],
        free_raw_data=False
    )

    # 5. Params
    params: Dict[str, str | int | float] = {
        "objective": "multiclass",
        "num_class": 3,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": 6,
        "metric": "multi_logloss",
        "feature_fraction": 0.8,
        "subsample": 0.8,
        "seed": RANDOM_STATE,
        "verbose": -1,
    }

    # 6. Train
    start_time: float = time.time()
    model = lgb.train(
        params,
        trn_data,
        valid_sets=[trn_data, val_data],
        valid_names=["train", "val"],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )
    train_time: float = time.time() - start_time

    # 7. Predict
    y_val_prob: np.ndarray = model.predict(X_vl, num_iteration=model.best_iteration)

    # 8. Metrics
    metrics = evaluate_model(y_true=y_val.values, y_prob=y_val_prob)
    metrics["best_iteration"] = model.best_iteration
    metrics["train_time_sec"] = round(train_time, 2)

    print(
        f"\nRESULTADOS: RPS: {metrics['rps']:.5f} | "
        f"LogLoss: {metrics['log_loss']:.5f} | Acc: {metrics['accuracy']:.4f}"
    )

    return metrics


def timeseries_cv_evaluate(
    model_cls: Union[Any, Callable[..., Any]],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    model_params: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Any]] = None,
    target_encode_col: Optional[str] = None,
    te_smoothing: float = 20.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Robust TimeSeries CV that returns metrics and Out-of-Fold predictions.

    Now supports leakage-free target encoding.

    Args:
        model_cls: The class or factory of the model (e.g., lgb.LGBMClassifier)
        X: Feature DataFrame
        y: Target Series
        n_splits: Number of splits for TimeSeriesSplit
        model_params: Dict passed to model_cls(..., **model_params)
        fit_params: Dict passed to model.fit(..., **fit_params)
        target_encode_col: Column name to apply target encoding (e.g.,
          'tournament')
        te_smoothing: Smoothing parameter for target encoding
        verbose: If True, prints fold progress and summary.

    Returns:
        Tuple containing:
        1. metrics_df: DataFrame with scores for each fold + summary stats.
        2. oof_df: DataFrame containing the validation predictions for all
        folds.
    """

    # Defaults
    actual_model_params: Dict[str, Any] = model_params or {}
    actual_fit_params: Dict[str, Any] = fit_params or {}

    tscv: TimeSeriesSplit = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics: List[Dict[str, Any]] = []
    oof_list: List[pd.DataFrame] = []

    if verbose:
        print(f"{'='*20} Starting TimeSeries CV ({n_splits} splits) {'='*20}")

    train_idx: np.ndarray
    val_idx: np.ndarray
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        # 1. Split Data
        X_tr: pd.DataFrame = X.iloc[train_idx].copy()
        X_val: pd.DataFrame = X.iloc[val_idx].copy()
        y_tr: pd.Series = y.iloc[train_idx]
        y_val: pd.Series = y.iloc[val_idx]

        # 2. Apply Target Encoding (if specified)
        if target_encode_col is not None:
            # Create a combined dataframe for training with proper indexing
            train_df: pd.DataFrame = pd.DataFrame(
                {
                    target_encode_col: X_tr[target_encode_col].values,
                    "_target_": y_tr.values,
                },
                index=pd.Index(range(len(X_tr))),
            )

            # Calculate global mean
            global_mean: float = float(train_df["_target_"].mean())

            # Apply expanding window target encoding on training
            te_feature_name: str = f"{target_encode_col}_te"
            train_te_values: List[float] = []

            idx: int
            for idx in range(len(train_df)):
                cat_value: Any = train_df.loc[idx, target_encode_col]

                # Get all previous rows with same category
                if idx == 0:
                    train_te_values.append(global_mean)
                else:
                    prev_data: pd.DataFrame = train_df.iloc[:idx]
                    cat_mask: pd.Series = (
                        prev_data[target_encode_col] == cat_value
                    )
                    cat_data: pd.DataFrame = prev_data[cat_mask]

                    if len(cat_data) == 0:
                        # First occurrence of this category
                        train_te_values.append(global_mean)
                    else:
                        # Calculate smoothed mean
                        cat_sum: float = float(cat_data["_target_"].sum())
                        cat_count: int = len(cat_data)
                        smoothed_mean: float = (
                            cat_sum + te_smoothing * global_mean
                        ) / (cat_count + te_smoothing)
                        train_te_values.append(smoothed_mean)

            X_tr[te_feature_name] = train_te_values

            # For validation: use full training statistics (leakage-free)
            train_stats: pd.DataFrame = (
                train_df.groupby(target_encode_col)["_target_"]
                .agg(["sum", "count"])
                .reset_index()
            )
            train_stats.columns = pd.Index([target_encode_col, "sum", "count"])

            # Merge validation data with training statistics
            val_df: pd.DataFrame = pd.DataFrame(
                {target_encode_col: X_val[target_encode_col].values}
            )
            val_merged: pd.DataFrame = val_df.merge(
                train_stats, on=target_encode_col, how="left"
            )

            # Calculate smoothed encoding for validation
            val_merged["sum"] = val_merged["sum"].fillna(0)
            val_merged["count"] = val_merged["count"].fillna(0)

            val_te_values: pd.Series = (
                val_merged["sum"] + te_smoothing * global_mean
            ) / (val_merged["count"] + te_smoothing)

            X_val[te_feature_name] = val_te_values.values

            # DROP the original categorical column after encoding
            X_tr = X_tr.drop(columns=[target_encode_col])
            X_val = X_val.drop(columns=[target_encode_col])

        # 3. Initialize and Fit
        model: Any = model_cls(**actual_model_params)
        model.fit(X_tr, y_tr, **actual_fit_params)

        # 4. Predict
        y_prob: np.ndarray
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)
        else:
            raise ValueError(
                "Model must support predict_proba for RPS/LogLoss calculation"
            )

        # 5. Evaluate Fold
        metrics: Dict[str, Any] = evaluate_model(y_val.values, y_prob)
        metrics["fold"] = fold
        metrics["train_size"] = len(train_idx)
        metrics["val_size"] = len(val_idx)
        fold_metrics.append(metrics)

        # 6. Store OOF Predictions
        fold_oof: pd.DataFrame = pd.DataFrame(
            y_prob,
            index=X_val.index,
            columns=[f"prob_{c}" for c in range(y_prob.shape[1])],
        )
        fold_oof["y_true"] = y_val.values
        fold_oof["fold"] = fold
        oof_list.append(fold_oof)

        if verbose:
            print(
                f"Fold {fold}/{n_splits} | "
                f"Train: {len(train_idx)} Val: {len(val_idx)} | "
                f"RPS: {metrics['rps']:.4f} | LogLoss: {metrics['log_loss']:.4f}"
            )

    # --- Aggregation ---
    metrics_df: pd.DataFrame = pd.DataFrame(fold_metrics)
    oof_df: pd.DataFrame = pd.concat(oof_list).sort_index()

    # Calculate Global Aggregate Score
    global_probs: np.ndarray = oof_df[
        [c for c in oof_df.columns if c.startswith("prob_")]
    ].values
    global_true: np.ndarray = oof_df["y_true"].values.astype(int)

    global_metrics: Dict[str, Any] = evaluate_model(global_true, global_probs)

    # Add summary row to metrics_df
    summary_stats: Dict[str, Any] = (
        metrics_df.drop(columns=["fold"]).mean().to_dict()
    )
    summary_stats["fold"] = "Mean"
    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([summary_stats])], ignore_index=True
    )

    std_stats: Dict[str, Any] = (
        metrics_df.iloc[:-1].drop(columns=["fold"]).std().to_dict()
    )
    std_stats["fold"] = "Std"
    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([std_stats])], ignore_index=True
    )

    if verbose:
        print(f"{'='*20} CV Complete {'='*20}")
        print(
            f"Average RPS:      {summary_stats['rps']:.4f} ±"
            f" {std_stats['rps']:.4f}"
        )
        print(f"Average Accuracy: {summary_stats['accuracy']:.4f}")
        print(f"Global RPS (OOF): {global_metrics['rps']:.4f}")

    return metrics_df, oof_df


def apply_target_encoding_to_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    target_encode_col: str = "tournament",
    te_smoothing: float = 10.0,
) -> pd.DataFrame:
    """Apply target encoding to validation set using statistics from training

    set.

    This is a leakage-free transformation that uses ONLY training data
    statistics.

    Args:
        X_train: Training features (must include target_encode_col)
        y_train: Training target
        X_val: Validation features (must include target_encode_col)
        target_encode_col: Column name to encode (e.g., 'tournament')
        te_smoothing: Smoothing parameter for Bayesian encoding

    Returns:
        X_val_encoded: Validation set with target encoding applied
    """

    # Calculate global mean from training data
    global_mean: float = float(y_train.mean())

    # Calculate category statistics from training data
    train_df: pd.DataFrame = pd.DataFrame(
        {
            target_encode_col: X_train[target_encode_col].values,
            "_target_": y_train.values,
        }
    )

    train_stats: pd.DataFrame = (
        train_df.groupby(target_encode_col)["_target_"]
        .agg([("sum", "sum"), ("count", "count")])
        .reset_index()
    )

    # Create encoded validation set
    X_val_encoded: pd.DataFrame = X_val.copy()

    # Merge validation data with training statistics
    val_df: pd.DataFrame = pd.DataFrame(
        {target_encode_col: X_val[target_encode_col].values}
    )

    val_merged: pd.DataFrame = val_df.merge(
        train_stats, on=target_encode_col, how="left"
    )

    # Fill NaN for categories not seen in training
    val_merged["sum"] = val_merged["sum"].fillna(0)
    val_merged["count"] = val_merged["count"].fillna(0)

    # Calculate smoothed encoding
    te_feature_name: str = f"{target_encode_col}_te"
    val_merged[te_feature_name] = (
        val_merged["sum"] + te_smoothing * global_mean
    ) / (val_merged["count"] + te_smoothing)

    # Add encoded feature to validation set
    X_val_encoded[te_feature_name] = val_merged[te_feature_name].values

    # Drop original categorical column
    X_val_encoded = X_val_encoded.drop(columns=[target_encode_col])

    print("  Target encoding applied to validation set:")
    print(f"   Original column: {target_encode_col}")
    print(f"   New column: {te_feature_name}")
    print(f"   Global mean: {global_mean:.4f}")
    print(f"   Categories in train: {len(train_stats)}")
    print(f"   Unique categories in val: {X_val[target_encode_col].nunique()}")

    return X_val_encoded


# ==============================================================================
# 1. LOGISTIC REGRESSION (The "Anchor" Model)
# ==============================================================================
def run_logistic_cv(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Pipeline:
    """Trains a Logistic Regression model with TimeSeriesSplit CV and computes

    RPS and LogLoss metrics.
    """
    print("\n" + "=" * 60)
    print("Training Logistic Regression with TimeSeriesCV...")
    print("=" * 60)

    tscv: TimeSeriesSplit = TimeSeriesSplit(n_splits=n_splits)

    # Pipeline: Impute -> Scale -> Model
    # C=0.05 is moderate regularization (prevents overfitting on noise)
    pipeline: Pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    solver="saga",
                    penalty="elasticnet",
                    l1_ratio=0.5,
                    C=0.05,
                    max_iter=3000,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rps_scores: List[float] = []
    log_losses: List[float] = []

    fold: int = 1
    train_index: np.ndarray
    val_index: np.ndarray

    for train_index, val_index in tscv.split(X):
        # Data split
        X_tr: pd.DataFrame = X.iloc[train_index]
        X_val: pd.DataFrame = X.iloc[val_index]
        y_tr: pd.Series = y.iloc[train_index]
        y_val: pd.Series = y.iloc[val_index]

        # Fit Pipeline
        pipeline.fit(X_tr, y_tr)

        # Predict Probs
        val_probs: np.ndarray = pipeline.predict_proba(X_val)

        # Metrics
        fold_rps: float = float(
            np.mean(get_vectorized_rps(val_probs, y_val.values))
        )
        fold_ll: float = float(sk_log_loss(y_val, val_probs))

        print(
            f"Fold {fold}/{n_splits} | Train: {len(X_tr)} Val: {len(X_val)} | "
            f"RPS: {fold_rps:.4f} | LogLoss: {fold_ll:.4f}"
        )

        rps_scores.append(fold_rps)
        log_losses.append(fold_ll)
        fold += 1

    print("-" * 60)
    print("Logistic Regression CV Summary:")
    print(
        f"Mean RPS:      {np.mean(rps_scores):.4f} ± {np.std(rps_scores):.4f}"
    )
    print(f"Mean Log Loss: {np.mean(log_losses):.4f}")

    return pipeline


def train_and_evaluate_hardcoded(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: Optional[List[str]],
    best_params_dict: Dict[str, Dict[str, Any]],
    RANDOM_STATE: int = 67,
) -> Dict[str, Any]:
    """
    Train model using hardcoded hyperparameters.

    Returns:
        dict with keys: 'model' (trained model instance),
                        'val_rps' (float),
                        'val_probs' (np.ndarray)
    """

    best_params: Dict[str, Any] = best_params_dict[model_name]

    # ======================================================
    # LOGISTIC REGRESSION
    # ======================================================
    if model_name == "LogisticRegression":

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                **best_params
            ))
        ])

        model.fit(X_train, y_train)
        val_probs: np.ndarray = model.predict_proba(X_val)

    # ======================================================
    # EXTRA TREES
    # ======================================================
    elif model_name == "ExtraTrees":

        model = ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_params
        )

        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)

    # ======================================================
    # XGBOOST
    # ======================================================
    elif model_name == "XGBoost":

        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(np.unique(y_train)),
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_params
        )

        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)

    # ======================================================
    # LIGHTGBM
    # ======================================================
    elif model_name == "LightGBM":

        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(np.unique(y_train)),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **best_params
        )

        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)

    # ======================================================
    # CATBOOST
    # ======================================================
    elif model_name == "CatBoost":

        # Mapping to native CatBoost parameter names
        model = CatBoostClassifier(
            loss_function="MultiClass",
            random_seed=RANDOM_STATE,
            verbose=False,
            **best_params
        )

        model.fit(X_train, y_train)
        val_probs = model.predict_proba(X_val)

    else:
        raise ValueError(f"{model_name} not supported.")

    # Compute RPS
    val_rps: float = float(np.mean(get_vectorized_rps(val_probs, y_val.values)))

    return {
        "model": model,
        "val_rps": val_rps,
        "val_probs": val_probs
    }

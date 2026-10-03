"""Entrenamiento y extraccion de parametros para modelos de distribucion.

Unifica la API de LightGBMLSS, XGBoostLSS y NGBoost detras de una interfaz
comun: cada backend sabe entrenarse y devolver los parametros predichos
normalizados a la convencion NB2 ({'lambda'} para Poisson, {'mu', 'alpha'}
para NegBin).

Agnostico al target: no asume goles ni la estructura home/away.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

import lightgbm as lgb
from lightgbmlss.model import LightGBMLSS
from lightgbmlss.distributions.Poisson import Poisson as LGBMLSSPoisson
from lightgbmlss.distributions.NegativeBinomial import (
    NegativeBinomial as LGBMLSSNB,
)

import xgboost as xgb
from xgboostlss.model import XGBoostLSS
from xgboostlss.distributions.Poisson import Poisson as XGBLSSPoisson
from xgboostlss.distributions.NegativeBinomial import (
    NegativeBinomial as XGBLSSNB,
)

from ngboost import NGBRegressor
from ngboost import scores as ngb_scores
from ngboost.distns import NegativeBinomial as NGBNegBin
from ngboost.distns import Poisson as NGBPoisson
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor

from src.metrics.distribution_metrics import DistFamily

MODEL_NAMES: Tuple[str, ...] = ("LightGBMLSS", "XGBoostLSS", "NGBoost")

# Modelos que no toleran NaN y necesitan imputacion previa.
_NEEDS_IMPUTATION: Tuple[str, ...] = ("NGBoost",)


# =============================================================================
# PATCH - NGBoost: gradiente natural con shapes inconsistentes
# =============================================================================

def _patched_grad(self, Y: Any, natural: bool = True) -> NDArray:
    """Parche al gradiente natural de NGBoost.

    La implementacion original asume que `metric()` devuelve siempre una matriz
    de Fisher cuadrada por observacion. Con distribuciones de un solo parametro
    (Poisson) devuelve shapes que rompen `np.linalg.solve`. Este parche cubre
    los casos escalares y (N, 1, 1) dividiendo directamente.
    """
    grad = self.d_score(Y)
    if not natural:
        return grad

    metric_arr: NDArray = np.asarray(self.metric())
    grad_flat: NDArray = np.asarray(grad).ravel()

    if metric_arr.size == 1:
        result = grad_flat / (float(metric_arr.ravel()[0]) + 1e-8)
    elif metric_arr.ndim == 2 and metric_arr.shape == (1, 1):
        result = grad_flat / (float(metric_arr[0, 0]) + 1e-8)
    elif metric_arr.ndim == 3 and metric_arr.shape[1] == 1:
        result = grad_flat / (metric_arr[:, 0, 0] + 1e-8)
    elif metric_arr.ndim == 2 and metric_arr.shape[0] == metric_arr.shape[1]:
        result = np.linalg.solve(metric_arr, grad_flat)
    else:
        result = grad_flat / (float(np.mean(np.abs(metric_arr))) + 1e-8)

    return result.reshape(grad.shape)


def apply_ngboost_patch() -> None:
    """Aplica el parche del gradiente natural. Idempotente."""
    ngb_scores.Score.grad = _patched_grad  # type: ignore[method-assign]


apply_ngboost_patch()


# =============================================================================
# Guardas numericas
# =============================================================================

def clip_params(
    params: Dict[str, NDArray],
    family: DistFamily,
    guards: Optional[Dict[str, float]] = None,
) -> Dict[str, NDArray]:
    """Sanea los parametros predichos: sin NaN/inf ni valores degenerados.

    Parameters
    ----------
    guards : {'mu_min', 'alpha_min', 'alpha_max'} desde el YAML.
    """
    g: Dict[str, float] = guards or {}
    mu_min: float = float(g.get("mu_min", 1e-6))
    alpha_min: float = float(g.get("alpha_min", 1e-6))
    alpha_max: float = float(g.get("alpha_max", 100.0))

    def _clean(arr: NDArray, lo: float, hi: float = np.inf) -> NDArray:
        vals: NDArray = np.asarray(arr, dtype=float).ravel()
        # NaN/inf -> cota inferior: un parametro invalido no debe propagarse.
        vals = np.where(np.isfinite(vals), vals, lo)
        return np.clip(vals, lo, hi)

    if family == "poisson":
        return {"lambda": _clean(params["lambda"], mu_min)}

    return {
        "mu": _clean(params["mu"], mu_min),
        # alpha->0 colapsa a Poisson; alpha muy grande explota la varianza.
        "alpha": _clean(params["alpha"], alpha_min, alpha_max),
    }


# =============================================================================
# Extraccion de parametros - normalizada a la convencion NB2
# =============================================================================

def _nb_from_columns(preds: pd.DataFrame, backend: str) -> Dict[str, NDArray]:
    """Normaliza la salida NegBin de los backends LSS a {'mu', 'alpha'}.

    Las versiones de lightgbmlss/xgboostlss difieren en la parametrizacion que
    exponen: (total_count, probs) o (concentration, rate).
    """
    cols = preds.columns

    if "total_count" in cols and "probs" in cols:
        n: NDArray = np.asarray(preds["total_count"], dtype=float).ravel()
        p: NDArray = np.asarray(preds["probs"], dtype=float).ravel()
        return {"mu": n * (1.0 - p) / p, "alpha": 1.0 / n}

    if "concentration" in cols and "rate" in cols:
        # Parametrizacion gamma-poisson: concentration=r, rate=beta
        r: NDArray = np.asarray(preds["concentration"], dtype=float).ravel()
        beta: NDArray = np.asarray(preds["rate"], dtype=float).ravel()
        return {"mu": r / beta, "alpha": 1.0 / r}

    raise ValueError(
        f"Columnas inesperadas de {backend} NegBin: {list(cols)}. "
        f"Inspeccionar con print(preds.columns) y extender este helper."
    )


def predict_params(
    model: Any,
    X: pd.DataFrame,
    model_name: str,
    family: DistFamily,
) -> Dict[str, NDArray]:
    """Parametros predichos, normalizados segun la familia.

    Returns
    -------
    {'lambda': ...} si family='poisson'; {'mu': ..., 'alpha': ...} si 'negbin'.
    """
    if model_name == "LightGBMLSS":
        preds: pd.DataFrame = model.predict(X, pred_type="parameters")
        if family == "poisson":
            return {"lambda": np.asarray(preds["rate"], dtype=float).ravel()}
        return _nb_from_columns(preds, "LightGBMLSS")

    if model_name == "XGBoostLSS":
        levels = getattr(model, "_categorical_levels", None)
        preds = model.predict(
            xgb.DMatrix(encode_categoricals(X, levels)), pred_type="parameters"
        )
        if family == "poisson":
            return {"lambda": np.asarray(preds["rate"], dtype=float).ravel()}
        return _nb_from_columns(preds, "XGBoostLSS")

    if model_name == "NGBoost":
        # Mismo encoding que en train. Con modelos serializados antes de que
        # NGBoost codificara categoricas, `levels` es None y esto es un no-op.
        levels = getattr(model, "_categorical_levels", None)
        X = encode_categoricals(to_categorical(X), levels)

        if family == "poisson":
            return {"lambda": np.asarray(model.predict(X)).ravel()}

        dist = model.pred_dist(X)
        dist_params = getattr(dist, "params", None)
        if dist_params is not None and "n" in dist_params and "p" in dist_params:
            n = np.asarray(dist_params["n"]).ravel()
            p = np.asarray(dist_params["p"]).ravel()
        elif hasattr(dist, "n") and hasattr(dist, "p"):
            n = np.asarray(dist.n).ravel()
            p = np.asarray(dist.p).ravel()
        else:
            raise ValueError(
                "No pude extraer (n, p) de NGBoost NegBin. "
                "Inspeccionar dist.__dict__."
            )
        return {"mu": n * (1.0 - p) / p, "alpha": 1.0 / n}

    raise ValueError(
        f"Modelo no soportado: '{model_name}'. Opciones: {list(MODEL_NAMES)}"
    )


# =============================================================================
# Imputacion
# =============================================================================

def impute_frames(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    strategy: str = "median",
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Imputa train/valid con un imputer fiteado SOLO en train (sin leakage).

    Las numericas van con `strategy`; las categoricas/object con
    'most_frequent', porque la mediana no esta definida sobre strings.

    El ColumnTransformer emite las columnas en orden num->cat, no en el del
    input. Se reindexa al orden original porque los backends dependen de el.
    """
    num_cols: List[str] = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols: List[str] = [c for c in X_train.columns if c not in num_cols]

    # SimpleImputer compara contra np.nan por identidad de valor: los None que
    # pandas deja en columnas object no se detectarian como faltantes.
    if cat_cols:
        X_train = X_train.copy()
        X_valid = X_valid.copy()
        for frame in (X_train, X_valid):
            for col in cat_cols:
                frame[col] = frame[col].where(frame[col].notna(), np.nan)

    imputer = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy=strategy), num_cols),
            ("cat", SimpleImputer(strategy="most_frequent"), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    # set_output(transform="pandas") garantiza DataFrame, pero el stub de
    # sklearn sigue declarando ndarray: se estrecha el tipo explicitamente.
    order: List[str] = list(X_train.columns)
    train_out = cast(pd.DataFrame, imputer.fit_transform(X_train))
    valid_out = cast(pd.DataFrame, imputer.transform(X_valid))
    train_clean = cast(pd.DataFrame, train_out[order])
    valid_clean = cast(pd.DataFrame, valid_out[order])
    return train_clean, valid_clean, imputer


def needs_imputation(model_name: str) -> bool:
    """True si el backend no tolera NaN."""
    return model_name in _NEEDS_IMPUTATION


def categorical_levels(X: pd.DataFrame) -> Dict[str, pd.Index]:
    """Mapa {columna: categorias} de las columnas 'category' de X."""
    return {
        col: X[col].cat.categories
        for col in X.select_dtypes(include="category").columns
    }


def to_categorical(X: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas object/string a dtype 'category'.

    `encode_categoricals` solo mira columnas ya tipadas como 'category'. Las
    features que llegan como strings crudos (p.ej. el nombre del torneo) pasan
    primero por aca para que el encoding las alcance.

    Devuelve el mismo objeto si no hay nada que convertir (sin copia).
    """
    obj_cols = X.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) == 0:
        return X

    out = X.copy()
    for col in obj_cols:
        out[col] = out[col].astype("category")
    return out


def encode_categoricals(
    X: pd.DataFrame,
    levels: Optional[Dict[str, pd.Index]] = None,
) -> pd.DataFrame:
    """Convierte columnas 'category' a codigos enteros para XGBoost.

    xgb.DMatrix rechaza dtypes 'category' salvo con enable_categorical=True,
    que XGBoostLSS no expone. Se codifica a int32 via .cat.codes.

    IMPORTANTE: .cat.codes indexa contra X[col].cat.categories, que depende de
    los valores presentes en ESE DataFrame. Sin `levels`, un mismo valor puede
    recibir codigos distintos en train y en valid (p.ej. si valid no contiene
    todas las categorias), y el modelo predeciria sobre un encoding que no es
    el que aprendio. Por eso al predecir hay que pasar los niveles del train.

    Parameters
    ----------
    X      : DataFrame a codificar.
    levels : {columna: categorias} de referencia (las del train). Si es None,
             se usan las categorias del propio X.

    Notes
    -----
    Los NaN y las categorias no vistas en `levels` quedan como -1, que XGBoost
    trata como un valor mas, no como faltante.

    Devuelve el mismo objeto si no hay columnas categoricas (sin copia).
    """
    cat_cols = X.select_dtypes(include="category").columns
    if len(cat_cols) == 0:
        return X

    out = X.copy()
    for col in cat_cols:
        series = out[col]
        if levels is not None and col in levels:
            series = series.cat.set_categories(levels[col])
        out[col] = series.cat.codes.astype("int32")
    return out


# =============================================================================
# Entrenamiento por backend
# =============================================================================

def _build_lss_distribution(
    backend: str,
    family: DistFamily,
    negbin_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Instancia la distribucion del backend LSS correspondiente."""
    nb_kwargs: Dict[str, Any] = dict(negbin_kwargs or {})

    if backend == "LightGBMLSS":
        return LGBMLSSPoisson() if family == "poisson" else LGBMLSSNB(**nb_kwargs)
    if backend == "XGBoostLSS":
        return XGBLSSPoisson() if family == "poisson" else XGBLSSNB(**nb_kwargs)

    raise ValueError(f"Backend LSS no soportado: '{backend}'")


def train_model(
    model_name: str,
    family: DistFamily,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Any]] = None,
    negbin_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Entrena un modelo de distribucion con early stopping sobre el valid dado.

    Parameters
    ----------
    model_name : 'LightGBMLSS' | 'XGBoostLSS' | 'NGBoost'.
    family     : 'poisson' | 'negbin'.
    params     : hiperparametros del modelo (desde el YAML).
    fit_params : {'num_boost_round', 'early_stopping_rounds'}.

    Returns
    -------
    El modelo entrenado (tipo dependiente del backend).
    """
    model_params: Dict[str, Any] = dict(params or {})
    fit_cfg: Dict[str, Any] = dict(fit_params or {})
    num_boost_round: int = int(fit_cfg.get("num_boost_round", 1000))
    early_stopping: int = int(fit_cfg.get("early_stopping_rounds", 50))

    if model_name == "LightGBMLSS":
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
        model = LightGBMLSS(
            _build_lss_distribution("LightGBMLSS", family, negbin_kwargs)
        )
        model.train(
            model_params, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(early_stopping, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        return model

    if model_name == "XGBoostLSS":
        # Los niveles del train mandan: valid y las predicciones futuras se
        # codifican contra ellos para que un valor no cambie de codigo.
        levels = categorical_levels(X_train)
        dtrain = xgb.DMatrix(encode_categoricals(X_train, levels), label=y_train)
        dvalid = xgb.DMatrix(encode_categoricals(X_valid, levels), label=y_valid)
        model = XGBoostLSS(
            _build_lss_distribution("XGBoostLSS", family, negbin_kwargs)
        )
        model.train(
            model_params, dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dvalid, "val")],
            early_stopping_rounds=early_stopping,
            verbose_eval=False,
        )
        # predict_params los recupera de aca para codificar igual que en train.
        setattr(model, "_categorical_levels", levels)
        return model

    if model_name == "NGBoost":
        ngb_params: Dict[str, Any] = dict(model_params)

        # max_depth / min_samples_leaf configuran el arbol base, no el regresor.
        base_kwargs: Dict[str, Any] = {}
        for key in ("max_depth", "min_samples_leaf"):
            if key in ngb_params:
                base_kwargs[key] = ngb_params.pop(key)

        base = DecisionTreeRegressor(**base_kwargs) if base_kwargs else None

        # El DecisionTreeRegressor de sklearn solo acepta numerico. Se codifica
        # igual que en XGBoost: los niveles del train mandan sobre valid y
        # sobre las predicciones futuras, para que un valor no cambie de codigo.
        X_train_cat = to_categorical(X_train)
        levels = categorical_levels(X_train_cat)
        X_train_enc = encode_categoricals(X_train_cat, levels)
        X_valid_enc = encode_categoricals(to_categorical(X_valid), levels)

        model = NGBRegressor(
            Dist=NGBPoisson if family == "poisson" else NGBNegBin,  # type: ignore[arg-type]
            Score=LogScore,
            **({"Base": base} if base is not None else {}),
            **ngb_params,
        )
        model.fit(
            X_train_enc, np.asarray(y_train).ravel(),
            X_val=X_valid_enc, Y_val=np.asarray(y_valid).ravel(),
            early_stopping_rounds=early_stopping,
        )
        # predict_params los recupera de aca para codificar igual que en train.
        setattr(model, "_categorical_levels", levels)
        return model

    raise ValueError(
        f"Modelo no soportado: '{model_name}'. Opciones: {list(MODEL_NAMES)}"
    )


def train_and_predict(
    model_name: str,
    family: DistFamily,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: Optional[Dict[str, Any]] = None,
    fit_params: Optional[Dict[str, Any]] = None,
    negbin_kwargs: Optional[Dict[str, Any]] = None,
    guards: Optional[Dict[str, float]] = None,
    return_imputer: bool = False,
) -> Tuple[Any, ...]:
    """Entrena, predice sobre el valid y aplica las guardas numericas.

    Maneja la imputacion cuando el backend la necesita, fiteando el imputer
    solo sobre train.

    Parameters
    ----------
    return_imputer : devuelve tambien el imputer fiteado (None si el backend
        no lo necesita). Hace falta para serializar el modelo: sin el imputer
        no se puede reproducir en produccion el preprocesado del train.

    Returns
    -------
    (modelo, parametros predichos ya saneados) o, con `return_imputer=True`,
    (modelo, parametros, imputer).
    """
    imputer: Optional[ColumnTransformer] = None
    if needs_imputation(model_name):
        X_tr, X_va, imputer = impute_frames(X_train, X_valid)
    else:
        X_tr, X_va = X_train, X_valid

    model = train_model(
        model_name, family, X_tr, y_train, X_va, y_valid,
        params=params, fit_params=fit_params, negbin_kwargs=negbin_kwargs,
    )
    raw: Dict[str, NDArray] = predict_params(model, X_va, model_name, family)
    predicted: Dict[str, NDArray] = clip_params(raw, family, guards)

    if return_imputer:
        return model, predicted, imputer
    return model, predicted

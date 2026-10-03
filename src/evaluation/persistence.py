"""Persistencia de resultados de evaluacion y de configuraciones.

Toda evaluacion guarda:
  - metricas -> `outputs/clubs/model_metrics/<target>/<nombre>.json`
  - configs  -> `configs/<target>/<nombre>.yaml`

Las rutas salen del `ModelConfig`, asi que cambiando `target.name` en el YAML
los resultados de otro target caen en su propia carpeta sin tocar codigo.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.config.config import ModelConfig, save_yaml


def _to_serializable(obj: Any) -> Any:
    """Convierte tipos numpy/pandas a equivalentes nativos para JSON/YAML.

    `json.dump` falla con np.float64, np.int64 y ndarray; esto los normaliza
    recursivamente.
    """
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return _to_serializable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return _to_serializable(obj.to_dict())
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        # NaN/Infinity no son JSON valido; se serializan como null.
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return _to_serializable(obj.tolist())
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def summarize_evaluation(
    results: Dict[str, Any],
    keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Aplana la salida de `evaluate_distribution_model` a metricas escalares.

    Descarta `rps_per_sample` (un array por observacion) para que el resumen
    sea liviano y apto para JSON.
    """
    component_keys: List[str] = keys or list(results.get("rps", {}))

    summary: Dict[str, Any] = {
        "label": results.get("label"),
        "family": results.get("family"),
        "total_method": results.get("total_method"),
        "ece_global": results.get("ece_global"),
    }

    for key in component_keys:
        rps: Dict[str, Any] = results.get("rps", {}).get(key, {})
        summary[f"rps_{key}"] = rps.get("rps_mean")
        summary[f"rps_{key}_std"] = rps.get("rps_std")
        summary[f"rps_{key}_p95"] = rps.get("rps_p95")
        summary[f"nll_{key}"] = results.get("nll", {}).get(key)

    n_samples: List[int] = [
        int(results["rps"][k]["n_samples"])
        for k in component_keys
        if k in results.get("rps", {})
    ]
    if n_samples:
        summary["n_samples"] = n_samples[0]

    return summary


def save_metrics(
    results: Union[Dict[str, Any], pd.DataFrame],
    name: str,
    config: ModelConfig,
    metadata: Optional[Dict[str, Any]] = None,
    metrics_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Guarda metricas en `outputs/clubs/model_metrics/<target>/<name>.json`.

    Parameters
    ----------
    results  : dict de metricas o DataFrame de resumen.
    name     : nombre del archivo, sin extension.
    config   : ModelConfig; determina la carpeta destino.
    metadata : campos extra a incluir (modelos evaluados, split, etc.).

    Returns
    -------
    Ruta del archivo escrito.
    """
    directory: Path = Path(metrics_dir) if metrics_dir else config.metrics_dir
    directory.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "target": config.target.name,
        "components": config.target.columns,
        "thresholds": config.target.thresholds,
        "total_method": config.target.total_method,
        "max_k": config.target.max_k,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    if metadata:
        payload.update(metadata)

    payload["results"] = results

    path: Path = directory / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(payload), f, indent=4, ensure_ascii=False)

    print(f" [METRICS] {name} -> {path}")
    return path


def save_config(
    data: Dict[str, Any],
    name: str,
    config: ModelConfig,
    configs_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Guarda un YAML de configuracion en `configs/<target>/<name>.yaml`."""
    directory: Path = Path(configs_dir) if configs_dir else config.configs_dir
    path: Path = directory / f"{name}.yaml"

    save_yaml(_to_serializable(data), path)

    print(f" [CONFIG] {name} -> {path}")
    return path


def best_params_stem(
    model_name: str,
    family: Optional[str] = None,
    component: Optional[str] = None,
) -> str:
    """Nombre canonico del archivo de best params, sin extension.

    Incluye familia y componente para que Poisson/NegBin y los distintos
    componentes del target no se pisen entre si:
        best_params_lightgbmlss_poisson_home
    """
    parts: List[str] = [model_name.lower()]
    if family:
        parts.append(family)
    if component:
        parts.append(component)
    return "best_params_" + "_".join(parts)


def save_best_params(
    best_params: Dict[str, Any],
    model_name: str,
    config: ModelConfig,
    family: Optional[str] = None,
    component: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Guarda los mejores hiperparametros de un modelo tras el tuning."""
    payload: Dict[str, Any] = dict(best_params)
    if extra:
        payload.update(extra)

    return save_config(
        payload, best_params_stem(model_name, family, component), config,
    )


def load_best_params(
    model_name: str,
    config: ModelConfig,
    family: Optional[str] = None,
    component: Optional[str] = None,
    params_only: bool = False,
) -> Dict[str, Any]:
    """Carga los hiperparametros guardados por el tuning.

    Parameters
    ----------
    params_only : devuelve solo el bloque `params` (listo para pasar a
        `evaluate_on_validation`), en vez del YAML completo con metadatos.
    """
    from src.config.config import load_yaml

    stem: str = best_params_stem(model_name, family, component)
    path: Path = config.configs_dir / f"{stem}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"No hay best params en {path}. Corre primero el tuning."
        )

    data: Dict[str, Any] = load_yaml(path)
    if params_only:
        return dict(data.get("params", {}))
    return data


def save_evaluation(
    results: Dict[str, Any],
    name: str,
    config: ModelConfig,
    metadata: Optional[Dict[str, Any]] = None,
    metrics_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Guarda la salida de `evaluate_distribution_model`: resumen + ECE por umbral.

    Parameters
    ----------
    metrics_dir : directorio alternativo. Por defecto `config.metrics_dir`.
    """
    payload: Dict[str, Any] = {
        "summary": summarize_evaluation(results, config.target.keys),
        "ece_by_threshold": results.get("ece_df"),
    }
    return save_metrics(
        payload, name, config, metadata=metadata, metrics_dir=metrics_dir,
    )

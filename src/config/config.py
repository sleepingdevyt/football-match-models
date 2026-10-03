"""Carga y validacion de la configuracion YAML de los modelos de distribucion.

Todo proceso que involucre parametros los lee desde un YAML en `configs/<target>/`.
Este modulo centraliza esa carga para que los notebooks no repitan `yaml.safe_load`
ni hardcodeen rutas.

Estructura esperada del YAML de target (ver `configs/goals/goals_config.yaml`):

    target:
      name: goals
      components:
        - {key: home, column: home_score_90}
        - {key: away, column: away_score_90}
      thresholds: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

El bloque `components` describe las dos variables que se suman para formar el
mercado Over/Under. Los nombres NO estan hardcodeados: cambiando `column` se
reutiliza todo el pipeline para corners, tarjetas, tiros, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Raiz del proyecto: .../EnvOficial (este archivo vive en EnvOficial/src/config/)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIGS_ROOT: Path = PROJECT_ROOT / "configs"
OUTPUTS_ROOT: Path = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Dataclasses de configuracion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Component:
    """Una de las dos variables que componen el target (ej. home / away).

    Attributes
    ----------
    key      : identificador corto usado como clave en los dicts de resultados.
    column   : nombre real de la columna en el DataFrame (ej. 'home_score_90').
    features : features especificas de este componente. Si es None, el caller
               pasa la lista de features explicitamente.
    """

    key: str
    column: str
    features: Optional[List[str]] = None


@dataclass(frozen=True)
class TargetConfig:
    """Definicion del target y del mercado Over/Under asociado."""

    name: str
    components: List[Component]
    thresholds: List[float]
    max_k: int = 8
    total_method: str = "convolution"

    @property
    def keys(self) -> List[str]:
        """Claves de los componentes, en orden (ej. ['home', 'away'])."""
        return [c.key for c in self.components]

    @property
    def columns(self) -> List[str]:
        """Columnas objetivo, en orden (ej. ['home_score_90', 'away_score_90'])."""
        return [c.column for c in self.components]

    def component(self, key: str) -> Component:
        """Devuelve el componente con la clave dada."""
        for c in self.components:
            if c.key == key:
                return c
        raise KeyError(
            f"No existe el componente '{key}'. Disponibles: {self.keys}"
        )


@dataclass
class ModelConfig:
    """Configuracion completa de un experimento de modelado."""

    target: TargetConfig
    cv: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    distributions: List[str] = field(default_factory=lambda: ["poisson", "negbin"])
    numeric_guards: Dict[str, float] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    # ── Accesos derivados ──────────────────────────────────────────────────

    @property
    def n_splits(self) -> int:
        return int(self.cv.get("n_splits", 5))

    @property
    def random_state(self) -> int:
        return int(self.cv.get("random_state", 67))

    def model_params(self, model_name: str) -> Dict[str, Any]:
        """Hiperparametros base de un modelo, tal como vienen del YAML."""
        entry: Dict[str, Any] = self.models.get(model_name, {})
        return dict(entry.get("params", {}))

    def fit_params(self, model_name: str) -> Dict[str, Any]:
        """Parametros de `fit`/`train` (early stopping, num_boost_round, ...)."""
        entry: Dict[str, Any] = self.models.get(model_name, {})
        return dict(entry.get("fit_params", {}))

    @property
    def configs_dir(self) -> Path:
        """Carpeta de configs del target (`configs/<name>/`)."""
        override: Optional[str] = self.paths.get("configs_dir")
        return Path(override) if override else CONFIGS_ROOT / self.target.name

    @property
    def metrics_dir(self) -> Path:
        """Carpeta de metricas del target (`outputs/clubs/model_metrics/<name>/`)."""
        override: Optional[str] = self.paths.get("metrics_dir")
        if override:
            return Path(override)
        return OUTPUTS_ROOT / "clubs" / "model_metrics" / self.target.name

    @property
    def models_dir(self) -> Path:
        """Carpeta de modelos serializados del target."""
        override: Optional[str] = self.paths.get("models_dir")
        if override:
            return Path(override)
        return OUTPUTS_ROOT / "clubs" / "models" / self.target.name


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

_VALID_TOTAL_METHODS = {"convolution", "poisson_approx"}
_VALID_FAMILIES = {"poisson", "negbin"}


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Lee un YAML y devuelve un dict (nunca None)."""
    p: Path = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo de configuracion: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data: Optional[Dict[str, Any]] = yaml.safe_load(f)

    return data or {}


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Escribe un dict a YAML creando el directorio padre si hace falta."""
    p: Path = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False,
                  allow_unicode=True)

    return p


def _parse_target(block: Dict[str, Any]) -> TargetConfig:
    """Construye el TargetConfig validando el bloque `target` del YAML."""
    if "name" not in block:
        raise ValueError("El bloque 'target' necesita una clave 'name'.")

    raw_components: List[Dict[str, Any]] = block.get("components", [])
    if len(raw_components) != 2:
        raise ValueError(
            f"'target.components' debe tener exactamente 2 entradas "
            f"(el mercado Over/Under se arma sumando dos variables); "
            f"recibi {len(raw_components)}."
        )

    components: List[Component] = []
    for entry in raw_components:
        if "key" not in entry or "column" not in entry:
            raise ValueError(
                f"Cada componente necesita 'key' y 'column'. Recibi: {entry}"
            )
        components.append(
            Component(
                key=str(entry["key"]),
                column=str(entry["column"]),
                features=entry.get("features"),
            )
        )

    keys: List[str] = [c.key for c in components]
    if len(set(keys)) != len(keys):
        raise ValueError(f"Las claves de los componentes se repiten: {keys}")

    thresholds: List[float] = [float(t) for t in block.get("thresholds", [])]
    if not thresholds:
        raise ValueError(
            "'target.thresholds' esta vacio: es la grilla Over/Under a evaluar."
        )

    total_method: str = str(block.get("total_method", "convolution"))
    if total_method not in _VALID_TOTAL_METHODS:
        raise ValueError(
            f"'target.total_method' invalido: '{total_method}'. "
            f"Opciones: {sorted(_VALID_TOTAL_METHODS)}"
        )

    max_k: int = int(block.get("max_k", 8))
    if max_k < 1:
        raise ValueError(f"'target.max_k' debe ser >= 1; recibi {max_k}.")

    # La grilla debe cubrir el umbral mas alto, si no el ECE queda truncado.
    if max_k < max(thresholds):
        raise ValueError(
            f"'target.max_k' ({max_k}) es menor que el threshold mas alto "
            f"({max(thresholds)}): la CDF quedaria truncada y el ECE mal calculado."
        )

    return TargetConfig(
        name=str(block["name"]),
        components=components,
        thresholds=thresholds,
        max_k=max_k,
        total_method=total_method,
    )


def load_config(path: Union[str, Path]) -> ModelConfig:
    """Carga y valida el YAML de configuracion de un target.

    Parameters
    ----------
    path : ruta al YAML (ej. `configs/goals/goals_config.yaml`).

    Returns
    -------
    ModelConfig ya validado.
    """
    raw: Dict[str, Any] = load_yaml(path)

    if "target" not in raw:
        raise ValueError(
            f"El YAML {path} no tiene bloque 'target'. Es obligatorio."
        )

    target: TargetConfig = _parse_target(raw["target"])

    distributions: List[str] = [
        str(d) for d in raw.get("distributions", ["poisson", "negbin"])
    ]
    invalid: List[str] = [d for d in distributions if d not in _VALID_FAMILIES]
    if invalid:
        raise ValueError(
            f"Distribuciones no soportadas: {invalid}. "
            f"Opciones: {sorted(_VALID_FAMILIES)}"
        )

    return ModelConfig(
        target=target,
        cv=raw.get("cv", {}),
        models=raw.get("models", {}),
        distributions=distributions,
        numeric_guards=raw.get("numeric_guards", {}),
        paths=raw.get("paths", {}),
        raw=raw,
    )


def load_target_config(
    target_name: str,
    filename: Optional[str] = None,
) -> ModelConfig:
    """Atajo: carga `configs/<target_name>/<target_name>_config.yaml`.

    Parameters
    ----------
    target_name : nombre del target (ej. 'goals', 'corners').
    filename    : nombre del archivo si difiere del default.
    """
    fname: str = filename or f"{target_name}_config.yaml"
    return load_config(CONFIGS_ROOT / target_name / fname)

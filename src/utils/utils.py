"""Utilidades compartidas: splits temporales, preparación de X/y y alineación
de esquemas entre train / val / test."""

from pathlib import Path
from typing import Dict, Final, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Separadores para las cabeceras de los reportes por consola.
SEP: Final[str] = "=" * 72
SUB: Final[str] = "-" * 72



def split_data(
    df: pd.DataFrame,
    splits: Dict[str, List[str]],
    VALID_TOURNAMENTS: List[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_copy = df.copy()

    if VALID_TOURNAMENTS is not None:
        df_copy = df_copy[df_copy["tournament"].isin(VALID_TOURNAMENTS)]

    df_copy["match_datetime_utc"] = pd.to_datetime(df_copy["match_datetime_utc"])

    df_copy.sort_values(["match_datetime_utc", "match_id"], inplace=True)

    return (
        df_copy[df_copy["name"].isin(splits["train"])],
        df_copy[df_copy["name"].isin(splits["val"])],
        df_copy[df_copy["name"].isin(splits["test"])],
    )

def clean_key_missing(df: pd.DataFrame, cols_to_check: List[str] = ['home_score', 'away_score'] ) -> pd.DataFrame:
    """
    Drops rows where ANY of the specified columns contain null values.
    Intended for TRAIN and VALIDATION only.
    """
    initial_rows = len(df)

    df_clean = df.dropna(subset=cols_to_check)

    dropped = initial_rows - len(df_clean)

    print(f"Removed {dropped} rows with missing key odds.")
    print(f"Remaining rows: {len(df_clean)}")

    return df_clean

def prepare_data_for_training(
    df: pd.DataFrame,
    EXCLUDE_COLS: list[str],
    leaky_cols: list[str],
    TARGET_COL: Union[str, list[str]],
    category_map: dict | None = None,
):

    cols_to_drop = []

    if leaky_cols is not None:
        cols_to_drop.extend(leaky_cols)

    if EXCLUDE_COLS is not None:
        cols_to_drop.extend(EXCLUDE_COLS)

    if isinstance(TARGET_COL, str):
        cols_to_drop.append(TARGET_COL)
    else:
        cols_to_drop.extend(TARGET_COL)

    # dict.fromkeys deduplica conservando el orden: una columna que aparezca
    # en leaky_cols y en EXCLUDE_COLS a la vez no debe romper el drop.
    cols_to_drop = list(dict.fromkeys(cols_to_drop))

    # errors="ignore" tolera columnas ausentes: las listas son compartidas
    # entre targets (1x2, goals, corners) y no todos los datasets las traen.
    faltantes = [c for c in cols_to_drop if c not in df.columns]
    if faltantes:
        print(f"  [PREP] {len(faltantes)} columnas a dropear no estan en el df "
              f"y se ignoran: {faltantes}")

    X = df.drop(columns=cols_to_drop, errors="ignore")
    y = df[TARGET_COL]

    # Detectar columnas categóricas
    cat_cols = X.columns[X.nunique() < 20].tolist()

    if "tournament" in X.columns and "tournament" not in cat_cols:
        cat_cols.append("tournament")

    if category_map is None:
        # TRAIN: crear categorías
        category_map = {}

        for col in cat_cols:
            X[col] = X[col].astype("category")
            category_map[col] = X[col].cat.categories

    else:
        # VALID / TEST: reutilizar categorías del train
        for col, cats in category_map.items():
            if col in X.columns:
                X[col] = pd.Categorical(
                    X[col],
                    categories=cats,
                )

    return X, y, category_map


# ================================================================================
# [F-CAT] ALINEACIÓN DE ESQUEMA ENTRE TRAIN / VAL / TEST
# ================================================================================
def is_categorical(series: pd.Series) -> bool:
    """True si la serie tiene dtype 'category' (compatible pandas 1.x-3.x)."""
    return isinstance(series.dtype, pd.CategoricalDtype)


def categorical_columns(X: pd.DataFrame) -> List[str]:
    """Columnas con dtype 'category', en orden de aparición."""
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def safe_cast_to_categorical(
    values: pd.Series,
    ref_dtype: pd.CategoricalDtype,
) -> pd.Series:
    """
    [F-CAST] Castea a una CategoricalDtype de referencia mapeando a NaN todo
    nivel que no exista en ella.

    `pd.Categorical(values, categories=cats)` hace lo mismo, pero está deprecado
    en pandas 3.x cuando aparecen valores fuera de `categories` (va a lanzar
    excepción en una versión futura). Enmascarar primero es equivalente y
    estable en 1.x / 2.x / 3.x.
    """
    raw: pd.Series = values.astype(object)
    allowed: List[object] = list(ref_dtype.categories)
    raw = raw.where(raw.isin(allowed), other=None)
    return raw.astype(
        pd.CategoricalDtype(categories=ref_dtype.categories, ordered=bool(ref_dtype.ordered))
    )


def align_frame_to_reference(
    X_ref: pd.DataFrame,
    X_other: pd.DataFrame,
    name: str,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    [F-CAT] Fuerza que `X_other` tenga el MISMO esquema que `X_ref`:
    mismas columnas, mismo orden, mismos dtypes y — en las categóricas — las
    mismas categorías y el mismo flag `ordered`.

    Casos que resuelve (todos producen el ValueError de LightGBM o un fallo
    silencioso más adelante):
      - columna 'category' en val/test que es numérica u object en train
        -> se devuelve a numérica (`to_numeric`, errores -> NaN);
      - columna 'category' en train con categorías distintas en val/test
        -> se recastea contra las categorías de train (niveles nuevos -> NaN);
      - columna presente en train y ausente en val/test
        -> se crea llena de NaN con el dtype correcto;
      - orden de columnas distinto -> se reordena como train.

    Retorna (frame_alineado, tabla_de_cambios).
    """
    ref_cols: List[str] = list(X_ref.columns)
    other_cols: Set[str] = set(X_other.columns)

    data: Dict[str, pd.Series] = {}
    missing: List[str] = []
    new_cat: List[str] = []       # object/num -> category (esperado, no es anomalía)
    relevelled: List[str] = []    # category con niveles distintos a los de train
    demoted_cat: List[str] = []   # category en val/test pero NO en train (anomalía)
    coerced: List[str] = []

    col: str
    for col in ref_cols:
        ref_dtype = X_ref[col].dtype
        ref_is_cat: bool = isinstance(ref_dtype, pd.CategoricalDtype)

        if col not in other_cols:
            missing.append(col)
            if ref_is_cat:
                data[col] = pd.Series(
                    pd.Categorical(
                        [None] * int(X_other.shape[0]),
                        categories=ref_dtype.categories,
                        ordered=bool(ref_dtype.ordered),
                    ),
                    index=X_other.index,
                )
            else:
                data[col] = pd.Series(np.nan, index=X_other.index, dtype="float64")
            continue

        cur: pd.Series = X_other[col]
        cur_is_cat: bool = isinstance(cur.dtype, pd.CategoricalDtype)

        if ref_is_cat:
            same_schema: bool = (
                cur_is_cat
                and list(cur.cat.categories) == list(ref_dtype.categories)
                and bool(cur.cat.ordered) == bool(ref_dtype.ordered)
            )
            if same_schema:
                data[col] = cur
            else:
                data[col] = safe_cast_to_categorical(cur, ref_dtype)
                (relevelled if cur_is_cat else new_cat).append(col)
        else:
            if cur_is_cat:
                # category en val/test pero NO en train -> desarmar
                data[col] = pd.to_numeric(cur.astype(object), errors="coerce")
                demoted_cat.append(col)
            elif cur.dtype != ref_dtype:
                try:
                    data[col] = cur.astype(ref_dtype)
                except (TypeError, ValueError):
                    data[col] = pd.to_numeric(cur, errors="coerce")
                coerced.append(col)
            else:
                data[col] = cur

    out: pd.DataFrame = pd.DataFrame(data, index=X_other.index)[ref_cols]

    changes: pd.DataFrame = pd.DataFrame(
        [
            {"frame": name, "issue": "object_a_category", "severity": "normal",
             "n": len(new_cat), "examples": ", ".join(new_cat[:5])},
            {"frame": name, "issue": "columna_faltante", "severity": "ANOMALIA",
             "n": len(missing), "examples": ", ".join(missing[:5])},
            {"frame": name, "issue": "niveles_distintos_a_train", "severity": "ANOMALIA",
             "n": len(relevelled), "examples": ", ".join(relevelled[:5])},
            {"frame": name, "issue": "category_no_presente_en_train", "severity": "ANOMALIA",
             "n": len(demoted_cat), "examples": ", ".join(demoted_cat[:5])},
            {"frame": name, "issue": "dtype_coercionado", "severity": "ANOMALIA",
             "n": len(coerced), "examples": ", ".join(coerced[:5])},
        ]
    )
    changes = changes[changes["n"] > 0].reset_index(drop=True)

    anomalies: pd.DataFrame = changes[changes["severity"] == "ANOMALIA"]
    if verbose and not anomalies.empty:
        print(f"\n  [SCHEMA] '{name}' no tenía el mismo esquema que train. Ajustes:")
        print(anomalies.to_string(index=False))
        if len(demoted_cat) > 0:
            print(f"  [SCHEMA] >> {len(demoted_cat)} columnas venían como 'category' en "
                  f"'{name}' pero no en train.")
            print("             Ésta es exactamente la causa del ValueError de LightGBM")
            print("             'train and valid dataset categorical_feature do not match'.")

    return out, changes


def assert_same_categorical_schema(
    X_a: pd.DataFrame,
    X_b: pd.DataFrame,
    name_a: str = "train",
    name_b: str = "valid",
) -> None:
    """
    [F-CAT] Falla temprano y con un mensaje legible en vez de dejar que
    LightGBM tire el ValueError opaco desde `_data_from_pandas`.
    """
    cats_a: List[str] = categorical_columns(X_a)
    cats_b: List[str] = categorical_columns(X_b)

    if list(X_a.columns) != list(X_b.columns):
        only_a = [c for c in X_a.columns if c not in set(X_b.columns)]
        only_b = [c for c in X_b.columns if c not in set(X_a.columns)]
        raise ValueError(
            f"Las columnas de '{name_a}' y '{name_b}' no coinciden.\n"
            f"  solo en {name_a}: {only_a[:10]}\n"
            f"  solo en {name_b}: {only_b[:10]}"
        )

    if cats_a != cats_b:
        only_a = [c for c in cats_a if c not in set(cats_b)]
        only_b = [c for c in cats_b if c not in set(cats_a)]
        raise ValueError(
            f"Esquema categórico desalineado entre '{name_a}' ({len(cats_a)} cols) y "
            f"'{name_b}' ({len(cats_b)} cols).\n"
            f"  category solo en {name_a}: {only_a[:10]}\n"
            f"  category solo en {name_b}: {only_b[:10]}\n"
            f"  -> LightGBM exige la misma cantidad de columnas 'category' en ambos."
        )

    for col in cats_a:
        if list(X_a[col].cat.categories) != list(X_b[col].cat.categories):
            raise ValueError(
                f"La columna categórica '{col}' tiene categorías distintas en "
                f"'{name_a}' y '{name_b}'."
            )

def print_header(title: str) -> None:
    """Cabecera consistente para cada paso."""
    print(f"\n{SEP}")
    print(f"  {title}")
    print(f"{SEP}")


def save_feature_datasets(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    output_dir: Union[str, Path],
    metadata_cols: List[str],
    datetime_col: str = "match_datetime_utc",
) -> Dict[str, Path]:
    """Guarda un parquet por conjunto de features, con las columnas de metadata.

    Generaliza el guardado a N conjuntos: en vez de home/away fijos, la clave
    de `feature_sets` nombra el archivo.

    Parameters
    ----------
    df            : DataFrame con las features ya construidas.
    feature_sets  : {nombre: lista de features}. Cada entrada -> un parquet.
    output_dir    : carpeta destino.
    metadata_cols : columnas de contexto que se incluyen en todos los archivos.
    datetime_col  : columna de fecha a normalizar; se omite si no existe.

    Returns
    -------
    {nombre: ruta escrita}
    """
    directory: Path = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    # Columnas duplicadas rompen el guardado a parquet.
    frame: pd.DataFrame = df.loc[:, ~df.columns.duplicated()].copy()

    if datetime_col in frame.columns and not pd.api.types.is_datetime64_any_dtype(
        frame[datetime_col]
    ):
        frame[datetime_col] = pd.to_datetime(
            frame[datetime_col], format="ISO8601", errors="coerce",
        )

    print(f"Shape antes de guardar: {frame.shape}")

    written: Dict[str, Path] = {}

    for name, features in feature_sets.items():
        # dict.fromkeys preserva el orden y elimina repetidos.
        requested: List[str] = list(dict.fromkeys(metadata_cols + list(features)))
        available: List[str] = [c for c in requested if c in frame.columns]

        missing: List[str] = [c for c in requested if c not in frame.columns]
        if missing:
            print(f"  [{name}] {len(missing)} columnas ausentes, excluidas: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

        path: Path = directory / f"{name}.parquet"
        frame[available].to_parquet(path, index=False)
        written[name] = path

        print(f"  [{name}] guardado con {len(available)} columnas -> {path.name}")

    return written
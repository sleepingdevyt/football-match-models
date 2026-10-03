"""Analisis de error para modelos de distribucion de conteos.

Una sola clase cubre Poisson y NegBin: la familia solo cambia como se calcula
el RPS y habilita el diagnostico de sobredispersion. Agnostica al target; los
bins, umbrales y nombres de columnas vienen del YAML.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from src.config.config import ModelConfig
from src.metrics.distribution_metrics import DistFamily, get_mean, rps_per_sample


def _section(title: str, width: int = 60) -> None:
    """Cabecera consistente para cada bloque del analisis."""
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def _spearman(a: NDArray, b: NDArray) -> Tuple[float, float]:
    """Correlacion de Spearman como (rho, p_value) de floats nativos."""
    result: Any = stats.spearmanr(a, b)
    return float(result[0]), float(result[1])


class DistributionErrorAnalyzer:
    """Analisis de error a partir de los parametros predichos.

    Recibe las predicciones ya calculadas (no el modelo), asi que sirve igual
    para LightGBMLSS, XGBoostLSS, NGBoost o un GLM.

    Parameters
    ----------
    params     : Poisson -> {'lambda': ...}; NegBin -> {'mu': ..., 'alpha': ...}.
    y_true     : valores observados.
    family     : 'poisson' | 'negbin'.
    meta       : DataFrame opcional con columnas de contexto (liga, fecha...).
    component  : nombre del componente analizado (ej. 'home').
    model_name : etiqueta para los titulos.
    config     : ModelConfig; de ahi salen bins, k_max y nombres de columnas.
    """

    def __init__(
        self,
        params: Dict[str, NDArray],
        y_true: NDArray,
        family: DistFamily = "poisson",
        meta: Optional[pd.DataFrame] = None,
        component: str = "",
        model_name: str = "Model",
        config: Optional[ModelConfig] = None,
        k_max: Optional[int] = None,
    ) -> None:

        self.family: DistFamily = family
        self.params: Dict[str, NDArray] = params
        self.model_name: str = model_name
        self.component: str = component.upper()
        self.config: Optional[ModelConfig] = config

        ea_cfg: Dict[str, Any] = (
            config.raw.get("error_analysis", {}) if config else {}
        )
        self.ea_cfg: Dict[str, Any] = ea_cfg
        self.k_max: int = int(
            k_max if k_max is not None else ea_cfg.get("k_max", 6)
        )

        self.y_true: NDArray = np.asarray(y_true, dtype=float).ravel()
        self.mean_pred: NDArray = get_mean(params, family)
        self.n: int = len(self.y_true)

        if self.n != len(self.mean_pred):
            raise ValueError(
                f"Desalineacion: y_true ({self.n}) vs predicciones "
                f"({len(self.mean_pred)})."
            )

        self.meta: Optional[pd.DataFrame] = (
            meta.reset_index(drop=True) if meta is not None else None
        )
        if self.meta is not None and len(self.meta) != self.n:
            raise ValueError(
                f"Desalineacion: meta ({len(self.meta)}) vs predicciones "
                f"({self.n})."
            )

        self.rps: NDArray = rps_per_sample(
            self.y_true, params, family, self.k_max,
        )
        self.residuals: NDArray = self.y_true - self.mean_pred
        self.log_residuals: NDArray = (
            np.log1p(self.y_true) - np.log1p(self.mean_pred)
        )

        frame: Dict[str, NDArray] = {
            "y_true": self.y_true,
            "mean_pred": self.mean_pred,
            "residual": self.residuals,
            "log_residual": self.log_residuals,
            "rps": self.rps,
        }

        # Residuo de Pearson: estandariza por la varianza NB2, lo que permite
        # diagnosticar sobredispersion no capturada (Var ~ 1 si ajusta bien).
        self.pearson_resid: Optional[NDArray] = None
        if family == "negbin":
            alpha: NDArray = np.asarray(params["alpha"], dtype=float).ravel()
            variance: NDArray = self.mean_pred + alpha * self.mean_pred ** 2
            self.pearson_resid = self.residuals / np.sqrt(variance)
            frame["alpha_pred"] = alpha
            frame["pearson_resid"] = self.pearson_resid

        self._df: pd.DataFrame = pd.DataFrame(frame)
        if self.meta is not None:
            self._df = pd.concat([self._df, self.meta], axis=1)

    # ── Punto de entrada ─────────────────────────────────────────────────────

    def run_all(self) -> None:
        """Corre todos los bloques del analisis."""
        print(f"\n{'=' * 70}")
        print(f"  {self.model_name} [{self.family}] - {self.component} | "
              f"N={self.n:,}")
        print(f"  RPS mean={self.rps.mean():.5f}  |  "
              f"media predicha={self.mean_pred.mean():.4f}")
        print(f"{'=' * 70}")

        self.plot_calibration()
        self.plot_residuals_by_ytrue()
        self.analyze_high_rps_tail()
        self.plot_bias_by_group()
        self.plot_temporal_drift()

        if self.family == "negbin":
            self.summarize_dispersion()

    # =========================================================================
    # 1. Calibracion por tramo de media predicha (reliability diagram)
    # =========================================================================

    def plot_calibration(
        self,
        bins: Optional[Sequence[float]] = None,
        figsize: Tuple[int, int] = (9, 5),
    ) -> pd.DataFrame:
        """Compara la media predicha contra la media observada por tramo."""
        _section(f"[1] Calibracion - {self.model_name} | {self.component}")

        bin_edges: List[float] = [float(b) for b in (
            bins if bins is not None
            else self.ea_cfg.get(
                "lambda_bins",
                [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
            )
        )]

        labels: List[str] = [
            f"[{bin_edges[i]:.1f}, {bin_edges[i + 1]:.1f})"
            for i in range(len(bin_edges) - 1)
        ]
        labels[-1] = f"[{bin_edges[-2]:.1f}, +inf)"

        df: pd.DataFrame = self._df.copy()
        df["pred_bin"] = pd.cut(
            df["mean_pred"], bins=bin_edges, labels=labels, right=False,
        )

        agg: pd.DataFrame = (
            df.groupby("pred_bin", observed=True)
            .agg(
                n=("y_true", "count"),
                pred_mean=("mean_pred", "mean"),
                ytrue_mean=("y_true", "mean"),
                rps_mean=("rps", "mean"),
            )
            .dropna()
            .reset_index()
        )
        print(agg.to_string(index=False))

        upper: float = float(max(agg["pred_mean"].max(),
                                 agg["ytrue_mean"].max())) + 0.5

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, upper], [0, upper], "k--", lw=1.2,
                label="Calibracion perfecta")

        scatter = ax.scatter(
            agg["pred_mean"], agg["ytrue_mean"],
            s=agg["n"] / max(self.n, 1) * 5000,
            c=agg["rps_mean"], cmap="RdYlGn_r",
            edgecolors="black", linewidths=0.6, zorder=3,
        )
        plt.colorbar(scatter, ax=ax, label="RPS medio del tramo")

        for _, row in agg.iterrows():
            ax.annotate(
                f"n={int(row['n'])}",
                (row["pred_mean"], row["ytrue_mean"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )

        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_xlabel("Media predicha")
        ax.set_ylabel("Media observada")
        ax.set_title(f"Reliability Diagram - {self.model_name} "
                     f"({self.component}) [{self.family}]")
        ax.legend()
        plt.tight_layout()
        plt.show()

        return agg

    # =========================================================================
    # 2. Residuos por valor observado
    # =========================================================================

    def plot_residuals_by_ytrue(
        self,
        max_ytrue: Optional[int] = None,
        figsize: Tuple[int, int] = (11, 5),
        n_boot: int = 2000,
    ) -> pd.DataFrame:
        """Distribucion y sesgo medio de los residuos, agrupados por y_true."""
        _section(f"[2] Residuos por y_true - {self.model_name} | "
                 f"{self.component}")

        cap: int = int(max_ytrue if max_ytrue is not None else self.k_max)

        df: pd.DataFrame = self._df.copy()
        df["ytrue_grp"] = df["y_true"].clip(upper=cap).astype(int)

        groups: List[int] = sorted(df["ytrue_grp"].unique())
        labels: List[str] = [f"{g}+" if g == cap else str(g) for g in groups]
        data_box: List[NDArray] = [
            df.loc[df["ytrue_grp"] == g, "residual"].to_numpy() for g in groups
        ]
        means: List[float] = [float(d.mean()) for d in data_box]

        # IC bootstrap del residuo medio por grupo.
        rng = np.random.default_rng(42)
        cis: List[Tuple[float, float]] = []
        for d in data_box:
            if len(d) < 5:
                cis.append((np.nan, np.nan))
                continue
            boots: NDArray = rng.choice(
                d, size=(n_boot, len(d)), replace=True,
            ).mean(axis=1)
            cis.append((
                float(np.percentile(boots, 2.5)),
                float(np.percentile(boots, 97.5)),
            ))

        summary: pd.DataFrame = pd.DataFrame({
            "y_true": labels,
            "n": [len(d) for d in data_box],
            "residual_mean": [round(m, 4) for m in means],
            "ci_low": [round(c[0], 4) if np.isfinite(c[0]) else None
                       for c in cis],
            "ci_high": [round(c[1], 4) if np.isfinite(c[1]) else None
                        for c in cis],
        })
        print(summary.to_string(index=False))

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].boxplot(data_box, tick_labels=labels, showfliers=False,
                        patch_artist=True,
                        boxprops=dict(facecolor="#cce5ff", color="steelblue"),
                        medianprops=dict(color="navy", lw=2))
        axes[0].axhline(0, color="red", lw=1.2, ls="--")
        axes[0].set_xlabel("Observado (y_true)")
        axes[0].set_ylabel("Residuo (y_true - media predicha)")
        axes[0].set_title(f"Distribucion de residuos - {self.component}")

        x: NDArray = np.arange(len(groups))
        axes[1].bar(x, means,
                    color=["#d9534f" if m > 0 else "#5bc0de" for m in means],
                    alpha=0.75, edgecolor="black")
        for i, (ci, m) in enumerate(zip(cis, means)):
            if np.isfinite(ci[0]):
                axes[1].errorbar(i, m, yerr=[[m - ci[0]], [ci[1] - m]],
                                 fmt="none", color="black", capsize=4)
        axes[1].axhline(0, color="red", lw=1.2, ls="--")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_xlabel("Observado (y_true)")
        axes[1].set_ylabel("Residuo medio")
        axes[1].set_title(f"Sesgo medio con IC 95% - {self.component}")

        plt.tight_layout()
        plt.show()

        return summary

    # =========================================================================
    # 3. Cola de errores altos
    # =========================================================================

    def analyze_high_rps_tail(
        self,
        percentile: Optional[float] = None,
        figsize: Tuple[int, int] = (12, 5),
    ) -> pd.DataFrame:
        """Compara las observaciones con RPS mas alto contra el resto."""
        pct: float = float(
            percentile if percentile is not None
            else self.ea_cfg.get("tail_percentile", 95.0)
        )
        _section(f"[3] Cola RPS > p{pct:.0f} - {self.model_name} | "
                 f"{self.component}")

        threshold: float = float(np.percentile(self.rps, pct))
        mask: NDArray[np.bool_] = self.rps >= threshold
        df_tail: pd.DataFrame = self._df[mask].copy()
        df_rest: pd.DataFrame = self._df[~mask].copy()

        print(f"  Umbral RPS p{pct:.0f} : {threshold:.5f}")
        print(f"  En la cola          : {int(mask.sum()):,} "
              f"({mask.mean() * 100:.1f}%)")

        summary: pd.DataFrame = pd.DataFrame({
            "grupo": ["cola", "resto"],
            "n": [len(df_tail), len(df_rest)],
            "rps_mean": [df_tail["rps"].mean(), df_rest["rps"].mean()],
            "ytrue_mean": [df_tail["y_true"].mean(), df_rest["y_true"].mean()],
            "pred_mean": [df_tail["mean_pred"].mean(),
                          df_rest["mean_pred"].mean()],
            "residual_mean": [df_tail["residual"].mean(),
                              df_rest["residual"].mean()],
        }).round(4)
        print(f"\n{summary.to_string(index=False)}")

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        bins: NDArray = np.arange(-0.5, self.k_max + 1.5)

        for data, lbl, color in [
            (df_tail, f"Cola p{pct:.0f}", "salmon"),
            (df_rest, "Resto", "steelblue"),
        ]:
            axes[0].hist(data["y_true"], bins=bins, density=True, alpha=0.6,
                         label=lbl, color=color, edgecolor="black")
        axes[0].set_xticks(range(self.k_max + 1))
        axes[0].set_xlabel("Observado (y_true)")
        axes[0].set_ylabel("Densidad")
        axes[0].set_title(f"Distribucion y_true - {self.component}")
        axes[0].legend()

        for data, lbl, color in [
            (df_tail, f"Cola p{pct:.0f}", "salmon"),
            (df_rest, "Resto", "steelblue"),
        ]:
            axes[1].hist(data["mean_pred"], bins=30, density=True, alpha=0.6,
                         label=lbl, color=color, edgecolor="black")
        axes[1].set_xlabel("Media predicha")
        axes[1].set_ylabel("Densidad")
        axes[1].set_title(f"Distribucion predicha - {self.component}")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        return df_tail

    # =========================================================================
    # 4. Sesgo por grupo (liga / competicion)
    # =========================================================================

    def plot_bias_by_group(
        self,
        group_column: Optional[str] = None,
        top_n: Optional[int] = None,
        min_count: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 7),
    ) -> Optional[pd.DataFrame]:
        """RPS y sesgo medio desagregados por grupo (por defecto, liga)."""
        column: str = str(
            group_column or self.ea_cfg.get("league_column", "league")
        )
        _section(f"[4] Sesgo por '{column}' - {self.model_name} | "
                 f"{self.component}")

        if column not in self._df.columns:
            print(f"  [SKIP] No hay columna '{column}' en meta.")
            return None

        limit: int = int(top_n if top_n is not None
                         else self.ea_cfg.get("top_n_leagues", 20))
        minimum: int = int(min_count if min_count is not None
                           else self.ea_cfg.get("min_matches_per_league", 20))

        agg: pd.DataFrame = (
            self._df.groupby(column)
            .agg(
                n=("y_true", "count"),
                rps_mean=("rps", "mean"),
                residual_mean=("residual", "mean"),
                ytrue_mean=("y_true", "mean"),
                pred_mean=("mean_pred", "mean"),
            )
            .query(f"n >= {minimum}")
            .sort_values("rps_mean", ascending=False)
            .head(limit)
            .reset_index()
        )

        if agg.empty:
            print(f"  [SKIP] Ningun grupo alcanza el minimo de {minimum}.")
            return None

        print(agg.to_string(index=False))

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        spread: float = float(
            agg["rps_mean"].max() - agg["rps_mean"].min()
        ) + 1e-9
        norm: NDArray = ((agg["rps_mean"] - agg["rps_mean"].min())
                         / spread).to_numpy()

        axes[0].barh(agg[column].astype(str), agg["rps_mean"],
                     color=plt.cm.RdYlGn_r(norm), edgecolor="black")
        axes[0].axvline(self.rps.mean(), color="navy", ls="--", lw=1.5,
                        label=f"Global={self.rps.mean():.4f}")
        axes[0].set_xlabel("RPS medio")
        axes[0].set_title(f"RPS por {column} (top {limit})")
        axes[0].invert_yaxis()
        axes[0].legend(fontsize=8)

        axes[1].barh(
            agg[column].astype(str), agg["residual_mean"],
            color=["#d9534f" if r > 0 else "#5bc0de"
                   for r in agg["residual_mean"]],
            edgecolor="black",
        )
        axes[1].axvline(0, color="black", ls="--", lw=1.2)
        axes[1].set_xlabel("Residuo medio")
        axes[1].set_title(f"Sesgo por {column}")
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.show()

        return agg

    # =========================================================================
    # 5. Drift temporal
    # =========================================================================

    def plot_temporal_drift(
        self,
        time_column: Optional[str] = None,
        window: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 6),
    ) -> Optional[pd.DataFrame]:
        """Evolucion del RPS y del sesgo a lo largo del tiempo."""
        _section(f"[5] Drift temporal - {self.model_name} | {self.component}")

        candidates: List[str] = (
            [time_column] if time_column
            else list(self.ea_cfg.get(
                "temporal_columns",
                ["year_month", "matchweek", "date", "match_date", "season_week"],
            ))
        )
        column: Optional[str] = next(
            (c for c in candidates if c and c in self._df.columns), None,
        )

        if column is None:
            print(f"  [SKIP] Ninguna columna temporal disponible: {candidates}")
            return None

        roll: int = int(window if window is not None
                        else self.ea_cfg.get("rolling_window", 3))

        df: pd.DataFrame = self._df[
            [column, "rps", "residual", "y_true", "mean_pred"]
        ].copy()

        # Normaliza fechas a periodos mensuales para agrupar.
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = df[column].dt.to_period("M").astype(str)
        elif df[column].dtype == object:
            try:
                df[column] = (pd.to_datetime(df[column])
                              .dt.to_period("M").astype(str))
            except (ValueError, TypeError):
                pass  # se agrupa por el valor crudo

        agg: pd.DataFrame = (
            df.groupby(column)
            .agg(
                n=("rps", "count"),
                rps_mean=("rps", "mean"),
                rps_std=("rps", "std"),
                residual_mean=("residual", "mean"),
                ytrue_mean=("y_true", "mean"),
                pred_mean=("mean_pred", "mean"),
            )
            .reset_index()
            .sort_values(column)
        )
        agg["rps_rolling"] = (agg["rps_mean"]
                              .rolling(roll, min_periods=1).mean())
        print(agg.to_string(index=False))

        rho, p_value = _spearman(
            np.arange(len(agg)), agg["rps_mean"].to_numpy(),
        )
        trend: str = "empeora" if rho > 0 else "mejora"
        sig: str = "significativa" if p_value < 0.05 else "no significativa"
        print(f"\n  Tendencia Spearman rho={rho:.3f} (p={p_value:.4f}) - "
              f"{trend} | {sig}")

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        x: NDArray = np.arange(len(agg))

        axes[0].bar(x, agg["rps_mean"], color="steelblue", alpha=0.6,
                    label="RPS medio")
        axes[0].plot(x, agg["rps_rolling"], color="navy", lw=2,
                     label=f"Rolling {roll}")
        axes[0].axhline(self.rps.mean(), color="red", ls="--", lw=1.2,
                        label=f"Global={self.rps.mean():.4f}")
        axes[0].set_ylabel("RPS medio")
        axes[0].set_title(f"Drift temporal - {self.model_name} "
                          f"({self.component}) | rho={rho:.3f} {trend} ({sig})")
        axes[0].legend(fontsize=8)

        axes[1].bar(x, agg["residual_mean"],
                    color=["#d9534f" if r > 0 else "#5bc0de"
                           for r in agg["residual_mean"]], alpha=0.8)
        axes[1].axhline(0, color="black", ls="--", lw=1.2)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(agg[column].astype(str), rotation=45,
                                ha="right", fontsize=8)
        axes[1].set_ylabel("Residuo medio")
        axes[1].set_title("Sesgo temporal")

        plt.tight_layout()
        plt.show()

        return agg

    # =========================================================================
    # 6. Sobredispersion (solo NegBin)
    # =========================================================================

    def summarize_dispersion(self) -> Optional[Dict[str, float]]:
        """Diagnostica si la NegBin esta capturando la sobredispersion real."""
        if self.family != "negbin" or self.pearson_resid is None:
            print("  [SKIP] Diagnostico de dispersion solo aplica a NegBin.")
            return None

        _section(f"[alpha] Sobredispersion - {self.model_name} | "
                 f"{self.component}")

        alpha: NDArray = np.asarray(self.params["alpha"], dtype=float).ravel()
        pearson_var: float = float(self.pearson_resid.var())

        print(f"  alpha mean    : {alpha.mean():.4f}")
        print(f"  alpha median  : {np.median(alpha):.4f}")
        print(f"  alpha p05/p95 : {np.percentile(alpha, 5):.4f} / "
              f"{np.percentile(alpha, 95):.4f}")
        print(f"  Var(Pearson)  : {pearson_var:.4f}   <- ~1 si NB ajusta bien")

        if alpha.mean() < 0.05:
            verdict = "alpha muy bajo: la NB colapsa a Poisson (no aporta)"
        elif pearson_var > 1.3:
            verdict = "Var(Pearson) >> 1: queda sobredispersion sin capturar"
        elif pearson_var < 0.7:
            verdict = "Var(Pearson) << 1: posible overfitting de alpha"
        else:
            verdict = "dispersion razonable"

        print(f"  Diagnostico   : {verdict}")

        return {
            "alpha_mean": float(alpha.mean()),
            "alpha_median": float(np.median(alpha)),
            "pearson_var": pearson_var,
        }

    # =========================================================================
    # 7. Correlacion del error entre componentes
    # =========================================================================

    @classmethod
    def correlation_between(
        cls,
        analyzer_a: "DistributionErrorAnalyzer",
        analyzer_b: "DistributionErrorAnalyzer",
        figsize: Tuple[int, int] = (10, 5),
    ) -> Dict[str, float]:
        """Correlacion de errores entre los dos componentes del target.

        Correlacion alta sugiere senal estructural del evento que ningun
        componente esta capturando (y rompe el supuesto de independencia que
        usa la convolucion del total).
        """
        _section(f"[6] Correlacion de error {analyzer_a.component} vs "
                 f"{analyzer_b.component} | {analyzer_a.model_name}")

        if analyzer_a.n != analyzer_b.n:
            raise ValueError(
                f"Distinto numero de observaciones: {analyzer_a.n} vs "
                f"{analyzer_b.n}."
            )

        rho_rps, p_rps = _spearman(analyzer_a.rps, analyzer_b.rps)
        rho_res, p_res = _spearman(analyzer_a.residuals,
                                   analyzer_b.residuals)
        rho_log, p_log = _spearman(analyzer_a.log_residuals,
                                   analyzer_b.log_residuals)

        print(f"  N observaciones    : {analyzer_a.n:,}")
        print(f"  Spearman rho (RPS) : {rho_rps:.4f}  p={p_rps:.4e}")
        print(f"  Spearman rho (res) : {rho_res:.4f}  p={p_res:.4e}")
        print(f"  Spearman rho (log) : {rho_log:.4f}  p={p_log:.4e}")

        interp: str = (
            "Correlacion alta: hay senal estructural del evento sin capturar; "
            "revisar el supuesto de independencia del total."
            if abs(rho_rps) > 0.3
            else "Correlacion baja: los componentes fallan de forma "
                 "independiente."
        )
        print(f"\n  Interpretacion: {interp}")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].scatter(analyzer_a.rps, analyzer_b.rps, alpha=0.3, s=12,
                        color="steelblue", edgecolors="none")
        axes[0].set_xlabel(f"RPS - {analyzer_a.component}")
        axes[0].set_ylabel(f"RPS - {analyzer_b.component}")
        axes[0].set_title(f"Correlacion RPS | rho={rho_rps:.3f}")

        axes[1].scatter(analyzer_a.residuals, analyzer_b.residuals, alpha=0.3,
                        s=12, color="salmon", edgecolors="none")
        axes[1].axhline(0, color="gray", ls="--", lw=0.8)
        axes[1].axvline(0, color="gray", ls="--", lw=0.8)
        axes[1].set_xlabel(f"Residuo - {analyzer_a.component}")
        axes[1].set_ylabel(f"Residuo - {analyzer_b.component}")
        axes[1].set_title(f"Correlacion residuos | rho={rho_res:.3f}")

        plt.tight_layout()
        plt.show()

        return {
            "rho_rps": float(rho_rps), "p_rps": float(p_rps),
            "rho_residual": float(rho_res), "p_residual": float(p_res),
            "rho_log": float(rho_log), "p_log": float(p_log),
        }

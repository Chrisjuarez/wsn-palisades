"""Static figure helpers for the k-sweep results.

Functions take either the long-format ``df_k_sweep`` (the metrics DataFrame
returned by :func:`wsn_palisades.ksweep.run_k_sweep_all`) or the nested
``res_k_sweep`` mapping. They return the matplotlib ``Figure`` so callers
can show / save / embed in Streamlit.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .optimizers import _distance_to_ideal, _prep_optimizer_data, _scalar_score4

OPTIMIZERS_DEFAULT = ("Random", "Greedy", "Simple-NSGA", "Seeded-NSGA")
SCENARIO_ORDER_DEFAULT = ("FLAT", "DEM", "DSM/CHM")


def _ieee_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 130,
            "savefig.dpi": 300,
        }
    )


def _normalize_df(df: pd.DataFrame, optimizers: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    df["scenario"] = df["scenario"].astype(str).str.strip()
    df["optimizer"] = df["optimizer"].astype(str).str.strip()
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    return df[df["optimizer"].isin(optimizers)].sort_values(["scenario", "optimizer", "K"])


def _scenarios_in_df(df: pd.DataFrame, prefer: Sequence[str]) -> list[str]:
    found = list(df["scenario"].unique())
    ordered = [s for s in prefer if s in found]
    extras = [s for s in found if s not in ordered]
    return ordered + extras


def plot_metric_vs_k(
    df_k_sweep: pd.DataFrame,
    metric: str = "coverage_pct",
    ylabel: Optional[str] = None,
    optimizers: Sequence[str] = OPTIMIZERS_DEFAULT,
    scenarios: Sequence[str] = SCENARIO_ORDER_DEFAULT,
    title: Optional[str] = None,
):
    """One subplot per scenario, lines per optimizer, ``metric`` vs K."""
    _ieee_style()
    df = _normalize_df(df_k_sweep, optimizers)
    scen_list = _scenarios_in_df(df, scenarios)

    fig, axes = plt.subplots(1, len(scen_list), figsize=(5 * len(scen_list), 4.6), sharey=True)
    if len(scen_list) == 1:
        axes = [axes]

    for ax, scen in zip(axes, scen_list):
        sub = df[df["scenario"] == scen]
        for opt in optimizers:
            g = sub[sub["optimizer"] == opt].sort_values("K")
            if g.empty:
                continue
            ax.plot(g["K"].astype(int), g[metric], marker="o", linewidth=2, label=opt)
        ax.set_title(scen)
        ax.set_xlabel("Number of sensors, K")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(ylabel or metric)
    axes[0].legend(frameon=True, loc="best")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_coverage_vs_k(df_k_sweep: pd.DataFrame, **kwargs):
    return plot_metric_vs_k(
        df_k_sweep,
        metric="coverage_pct",
        ylabel="Coverage (%)",
        title=kwargs.pop("title", "Coverage vs K"),
        **kwargs,
    )


def plot_gamma_vs_k(df_k_sweep: pd.DataFrame, **kwargs):
    return plot_metric_vs_k(
        df_k_sweep,
        metric="gamma_mean",
        ylabel=r"Mean visibility $\bar{\gamma}$",
        title=kwargs.pop("title", "Visibility vs K"),
        **kwargs,
    )


def plot_spacing_vs_k(df_k_sweep: pd.DataFrame, **kwargs):
    return plot_metric_vs_k(
        df_k_sweep,
        metric="d_mean_m",
        ylabel=r"Mean spacing $\bar{d}$ (m)",
        title=kwargs.pop("title", "Spacing vs K"),
        **kwargs,
    )


def plot_solar_vs_k(df_k_sweep: pd.DataFrame, **kwargs):
    return plot_metric_vs_k(
        df_k_sweep,
        metric="solar_mean",
        ylabel="Mean solar (normalized)",
        title=kwargs.pop("title", "Solar vs K"),
        **kwargs,
    )


def plot_distance_to_ideal_vs_k(
    df_k_sweep: pd.DataFrame,
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    Dmax_per_scenario: Optional[dict] = None,
    optimizers: Sequence[str] = OPTIMIZERS_DEFAULT,
    scenarios: Sequence[str] = SCENARIO_ORDER_DEFAULT,
):
    """Distance-to-ideal in normalised 4-objective space (lower is better)."""
    _ieee_style()
    df = _normalize_df(df_k_sweep, optimizers).copy()
    if Dmax_per_scenario is None:
        Dmax_per_scenario = (
            df.groupby("scenario")["d_mean_m"].max().replace(0.0, 1.0).to_dict()
        )
    df["dist_to_ideal"] = [
        _distance_to_ideal(
            row.coverage_pct, row.gamma_mean, row.d_mean_m, row.solar_mean,
            Dmax_per_scenario.get(row.scenario, 1.0), weights,
        )
        for row in df.itertuples()
    ]
    fig = plot_metric_vs_k(
        df,
        metric="dist_to_ideal",
        ylabel="Distance to ideal (lower = better)",
        title="Distance to ideal vs K",
        optimizers=optimizers,
        scenarios=scenarios,
    )
    return fig


def plot_pareto_tradeoff(
    df_k_sweep: pd.DataFrame,
    x: str = "coverage_pct",
    y: str = "gamma_mean",
    optimizers: Sequence[str] = OPTIMIZERS_DEFAULT,
    scenarios: Sequence[str] = SCENARIO_ORDER_DEFAULT,
):
    """Scatter: x vs y, color = optimizer, marker = scenario, size scales with K."""
    _ieee_style()
    df = _normalize_df(df_k_sweep, optimizers)
    scen_list = _scenarios_in_df(df, scenarios)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    color_map = {o: c for o, c in zip(optimizers, plt.cm.tab10.colors)}
    marker_map = {"FLAT": "o", "DEM": "s", "DSM/CHM": "^"}

    for opt in optimizers:
        for scen in scen_list:
            g = df[(df["optimizer"] == opt) & (df["scenario"] == scen)]
            if g.empty:
                continue
            ax.scatter(
                g[x],
                g[y],
                c=[color_map.get(opt, "gray")],
                marker=marker_map.get(scen, "o"),
                s=20 + (g["K"].astype(float) - 10) * 2.5,
                alpha=0.85,
                edgecolors="black",
                linewidths=0.4,
                label=f"{opt} ({scen})",
            )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x} — by optimizer × scenario")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, frameon=True, loc="best")
    fig.tight_layout()
    return fig


def plot_summary_four_panel(
    df_k_sweep: pd.DataFrame,
    optimizers: Sequence[str] = OPTIMIZERS_DEFAULT,
    scenarios: Sequence[str] = SCENARIO_ORDER_DEFAULT,
):
    """One figure with the four primary metrics vs K, three rows of scenarios."""
    _ieee_style()
    df = _normalize_df(df_k_sweep, optimizers)
    scen_list = _scenarios_in_df(df, scenarios)

    metrics = [
        ("coverage_pct", "Coverage (%)"),
        ("gamma_mean", r"$\bar{\gamma}$"),
        ("d_mean_m", r"$\bar{d}$ (m)"),
        ("solar_mean", "Solar (norm.)"),
    ]
    fig, axes = plt.subplots(
        len(scen_list), 4, figsize=(16, 3.5 * len(scen_list)), sharex=True
    )
    if len(scen_list) == 1:
        axes = np.expand_dims(axes, 0)

    for r, scen in enumerate(scen_list):
        sub = df[df["scenario"] == scen]
        for c, (col, lbl) in enumerate(metrics):
            ax = axes[r, c]
            for opt in optimizers:
                gg = sub[sub["optimizer"] == opt].sort_values("K")
                if gg.empty:
                    continue
                ax.plot(gg["K"].astype(int), gg[col], marker="o", linewidth=1.8, label=opt)
            ax.set_title(f"{scen} — {lbl}")
            ax.grid(True, alpha=0.3)
            if r == len(scen_list) - 1:
                ax.set_xlabel("K")
            if c == 0:
                ax.set_ylabel("value")

    axes[0, 0].legend(fontsize=8, ncol=2, frameon=True)
    fig.tight_layout()
    return fig


__all__ = [
    "plot_metric_vs_k",
    "plot_coverage_vs_k",
    "plot_gamma_vs_k",
    "plot_spacing_vs_k",
    "plot_solar_vs_k",
    "plot_distance_to_ideal_vs_k",
    "plot_pareto_tradeoff",
    "plot_summary_four_panel",
]

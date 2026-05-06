"""Render the canonical IEEE figures from a saved ``df_k_sweep.csv``.

Example:
    python scripts/make_figures.py \\
        --csv results/csv/df_k_sweep_k=10to60_PalisadesFinal.csv \\
        --out results/figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wsn_palisades import plotting as wp_plotting


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        default="results/csv/df_k_sweep_k=10to60_PalisadesFinal.csv",
        help="Path to df_k_sweep CSV",
    )
    ap.add_argument("--out", default="results/figures", help="Output directory")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} rows from {args.csv}")

    figures = {
        "fig1_coverage_vs_k.png": wp_plotting.plot_coverage_vs_k(df),
        "fig2_distance_to_ideal_vs_k.png": wp_plotting.plot_distance_to_ideal_vs_k(df),
        "fig3_pareto_coverage_gamma.png": wp_plotting.plot_pareto_tradeoff(
            df, x="coverage_pct", y="gamma_mean"
        ),
        "fig4_summary_panel.png": wp_plotting.plot_summary_four_panel(df),
        "fig5_gamma_vs_k.png": wp_plotting.plot_gamma_vs_k(df),
        "fig6_spacing_vs_k.png": wp_plotting.plot_spacing_vs_k(df),
        "fig7_solar_vs_k.png": wp_plotting.plot_solar_vs_k(df),
    }
    for name, fig in figures.items():
        path = out / name
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()

"""Run all four optimizers across a range of K values, across all scenarios."""

from __future__ import annotations

import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .optimizers import (
    NSGA_THREADS,
    _normalize_weights4,
    _prep_optimizer_data,
    _scalar_score4,
    greedy_select,
    nsga_select,
    random_select,
    simple_nsga_select,
)
from .params import RANDOM_TRIALS_DEFAULT, SensorParams, nsga_params_for_k


def _idxs_to_str(idxs) -> str:
    return ",".join(map(str, list(map(int, idxs))))


def _mean_pairwise_from_packs(packs, idxs) -> float:
    _, _, _, D, _ = _prep_optimizer_data(packs)
    idxs = list(map(int, idxs))
    if len(idxs) <= 1:
        return 0.0
    sub = D[np.ix_(idxs, idxs)]
    iu = np.triu_indices(len(idxs), 1)
    return float(sub[iu].mean())


def _row_from_res(label: str, K: int, optimizer_name: str, res, packs, weights) -> dict:
    cov = -float(res["F"][0])
    g = -float(res["F"][1])
    s = -float(res["F"][3])
    d_mean = _mean_pairwise_from_packs(packs, res["idxs"])
    _, _, _, _, Dmax = _prep_optimizer_data(packs)
    scalar4 = _scalar_score4(cov, g, d_mean, s, Dmax, weights)
    idxs = np.array(res["idxs"], dtype=int)
    return {
        "scenario": label,
        "K": int(K),
        "optimizer": optimizer_name,
        "coverage_pct": float(cov),
        "gamma_mean": float(g),
        "d_mean_m": float(d_mean),
        "solar_mean": float(s),
        "scalar_score4": float(scalar4),
        "idxs_str": _idxs_to_str(idxs),
        "idxs": idxs,
        "weights": tuple(weights),
    }


def run_k_sweep_all(
    packs_by_label: dict,
    SP: SensorParams,
    K_values: Iterable[int] = range(10, 61, 10),
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    random_seed: int = 42,
    spacing_metric: str = "mean",
    deterministic_per_case: bool = True,
    random_trials: int = RANDOM_TRIALS_DEFAULT,
    n_threads: int = NSGA_THREADS,
    verbose: bool = True,
):
    """Run random / greedy / simple-NSGA / seeded-NSGA for every (scenario, K).

    Returns ``(df_metrics, results_nested)`` where ``df_metrics`` has one row per
    (scenario, K, optimizer) and ``results_nested[label][K][optimizer]`` holds
    the full result dict.
    """
    weights = _normalize_weights4(weights, strict=False)

    rows: list[dict] = []
    results: dict = {label: {} for label in packs_by_label}

    for K in K_values:
        for label, packs in packs_by_label.items():
            if verbose:
                print(f"--- {label} K={K} ---")
            t0 = time.time()
            params = nsga_params_for_k(K)

            if deterministic_per_case:
                case_seed = (
                    int(random_seed) * 1_000_003
                    + int(K) * 9_176
                    + (abs(hash(label)) % 100_000)
                ) % (2**32 - 1)
                np.random.seed(int(case_seed))
                rs_for_nsga = int(case_seed)
            else:
                rs_for_nsga = int(random_seed)

            r = random_select(packs, K, SP, weights=weights, trials=random_trials)
            g = greedy_select(packs, K, SP, weights=weights)
            sn = simple_nsga_select(
                packs, K, SP,
                weights=weights,
                max_gen=max(250, params["max_gen"] // 3),
                pop_mult=params["pop_mult"],
                partitions=params["partitions"],
                spacing_metric=spacing_metric,
                use_threads=True,
                n_threads=n_threads,
                random_seed=rs_for_nsga,
            )
            sn2 = nsga_select(
                packs, K, SP,
                weights=weights,
                external_seeds=g["idxs"],
                max_gen=params["max_gen"],
                pop_mult=params["pop_mult"],
                partitions=params["partitions"],
                spacing_metric=spacing_metric,
                multi_seed=True,
                wgrid_L=4,
                use_threads=True,
                n_threads=n_threads,
                random_seed=rs_for_nsga,
            )

            results[label][int(K)] = {
                "random": r,
                "greedy": g,
                "simple_nsga": sn,
                "seeded_nsga": sn2,
            }

            if verbose:
                print(f"  done in {time.time() - t0:.1f}s")

            rows.append(_row_from_res(label, K, "Random", r, packs, weights))
            rows.append(_row_from_res(label, K, "Greedy", g, packs, weights))
            rows.append(_row_from_res(label, K, "Simple-NSGA", sn, packs, weights))
            rows.append(_row_from_res(label, K, "Seeded-NSGA", sn2, packs, weights))

    return pd.DataFrame(rows), results


__all__ = ["run_k_sweep_all"]

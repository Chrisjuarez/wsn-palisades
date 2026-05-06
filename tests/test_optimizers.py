"""Smoke tests for the K-subset selectors using a tiny synthetic ``packs`` dict."""

import numpy as np
import pandas as pd

from wsn_palisades import SensorParams
from wsn_palisades.optimizers import (
    _cov_union_fast,
    _distance_to_ideal,
    _normalize_weights4,
    _scalar_score4,
    greedy_select,
    random_select,
)


def _synthetic_packs(n: int = 30, n_grid: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    # spread n candidates across a 4 km grid (degrees ≈ meters / 111000)
    lon0, lat0 = -118.55, 34.05
    metres = rng.uniform(-2000, 2000, size=(n, 2))
    lons = lon0 + metres[:, 0] / 92_000.0
    lats = lat0 + metres[:, 1] / 111_000.0

    masks = rng.random((n, n_grid)) < 0.35
    metrics = pd.DataFrame(
        {
            "lon": lons,
            "lat": lats,
            "gamma_mean": rng.uniform(0.4, 0.95, size=n),
            "Aeff_m2": rng.uniform(50_000, 800_000, size=n),
            "solar_kwhm2_yr": rng.uniform(1500, 2200, size=n),
        }
    )
    s = metrics["solar_kwhm2_yr"].to_numpy()
    metrics["solar_norm"] = (s - s.min()) / (s.max() - s.min())
    metrics["solar_norm_robust"] = metrics["solar_norm"]
    return {"candidates": list(zip(lons, lats)), "masks": masks, "metrics": metrics}


def test_random_select_returns_k_unique():
    SP = SensorParams(R_m=300.0, min_sep_m=100.0)
    packs = _synthetic_packs()
    out = random_select(packs, k=6, SP=SP, trials=30)
    assert out["idxs"].shape == (6,)
    assert len(set(out["idxs"].tolist())) == 6
    assert 0.0 <= out["coverage_pct"] <= 100.0


def test_greedy_select_returns_k_unique():
    SP = SensorParams(R_m=300.0, min_sep_m=100.0)
    packs = _synthetic_packs()
    out = greedy_select(packs, k=6, SP=SP)
    assert len(set(out["idxs"].tolist())) == 6
    assert 0.0 <= out["coverage_pct"] <= 100.0
    assert 0.0 <= out["gamma_mean"] <= 1.0


def test_score_helpers_bounded():
    weights = (0.25, 0.25, 0.25, 0.25)
    s = _scalar_score4(50.0, 0.6, 1500.0, 0.7, Dmax=3000.0, weights4=weights)
    d = _distance_to_ideal(50.0, 0.6, 1500.0, 0.7, Dmax=3000.0, weights4=weights)
    assert 0.0 <= s <= 1.0
    assert 0.0 <= d <= 1.0


def test_normalize_weights_sums_to_one():
    w = _normalize_weights4((1, 2, 3, 4), strict=False)
    assert abs(sum(w) - 1.0) < 1e-9


def test_cov_union_fast_consistent():
    SP = SensorParams(R_m=300.0, min_sep_m=100.0)
    packs = _synthetic_packs()
    # Trigger the bit-packing path
    out = greedy_select(packs, k=4, SP=SP)
    fast = _cov_union_fast(packs, out["idxs"])
    assert 0.0 <= fast <= 100.0

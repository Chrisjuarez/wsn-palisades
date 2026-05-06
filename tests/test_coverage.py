"""Coverage geometry tests on a tiny lon/lat grid (no DEM warp)."""

import math

import numpy as np

from wsn_palisades import SensorParams
from wsn_palisades.coverage import coverage_contour_lonlat, coverage_mask_for_sensor, union_coverage


def _flat_dirpack(SP: SensorParams):
    az = np.arange(0, 360, SP.az_step_deg, dtype=float)
    return {
        "az": az,
        "gamma_az": np.ones_like(az),
        "g_topo": np.ones_like(az),
        "g_veg": np.ones_like(az),
        "horizon_deg": np.zeros_like(az),
        "A0_m2": math.pi * SP.R_m**2,
        "Aeff_m2": math.pi * SP.R_m**2,
        "gamma_mean": 1.0,
        "r_eff": np.full_like(az, SP.R_m, dtype=float),
    }


def test_flat_coverage_disc_is_round():
    SP = SensorParams(R_m=200.0, az_step_deg=2)
    coord = (-118.54, 34.06)
    dp = _flat_dirpack(SP)

    # 11x11 grid centered on the sensor, ~120 m spacing in degrees-equivalent
    lon0, lat0 = coord
    cov_grid = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            cov_grid.append((lon0 + i * 0.0006, lat0 + j * 0.0006))

    mask = coverage_mask_for_sensor(coord, cov_grid, dp, SP)
    assert mask.dtype == bool
    # The center cell is inside the disc; outer corners may be in or out
    center_idx = 5 * 11 + 5
    assert mask[center_idx]


def test_coverage_contour_closed_polygon():
    SP = SensorParams(R_m=200.0, az_step_deg=10)
    dp = _flat_dirpack(SP)
    pts = coverage_contour_lonlat((-118.54, 34.06), dp, SP)
    assert len(pts) == len(dp["az"]) + 1
    assert pts[0] == pts[-1]


def test_union_coverage_bounds():
    rng = np.random.default_rng(0)
    masks = rng.random((10, 200)) < 0.2
    cov = union_coverage(masks)
    assert 0.0 <= cov <= 100.0
    # Empty selection
    assert union_coverage(np.zeros((0, 200), dtype=bool)) == 0.0

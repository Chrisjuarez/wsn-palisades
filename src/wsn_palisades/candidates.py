"""Candidate sensor locations + per-scenario precompute (visibility + solar)."""

from __future__ import annotations

import json
import os
import random
import time
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from shapely.geometry import Point, Polygon, shape

from .coverage import coverage_mask_for_sensor
from .params import SensorParams, SolarParams
from .solar import compute_poa_clearsky_candidate
from .surfaces import DEMManager, elev_surface_at_utm, slope_aspect_from_dem_utm
from .visibility import directional_gamma


# -- AOI helpers --------------------------------------------------------------


def load_aoi(geojson_path: str) -> Polygon:
    """Read a GeoJSON polygon (Feature or Geometry) and return a shapely Polygon."""
    with open(geojson_path) as f:
        gj = json.load(f)
    geom = gj.get("geometry", gj) if "geometry" in gj else gj
    if geom.get("type") == "FeatureCollection":
        geom = gj["features"][0]["geometry"]
    return shape(geom)


def generate_coverage_grid(polygon: Polygon, grid_size: int = 60) -> List[Tuple[float, float]]:
    """Uniform lon/lat grid clipped to the polygon."""
    min_x, min_y, max_x, max_y = polygon.bounds
    xs = np.linspace(min_x, max_x, grid_size)
    ys = np.linspace(min_y, max_y, grid_size)
    pts = []
    for x in xs:
        for y in ys:
            if polygon.contains(Point(x, y)):
                pts.append((x, y))
    return pts


def sample_points_in_poly(polygon: Polygon, n_points: int) -> List[Tuple[float, float]]:
    minx, miny, maxx, maxy = polygon.bounds
    pts = []
    while len(pts) < n_points:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            pts.append((p.x, p.y))
    return pts


def build_sensor_candidates(
    aoi: Polygon, mode: str = "grid", grid_size: int = 30, n_random: int = 400
):
    if mode == "grid":
        return generate_coverage_grid(aoi, grid_size=grid_size)
    return sample_points_in_poly(aoi, n_random)


# -- Single-process scenario precompute (kept for clarity / tests) ------------


def precompute_scenario(
    aoi: Polygon,
    dmgr: DEMManager,
    scenario: str,
    SP: SensorParams,
    candidate_mode: str = "grid",
    grid_size: int = 30,
    n_random: int = 400,
    cov_grid_size: int = 80,
):
    """Sequential reference implementation. Returns the same schema as the parallel version.

    Output dict keys: ``candidates``, ``cov_grid``, ``dirpacks``, ``masks``, ``metrics``.
    """
    candidates = build_sensor_candidates(
        aoi, mode=candidate_mode, grid_size=grid_size, n_random=n_random
    )
    cov_grid = generate_coverage_grid(aoi, grid_size=cov_grid_size)

    dirpacks, masks, rows = [], [], []
    for c in candidates:
        dp = directional_gamma(dmgr, c, scenario, SP)
        dirpacks.append(dp)
        masks.append(coverage_mask_for_sensor(c, cov_grid, dp, SP, dmgr))
        rows.append({"lon": c[0], "lat": c[1], "gamma_mean": dp["gamma_mean"], "Aeff_m2": dp["Aeff_m2"]})
    masks = np.vstack(masks).astype(bool)
    return {
        "candidates": candidates,
        "cov_grid": cov_grid,
        "dirpacks": dirpacks,
        "masks": masks,
        "metrics": pd.DataFrame(rows),
    }


# -- Parallel scenario precompute (loky backend) ------------------------------


def _scenario_worker(
    idx: int,
    coord,
    scenario: str,
    SP: SensorParams,
    cov_grid,
    dmgr: DEMManager,
    solar_params: SolarParams,
):
    """Top-level worker so loky can pickle it."""
    dp = directional_gamma(dmgr, coord, scenario, SP)
    m = coverage_mask_for_sensor(coord, cov_grid, dp, SP, dmgr)

    lon, lat = coord
    try:
        x, y = dmgr._ll2utm.transform(lon, lat)
        elev_m = float(elev_surface_at_utm(dmgr, x, y, mode="dem"))
        slope_deg, aspect_deg = slope_aspect_from_dem_utm(dmgr, x, y)
    except Exception:
        elev_m = float(dmgr.get_elevation((lon, lat)))
        slope_deg, aspect_deg = 0.0, 180.0

    if scenario == "flat":
        solar_kwhm2_yr = 1.0
        svf = 1.0
        dni_block_frac = 0.0
    else:
        try:
            solar_res = compute_poa_clearsky_candidate(
                lat=lat,
                lon=lon,
                elev_m=elev_m,
                slope_deg=slope_deg,
                aspect_deg=aspect_deg,
                horizon_az_deg=dp["az"],
                horizon_elev_deg=dp["horizon_deg"],
                params=solar_params,
            )
            solar_kwhm2_yr = float(solar_res.get("poa_kwh_m2_yr", 0.0))
            svf = float(solar_res.get("svf", float("nan")))
            dni_block_frac = float(solar_res.get("dni_block_frac", float("nan")))
        except Exception:
            solar_kwhm2_yr = 0.0
            svf = float("nan")
            dni_block_frac = float("nan")

    r = {
        "lon": float(lon),
        "lat": float(lat),
        "gamma_mean": float(dp.get("gamma_mean", 0.0)),
        "Aeff_m2": float(dp.get("Aeff_m2", 0.0)),
        "solar_kwhm2_yr": solar_kwhm2_yr,
        "svf": svf if np.isfinite(svf) else np.nan,
        "dni_block_frac": dni_block_frac if np.isfinite(dni_block_frac) else np.nan,
    }
    return idx, dp, m, r


def precompute_scenario_loky(
    aoi: Polygon,
    dmgr: DEMManager,
    scenario: str,
    SP: SensorParams,
    candidate_mode: str = "grid",
    grid_size: int = 30,
    n_random: int = 400,
    cov_grid_size: int = 80,
    n_jobs: Optional[int] = None,
    batch_size: int = 1,
    solar_params: Optional[SolarParams] = None,
    verbose: bool = True,
) -> dict:
    """Process-parallel precompute via joblib's loky backend.

    Returns a dict with keys ``candidates``, ``cov_grid``, ``dirpacks``, ``masks``,
    ``metrics``. The metrics DataFrame is augmented with ``solar_norm`` (min-max)
    and ``solar_norm_robust`` (2nd–98th percentile range).
    """
    candidates = build_sensor_candidates(
        aoi, mode=candidate_mode, grid_size=grid_size, n_random=n_random
    )
    cov_grid = generate_coverage_grid(aoi, grid_size=cov_grid_size)
    if solar_params is None:
        solar_params = SolarParams()

    n_jobs = n_jobs or max(1, (os.cpu_count() or 4) - 1)
    if verbose:
        print(f"[{scenario}] {len(candidates)} candidates, cov_grid={len(cov_grid)} (n_jobs={n_jobs})")
    t0 = time.time()

    dirpacks: List = [None] * len(candidates)
    masks: List = [None] * len(candidates)
    rows: List = [None] * len(candidates)

    with parallel_backend("loky", n_jobs=n_jobs):
        results = Parallel(batch_size=batch_size)(
            delayed(_scenario_worker)(i, c, scenario, SP, cov_grid, dmgr, solar_params)
            for i, c in enumerate(candidates)
        )

    for i, dp, m, r in results:
        dirpacks[i] = dp
        masks[i] = m
        rows[i] = r

    masks = np.vstack(masks).astype(bool)
    metrics = pd.DataFrame(rows)
    for col in ["lon", "lat", "gamma_mean", "Aeff_m2", "solar_kwhm2_yr", "svf", "dni_block_frac"]:
        if col in metrics.columns:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")

    if "solar_kwhm2_yr" in metrics.columns:
        s = metrics["solar_kwhm2_yr"].to_numpy(float)
        if np.all(~np.isfinite(s)):
            metrics["solar_norm"] = 0.0
            metrics["solar_norm_robust"] = 0.0
        else:
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            rng = np.ptp(s)
            if rng < 1e-6:
                metrics["solar_norm"] = 1.0
                metrics["solar_norm_robust"] = 1.0
            else:
                metrics["solar_norm"] = (s - s.min()) / (rng + 1e-12)
                lo, hi = np.quantile(s, [0.02, 0.98])
                rngr = max(1e-12, hi - lo)
                metrics["solar_norm_robust"] = np.clip((s - lo) / rngr, 0.0, 1.0)

    if verbose:
        print(f"[{scenario}] done in {time.time() - t0:.1f}s")

    return {
        "candidates": candidates,
        "cov_grid": cov_grid,
        "dirpacks": dirpacks,
        "masks": masks,
        "metrics": metrics,
    }


def precompute_all(
    aoi: Polygon,
    dmgr: DEMManager,
    SP: SensorParams,
    grid_size: int = 30,
    cov_grid_size: int = 80,
    n_jobs: Optional[int] = None,
    solar_params: Optional[SolarParams] = None,
    verbose: bool = True,
):
    """Run the three canonical scenarios and return (packs_flat, packs_dem, packs_dsmchm)."""
    common = dict(
        SP=SP,
        grid_size=grid_size,
        cov_grid_size=cov_grid_size,
        n_jobs=n_jobs,
        batch_size=4,
        solar_params=solar_params,
        verbose=verbose,
    )
    packs_flat = precompute_scenario_loky(aoi, dmgr, "flat", **common)
    packs_dem = precompute_scenario_loky(aoi, dmgr, "dem", **common)
    packs_dsmchm = precompute_scenario_loky(aoi, dmgr, "dsm_chm", **common)
    return packs_flat, packs_dem, packs_dsmchm


__all__ = [
    "load_aoi",
    "generate_coverage_grid",
    "sample_points_in_poly",
    "build_sensor_candidates",
    "precompute_scenario",
    "precompute_scenario_loky",
    "precompute_all",
    "_scenario_worker",
]

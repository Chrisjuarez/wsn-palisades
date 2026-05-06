"""Per-azimuth visibility γ(θ): topographic openness and vegetation attenuation."""

from __future__ import annotations

import math

import numpy as np

from .params import SensorParams
from .surfaces import DEMManager, horizon_and_veg_profiles_utm


def meters_per_deg(lat: float):
    m_per_deg_lat = 111_000.0
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat))
    return m_per_deg_lon, m_per_deg_lat


def _step_from(coord, az_deg: float, step_m: float):
    lon0, lat0 = coord
    mlon, mlat = meters_per_deg(lat0)
    dx = math.sin(math.radians(az_deg)) * step_m
    dy = math.cos(math.radians(az_deg)) * step_m
    return (lon0 + dx / mlon, lat0 + dy / mlat)


def horizon_profile(
    dmgr: DEMManager,
    coord,
    radius_m: float,
    az_step_deg: int,
    step_m: float,
    sensor_h_m: float,
    mode: str = "dem",
    stop_at_deg: float | None = None,
):
    """Lon/lat fallback path — only used when the DEM hasn't been UTM-warped.

    Returns (az, horizon_deg).
    """
    az = np.arange(0, 360, az_step_deg, dtype=float)
    horz = np.zeros_like(az)

    def zsurf(c):
        if mode == "flat":
            return dmgr.get_elevation(coord)
        if mode == "dem":
            return dmgr.get_elevation(c)
        if dmgr.dsm_array is not None:
            return dmgr.get_surface_elevation(c)
        if dmgr.chm_array is not None:
            return dmgr.get_elevation(c) + dmgr.get_canopy_height(c)
        return dmgr.get_elevation(c)

    z0 = zsurf(coord) + sensor_h_m
    steps = max(2, int(radius_m // step_m))
    for i, a in enumerate(az):
        max_alpha = 0.0
        c = coord
        for s in range(1, steps + 1):
            c = _step_from(c, a, step_m)
            dist = s * step_m
            z_here = zsurf(c) + sensor_h_m
            alpha = math.degrees(math.atan2(z_here - z0, dist))
            if alpha > max_alpha:
                max_alpha = alpha
                if stop_at_deg is not None and max_alpha >= stop_at_deg:
                    break
        horz[i] = max(0.0, max_alpha)
    return az, horz


def gamma_topo_az(dmgr: DEMManager, coord, SP: SensorParams, mode: str = "dem"):
    """Per-azimuth topographic openness g_topo(θ) ∈ {0, 1}.

    1 if the local horizon stays below SP.theta_topo_deg, else 0.
    """
    stop_at = SP.theta_topo_deg + 0.25
    az, horz = horizon_profile(
        dmgr,
        coord,
        SP.R_m,
        SP.az_step_deg,
        SP.step_m,
        SP.sensor_height_m,
        mode=mode,
        stop_at_deg=stop_at,
    )
    open_mask = horz <= SP.theta_topo_deg
    return az, open_mask.astype(float), horz


def vegetation_gamma_az(dmgr: DEMManager, coord, SP: SensorParams):
    """Per-azimuth vegetation attenuation γ_veg(θ).

    Density along ray: V(θ) = mean( min(1, CHM/h_ref) ) over CHM ≥ h_thr.
    γ_veg(θ) = exp(-α V) (default) or 1 - α V (if SP.veg_mode == "linear").
    """
    if dmgr.chm_array is None and dmgr.chm_utm is None:
        az = np.arange(0, 360, SP.az_step_deg, dtype=float)
        return az, np.ones_like(az, dtype=float), np.zeros_like(az, dtype=float)

    az = np.arange(0, 360, SP.az_step_deg, dtype=float)
    V = np.zeros_like(az, dtype=float)

    steps = max(2, int(SP.R_m // SP.step_m))
    for i, a in enumerate(az):
        vals = []
        c = coord
        for _ in range(1, steps + 1):
            c = _step_from(c, a, SP.step_m)
            ch = dmgr.get_canopy_height(c)
            if ch >= SP.h_thr_m:
                vals.append(min(1.0, ch / SP.h_ref_m))
        V[i] = float(np.mean(vals)) if vals else 0.0

    if SP.veg_mode == "linear":
        g = np.clip(1.0 - SP.alpha_veg * V, 0.0, 1.0)
    else:
        g = np.exp(-SP.alpha_veg * V)
    return az, g, V


def directional_gamma(dmgr: DEMManager, coord, scenario: str, SP: SensorParams) -> dict:
    """Combine topography + vegetation into γ(θ) and effective radius r_eff(θ).

    scenario ∈ {"flat", "dem", "dsm_chm"}. Uses UTM-meters geometry when
    `warp_surfaces_to_utm` has been run on `dmgr`; otherwise falls back to
    lon/lat sampling.
    """
    A0 = math.pi * SP.R_m**2

    if dmgr.utm_crs is not None:
        x0, y0 = dmgr._ll2utm.transform(coord[0], coord[1])
        if scenario == "flat":
            az = np.arange(0, 360, SP.az_step_deg, dtype=float)
            gamma_az = np.ones_like(az)
            g_topo = np.ones_like(az)
            g_veg = np.ones_like(az)
            horz = np.zeros_like(az)
        elif scenario == "dem":
            az, gamma_az, g_topo, horz = horizon_and_veg_profiles_utm(dmgr, x0, y0, SP, mode="dem")
            g_veg = np.ones_like(az)
        elif scenario == "dsm_chm":
            az, gamma_az, g_topo, horz = horizon_and_veg_profiles_utm(
                dmgr, x0, y0, SP, mode="dsm_chm"
            )
            g_veg = np.clip(gamma_az / np.maximum(g_topo, 1e-6), 0.0, 1.0)
        else:
            raise ValueError("scenario must be 'flat'|'dem'|'dsm_chm'")
    else:
        if scenario == "flat":
            az = np.arange(0, 360, SP.az_step_deg, dtype=float)
            g_topo = np.ones_like(az)
            g_veg = np.ones_like(az)
            horz = np.zeros_like(az)
        elif scenario == "dem":
            az, g_topo, horz = gamma_topo_az(dmgr, coord, SP, mode="dem")
            g_veg = np.ones_like(az)
        elif scenario == "dsm_chm":
            az, g_topo, horz = gamma_topo_az(dmgr, coord, SP, mode="dsm_chm")
            _, g_veg, _ = vegetation_gamma_az(dmgr, coord, SP)
        else:
            raise ValueError("scenario must be 'flat'|'dem'|'dsm_chm'")
        gamma_az = np.clip(g_topo * g_veg, 0.0, 1.0)

    gamma_mean = float(np.nanmean(gamma_az))
    Aeff = A0 * gamma_mean
    r_eff = SP.R_m * np.sqrt(gamma_az)
    return {
        "az": az,
        "gamma_az": gamma_az,
        "g_topo": g_topo,
        "g_veg": g_veg,
        "horizon_deg": horz,
        "A0_m2": A0,
        "Aeff_m2": Aeff,
        "gamma_mean": gamma_mean,
        "r_eff": r_eff,
    }


__all__ = [
    "meters_per_deg",
    "horizon_profile",
    "gamma_topo_az",
    "vegetation_gamma_az",
    "directional_gamma",
]

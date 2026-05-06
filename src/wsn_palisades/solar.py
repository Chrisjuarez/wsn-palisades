"""Per-candidate solar irradiance under terrain horizon shading.

Wraps pvlib clearsky + Perez transposition with a horizon mask derived from the
per-azimuth elevation profile we already compute for visibility. Returns annual
plane-of-array energy (kWh/m^2/yr) plus a sky-view factor and direct-beam
blocking fractions.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

import pvlib
from pvlib.atmosphere import get_relative_airmass
from pvlib.clearsky import lookup_linke_turbidity
from pvlib.irradiance import get_extra_radiation, get_total_irradiance
from pvlib.location import Location

from .params import SolarParams


def _wrap_360(a):
    a = np.asarray(a, dtype=float)
    return np.mod(a, 360.0)


def _interp_horizon_at_az(
    h_az_deg: np.ndarray, h_elev_deg: np.ndarray, query_az_deg: np.ndarray
) -> np.ndarray:
    """Linear interpolation of horizon elevation (deg) at arbitrary azimuths (deg)."""
    az = _wrap_360(h_az_deg)
    elev = np.asarray(h_elev_deg, dtype=float)
    order = np.argsort(az)
    az, elev = az[order], elev[order]
    az_ext = np.concatenate([az, az[:1] + 360.0])
    elev_ext = np.concatenate([elev, elev[:1]])
    q = _wrap_360(query_az_deg)
    return np.interp(q, az_ext, elev_ext)


def sky_view_factor_cos2(h_elev_deg: np.ndarray) -> float:
    """Cos^2 horizon-based isotropic SVF approximation."""
    h = np.radians(np.asarray(h_elev_deg, dtype=float))
    return float(np.clip(np.mean(np.cos(h) ** 2), 0.0, 1.0))


def build_times(params: SolarParams) -> pd.DatetimeIndex:
    return pd.date_range(
        f"{params.year}-01-01",
        f"{params.year + 1}-01-01",
        inclusive="left",
        freq=params.freq,
        tz=params.tz,
    )


def compute_poa_clearsky_candidate(
    lat: float,
    lon: float,
    elev_m: float,
    slope_deg: float,
    aspect_deg: float,
    horizon_az_deg: Sequence[float],
    horizon_elev_deg: Sequence[float],
    params: SolarParams,
) -> Dict[str, float]:
    """Annual plane-of-array irradiance for one candidate sensor location.

    Returns dict with keys: poa_kwh_m2_yr, svf, dni_block_frac (all hours),
    dni_block_frac_day (daylight only).
    """
    times = build_times(params)

    loc = Location(latitude=lat, longitude=lon, tz=params.tz, altitude=elev_m)
    sp = loc.get_solarposition(times)
    zen = np.asarray(
        sp["apparent_zenith" if params.use_apparent_zenith else "zenith"].values, dtype=float
    )
    az = np.asarray(sp["azimuth"].values, dtype=float)
    alt = 90.0 - zen
    day = alt > 0.0

    try:
        linke = lookup_linke_turbidity(times, lat, lon)
        cs = loc.get_clearsky(times, model="ineichen", linke_turbidity=linke)
    except Exception:
        cs = loc.get_clearsky(times, model="haurwitz")

    dni = cs["dni"].to_numpy(float)
    ghi = cs["ghi"].to_numpy(float)
    dhi = cs["dhi"].to_numpy(float)

    hz = _interp_horizon_at_az(np.asarray(horizon_az_deg), np.asarray(horizon_elev_deg), az)
    blocked_hz = alt <= hz
    blocked_all = blocked_hz | (alt <= 0.0)

    dni_block_frac = float(np.mean(blocked_all)) if blocked_all.size else float("nan")
    dni_block_frac_day = float(np.mean(blocked_hz[day])) if np.any(day) else float("nan")

    if params.use_horizon_for_direct:
        dni = np.where(blocked_all, 0.0, dni)
        mu0 = np.clip(np.cos(np.radians(zen)), 0.0, 1.0)
        ghi = dni * mu0 + dhi

    svf = sky_view_factor_cos2(horizon_elev_deg)
    if params.use_svf_for_diffuse:
        dhi = dhi * svf

    dni_extra = get_extra_radiation(times).to_numpy(float)
    am_rel = get_relative_airmass(zen, model="kastenyoung1989")

    model = (
        "isotropic"
        if (params.use_svf_for_diffuse and params.svf_pair_isotropic)
        else params.diffuse_model
    )

    poa = get_total_irradiance(
        surface_tilt=max(0.0, float(slope_deg)),
        surface_azimuth=float(_wrap_360(aspect_deg)),
        solar_zenith=zen,
        solar_azimuth=az,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        dni_extra=dni_extra,
        airmass=am_rel,
        albedo=params.albedo,
        model=model,
    )
    poa_glob = np.asarray(poa["poa_global"], float)

    if len(times) >= 2:
        dt_h = np.diff(times.asi8) / 3_600_000_000_000.0
        energy_wh_m2 = float(np.sum(0.5 * (poa_glob[:-1] + poa_glob[1:]) * dt_h))
    else:
        step_h = pd.Timedelta(params.freq).total_seconds() / 3600.0
        energy_wh_m2 = float(poa_glob[0]) * step_h

    return {
        "poa_kwh_m2_yr": energy_wh_m2 / 1000.0,
        "svf": float(svf),
        "dni_block_frac": dni_block_frac,
        "dni_block_frac_day": dni_block_frac_day,
    }

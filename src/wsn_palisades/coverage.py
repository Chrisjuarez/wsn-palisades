"""Per-sensor coverage masks and contour helpers."""

from __future__ import annotations

import math

import numpy as np

from .params import SensorParams
from .surfaces import DEMManager
from .visibility import meters_per_deg


def coverage_mask_for_sensor(coord, cov_grid, dirpack, SP: SensorParams, dmgr: DEMManager = None):
    """Boolean mask: which cells of cov_grid are inside this sensor's directional radius.

    Uses UTM meters when `warp_surfaces_to_utm` has run on `dmgr`, else falls back
    to a meters-per-degree approximation around the sensor latitude.
    """
    az = dirpack["az"]
    r_eff = dirpack["r_eff"]

    if dmgr is not None and dmgr.utm_crs is not None:
        x0, y0 = dmgr._ll2utm.transform(coord[0], coord[1])
        xs, ys = zip(*[dmgr._ll2utm.transform(x, y) for (x, y) in cov_grid])
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        dx = xs - x0
        dy = ys - y0
    else:
        lon0, lat0 = coord
        mlon, mlat = meters_per_deg(lat0)
        dx = (np.array([p[0] for p in cov_grid]) - lon0) * mlon
        dy = (np.array([p[1] for p in cov_grid]) - lat0) * mlat

    d = np.hypot(dx, dy)
    az_pt = (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0
    k = (np.round(az_pt / SP.az_step_deg).astype(int)) % len(az)
    return d <= (r_eff[k] + 1e-9)


def coverage_contour_lonlat(coord, dirpack, SP: SensorParams):
    """Closed lon/lat polygon tracing the per-azimuth effective radius."""
    lon0, lat0 = coord
    az = dirpack["az"]
    r_eff = dirpack["r_eff"]
    mlon, mlat = meters_per_deg(lat0)
    pts = []
    for k, a in enumerate(az):
        d = r_eff[k]
        dx = math.sin(math.radians(a)) * d
        dy = math.cos(math.radians(a)) * d
        pts.append((lon0 + dx / mlon, lat0 + dy / mlat))
    pts.append(pts[0])
    return pts


def union_coverage(mask_rows: np.ndarray) -> float:
    """Union coverage (% of grid covered by ANY sensor in the selection)."""
    if mask_rows.size == 0:
        return 0.0
    return 100.0 * float(np.any(mask_rows, axis=0).mean())


__all__ = [
    "coverage_mask_for_sensor",
    "coverage_contour_lonlat",
    "union_coverage",
]

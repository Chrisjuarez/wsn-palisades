"""Raster surfaces: DTM/DSM/CHM loading, UTM reprojection, and bilinear sampling.

Supports both local file paths and ``s3://bucket/key`` URIs. Public-read S3
buckets are accessed anonymously via ``AWS_NO_SIGN_REQUEST=YES`` so the app
needs no AWS credentials at runtime.
"""

from __future__ import annotations

import contextlib
import os
from typing import Optional, Tuple

import numpy as np
import rasterio
from pyproj import CRS, Geod, Transformer
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds
from shapely.geometry import Polygon, box, mapping
from shapely.ops import transform as shapely_transform

from .params import SensorParams


def _is_s3(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def _path_exists(path: str) -> bool:
    """os.path.exists that also returns True for s3:// URIs (existence is verified at open time)."""
    if _is_s3(path):
        return True  # rasterio.open will raise if the key is missing
    return os.path.exists(path)


def _rio_env(path: str):
    """rasterio.Env tuned for the path.

    For s3://, enables anonymous reads and range-request friendly settings.
    For all paths, sets GTIFF_SRS_SOURCE=EPSG so GDAL uses the canonical
    EPSG definition when a GeoTIFF's embedded WKT slightly differs from
    the registry (which is the case for our Palisades rasters tagged
    EPSG:6340 / NAD83(2011) UTM 11N).
    """
    common = {"GTIFF_SRS_SOURCE": "EPSG"}
    if _is_s3(path):
        return rasterio.Env(
            AWS_NO_SIGN_REQUEST="YES",
            GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            CPL_VSIL_CURL_USE_HEAD="NO",
            **common,
        )
    return rasterio.Env(**common)


# -- AOI helpers --------------------------------------------------------------


def reproject_aoi_to_raster(aoi_poly: Polygon, raster_crs) -> Polygon:
    """Project a WGS84 lon/lat polygon into the raster's CRS.

    Resilient to ``raster_crs`` arriving as a rasterio.CRS, pyproj.CRS,
    EPSG int, or a WKT/PROJ string. Raises with a useful message if pyproj
    can't build the transformer (e.g. missing PROJ data on the host).
    """
    if raster_crs is None:
        return aoi_poly

    # Normalise to a form pyproj definitely accepts.
    target = None
    for getter in ("to_epsg", "to_wkt", "to_proj4"):
        if hasattr(raster_crs, getter):
            try:
                val = getattr(raster_crs, getter)()
                if val:
                    target = f"EPSG:{val}" if getter == "to_epsg" else val
                    break
            except Exception:
                continue
    if target is None:
        target = raster_crs  # last-ditch — let pyproj try the original object

    try:
        project = Transformer.from_crs("EPSG:4326", target, always_xy=True).transform
    except Exception as e:
        raise RuntimeError(
            f"Could not build transformer EPSG:4326 -> {target!r}: {e}. "
            "If the raster is on S3, make sure it's a Cloud-Optimized GeoTIFF "
            "with a CRS that pyproj's bundled PROJ database understands."
        ) from e
    return shapely_transform(project, aoi_poly)


def fix_aoi_bounds(aoi_poly: Polygon, dem_manager: "DEMManager") -> Polygon:
    """Trim AOI so it stays inside the DEM bounds (with a tiny margin)."""
    aoi_b, dem_b = aoi_poly.bounds, dem_manager.bounds
    if (
        aoi_b[0] < dem_b[0]
        or aoi_b[1] < dem_b[1]
        or aoi_b[2] > dem_b[2]
        or aoi_b[3] > dem_b[3]
    ):
        safe = (
            max(aoi_b[0], dem_b[0] + 1e-4),
            max(aoi_b[1], dem_b[1] + 1e-4),
            min(aoi_b[2], dem_b[2] - 1e-4),
            min(aoi_b[3], dem_b[3] - 1e-4),
        )
        return box(*safe)
    return aoi_poly


def _utm_crs_for_ll(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg(32600 + zone if lat >= 0 else 32700 + zone)


# -- DEM manager --------------------------------------------------------------


class DEMManager:
    """Holds DEM (DTM) plus optional DSM/CHM rasters and provides sampling helpers.

    Two construction paths:
      - DEMManager(dem_bytes=...)   for in-memory bytes (e.g. an OpenTopography fetch)
      - DEMManager.from_files(...)  for local GeoTIFFs (DTM + optional DSM/CHM)
    """

    def __init__(self, dem_bytes: Optional[bytes] = None):
        self.dem_array = None
        self.transform = None
        self.bounds = None
        self.profile = None
        self.crs = None
        self.shape = None
        self.slope_arr = None
        self.aspect_arr = None

        self.dsm_array = None
        self.dsm_transform = None
        self.dsm_crs = None

        self.chm_array = None
        self.chm_transform = None
        self.chm_crs = None

        # UTM warp products, populated by warp_surfaces_to_utm
        self.utm_crs = None
        self.utm_transform = None
        self.dem_utm = None
        self.dsm_utm = None
        self.chm_utm = None
        self._ll2utm = None
        self._utm2ll = None

        # cached lon/lat -> raster CRS transformers for fast sampling
        self._tf_ll2dsm = None
        self._tf_ll2chm = None

        if dem_bytes is not None:
            self._load_dem_bytes(dem_bytes)

    def _load_dem_bytes(self, dem_bytes: bytes) -> None:
        with MemoryFile(dem_bytes) as memfile:
            with memfile.open() as src:
                self.dem_array = src.read(1)
                self.transform = src.transform
                self.bounds = src.bounds
                self.profile = src.profile
                self.crs = src.crs
        self.shape = self.dem_array.shape

    @classmethod
    def from_files(
        cls,
        aoi_poly: Polygon,
        dtm_path: str,
        dsm_path: Optional[str] = None,
        chm_path: Optional[str] = None,
    ) -> "DEMManager":
        """Load DTM/DSM/CHM from local GeoTIFFs **or s3:// URIs**, masking each to the AOI."""
        dm = cls()
        dm._load_dtm_file(dtm_path, aoi_poly)
        if dsm_path and _path_exists(dsm_path):
            dm._load_dsm_file(dsm_path, aoi_poly)
        if chm_path and _path_exists(chm_path):
            dm._load_chm_file(chm_path, aoi_poly)
        return dm

    def _load_dtm_file(self, path: str, aoi_poly: Polygon) -> None:
        with _rio_env(path), rasterio.open(path) as src:
            aoi_proj = reproject_aoi_to_raster(aoi_poly, src.crs)
            arr, tr = rio_mask(src, [mapping(aoi_proj)], crop=True)
            self.dem_array = arr[0]
            self.transform = tr
            self.crs = src.crs
            self.profile = src.profile
            nodata = src.nodata if src.nodata is not None else -999999.0
        self.dem_array = np.where(
            (self.dem_array == nodata) | (self.dem_array < -1000), np.nan, self.dem_array
        )
        h, w = self.dem_array.shape
        self.shape = (h, w)
        # Recover bounds from the masked transform
        minx, miny = self.transform * (0, h)
        maxx, maxy = self.transform * (w, 0)
        self.bounds = (minx, miny, maxx, maxy)

    def _load_dsm_file(self, path: str, aoi_poly: Polygon) -> None:
        with _rio_env(path), rasterio.open(path) as src:
            aoi_proj = reproject_aoi_to_raster(aoi_poly, src.crs)
            arr, tr = rio_mask(src, [mapping(aoi_proj)], crop=True)
            self.dsm_array = arr[0]
            self.dsm_transform = tr
            self.dsm_crs = src.crs
            nodata = src.nodata if src.nodata is not None else -999999.0
        self.dsm_array = np.where(
            (self.dsm_array == nodata) | (self.dsm_array < -1000), np.nan, self.dsm_array
        )

    def _load_chm_file(self, path: str, aoi_poly: Polygon) -> None:
        with _rio_env(path), rasterio.open(path) as src:
            aoi_proj = reproject_aoi_to_raster(aoi_poly, src.crs)
            arr, tr = rio_mask(src, [mapping(aoi_proj)], crop=True)
            self.chm_array = arr[0]
            self.chm_transform = tr
            self.chm_crs = src.crs
            nodata = src.nodata if src.nodata is not None else -999999.0
        self.chm_array = np.where(
            (self.chm_array == nodata) | (self.chm_array < -1000), np.nan, self.chm_array
        )

    # -- legacy DSM/CHM setters (kept for compatibility) ---
    def set_dsm(self, dsm_array, dsm_transform):
        self.dsm_array = dsm_array
        self.dsm_transform = dsm_transform

    def set_chm(self, chm_array, chm_transform):
        self.chm_array = chm_array
        self.chm_transform = chm_transform

    # -- slope / aspect from base DTM ---------------------------------------
    def degree_to_meter(self, lon: float, lat: float) -> Tuple[float, float]:
        geod = Geod(ellps="WGS84")
        dx = abs(self.transform.a)
        dy = abs(self.transform.e)
        _, _, x_meter = geod.inv(lon, lat, lon + dx, lat)
        _, _, y_meter = geod.inv(lon, lat, lon, lat + dy)
        return abs(x_meter), abs(y_meter)

    def calculate_slope_and_aspect(self) -> None:
        center_col = self.dem_array.shape[1] // 2
        center_row = self.dem_array.shape[0] // 2
        lon0, lat0 = self.transform * (center_col, center_row)
        dx_meter, dy_meter = self.degree_to_meter(lon0, lat0)
        gy, gx = np.gradient(self.dem_array.astype(float), dy_meter, dx_meter)
        self.slope_arr = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
        aspect_rad = np.arctan2(-gx, gy)
        self.aspect_arr = np.degrees(aspect_rad)
        self.aspect_arr[self.aspect_arr < 0] += 360.0

    # -- nearest-neighbour samplers (DEM/DSM/CHM in source CRS) -------------
    def _sample(self, arr, tr, coord):
        if arr is None or tr is None:
            return float("nan")
        col, row = ~tr * coord
        row, col = int(row), int(col)
        if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
            v = arr[row, col]
            return float(v) if np.isfinite(v) else float("nan")
        return float("nan")

    def get_elevation(self, coord: Tuple[float, float]) -> float:
        return self._sample(self.dem_array, self.transform, coord)

    def _ensure_tf_cache(self) -> None:
        if self._tf_ll2dsm is None and self.dsm_crs is not None and str(self.dsm_crs).lower() != "epsg:4326":
            self._tf_ll2dsm = Transformer.from_crs("EPSG:4326", self.dsm_crs, always_xy=True)
        if self._tf_ll2chm is None and self.chm_crs is not None and str(self.chm_crs).lower() != "epsg:4326":
            self._tf_ll2chm = Transformer.from_crs("EPSG:4326", self.chm_crs, always_xy=True)

    def get_surface_elevation(self, coord: Tuple[float, float]) -> float:
        if self.dsm_array is None:
            return self.get_elevation(coord)
        self._ensure_tf_cache()
        if self._tf_ll2dsm is not None:
            x, y = self._tf_ll2dsm.transform(coord[0], coord[1])
        else:
            x, y = coord
        return self._sample(self.dsm_array, self.dsm_transform, (x, y))

    def get_canopy_height(self, coord: Tuple[float, float]) -> float:
        if self.chm_array is None:
            return 0.0
        self._ensure_tf_cache()
        if self._tf_ll2chm is not None:
            x, y = self._tf_ll2chm.transform(coord[0], coord[1])
        else:
            x, y = coord
        v = self._sample(self.chm_array, self.chm_transform, (x, y))
        return float(v) if np.isfinite(v) else 0.0


# -- UTM warp + bilinear sampling --------------------------------------------


def warp_surfaces_to_utm(
    dmgr: DEMManager, aoi_poly: Polygon, target_res_m: float = 2.0
) -> None:
    """Warp DEM/DSM/CHM to a single aligned UTM grid (meters)."""
    cx, cy = aoi_poly.centroid.x, aoi_poly.centroid.y
    utm = _utm_crs_for_ll(cx, cy)

    transform, width, height = calculate_default_transform(
        dmgr.crs,
        utm,
        dmgr.dem_array.shape[1],
        dmgr.dem_array.shape[0],
        *dmgr.bounds,
        resolution=target_res_m,
    )

    def _rp(src_arr, src_tr, src_crs):
        dst = np.empty((height, width), dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src_tr,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=utm,
            resampling=Resampling.bilinear,
            num_threads=2,
        )
        return dst

    dmgr.dem_utm = _rp(dmgr.dem_array, dmgr.transform, dmgr.crs)
    dmgr.dsm_utm = (
        _rp(dmgr.dsm_array, dmgr.dsm_transform, dmgr.dsm_crs or dmgr.crs)
        if dmgr.dsm_array is not None
        else None
    )
    dmgr.chm_utm = (
        _rp(dmgr.chm_array, dmgr.chm_transform, dmgr.chm_crs or dmgr.crs)
        if dmgr.chm_array is not None
        else None
    )

    dmgr.utm_crs = utm
    dmgr.utm_transform = transform
    dmgr._ll2utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    dmgr._utm2ll = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)


def _bilinear(arr, tr, x, y):
    """Bilinear sample at UTM (x, y) meters. Returns NaN if outside."""
    if arr is None:
        return float("nan")
    colf, rowf = (~tr) * (x, y)
    r0, c0 = int(np.floor(rowf)), int(np.floor(colf))
    if r0 < 0 or c0 < 0 or r0 + 1 >= arr.shape[0] or c0 + 1 >= arr.shape[1]:
        return float("nan")
    dr, dc = rowf - r0, colf - c0
    a = arr[r0, c0]
    b = arr[r0, c0 + 1]
    c = arr[r0 + 1, c0]
    d = arr[r0 + 1, c0 + 1]
    return a * (1 - dc) * (1 - dr) + b * dc * (1 - dr) + c * (1 - dc) * dr + d * dc * dr


def elev_surface_at_utm(dmgr: DEMManager, x: float, y: float, mode: str) -> float:
    if mode == "flat" or mode == "dem":
        return _bilinear(dmgr.dem_utm, dmgr.utm_transform, x, y)
    # dsm_chm: prefer DSM; else DEM + CHM
    if dmgr.dsm_utm is not None:
        return _bilinear(dmgr.dsm_utm, dmgr.utm_transform, x, y)
    base = _bilinear(dmgr.dem_utm, dmgr.utm_transform, x, y)
    ch = 0.0 if dmgr.chm_utm is None else _bilinear(dmgr.chm_utm, dmgr.utm_transform, x, y)
    return base + max(0.0, ch)


def horizon_and_veg_profiles_utm(
    dmgr: DEMManager, x0: float, y0: float, SP: SensorParams, mode: str = "dem"
):
    """Per-azimuth gamma (= openness * vegetation) using UTM samplers.

    Returns (az, gamma_az, g_topo, horizon_deg).
    """
    az = np.arange(0, 360, SP.az_step_deg, dtype=float)
    steps = max(2, int(SP.R_m // SP.step_m))
    s = np.linspace(SP.step_m, steps * SP.step_m, steps, dtype=float)
    vx = np.sin(np.deg2rad(az))
    vy = np.cos(np.deg2rad(az))

    z0 = elev_surface_at_utm(dmgr, x0, y0, mode) + SP.sensor_height_m

    horizon = np.zeros_like(az, dtype=float)
    g_topo = np.zeros_like(az, dtype=float)
    g_veg = np.ones_like(az, dtype=float)

    theta_thr = SP.theta_topo_deg
    R = SP.R_m

    for i in range(az.size):
        xs = x0 + vx[i] * s
        ys = y0 + vy[i] * s

        z_s = (
            np.array(
                [elev_surface_at_utm(dmgr, xs[k], ys[k], mode) for k in range(steps)],
                dtype=float,
            )
            + SP.sensor_height_m
        )
        alpha = np.degrees(np.arctan2(z_s - z0, s))
        horizon[i] = np.nanmax(alpha)

        ok = np.where(alpha <= theta_thr)[0]
        r_open = float(s[ok[-1]]) if ok.size else 0.0
        g_topo[i] = (r_open / R) ** 2

        if mode == "dsm_chm" and dmgr.chm_utm is not None and r_open > 0:
            ch = np.array(
                [_bilinear(dmgr.chm_utm, dmgr.utm_transform, xs[k], ys[k]) for k in range(steps)],
                dtype=float,
            )
            ch = np.nan_to_num(ch, nan=0.0)
            use = (s <= r_open) & (ch >= SP.h_thr_m)
            if np.any(use):
                v = np.minimum(1.0, ch[use] / SP.h_ref_m)
                dens = float(np.mean(v))
            else:
                dens = 0.0
            g_veg[i] = (
                np.exp(-SP.alpha_veg * dens)
                if SP.veg_mode != "linear"
                else float(np.clip(1.0 - SP.alpha_veg * dens, 0.0, 1.0))
            )
        else:
            g_veg[i] = 1.0

    gamma = np.clip(g_topo * g_veg, 0.0, 1.0)
    return az, gamma, g_topo, horizon


def slope_aspect_from_dem_utm(dmgr: DEMManager, x: float, y: float) -> Tuple[float, float]:
    """Local slope (deg) and aspect (deg, 0=N, CW) from the UTM DEM via a 3x3 patch."""
    if dmgr.dem_utm is None or dmgr.utm_transform is None:
        return 0.0, 180.0
    colf, rowf = (~dmgr.utm_transform) * (x, y)
    r0, c0 = int(np.floor(rowf)), int(np.floor(colf))
    arr = dmgr.dem_utm
    if r0 < 1 or c0 < 1 or r0 + 1 >= arr.shape[0] or c0 + 1 >= arr.shape[1]:
        return 0.0, 180.0
    dy = abs(dmgr.utm_transform.e)
    dx = abs(dmgr.utm_transform.a)
    patch = arr[r0 - 1 : r0 + 2, c0 - 1 : c0 + 2].astype(float)
    gy = (patch[2, :].mean() - patch[0, :].mean()) / (2 * dy)
    gx = (patch[:, 2].mean() - patch[:, 0].mean()) / (2 * dx)
    slope = float(np.degrees(np.arctan(np.hypot(gx, gy))))
    aspect = float(np.degrees(np.arctan2(-gx, gy)))
    if aspect < 0:
        aspect += 360.0
    return slope, aspect


__all__ = [
    "DEMManager",
    "warp_surfaces_to_utm",
    "_utm_crs_for_ll",
    "_bilinear",
    "elev_surface_at_utm",
    "horizon_and_veg_profiles_utm",
    "slope_aspect_from_dem_utm",
    "reproject_aoi_to_raster",
    "fix_aoi_bounds",
]

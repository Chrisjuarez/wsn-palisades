"""Build per-scenario coverage contours for the Explore Results page.

Loads the canonical Palisades AOI, fetches the DTM/DSM/CHM rasters from
S3 (or uses local copies in ``data/`` if present), runs the same per-azimuth
visibility ray-cast that ``precompute_scenario_loky`` uses, and emits
``results/contours_palisades.pkl.gz`` — a small lookup of:

    {
        "FLAT":     {idx: [(lon, lat), ...closed polygon], ...},
        "DEM":      {idx: [(lon, lat), ...], ...},
        "DSM/CHM":  {idx: [(lon, lat), ...], ...},
    }

Streamlit's Explore Results page reads this file once at startup so the
canonical sweep can render the irregular footprints without re-running the
optimizer or shipping the heavy ``packs_*`` objects.

Run once locally::

    python scripts/save_contours.py

Takes ~10-25 minutes at grid_size=30 depending on CPU count. Re-run only if
the AOI, sensor params, or rasters change.
"""

from __future__ import annotations

import gzip
import os
import pickle
import sys
from pathlib import Path

# Conda's base env often exports PROJ_DATA pointing at an old proj.db that
# rasterio/pyproj wheels can't read. Clear those before the geospatial imports
# so the wheels use their own bundled PROJ database.
for _v in ("PROJ_DATA", "PROJ_LIB"):
    os.environ.pop(_v, None)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from dotenv import load_dotenv  # noqa: E402

from wsn_palisades.candidates import (  # noqa: E402
    load_aoi,
    precompute_scenario_loky,
)
from wsn_palisades.coverage import coverage_contour_lonlat  # noqa: E402
from wsn_palisades.data_uris import chm_uri, dsm_uri, dtm_uri  # noqa: E402
from wsn_palisades.params import SensorParams, SolarParams  # noqa: E402
from wsn_palisades.surfaces import DEMManager, warp_surfaces_to_utm  # noqa: E402

GRID_SIZE = 30  # must match the canonical sweep's grid_size
COV_GRID_SIZE = 40
SCENARIOS = [("FLAT", "flat"), ("DEM", "dem"), ("DSM/CHM", "dsm_chm")]


def _resolve_raster_paths() -> dict:
    """Prefer local ./data rasters; fall back to s3:// URIs."""
    data = REPO / "data"
    local = {
        "dtm": data / "palisadesoutput.dtm.tif",
        "dsm": data / "palisadesoutput.dsm.tif",
        "chm": data / "palisadesCHM.tif",
    }
    if all(p.exists() for p in local.values()):
        print(f"Using local rasters from {data}/")
        return {k: str(v) for k, v in local.items()}
    print("Local rasters not present — streaming from S3.")
    return {"dtm": dtm_uri(), "dsm": dsm_uri(), "chm": chm_uri()}


def main():
    load_dotenv()
    n_jobs = int(os.environ.get("WSN_N_JOBS", "4"))

    aoi_path = REPO / "sample" / "aoi_palisades.geojson"
    aoi = load_aoi(str(aoi_path))
    print(f"AOI: {aoi_path.name}, area ~{aoi.area * 111**2:.2f} deg² (rough)")

    SP = SensorParams()
    solar = SolarParams()

    paths = _resolve_raster_paths()
    print("Loading rasters and warping to UTM...")
    dmgr = DEMManager.from_files(
        aoi_poly=aoi,
        dtm_path=paths["dtm"],
        dsm_path=paths["dsm"],
        chm_path=paths["chm"],
    )
    dmgr.calculate_slope_and_aspect()
    warp_surfaces_to_utm(dmgr, aoi, target_res_m=2.0)

    contours: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for label, mode in SCENARIOS:
        print(f"\n=== {label} ({mode}) ===")
        packs = precompute_scenario_loky(
            aoi, dmgr, mode, SP,
            grid_size=GRID_SIZE, cov_grid_size=COV_GRID_SIZE,
            n_jobs=n_jobs, solar_params=solar, verbose=True,
        )
        cands = packs["candidates"]
        dirpacks = packs["dirpacks"]
        per_scenario: dict[int, list[tuple[float, float]]] = {}
        for i, (c, dp) in enumerate(zip(cands, dirpacks)):
            poly = coverage_contour_lonlat(c, dp, SP)
            per_scenario[i] = [(float(lon), float(lat)) for lon, lat in poly]
        contours[label] = per_scenario
        print(f"  -> {len(per_scenario)} contours")

    out_path = REPO / "results" / "contours_palisades.pkl.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wb") as f:
        pickle.dump(contours, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {sum(len(v) for v in contours.values())} contours")
    print(f"  -> {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

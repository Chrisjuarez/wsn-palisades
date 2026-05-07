"""S3 URIs and bounds for the canonical Palisades rasters.

The bucket is configured public-read, so reads happen anonymously
(``AWS_NO_SIGN_REQUEST=YES``). Used by the Streamlit Live AOI page to
stream the COG-encoded DTM/DSM/CHM rasters without downloading them.
"""

from __future__ import annotations

import os
from typing import Final

# Bucket comes from .env / Streamlit Secrets so it's overridable per env.
DEFAULT_BUCKET: Final[str] = "wsn-palisades-data"


def bucket_name() -> str:
    return os.environ.get("WSN_DATA_BUCKET", DEFAULT_BUCKET)


def _uri(key: str) -> str:
    return f"s3://{bucket_name()}/{key}"


# Raster URIs --------------------------------------------------------------

def dtm_uri() -> str:
    return _uri("palisadesoutput.dtm.tif")


def dsm_uri() -> str:
    return _uri("palisadesoutput.dsm.tif")


def chm_uri() -> str:
    return _uri("palisadesCHM.tif")


# Palisades raster coverage bounds (lon_min, lat_min, lon_max, lat_max) in WGS84.
# Taken directly from the DSM/CHM/DTM rasters (CRS:EPSG:6340, 0.5 m resolution,
# 21761 x 23453 px). Drawn AOIs in the Live page must fall inside this box.
PALISADES_BOUNDS: Final[tuple[float, float, float, float]] = (
    -118.584253, 34.033582, -118.464458, 34.140766,
)
# (lat, lon) for folium initial centering — uses the raster's natural center.
PALISADES_CENTER: Final[tuple[float, float]] = (34.087174, -118.524355)

# The Palisades rasters are tagged NAD83(2011) / UTM zone 11N (EPSG:6340).
# Older PROJ databases (e.g. Streamlit Cloud's) lose this on read and surface
# the CRS as a useless ``LOCAL_CS`` WKT, so we substitute the canonical EPSG
# code in ``surfaces.py``. The pixel data and transforms are unchanged.
PALISADES_RASTER_EPSG: Final[int] = 6340


__all__ = [
    "DEFAULT_BUCKET",
    "bucket_name",
    "dtm_uri",
    "dsm_uri",
    "chm_uri",
    "PALISADES_BOUNDS",
    "PALISADES_CENTER",
    "PALISADES_RASTER_EPSG",
]

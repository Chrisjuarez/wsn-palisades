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


# Palisades coverage bounds (lon_min, lat_min, lon_max, lat_max) in WGS84.
# Drawn AOIs in the Live page must fall inside this box.
PALISADES_BOUNDS: Final[tuple[float, float, float, float]] = (
    -118.5550, 34.0450, -118.5100, 34.0750,
)
PALISADES_CENTER: Final[tuple[float, float]] = (
    (PALISADES_BOUNDS[1] + PALISADES_BOUNDS[3]) / 2,  # lat
    (PALISADES_BOUNDS[0] + PALISADES_BOUNDS[2]) / 2,  # lon
)


__all__ = [
    "DEFAULT_BUCKET",
    "bucket_name",
    "dtm_uri",
    "dsm_uri",
    "chm_uri",
    "PALISADES_BOUNDS",
    "PALISADES_CENTER",
]

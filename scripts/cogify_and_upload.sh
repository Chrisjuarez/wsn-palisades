#!/usr/bin/env bash
# Convert the three Palisades rasters to Cloud-Optimized GeoTIFFs and re-upload
# them to s3://${WSN_DATA_BUCKET}/. COGs let the Streamlit app stream only the
# AOI window over HTTP via rasterio + GDAL's /vsis3/ driver.
#
# Run this once after you've installed rio-cogeo.
#
# Prereqs:
#   pip install rio-cogeo
#   AWS CLI configured (`aws sts get-caller-identity` works)
#
# Usage:
#   ./scripts/cogify_and_upload.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${HOME}/moo_node"
WORK_DIR="${TMPDIR:-/tmp}/wsn_cog"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a; source "${REPO_ROOT}/.env"; set +a
fi
: "${WSN_DATA_BUCKET:?WSN_DATA_BUCKET must be set in .env}"

FILES=(
  "palisadesoutput.dtm.tif"
  "palisadesoutput.dsm.tif"
  "palisadesCHM.tif"
)

mkdir -p "${WORK_DIR}"
echo "==> Working dir: ${WORK_DIR}"

# Sanity check rio is installed
if ! command -v rio >/dev/null 2>&1; then
  echo "rio-cogeo not installed. Run: pip install rio-cogeo" >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  src="${SOURCE_DIR}/${f}"
  dst="${WORK_DIR}/${f}"
  if [[ ! -f "${src}" ]]; then
    echo "  MISSING: ${src} (skipping)" >&2
    continue
  fi
  echo "==> Converting ${f} to COG..."
  # deflate compression + bilinear overviews (good for continuous DEMs)
  # Use 'nearest' for the CHM which can have sharp edges
  if [[ "${f}" == *"CHM"* ]]; then
    OVR=nearest
  else
    OVR=bilinear
  fi
  rio cogeo create \
    --overview-resampling "${OVR}" \
    --cog-profile deflate \
    "${src}" "${dst}"

  echo "==> Validating COG ${f}..."
  rio cogeo validate "${dst}"

  echo "==> Uploading s3://${WSN_DATA_BUCKET}/${f}..."
  aws s3 cp "${dst}" "s3://${WSN_DATA_BUCKET}/${f}"

  rm -f "${dst}"
done

echo
echo "==> Done. Quick sanity check:"
echo "    aws s3 ls s3://${WSN_DATA_BUCKET}/"
echo "    rio cogeo info /vsis3/${WSN_DATA_BUCKET}/palisadesCHM.tif"

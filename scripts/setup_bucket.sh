#!/usr/bin/env bash
# Create the WSN-Palisades S3 bucket, configure it for public-read,
# and upload the three large rasters from ~/moo_node.
#
# Prerequisites:
#   - AWS CLI v2 installed (`brew install awscli`)
#   - Credentials configured (`aws configure` or AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env)
#   - .env in repo root with WSN_DATA_BUCKET and AWS_DEFAULT_REGION set
#
# Usage:
#   ./scripts/setup_bucket.sh                # create + configure + upload
#   ./scripts/setup_bucket.sh --skip-upload  # create + configure only
#   ./scripts/setup_bucket.sh --upload-only  # skip creation, just upload

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${HOME}/moo_node"
FILES=(
  "palisadesoutput.dtm.tif"
  "palisadesoutput.dsm.tif"
  "palisadesCHM.tif"
)

# Load .env
if [[ -f "${REPO_ROOT}/.env" ]]; then
  # shellcheck disable=SC1090
  set -a; source "${REPO_ROOT}/.env"; set +a
fi

: "${WSN_DATA_BUCKET:?WSN_DATA_BUCKET must be set in .env}"
: "${AWS_DEFAULT_REGION:?AWS_DEFAULT_REGION must be set in .env}"

BUCKET="${WSN_DATA_BUCKET}"
REGION="${AWS_DEFAULT_REGION}"

SKIP_UPLOAD=0
SKIP_CREATE=0
for arg in "$@"; do
  case "$arg" in
    --skip-upload)  SKIP_UPLOAD=1 ;;
    --upload-only)  SKIP_CREATE=1 ;;
    -h|--help)      sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "Unknown flag: $arg" >&2; exit 1 ;;
  esac
done

echo "==> Bucket: ${BUCKET} (region ${REGION})"
aws sts get-caller-identity >/dev/null

if [[ "${SKIP_CREATE}" -eq 0 ]]; then
  if aws s3api head-bucket --bucket "${BUCKET}" 2>/dev/null; then
    echo "==> Bucket already exists, skipping create."
  else
    echo "==> Creating bucket..."
    if [[ "${REGION}" == "us-east-1" ]]; then
      aws s3api create-bucket --bucket "${BUCKET}" --region "${REGION}"
    else
      aws s3api create-bucket \
        --bucket "${BUCKET}" \
        --region "${REGION}" \
        --create-bucket-configuration "LocationConstraint=${REGION}"
    fi
  fi

  echo "==> Disabling Block Public Access..."
  aws s3api put-public-access-block \
    --bucket "${BUCKET}" \
    --public-access-block-configuration \
      "BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false"

  echo "==> Applying public-read bucket policy..."
  POLICY=$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGetObject",
      "Effect": "Allow",
      "Principal": "*",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::${BUCKET}/*"
    }
  ]
}
EOF
)
  aws s3api put-bucket-policy --bucket "${BUCKET}" --policy "${POLICY}"

  echo "==> Enabling CORS (so the Streamlit app can fetch tiles from a browser)..."
  CORS=$(cat <<'EOF'
{
  "CORSRules": [
    {
      "AllowedOrigins": ["*"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedHeaders": ["*"],
      "MaxAgeSeconds": 3000
    }
  ]
}
EOF
)
  aws s3api put-bucket-cors --bucket "${BUCKET}" --cors-configuration "${CORS}"
fi

if [[ "${SKIP_UPLOAD}" -eq 0 ]]; then
  echo "==> Uploading rasters from ${SOURCE_DIR}..."
  for f in "${FILES[@]}"; do
    src="${SOURCE_DIR}/${f}"
    if [[ ! -f "${src}" ]]; then
      echo "  MISSING: ${src} (skipping)" >&2
      continue
    fi
    size=$(du -h "${src}" | cut -f1)
    echo "  uploading ${f} (${size})..."
    aws s3 cp "${src}" "s3://${BUCKET}/${f}"
  done
fi

echo
echo "==> Done. Verify with:"
echo "    aws s3 ls s3://${BUCKET}/"
echo "    curl -I https://${BUCKET}.s3.${REGION}.amazonaws.com/palisadesCHM.tif"

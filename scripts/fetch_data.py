"""Fetch the Palisades DEM/DSM/CHM rasters from S3 into ./data/.

Reads the bucket name from the ``WSN_DATA_BUCKET`` env var (or ``.env``).
Public-read buckets work without AWS credentials. Files already present
with a matching size are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv

DEFAULT_FILES = [
    "palisadesoutput.dtm.tif",
    "palisadesoutput.dsm.tif",
    "palisadesCHM.tif",
]


def _client(unsigned: bool):
    if unsigned:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return boto3.client("s3")


def _head_size(client, bucket: str, key: str) -> int | None:
    try:
        return int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
    except Exception:
        return None


def _download(client, bucket: str, key: str, dest: Path) -> None:
    print(f"  downloading s3://{bucket}/{key} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))


def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default=os.environ.get("WSN_DATA_BUCKET"),
                    help="S3 bucket (defaults to $WSN_DATA_BUCKET)")
    ap.add_argument("--prefix", default="",
                    help="Optional key prefix inside the bucket")
    ap.add_argument("--dest", default="data", help="Local destination directory")
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES,
                    help="Specific file names to fetch (default: the three Palisades rasters)")
    ap.add_argument("--public", action="store_true",
                    help="Use unsigned/anonymous S3 (public-read bucket)")
    args = ap.parse_args()

    if not args.bucket:
        print("error: bucket not set (pass --bucket or set WSN_DATA_BUCKET in .env)", file=sys.stderr)
        sys.exit(2)

    client = _client(unsigned=args.public)
    dest_dir = Path(args.dest).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"fetching {len(args.files)} files from s3://{args.bucket}/{args.prefix} -> {dest_dir}/")
    for name in args.files:
        key = (args.prefix.rstrip("/") + "/" + name).lstrip("/")
        local = dest_dir / name
        remote_size = _head_size(client, args.bucket, key)
        if local.exists() and remote_size is not None and local.stat().st_size == remote_size:
            print(f"  ok        {name} (already {remote_size} bytes)")
            continue
        _download(client, args.bucket, key, local)
    print("done.")


if __name__ == "__main__":
    main()

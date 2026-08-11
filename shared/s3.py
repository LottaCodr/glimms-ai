"""Small, lazy S3 helpers shared by the image-processing services.

The services deliberately do not create an AWS client at import time.  Import-time
network calls make health checks, tests, and local development unnecessarily
fragile.  A client is created on the first operation instead.
"""

from __future__ import annotations

import os
from functools import lru_cache
from urllib.parse import quote

import boto3
from botocore.config import Config


class StorageConfigurationError(RuntimeError):
    """Raised when an image operation cannot be configured safely."""


def _bucket() -> str:
    bucket = os.getenv("S3_BUCKET", "glimms-images").strip()
    if not bucket:
        raise StorageConfigurationError("S3_BUCKET must not be empty")
    return bucket


@lru_cache(maxsize=1)
def get_s3_client():
    """Return a cached boto3 client, without contacting S3 during construction."""

    region = os.getenv("AWS_REGION", "us-east-1").strip() or "us-east-1"
    kwargs = {
        "region_name": region,
        "config": Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    }
    endpoint = os.getenv("S3_ENDPOINT_URL", "").strip()
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def fetch_image_bytes(key: str) -> bytes:
    """Read an image object from the configured bucket.

    ``key`` is intentionally a key rather than an arbitrary URL.  This keeps
    the internal services from becoming an SSRF proxy and makes authorization
    live at the API/S3 boundary.
    """

    if not isinstance(key, str) or not key.strip():
        raise ValueError("image key must be a non-empty string")
    response = get_s3_client().get_object(Bucket=_bucket(), Key=key.strip())
    body = response.get("Body")
    if body is None:
        raise StorageConfigurationError(f"S3 returned no body for key {key!r}")
    data = body.read()
    if not data:
        raise ValueError(f"image object {key!r} is empty")
    return data


def upload_image_bytes(
    key: str,
    data: bytes,
    content_type: str = "image/jpeg",
) -> str:
    """Upload image bytes and return a URL-safe object URL."""

    if not isinstance(key, str) or not key.strip():
        raise ValueError("output key must be a non-empty string")
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("image data must be non-empty bytes")
    if not content_type or "/" not in content_type:
        raise ValueError("content_type must be a valid MIME type")

    clean_key = key.strip().lstrip("/")
    get_s3_client().put_object(
        Bucket=_bucket(),
        Key=clean_key,
        Body=bytes(data),
        ContentType=content_type,
    )

    endpoint = os.getenv("S3_ENDPOINT_URL", "").strip()
    if endpoint:
        return f"{endpoint.rstrip('/')}/{quote(_bucket())}/{quote(clean_key, safe='/')}"
    region = os.getenv("AWS_REGION", "us-east-1").strip() or "us-east-1"
    if region == "us-east-1":
        return f"https://{_bucket()}.s3.amazonaws.com/{quote(clean_key, safe='/')}"
    return f"https://{_bucket()}.s3.{region}.amazonaws.com/{quote(clean_key, safe='/')}"


def clear_s3_client_cache() -> None:
    """Clear the client cache; useful for tests and credential rotation."""

    get_s3_client.cache_clear()

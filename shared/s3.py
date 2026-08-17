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


def _prefix_list(name: str) -> list[str]:
    return [
        prefix.strip().lstrip("/")
        for prefix in os.getenv(name, "").split(",")
        if prefix.strip()
    ]


def normalise_key(key: str, *, what: str = "image key") -> str:
    """Validate and normalise an S3 object key supplied by a caller."""

    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"{what} must be a non-empty string")
    clean = key.strip().lstrip("/")
    if "\x00" in clean or any(part in {"..", ""} for part in clean.split("/")):
        raise ValueError(f"{what} contains an unsafe path")
    if len(clean) > 1024:
        raise ValueError(f"{what} is too long")
    return clean


def enforce_key_prefix(key: str, env_name: str, *, what: str) -> str:
    """Reject keys outside the allow-listed prefixes for this deployment.

    Unset means "no restriction", which keeps local development simple.  Set
    ``S3_ALLOWED_KEY_PREFIXES`` / ``S3_OUTPUT_KEY_PREFIXES`` in production so
    a compromised or buggy caller cannot read or overwrite arbitrary objects
    in the bucket.
    """

    clean = normalise_key(key, what=what)
    allowed = _prefix_list(env_name)
    if allowed and not any(clean.startswith(prefix) for prefix in allowed):
        raise ValueError(f"{what} is outside the permitted prefixes")
    return clean


def is_configured() -> bool:
    """True when a bucket and usable credentials appear to be present."""

    if not os.getenv("S3_BUCKET", "").strip():
        return False
    if os.getenv("AWS_ACCESS_KEY_ID", "").strip():
        return True
    # Instance/task roles and shared config files are equally valid.
    return bool(
        os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE", "").strip()
        or os.getenv("AWS_ROLE_ARN", "").strip()
        or os.getenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "").strip()
        or os.getenv("AWS_PROFILE", "").strip()
        or os.getenv("AWS_SHARED_CREDENTIALS_FILE", "").strip()
    )


def health() -> dict[str, object]:
    """Storage configuration summary for a service ``/health`` payload."""

    configured = is_configured()
    payload: dict[str, object] = {
        "s3_configured": configured,
        "bucket": os.getenv("S3_BUCKET", "").strip() or None,
    }
    if not configured:
        payload["warning"] = (
            "S3 is not configured; image endpoints will fail until "
            "S3_BUCKET and AWS credentials are set"
        )
    return payload


def presigned_get_url(key: str, expires_in: int = 900) -> str:
    """Return a short-lived signed download URL for a generated artifact."""

    clean = normalise_key(key, what="object key")
    expires_in = max(60, min(int(expires_in), 604800))
    return get_s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": _bucket(), "Key": clean},
        ExpiresIn=expires_in,
    )


def fetch_image_bytes(key: str) -> bytes:
    """Read an image object from the configured bucket.

    ``key`` is intentionally a key rather than an arbitrary URL.  This keeps
    the internal services from becoming an SSRF proxy and makes authorization
    live at the API/S3 boundary.
    """

    clean_key = enforce_key_prefix(key, "S3_ALLOWED_KEY_PREFIXES", what="image key")
    response = get_s3_client().get_object(Bucket=_bucket(), Key=clean_key)
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

    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("image data must be non-empty bytes")
    if not content_type or "/" not in content_type:
        raise ValueError("content_type must be a valid MIME type")

    clean_key = enforce_key_prefix(key, "S3_OUTPUT_KEY_PREFIXES", what="output key")
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

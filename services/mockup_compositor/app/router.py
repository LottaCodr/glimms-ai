from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared.s3 import fetch_image_bytes, presigned_get_url, upload_image_bytes

from .compositor import Compositor

router = APIRouter()
logger = logging.getLogger(__name__)
compositor = Compositor()


class ComposeRequest(BaseModel):
    layers: list[dict[str, Any]] = Field(default_factory=list, max_length=30)
    # ``items`` is accepted for direct use with permutation-engine output.
    items: list[dict[str, Any]] | None = Field(default=None, max_length=30)
    output_key: str | None = Field(default=None, max_length=512)
    width: int = Field(default=1200, ge=1, le=4000)
    height: int = Field(default=900, ge=1, le=4000)
    background: str = Field(default="#f7f4ef", min_length=1, max_length=30)
    format: Literal["jpg", "jpeg", "png"] = "jpg"
    output_format: Literal["jpg", "jpeg", "png"] | None = None
    remove_background: bool = False

    def selected_format(self) -> str:
        return self.output_format or self.format

    def selected_layers(self) -> list[dict[str, Any]]:
        return self.layers or (self.items or [])


def _safe_output_key(value: str) -> str:
    key = value.strip().lstrip("/")
    if not key or "\x00" in key or any(part == ".." for part in key.split("/")):
        raise ValueError("output_key contains an unsafe path")
    return key


@router.post("/compose")
@router.post("/mockup")
async def compose(body: ComposeRequest) -> dict[str, Any]:
    layers = body.selected_layers()
    if not layers:
        raise HTTPException(status_code=422, detail="layers or items must contain at least one image")

    fetched: list[tuple[dict[str, Any], bytes]] = []
    try:
        for layer in layers:
            key = layer.get("image_key")
            if not isinstance(key, str) or not key.strip():
                raise ValueError("every layer needs image_key")
            fetched.append((layer, fetch_image_bytes(key)))
        image_bytes, metadata = compositor.compose(
            fetched,
            width=body.width,
            height=body.height,
            background=body.background,
            remove_background=body.remove_background,
            output_format="PNG" if body.selected_format() == "png" else "JPEG",
        )
        extension = "png" if body.selected_format() == "png" else "jpg"
        output_key = body.output_key
        if output_key:
            output_key = _safe_output_key(output_key)
        else:
            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "layers": layers,
                        "width": body.width,
                        "height": body.height,
                        "background": body.background,
                        "format": body.selected_format(),
                    },
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()[:16]
            output_key = f"mockups/{fingerprint}.{extension}"
        content_type = "image/png" if extension == "png" else "image/jpeg"
        url = upload_image_bytes(output_key, image_bytes, content_type=content_type)
        # The bucket should be private, so the plain object URL is not usable
        # on its own.  Hand back a short-lived signed URL as well and let the
        # caller persist ``output_key`` as the durable reference.
        signed_url: str | None = None
        expires_in = int(os.getenv("ARTIFACT_URL_TTL_SECONDS", "900"))
        try:
            signed_url = presigned_get_url(output_key, expires_in=expires_in)
        except Exception as exc:  # noqa: BLE001 - signing is best-effort
            logger.warning("Could not presign %s: %s", output_key, exc)
        return {
            "output_key": output_key,
            "url": signed_url or url,
            "object_url": url,
            "signed_url": signed_url,
            "expires_in": expires_in if signed_url else None,
            "width": metadata["width"],
            "height": metadata["height"],
            "layers": metadata["layers"],
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Mockup composition failed: %s", exc)
        raise HTTPException(status_code=502, detail="mockup could not be composed") from exc

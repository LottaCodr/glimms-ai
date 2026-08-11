from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from shared.s3 import fetch_image_bytes

from .schemas import DetectionError, DetectionRequest, DetectionResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/detect", response_model=DetectionResponse)
async def detect(request: Request, body: DetectionRequest) -> DetectionResponse:
    detector = getattr(request.app.state, "detector", None)
    if detector is None:
        # This makes direct ASGI tests useful even when a lifespan is disabled.
        from .detector import Detector

        detector = Detector()
        request.app.state.detector = detector

    all_items = []
    errors: list[DetectionError] = []
    for key in body.image_keys:
        try:
            image_bytes = fetch_image_bytes(key)
            all_items.extend(detector.detect(image_bytes, body.vertical, key))
        except Exception as exc:  # noqa: BLE001 - process remaining image keys
            logger.warning("Detection failed for %s: %s", key, exc)
            errors.append(DetectionError(image_key=key, error="image could not be processed"))

    return DetectionResponse(
        items=all_items,
        image_count=len(body.image_keys),
        detected_count=len(all_items),
        failed_count=len(errors),
        errors=errors,
    )

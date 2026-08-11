from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared.s3 import fetch_image_bytes

from .analyzer import QualityAnalyzer

router = APIRouter()
logger = logging.getLogger(__name__)
analyzer = QualityAnalyzer()


class QualityRequest(BaseModel):
    image_key: str | None = None
    image_keys: list[str] = Field(default_factory=list, max_length=20)

    def keys(self) -> list[str]:
        return self.image_keys or ([self.image_key] if self.image_key else [])


class QualityResponse(BaseModel):
    results: list[dict[str, Any]]
    image_count: int
    passed_count: int
    failed_count: int
    passed: bool


@router.post("/check", response_model=QualityResponse)
@router.post("/assess", response_model=QualityResponse)
@router.post("/quality", response_model=QualityResponse)
async def check(body: QualityRequest) -> QualityResponse:
    keys = body.keys()
    if not keys:
        raise HTTPException(status_code=422, detail="image_key or image_keys is required")

    results: list[dict[str, Any]] = []
    for key in keys:
        try:
            if not key or not key.strip():
                raise ValueError("image key is empty")
            result = analyzer.analyze(fetch_image_bytes(key))
            results.append({"image_key": key, **result})
        except Exception as exc:  # noqa: BLE001 - return per-image quality results
            logger.warning("Quality check failed for %s: %s", key, exc)
            results.append(
                {
                    "image_key": key,
                    "acceptable": False,
                    "issues": ["unreadable"],
                    "guidance": ["Upload a valid JPEG, PNG, or WebP image."],
                    "error": "image could not be read",
                }
            )

    passed_count = sum(1 for result in results if result.get("acceptable") is True)
    return QualityResponse(
        results=results,
        image_count=len(results),
        passed_count=passed_count,
        failed_count=len(results) - passed_count,
        passed=passed_count == len(results),
    )

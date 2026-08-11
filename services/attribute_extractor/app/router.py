from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from shared.s3 import fetch_image_bytes

from .clip_extractor import CLIPExtractor
from .color_extractor import ColorExtractor
from .texture_extractor import TextureExtractor

router = APIRouter()
logger = logging.getLogger(__name__)

# These objects are lightweight: CLIP itself is lazy and will not download at
# import time.
clip = CLIPExtractor()
color = ColorExtractor()
texture = TextureExtractor()


class ExtractionRequest(BaseModel):
    items: list[dict] = Field(..., min_length=1, max_length=50)


class ExtractionError(BaseModel):
    image_key: str
    error: str


class ExtractionResponse(BaseModel):
    items: list[dict]
    processed_count: int
    failed_count: int
    errors: list[ExtractionError] = Field(default_factory=list)


@router.post("/extract", response_model=ExtractionResponse)
async def extract(body: ExtractionRequest) -> ExtractionResponse:
    enriched: list[dict] = []
    errors: list[ExtractionError] = []
    for item in body.items:
        key = item.get("image_key")
        try:
            if not isinstance(key, str) or not key.strip():
                raise ValueError("image_key is required")
            image_bytes = fetch_image_bytes(key)
            embedding = clip.embed(image_bytes)
            color_info = color.extract(image_bytes)
            texture_info = texture.extract(image_bytes)
            enriched.append(
                {
                    **item,
                    "embedding": embedding,
                    "embedding_dimension": len(embedding),
                    "color": color_info,
                    "texture": texture_info,
                    "style_tags": clip.get_style_tags(embedding),
                }
            )
        except Exception as exc:  # noqa: BLE001 - process the remaining batch items
            logger.warning("Extraction failed for %s: %s", key, exc)
            errors.append(
                ExtractionError(
                    image_key=str(key or ""),
                    error="image could not be processed",
                )
            )
            # Preserve the item so a downstream caller can associate the error
            # with the original detection.
            enriched.append(
                {
                    **item,
                    "embedding": [],
                    "embedding_dimension": 0,
                    "color": {},
                    "texture": {},
                    "style_tags": [],
                }
            )
    return ExtractionResponse(
        items=enriched,
        processed_count=len(body.items) - len(errors),
        failed_count=len(errors),
        errors=errors,
    )

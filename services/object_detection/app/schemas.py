from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class BBox(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class DetectedItem(BaseModel):
    label: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    bbox: BBox
    category: str = Field(min_length=1)
    image_key: str = Field(min_length=1)


class DetectionRequest(BaseModel):
    image_keys: list[str] = Field(..., min_length=1, max_length=20)
    vertical: Literal["wardrobe", "room", "garden"] = "wardrobe"


class DetectionError(BaseModel):
    image_key: str
    error: str


class DetectionResponse(BaseModel):
    items: list[DetectedItem]
    image_count: int
    detected_count: int
    failed_count: int = 0
    errors: list[DetectionError] = Field(default_factory=list)

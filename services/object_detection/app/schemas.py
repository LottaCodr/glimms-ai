from pydantic import BaseModel, Field
from typing import Literal

class BBox(BaseModel):
    x: int; y: int; width: int; height: int

class DetectedItem(BaseModel):
    label: str
    confidence: float
    bbox: BBox
    category: str
    image_key: str  # original S3 key — needed for attribute extraction

class DetectionRequest(BaseModel):
    image_keys: list[str] = Field(..., min_length=1, max_length=5)
    vertical: Literal["wardrobe", "room", "garden"] = "wardrobe"

class DetectionResponse(BaseModel):
    items: list[DetectedItem]
    image_count: int
    detected_count: int

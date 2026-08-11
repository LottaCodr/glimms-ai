from fastapi import APIRouter, HTTPException
from app.clip_extractor import CLIPExtractor
from app.color_extractor import ColorExtractor
from app.texture_extractor import TextureExtractor
from shared.s3 import fetch_image_bytes
from pydantic import BaseModel
import logging, io

router = APIRouter()
logger = logging.getLogger(__name__)

clip = CLIPExtractor()
color = ColorExtractor()
texture = TextureExtractor()

class ExtractionRequest(BaseModel):
    items: list[dict]

class ExtractionResponse(BaseModel):
    items: list[dict]

@router.post("/extract", response_model=ExtractionResponse)
async def extract(body: ExtractionRequest):
    enriched = []
    for item in body.items:
        try:
            img_bytes = fetch_image_bytes(item["image_key"])
            embedding   = clip.embed(img_bytes)
            color_info  = color.extract(img_bytes)
            texture_info = texture.extract(img_bytes)
            enriched.append({
                **item,
                "embedding":   embedding,
                "color":       color_info,
                "texture":     texture_info,
                "style_tags":  clip.get_style_tags(embedding),
            })
        except Exception as e:
            logger.warning(f"Extraction failed for item {item.get('label')}: {e}")
            enriched.append({**item, "embedding": [], "color": {}, "texture": {}, "style_tags": []})
    return ExtractionResponse(items=enriched)

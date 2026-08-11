from fastapi import APIRouter, HTTPException, Request
from app.schemas import DetectionRequest, DetectionResponse
from shared.s3 import fetch_image_bytes
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/detect", response_model=DetectionResponse)
async def detect(request: Request, body: DetectionRequest):
    detector = request.app.state.detector
    all_items = []
    for key in body.image_keys:
        try:
            img_bytes = fetch_image_bytes(key)
            items = detector.detect(img_bytes, body.vertical, key)
            all_items.extend(items)
        except Exception as e:
            logger.error(f"Detection failed for {key}: {e}")
    return DetectionResponse(items=all_items, image_count=len(body.image_keys), detected_count=len(all_items))

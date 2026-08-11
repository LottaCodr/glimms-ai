from fastapi import APIRouter, HTTPException
from app.reasoner import enrich_designs
from pydantic import BaseModel
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class ReasonRequest(BaseModel):
    permutations: list[dict]
    context: dict
    vertical: str = "wardrobe"

@router.post("/reason")
async def reason(body: ReasonRequest):
    try:
        designs = await enrich_designs(body.permutations, body.context, body.vertical)
        return {"designs": designs}
    except Exception as e:
        logger.error(f"Reasoning failed: {e}")
        raise HTTPException(500, str(e))

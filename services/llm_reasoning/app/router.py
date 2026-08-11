from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .reasoner import enrich_designs

router = APIRouter()


class ReasonRequest(BaseModel):
    permutations: list[dict] = Field(default_factory=list, max_length=100)
    context: dict[str, Any] = Field(default_factory=dict)
    vertical: Literal["wardrobe", "room", "garden"] = "wardrobe"


@router.post("/reason")
async def reason(body: ReasonRequest) -> dict[str, Any]:
    try:
        designs = await enrich_designs(body.permutations, body.context, body.vertical)
        return {"designs": designs, "count": len(designs)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="reasoning failed") from exc

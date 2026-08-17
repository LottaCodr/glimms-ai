from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared.runtime import DevFallbackBlocked, guard_dev_fallback

from .reasoner import enrich_designs

router = APIRouter()


def _has_paid_provider() -> bool:
    return bool(
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "").strip()
    )


class ReasonRequest(BaseModel):
    permutations: list[dict] = Field(default_factory=list, max_length=100)
    context: dict[str, Any] = Field(default_factory=dict)
    vertical: Literal["wardrobe", "room", "garden"] = "wardrobe"


@router.post("/reason")
async def reason(body: ReasonRequest) -> dict[str, Any]:
    # Free tiers are rate-limited and can silently degrade to the offline
    # template; production traffic should not depend on them.
    try:
        guard_dev_fallback(
            "llm-reasoning",
            degraded=not _has_paid_provider(),
            reason="no OpenAI/Anthropic key is configured; only free or offline fallbacks remain",
            remedy="set OPENAI_API_KEY or ANTHROPIC_API_KEY",
        )
    except DevFallbackBlocked as exc:
        raise HTTPException(status_code=503, detail=exc.detail()) from exc

    try:
        designs = await enrich_designs(body.permutations, body.context, body.vertical)
        return {"designs": designs, "count": len(designs)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="reasoning failed") from exc

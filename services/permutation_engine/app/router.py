from __future__ import annotations

import os
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from .engine import generate_permutations

router = APIRouter()


class PermutationRequest(BaseModel):
    items: list[dict] = Field(..., min_length=1, max_length=100)
    vertical: Literal["wardrobe", "room", "garden"] = "wardrobe"
    context: dict[str, Any] = Field(default_factory=dict)
    # Top-level aliases make the service easy to call without a separate
    # context-inference request.
    climate: str | dict[str, Any] | None = None
    culture: str | None = None
    occasion: str | None = None
    max_permutations: int | None = Field(default=None, ge=1, le=500)

    def effective_context(self) -> dict[str, Any]:
        result = dict(self.context)
        if self.climate is not None:
            result.setdefault("climate", self.climate)
        if self.culture is not None:
            result.setdefault("culture", self.culture)
        if self.occasion is not None:
            result.setdefault("occasion", self.occasion)
        return result


class PermutationResponse(BaseModel):
    permutations: list[dict]
    count: int
    truncated: bool


@router.post("/generate", response_model=PermutationResponse)
@router.post("/permute", response_model=PermutationResponse)
@router.post("/permutations", response_model=PermutationResponse)
async def permutations(body: PermutationRequest) -> PermutationResponse:
    results = generate_permutations(
        body.items,
        vertical=body.vertical,
        context=body.effective_context(),
        max_permutations=body.max_permutations,
    )
    try:
        configured_limit = int(os.getenv("MAX_PERMUTATIONS", "50"))
    except ValueError:
        configured_limit = 50
    requested_limit = body.max_permutations or configured_limit
    return PermutationResponse(
        permutations=results,
        count=len(results),
        truncated=len(results) >= min(requested_limit, configured_limit),
    )

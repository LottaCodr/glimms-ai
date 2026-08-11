from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from .provider_router import ProviderRouter

logger = logging.getLogger(__name__)

PROMPTS = {
    "wardrobe": (Path(__file__).parent / "prompts" / "wardrobe.txt").read_text(encoding="utf-8"),
    "room": (Path(__file__).parent / "prompts" / "room.txt").read_text(encoding="utf-8"),
    "garden": (Path(__file__).parent / "prompts" / "garden.txt").read_text(encoding="utf-8"),
}

provider_router = ProviderRouter()
# Backwards-compatible name for callers that imported the original module global.
router = provider_router
# Keep LLM spend and request latency bounded when a caller submits a large
# permutation set.  The permutation service can still return more candidates.
try:
    MAX_REASONING_DESIGNS = min(max(int(os.getenv("MAX_REASONING_DESIGNS", "20")), 1), 50)
except ValueError:
    MAX_REASONING_DESIGNS = 20


def _description(value: Any, default: str) -> str:
    if isinstance(value, dict):
        return str(value.get("description") or value.get("type") or default)
    return str(value or default)


def _culture(context: dict[str, Any]) -> str:
    value = context.get("culturalCtx") or context.get("culture") or context.get("region") or "global"
    if isinstance(value, dict):
        value = value.get("name") or value.get("description") or "global"
    return str(value)


def _normalise_enrichment(value: dict[str, Any], vertical: str) -> dict[str, Any]:
    """Keep model output within the public design contract."""

    title = str(value.get("title") or "Curated Design").strip()[:120]
    explanation = str(value.get("explanation") or "A considered combination tailored to the supplied context.").strip()[:1000]
    raw_tips = value.get("tips") if isinstance(value.get("tips"), list) else []
    tips = [str(tip).strip()[:300] for tip in raw_tips if str(tip).strip()][:6]
    if vertical == "wardrobe":
        try:
            occasion_fit = float(value.get("occasion_fit", 7))
        except (TypeError, ValueError):
            occasion_fit = 7.0
        return {
            "title": title,
            "explanation": explanation,
            "tips": tips,
            "occasion_fit": round(max(0.0, min(10.0, occasion_fit)), 1),
            "vibe": str(value.get("vibe") or "casual").strip().lower()[:40],
        }
    if vertical == "room":
        return {
            "title": title,
            "explanation": explanation,
            "tips": tips,
            "mood": str(value.get("mood") or "minimal").strip().lower()[:40],
            "style": str(value.get("style") or "Contemporary").strip()[:60],
        }
    return {
        "title": title,
        "explanation": explanation,
        "tips": tips,
        "season_fit": str(value.get("season_fit") or "year-round").strip().lower()[:40],
    }


async def enrich_designs(permutations: list[dict], context: dict, vertical: str) -> list[dict]:
    if vertical not in PROMPTS:
        raise ValueError(f"unsupported vertical: {vertical}")
    template = PROMPTS[vertical]
    selected = permutations[:MAX_REASONING_DESIGNS]
    semaphore = asyncio.Semaphore(4)

    async def enrich_one(permutation: dict) -> dict:
        items = permutation.get("items", []) if isinstance(permutation, dict) else []
        items_desc = ", ".join(
            str(item.get("label") or item.get("name") or "item")
            for item in items
            if isinstance(item, dict)
        ) or "selected elements"
        climate = context.get("climate") or context.get("weather")
        prompt = template.format(
            items=items_desc,
            occasion=str(context.get("occasion") or "casual"),
            weather=_description(climate, "mild weather"),
            culture=_culture(context),
            occupation=str(context.get("occupation") or "professional"),
        )
        async with semaphore:
            try:
                enrichment = await router.complete(prompt, vertical=vertical)
            except TypeError as exc:
                # Keep compatibility with small test/client adapters that
                # implement the original complete(prompt) signature.
                if "vertical" not in str(exc):
                    raise
                enrichment = await router.complete(prompt)
        safe_enrichment = _normalise_enrichment(enrichment if isinstance(enrichment, dict) else {}, vertical)
        # Never let an LLM overwrite the source items, id, score, or vertical.
        return {**permutation, **safe_enrichment}

    if not selected:
        return []
    return await asyncio.gather(*(enrich_one(permutation) for permutation in selected))

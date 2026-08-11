import os, logging
from pathlib import Path
from app.provider_router import ProviderRouter

logger = logging.getLogger(__name__)

PROMPTS = {
    "wardrobe": (Path(__file__).parent / "prompts" / "wardrobe.txt").read_text(),
    "room":     (Path(__file__).parent / "prompts" / "room.txt").read_text(),
    "garden":   (Path(__file__).parent / "prompts" / "garden.txt").read_text(),
}

router = ProviderRouter()

async def enrich_designs(permutations: list, context: dict, vertical: str) -> list:
    template = PROMPTS.get(vertical, PROMPTS["wardrobe"])
    results = []
    for perm in permutations:
        items_desc = ", ".join(i.get("label","item") for i in perm.get("items",[]))
        prompt = template.format(
            items=items_desc,
            occasion=context.get("occasion","casual"),
            weather=context.get("climate",{}).get("description","mild weather"),
            culture=context.get("culturalCtx","global"),
            occupation=context.get("occupation","professional"),
        )
        enrichment = await router.complete(prompt)
        results.append({**perm, **enrichment})
    return results

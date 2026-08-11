"""Rule-based context inference.

This service intentionally does not pretend to infer a person's culture from an
image.  Culture is an explicit user/product input; the rules only translate
that input, climate, and occasion into constraints that other services can use.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

CLIMATE_RULES: dict[str, dict[str, Any]] = {
    "tropical": {
        "description": "warm and humid",
        "temperature_range_c": [24, 38],
        "preferred_materials": ["linen", "cotton", "bamboo"],
        "avoid_materials": ["heavy wool", "thick fleece"],
        "layering": "light",
        "colors": ["white", "cream", "pastel", "earth tones"],
    },
    "arid": {
        "description": "hot and dry with strong sun exposure",
        "temperature_range_c": [20, 42],
        "preferred_materials": ["linen", "cotton", "breathable blends"],
        "avoid_materials": ["non-breathable synthetics"],
        "layering": "light protective",
        "colors": ["sand", "cream", "terracotta", "olive"],
    },
    "temperate": {
        "description": "mild with moderate seasonal variation",
        "temperature_range_c": [8, 24],
        "preferred_materials": ["cotton", "denim", "light wool", "linen"],
        "avoid_materials": [],
        "layering": "moderate",
        "colors": ["seasonal neutrals", "muted tones"],
    },
    "cold": {
        "description": "cold and often windy",
        "temperature_range_c": [-15, 10],
        "preferred_materials": ["wool", "fleece", "insulated fabrics"],
        "avoid_materials": ["very light single layers"],
        "layering": "heavy",
        "colors": ["charcoal", "navy", "deep green", "winter neutrals"],
    },
    "polar": {
        "description": "very cold with a high insulation requirement",
        "temperature_range_c": [-50, -5],
        "preferred_materials": ["insulated outerwear", "wool", "weatherproof fabrics"],
        "avoid_materials": ["uninsulated garments"],
        "layering": "technical heavy",
        "colors": ["dark neutrals", "high-visibility accents"],
    },
    "humid": {
        "description": "humid with limited evaporative cooling",
        "temperature_range_c": [18, 35],
        "preferred_materials": ["linen", "cotton", "moisture-wicking fabrics"],
        "avoid_materials": ["heavy non-breathable fabrics"],
        "layering": "light",
        "colors": ["light neutrals", "cool tones"],
    },
}

CULTURE_ALIASES = {
    "south asia": "south-asian",
    "south asian": "south-asian",
    "middle east": "middle-eastern",
    "middle eastern": "middle-eastern",
    "east asia": "east-asian",
    "east asian": "east-asian",
    "west africa": "west-african",
    "africa": "african",
    "north america": "north-american",
}

# These are optional starting points, not assumptions about an individual.
CULTURE_RULES: dict[str, dict[str, Any]] = {
    "south-asian": {
        "notes": "Offer user-selected traditional or contemporary options and preserve requested coverage.",
        "preferred_patterns": ["user-selected prints", "woven textures"],
        "coverage": "user preference",
    },
    "middle-eastern": {
        "notes": "Prioritize user-selected modesty, coverage, and occasion requirements.",
        "preferred_patterns": ["user-selected geometric or tonal patterns"],
        "coverage": "user preference",
    },
    "east-asian": {
        "notes": "Offer both contemporary and traditional references only when requested by the user.",
        "preferred_patterns": ["user-selected tonal or geometric patterns"],
        "coverage": "user preference",
    },
    "african": {
        "notes": "Offer a broad range of textiles, color, and pattern choices without assuming a single regional style.",
        "preferred_patterns": ["user-selected woven or graphic patterns"],
        "coverage": "user preference",
    },
}


class ContextRequest(BaseModel):
    vertical: Literal["wardrobe", "room", "garden"] = "wardrobe"
    climate: str | dict[str, Any] | None = None
    # ``weather`` is accepted because older gateway clients used that name.
    weather: str | dict[str, Any] | None = None
    culture: str | dict[str, Any] | None = None
    region: str | None = None
    occasion: str = Field(default="casual", min_length=1, max_length=100)
    occupation: str = Field(default="", max_length=100)
    season: str | None = Field(default=None, max_length=30)
    temperature_c: float | None = Field(default=None, ge=-100, le=70)
    humidity: float | None = Field(default=None, ge=0, le=100)


def _normalise_name(value: str | dict[str, Any] | None) -> str:
    if isinstance(value, dict):
        value = value.get("name") or value.get("type") or value.get("region") or ""
    value = re.sub(r"\s+", " ", (value or "").strip().lower())
    return CULTURE_ALIASES.get(value, value.replace("_", "-"))


def _infer_climate(body: ContextRequest) -> tuple[str, dict[str, Any]]:
    raw: Any = body.climate if body.climate is not None else body.weather
    climate_name = "temperate"
    supplied: dict[str, Any] = {}
    explicit_name = False
    if isinstance(raw, dict):
        supplied = raw
        raw_name = raw.get("type") or raw.get("climate") or raw.get("name") or ""
        if raw_name:
            climate_name = str(raw_name).strip().lower().replace(" ", "-")
            explicit_name = True
        temperature = raw.get("temperature_c", raw.get("temperature"))
        if body.temperature_c is None and isinstance(temperature, (int, float)):
            body.temperature_c = float(temperature)
        if body.humidity is None and isinstance(raw.get("humidity"), (int, float)):
            body.humidity = float(raw["humidity"])
    elif isinstance(raw, str) and raw.strip():
        climate_name = raw.strip().lower().replace(" ", "-")
        explicit_name = True

    # Numeric weather data is more precise than the default label.  If a
    # caller supplied an explicit climate name, keep it, but never return an
    # internally inconsistent default for a numeric-only request.
    if not explicit_name or climate_name not in CLIMATE_RULES:
        if body.humidity is not None and body.humidity >= 75 and body.temperature_c is not None and body.temperature_c >= 18:
            climate_name = "humid"
        elif body.temperature_c is not None:
            temperature = body.temperature_c
            climate_name = "polar" if temperature <= -5 else "cold" if temperature <= 10 else "temperate" if temperature <= 24 else "arid"
        elif body.humidity is not None and body.humidity >= 75:
            climate_name = "humid"
        else:
            climate_name = "temperate"
    rule = dict(CLIMATE_RULES[climate_name])
    rule["type"] = climate_name
    if body.temperature_c is not None:
        rule["temperature_c"] = body.temperature_c
    if body.humidity is not None:
        rule["humidity"] = body.humidity
    if body.season:
        rule["season"] = body.season.strip().lower()
    if supplied.get("description"):
        rule["description"] = str(supplied["description"])
    return climate_name, rule


def infer_context(body: ContextRequest) -> dict[str, Any]:
    climate_name, climate = _infer_climate(body)
    culture_name = _normalise_name(body.culture or body.region)
    cultural = dict(CULTURE_RULES.get(culture_name, {}))
    cultural["name"] = culture_name or "global"
    cultural["explicit_input"] = bool(body.culture or body.region)

    preferred_styles = ["comfortable", "context-aware"]
    if body.vertical == "wardrobe":
        preferred_styles.extend(["breathable" if climate_name in {"tropical", "humid", "arid"} else "layered"])
    elif body.vertical == "room":
        preferred_styles.extend(["functional", "balanced"])
    else:
        preferred_styles.extend(["climate-resilient", "low-maintenance"])

    return {
        "vertical": body.vertical,
        "occasion": body.occasion.strip(),
        "occupation": body.occupation.strip(),
        "climate": climate,
        "weather": climate,
        "culture": culture_name or "global",
        "culturalCtx": culture_name or "global",
        "cultural_context": cultural,
        "style_constraints": {
            "preferred_styles": preferred_styles,
            "preferred_materials": climate["preferred_materials"],
            "avoid_materials": climate["avoid_materials"],
            "preferred_colors": climate["colors"],
            "layering": climate["layering"],
            "coverage": cultural.get("coverage", "user preference"),
        },
        "source": "rule-based",
    }


@router.post("/infer")
@router.post("/context")
async def context(body: ContextRequest) -> dict[str, Any]:
    return infer_context(body)

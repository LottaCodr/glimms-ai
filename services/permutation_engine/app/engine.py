"""Deterministic, explainable design permutation generation."""

from __future__ import annotations

import colorsys
import hashlib
import itertools
import json
import os
from collections.abc import Iterable
from typing import Any

import numpy as np

WARDROBE_CATEGORIES = {
    "shirt": "top",
    "top": "top",
    "jacket": "top",
    "sweater": "top",
    "coat": "top",
    "blazer": "top",
    "pants": "bottom",
    "bottom": "bottom",
    "jeans": "bottom",
    "shorts": "bottom",
    "skirt": "bottom",
    "dress": "full",
    "suit": "full",
    "shoes": "footwear",
    "shoe": "footwear",
    "sneakers": "footwear",
    "boots": "footwear",
    "bag": "accessory",
    "hat": "accessory",
    "accessory": "accessory",
}


def _max_limit(requested: int | None) -> int:
    try:
        configured = int(os.getenv("MAX_PERMUTATIONS", "50"))
    except ValueError:
        configured = 50
    configured = min(max(configured, 1), 500)
    if requested is None:
        return configured
    return min(max(int(requested), 1), configured)


def _label(item: dict[str, Any]) -> str:
    return str(item.get("label") or item.get("name") or item.get("type") or "item").strip().lower()


def _category(item: dict[str, Any], vertical: str) -> str:
    explicit = str(item.get("category") or "").strip().lower()
    if explicit:
        # Detection uses "top", while some clients send a concrete label.
        return WARDROBE_CATEGORIES.get(explicit, explicit)
    if vertical == "wardrobe":
        return WARDROBE_CATEGORIES.get(_label(item), "item")
    return _label(item)


def _item_id(item: dict[str, Any], index: int) -> str:
    value = item.get("id") or item.get("item_id") or item.get("image_key")
    if value:
        return str(value)
    return f"item-{index}"


def _style_tags(item: dict[str, Any]) -> set[str]:
    tags = item.get("style_tags") or item.get("styles") or []
    if isinstance(tags, str):
        tags = [tags]
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def _rgb(item: dict[str, Any]) -> tuple[float, float, float] | None:
    color: Any = item.get("color") or item.get("colour")
    if isinstance(color, dict):
        color = color.get("dominant", color)
    if isinstance(color, dict):
        rgb = color.get("rgb")
        if isinstance(rgb, dict):
            try:
                return tuple(float(np.clip(rgb[channel], 0, 255)) / 255 for channel in ("r", "g", "b"))  # type: ignore[return-value]
            except (KeyError, TypeError, ValueError):
                return None
        color = color.get("hex")
    if isinstance(color, str):
        value = color.strip().lstrip("#")
        if len(value) == 6:
            try:
                return tuple(int(value[index : index + 2], 16) / 255 for index in (0, 2, 4))  # type: ignore[return-value]
            except ValueError:
                return None
    return None


def _colour_score(items: list[dict[str, Any]]) -> float:
    rgbs = [value for item in items if (value := _rgb(item)) is not None]
    if len(rgbs) < 2:
        return 0.65
    hues = []
    saturations = []
    for red, green, blue in rgbs:
        hue, saturation, _ = colorsys.rgb_to_hsv(red, green, blue)
        if saturation >= 0.12:  # neutrals do not create a clash
            hues.append(hue)
            saturations.append(saturation)
    if len(hues) < 2:
        return 0.82
    distances = []
    for first, second in itertools.combinations(hues, 2):
        distance = abs(first - second)
        distances.append(min(distance, 1 - distance))
    mean_distance = float(np.mean(distances))
    # Close hues, complementary hues, and neutral combinations are usually
    # coherent.  Penalise the ambiguous middle where colours often clash.
    harmony = 1.0 if mean_distance < 0.10 or mean_distance > 0.40 else 0.60
    if len(saturations) >= 3 and max(saturations) - min(saturations) > 0.65:
        harmony -= 0.08
    return max(0.0, min(1.0, harmony))


def _context_score(items: list[dict[str, Any]], context: dict[str, Any]) -> tuple[float, list[str]]:
    constraints = context.get("style_constraints") or {}
    if not isinstance(constraints, dict):
        constraints = {}
    raw_preferred_styles = constraints.get("preferred_styles", [])
    if isinstance(raw_preferred_styles, str):
        raw_preferred_styles = [raw_preferred_styles]
    preferred_styles = {str(value).lower() for value in raw_preferred_styles if value}
    item_styles = set().union(*(_style_tags(item) for item in items)) if items else set()
    matched_styles = preferred_styles & item_styles
    score = 0.5
    reasons: list[str] = []
    if matched_styles:
        score += 0.18
        reasons.append(f"matches {', '.join(sorted(matched_styles))} context")
    if item_styles and len(item_styles) == 1:
        score += 0.12
        reasons.append("keeps a consistent style direction")
    elif len(item_styles) > 2:
        score -= 0.08
    preferred_colors = constraints.get("preferred_colors", [])
    if isinstance(preferred_colors, str):
        preferred_colors = [preferred_colors]
    if preferred_colors:
        # Color names are descriptive rather than a forced classifier.  A
        # dominant palette that has a matching mood receives a small bonus.
        preferred_text = " ".join(str(value).lower() for value in preferred_colors)
        labels = " ".join(_label(item) for item in items)
        if any(token in preferred_text for token in labels.split()):
            score += 0.04
    return max(0.0, min(1.0, score)), reasons


def _score(items: list[dict[str, Any]], vertical: str, context: dict[str, Any]) -> tuple[float, list[str]]:
    colour = _colour_score(items)
    contextual, reasons = _context_score(items, context)
    score = 0.38 * colour + 0.34 * contextual + 0.28
    if vertical == "wardrobe":
        categories = {_category(item, vertical) for item in items}
        if {"top", "bottom"} <= categories or "full" in categories:
            score += 0.08
            reasons.insert(0, "has a complete base")
        if "footwear" in categories:
            score += 0.04
            reasons.append("includes footwear")
    elif vertical == "garden":
        if any(_label(item) in {"plant", "flower", "tree", "shrub", "herb", "succulent"} for item in items):
            score += 0.06
            reasons.append("includes living elements")
    else:
        if len(items) >= 2:
            score += 0.04
            reasons.append("balances multiple room elements")
    return round(max(0.0, min(1.0, score)), 4), reasons


def _unique_items(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        copied = dict(item)
        identifier = _item_id(copied, index)
        if identifier in seen:
            # Two detections can share an image key; their bbox makes a useful
            # deterministic distinction.
            identifier = f"{identifier}-{index}"
        copied["id"] = identifier
        seen.add(identifier)
        result.append(copied)
    return result


def _wardrobe_candidates(items: list[dict[str, Any]], limit: int) -> Iterable[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(_category(item, "wardrobe"), []).append(item)
    tops, bottoms = groups.get("top", []), groups.get("bottom", [])
    full = groups.get("full", [])
    footwear = groups.get("footwear", [])
    accessories = groups.get("accessory", [])

    yielded = 0
    if full:
        footwear_options = [None, *footwear]
        accessory_options = [None, *accessories]
        for one, shoe, accessory in itertools.product(full, footwear_options, accessory_options):
            combo = [one] + ([shoe] if shoe else []) + ([accessory] if accessory else [])
            yield combo
            yielded += 1
            if yielded >= limit * 4:
                return
    if tops and bottoms:
        footwear_options = [None, *footwear]
        accessory_options = [None, *accessories]
        for top, bottom, shoe, accessory in itertools.product(tops, bottoms, footwear_options, accessory_options):
            combo = [top, bottom] + ([shoe] if shoe else []) + ([accessory] if accessory else [])
            yield combo
            yielded += 1
            if yielded >= limit * 4:
                return

    # If detection found an incomplete wardrobe, still produce useful singles
    # and small combinations instead of an empty result.
    if yielded == 0:
        for size in range(min(3, len(items)), 0, -1):
            for combo in itertools.combinations(items, size):
                yield list(combo)
                yielded += 1
                if yielded >= limit * 4:
                    return


def _general_candidates(items: list[dict[str, Any]], limit: int) -> Iterable[list[dict[str, Any]]]:
    if len(items) == 1:
        yield items
        return
    yielded = 0
    for size in range(min(4, len(items)), 1, -1):
        for combo in itertools.combinations(items, size):
            yield list(combo)
            yielded += 1
            if yielded >= limit * 4:
                return


def generate_permutations(
    items: list[dict[str, Any]],
    vertical: str = "wardrobe",
    context: dict[str, Any] | None = None,
    max_permutations: int | None = None,
) -> list[dict[str, Any]]:
    if vertical not in {"wardrobe", "room", "garden"}:
        raise ValueError(f"unsupported vertical: {vertical}")
    unique = _unique_items(items)
    if not unique:
        return []
    context = context or {}
    limit = _max_limit(max_permutations)
    candidates = _wardrobe_candidates(unique, limit) if vertical == "wardrobe" else _general_candidates(unique, limit)

    constraints = context.get("style_constraints") or {}
    if not isinstance(constraints, dict):
        constraints = {}
    cultural = context.get("cultural_context") or context.get("cultural_filters") or {}
    if not isinstance(cultural, dict):
        cultural = {}
    excluded_values = (
        context.get("excluded_labels")
        or constraints.get("excluded_labels")
        or constraints.get("avoid_labels")
        or cultural.get("excluded_labels", cultural.get("avoid_labels", cultural.get("avoid", [])))
        or []
    )
    if isinstance(excluded_values, str):
        excluded_values = [excluded_values]
    excluded = {str(value).strip().lower() for value in excluded_values if value}
    ranked: list[dict[str, Any]] = []
    seen_combinations: set[tuple[str, ...]] = set()
    for combo in candidates:
        labels = tuple(_item_id(item, index) for index, item in enumerate(combo))
        if labels in seen_combinations or any(_label(item) in excluded for item in combo):
            continue
        seen_combinations.add(labels)
        score, reasons = _score(combo, vertical, context)
        fingerprint = hashlib.sha1(json.dumps(labels).encode("utf-8")).hexdigest()[:12]
        ranked.append(
            {
                "id": f"perm-{fingerprint}",
                "items": combo,
                "score": score,
                "reasons": reasons,
                "vertical": vertical,
            }
        )

    ranked.sort(key=lambda value: (-value["score"], value["id"]))
    return ranked[:limit]

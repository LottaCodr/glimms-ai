"""Provider failover and defensive JSON parsing for LLM responses."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


class ProviderRouter:
    def __init__(self) -> None:
        configured = os.getenv("DEFAULT_LLM_PROVIDER", "openai").strip().lower()
        self.provider = configured if configured in {"openai", "anthropic"} else "openai"
        try:
            configured_tokens = int(os.getenv("LLM_MAX_TOKENS", "600"))
        except ValueError:
            configured_tokens = 600
        try:
            configured_timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        except ValueError:
            configured_timeout = 30.0
        self.max_tokens = max(configured_tokens, 64)
        self.timeout = max(configured_timeout, 1.0)

    async def complete(self, prompt: str, vertical: str = "wardrobe") -> dict[str, Any]:
        """Complete a prompt, falling back to a useful offline response.

        Missing credentials are handled before constructing an SDK client.  In
        particular, a local health check must not wait for two failed network
        calls just to discover that no API key was supplied.
        """

        providers = [self.provider, "anthropic" if self.provider == "openai" else "openai"]
        last_error: Exception | None = None
        for provider in providers:
            if not self._configured(provider):
                continue
            try:
                text = await (self._openai(prompt) if provider == "openai" else self._anthropic(prompt))
                parsed = self._parse(text)
                if parsed:
                    return parsed
                raise ValueError("provider returned an empty JSON object")
            except Exception as exc:  # noqa: BLE001 - fail over to the next provider
                logger.warning("LLM provider %s failed: %s", provider, exc)
                last_error = exc
        if last_error:
            logger.warning("Using offline LLM fallback after provider failures")
        return self._fallback(vertical)

    @staticmethod
    def _configured(provider: str) -> bool:
        return bool(os.getenv("OPENAI_API_KEY", "").strip()) if provider == "openai" else bool(os.getenv("ANTHROPIC_API_KEY", "").strip())

    async def _openai(self, prompt: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=self.timeout,
        )
        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional fashion and design AI. Always respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"

    async def _anthropic(self, prompt: str) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=self.timeout,
        )
        message = await client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            max_tokens=self.max_tokens,
            system="You are a professional fashion and design AI. Always respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(block, "text", "") for block in message.content)

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        """Parse JSON even when a provider wrapped it in a markdown fence."""

        if not isinstance(text, str) or not text.strip():
            return {}
        clean = text.strip()
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean).strip()
        try:
            value = json.loads(clean)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            # Some models prepend a sentence despite the instruction.  Locate
            # the outermost JSON object without using the unsafe old lstrip()
            # character-set behaviour.
            start, end = clean.find("{"), clean.rfind("}")
            if start < 0 or end <= start:
                return {}
            try:
                value = json.loads(clean[start : end + 1])
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}

    @staticmethod
    def _fallback(vertical: str) -> dict[str, Any]:
        if vertical == "room":
            return {
                "title": "Balanced Everyday Space",
                "explanation": "The selected pieces create a practical base with a calm visual rhythm. Repeated materials and a clear focal point keep the room cohesive.",
                "tips": ["Keep one clear focal point.", "Repeat one material or finish.", "Leave comfortable walking space."],
                "mood": "minimal",
                "style": "Contemporary",
            }
        if vertical == "garden":
            return {
                "title": "Layered Living Garden",
                "explanation": "The selected elements create a balanced composition with varied height and texture. Grouping plants by light and water needs will make the design easier to maintain.",
                "tips": ["Group plants with similar water needs.", "Use varied heights to create depth.", "Check seasonal light changes."],
                "season_fit": "year-round",
            }
        return {
            "title": "Curated Everyday Look",
            "explanation": "The selected pieces create a flexible, considered base for the requested setting. Small changes in accessories and layering can adapt the look through the day.",
            "tips": ["Repeat one color in an accessory.", "Check the final layer for comfort.", "Choose footwear suited to the occasion."],
            "occasion_fit": 7,
            "vibe": "casual",
        }

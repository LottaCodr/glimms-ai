"""Provider failover and defensive JSON parsing for LLM responses."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Free OpenAI-compatible providers built into the fallback tier. Pollinations
# is keyless and always active; the others activate automatically once their
# dedicated API key environment variable is set. Each provider can also be
# re-pointed at a different model via its ``*_MODEL`` override.
_DEFAULT_FALLBACK_PROVIDERS: tuple[dict[str, Any], ...] = (
    {
        "name": "text.pollinations.ai",
        "base_url": "https://text.pollinations.ai/openai",
        "api_key_envs": (),
        "model_env": "POLLINATIONS_MODEL",
        "model": "openai",
        "keyless": True,
    },
    {
        "name": "openrouter.ai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_envs": ("OPENROUTER_API_KEY",),
        "model_env": "OPENROUTER_MODEL",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "keyless": False,
    },
    {
        "name": "api.groq.com",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_envs": ("GROQ_API_KEY",),
        "model_env": "GROQ_MODEL",
        "model": "llama-3.3-70b-versatile",
        "keyless": False,
    },
    {
        "name": "generativelanguage.googleapis.com",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "api_key_envs": ("GOOGLE_AI_API_KEY", "GEMINI_API_KEY"),
        "model_env": "GOOGLE_AI_MODEL",
        "model": "gemini-2.5-flash",
        "keyless": False,
    },
    {
        "name": "integrate.api.nvidia.com",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_envs": ("NVIDIA_NIM_API_KEY",),
        "model_env": "NVIDIA_NIM_MODEL",
        "model": "meta/llama-3.3-70b-instruct",
        "keyless": False,
    },
    {
        "name": "models.github.ai",
        "base_url": "https://models.github.ai/inference",
        "api_key_envs": ("GITHUB_MODELS_API_KEY", "GITHUB_TOKEN"),
        "model_env": "GITHUB_MODELS_MODEL",
        "model": "gpt-4.1",
        "keyless": False,
    },
)


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
        self.fallback_providers = self._parse_fallback_providers()

    async def complete(self, prompt: str, vertical: str = "wardrobe") -> dict[str, Any]:
        """Complete a prompt, falling back to a useful offline response.

        Missing credentials are handled before constructing an SDK client.  In
        particular, a local health check must not wait for two failed network
        calls just to discover that no API key was supplied.

        Failover order: primary provider, secondary provider, every configured
        free OpenAI-compatible provider (Pollinations by default), then the
        deterministic offline fallback.
        """

        providers: list[Any] = [self.provider, "anthropic" if self.provider == "openai" else "openai"]
        providers.extend(self.fallback_providers)
        last_error: Exception | None = None
        for provider in providers:
            if isinstance(provider, str):
                if not self._configured(provider):
                    continue
                label = provider
                coro = self._openai(prompt) if provider == "openai" else self._anthropic(prompt)
            else:
                label = provider["name"]
                coro = self._free_fallback(prompt, provider)
            try:
                text = await coro
                parsed = self._parse(text)
                if parsed:
                    return parsed
                raise ValueError("provider returned an empty JSON object")
            except Exception as exc:  # noqa: BLE001 - fail over to the next provider
                logger.warning("LLM provider %s failed: %s", label, exc)
                last_error = exc
        if last_error:
            logger.warning("Using offline LLM fallback after provider failures")
        return self._fallback(vertical)

    @staticmethod
    def _configured(provider: str) -> bool:
        return bool(os.getenv("OPENAI_API_KEY", "").strip()) if provider == "openai" else bool(os.getenv("ANTHROPIC_API_KEY", "").strip())

    @staticmethod
    def _parse_fallback_providers() -> list[dict[str, str]]:
        """Free OpenAI-compatible providers, tried after the paid ones.

        Pollinations (``https://text.pollinations.ai/openai``) needs no API
        key and is always available, so the pipeline keeps working with zero
        configuration. OpenRouter, Groq, Google AI Studio, NVIDIA NIM and
        GitHub Models are built in and activate automatically when their
        dedicated API key is set. ``FALLBACK_LLM_BASE_URL``,
        ``FALLBACK_LLM_API_KEY`` and ``FALLBACK_LLM_MODEL`` are comma-separated
        lists that add further OpenAI-compatible providers; entries align by
        position and an API key may be empty for keyless providers.
        ``FALLBACK_LLM_DISABLE=true`` switches the whole free tier off.
        """
        if os.getenv("FALLBACK_LLM_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}:
            return []
        providers: list[dict[str, str]] = []
        for entry in _DEFAULT_FALLBACK_PROVIDERS:
            api_key = ""
            if not entry["keyless"]:
                for env_name in entry["api_key_envs"]:
                    api_key = os.getenv(env_name, "").strip()
                    if api_key:
                        break
            if entry["keyless"] or api_key:
                providers.append(
                    {
                        "name": entry["name"],
                        "base_url": entry["base_url"],
                        "api_key": api_key,
                        "model": os.getenv(entry["model_env"], "").strip() or entry["model"],
                    }
                )
        seen = {provider["base_url"] for provider in providers}
        urls = [url.strip().rstrip("/") for url in os.getenv("FALLBACK_LLM_BASE_URL", "").split(",") if url.strip()]
        if urls:
            keys = [key.strip() for key in os.getenv("FALLBACK_LLM_API_KEY", "").split(",")]
            models = [model.strip() for model in os.getenv("FALLBACK_LLM_MODEL", "").split(",")]
            for index, url in enumerate(urls):
                if url in seen:
                    continue
                seen.add(url)
                key = keys[index] if index < len(keys) else ""
                model = models[index] if index < len(models) and models[index] else "openai"
                providers.append(
                    {
                        "name": url.split("//", 1)[-1].split("/", 1)[0],
                        "base_url": url,
                        "api_key": key,
                        "model": model,
                    }
                )
        return providers

    async def _openai(self, prompt: str) -> str:
        return await self._openai_compatible(
            prompt,
            base_url=None,
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            json_mode=True,
        )

    async def _free_fallback(self, prompt: str, entry: dict[str, str]) -> str:
        """Call one free OpenAI-compatible provider from the fallback list.

        Any OpenAI-compatible endpoint works: Pollinations (no key), OpenRouter
        free models, Groq, Google AI Studio, NVIDIA NIM, GitHub Models, ...
        ``response_format`` is deliberately not sent because several free
        endpoints reject it; ``_parse`` extracts JSON from prose-wrapped
        responses instead.
        """
        return await self._openai_compatible(
            prompt,
            base_url=entry["base_url"],
            api_key=entry["api_key"],
            model=entry["model"],
            json_mode=False,
        )

    async def _openai_compatible(
        self,
        prompt: str,
        *,
        base_url: str | None,
        api_key: str | None,
        model: str,
        json_mode: bool = False,
    ) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=api_key or "not-needed",
            base_url=base_url,
            timeout=self.timeout,
        )
        request: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional fashion and design AI. Always respond with valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        response = await client.chat.completions.create(**request)
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

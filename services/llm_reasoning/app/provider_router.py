import os, json, logging, asyncio
logger = logging.getLogger(__name__)

class ProviderRouter:
    def __init__(self):
        self.provider = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
        self.max_tokens = 600

    async def complete(self, prompt: str) -> dict:
        providers = [self.provider, "anthropic" if self.provider == "openai" else "openai"]
        last_err = None
        for p in providers:
            try:
                text = await (self._openai(prompt) if p == "openai" else self._anthropic(prompt))
                return self._parse(text)
            except Exception as e:
                logger.warning(f"Provider {p} failed: {e}")
                last_err = e
        logger.error(f"All providers failed: {last_err}")
        return {"title": "Curated Look", "explanation": "A carefully selected combination for your style.", "tips": [], "vibe": "casual"}

    async def _openai(self, prompt: str) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional fashion and design AI. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or "{}"

    async def _anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=self.max_tokens,
            system="You are a professional fashion and design AI. Always respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def _parse(self, text: str) -> dict:
        try:
            clean = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            lines = text.strip().split("\n")
            return {"title": lines[0] if lines else "Style", "explanation": text, "tips": []}

from __future__ import annotations

import os

from fastapi import FastAPI

from .provider_router import ProviderRouter
from .router import router

app = FastAPI(title="Glimms — LLM Reasoning", version="1.2.0")
app.include_router(router)


@app.get("/health")
def health():
    configured = []
    if os.getenv("OPENAI_API_KEY", "").strip():
        configured.append("openai")
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        configured.append("anthropic")
    free_fallback = [provider["name"] for provider in ProviderRouter().fallback_providers]
    if free_fallback:
        configured.append("free-fallback")
    return {
        "status": "ok",
        "service": "llm-reasoning",
        "port": int(os.getenv("PORT", "8005")),
        "providers_configured": configured,
        "free_fallback_providers": free_fallback,
        "offline_fallback": True,
    }

from __future__ import annotations

import os

from fastapi import FastAPI

from .router import router

app = FastAPI(title="Glimms — LLM Reasoning", version="1.1.0")
app.include_router(router)


@app.get("/health")
def health():
    configured = []
    if os.getenv("OPENAI_API_KEY", "").strip():
        configured.append("openai")
    if os.getenv("ANTHROPIC_API_KEY", "").strip():
        configured.append("anthropic")
    return {
        "status": "ok",
        "service": "llm-reasoning",
        "port": int(os.getenv("PORT", "8005")),
        "providers_configured": configured,
        "offline_fallback": True,
    }

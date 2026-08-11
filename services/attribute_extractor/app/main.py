from __future__ import annotations

import os

from fastapi import FastAPI

from .router import router

app = FastAPI(title="Glimms — Attribute Extractor", version="1.1.0")
app.include_router(router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "attribute-extractor",
        "port": int(os.getenv("PORT", "8002")),
        "clip_enabled": os.getenv("CLIP_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
    }

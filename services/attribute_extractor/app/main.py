from __future__ import annotations

import os

from fastapi import FastAPI

from shared import s3
from shared.runtime import allow_dev_fallbacks
from shared.auth import install_service_auth

from .router import router

app = FastAPI(title="Glimms — Attribute Extractor", version="1.1.0")
app.include_router(router)
install_service_auth(app, "attribute-extractor")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "attribute-extractor",
        "port": int(os.getenv("PORT", "8002")),
        "clip_enabled": os.getenv("CLIP_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
        "dev_fallbacks_allowed": allow_dev_fallbacks(),
        **s3.health(),
    }


@app.get("/livez")
def livez():
    """Liveness probe; intentionally unauthenticated."""

    return {"status": "ok", "service": "attribute-extractor"}

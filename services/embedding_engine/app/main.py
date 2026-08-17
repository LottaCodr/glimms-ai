from __future__ import annotations

import os

from fastapi import FastAPI

from shared.runtime import allow_dev_fallbacks
from shared.auth import install_service_auth

from .router import router, store

app = FastAPI(title="Glimms — Embedding Engine", version="1.1.0")
app.include_router(router)
install_service_auth(app, "embedding-engine")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "embedding-engine",
        "port": int(os.getenv("PORT", "8003")),
        "dev_fallbacks_allowed": allow_dev_fallbacks(),
        **store.health(),
    }


@app.get("/livez")
def livez():
    """Liveness probe; intentionally unauthenticated."""

    return {"status": "ok", "service": "embedding-engine"}

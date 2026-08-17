from __future__ import annotations

import os

from fastapi import FastAPI

from shared import s3
from shared.auth import install_service_auth

from .router import router

app = FastAPI(title="Glimms — Mockup Compositor", version="1.1.0")
app.include_router(router)
install_service_auth(app, "mockup-compositor")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "mockup-compositor",
        "port": int(os.getenv("PORT", "8006")),
        "backend": "pillow",
        **s3.health(),
    }


@app.get("/livez")
def livez():
    """Liveness probe; intentionally unauthenticated."""

    return {"status": "ok", "service": "mockup-compositor"}

from __future__ import annotations

import os

from fastapi import FastAPI

from .router import router

app = FastAPI(title="Glimms — Mockup Compositor", version="1.1.0")
app.include_router(router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "mockup-compositor",
        "port": int(os.getenv("PORT", "8006")),
        "backend": "pillow",
    }

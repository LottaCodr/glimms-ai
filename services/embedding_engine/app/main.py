from __future__ import annotations

import os

from fastapi import FastAPI

from .router import router, store

app = FastAPI(title="Glimms — Embedding Engine", version="1.1.0")
app.include_router(router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "embedding-engine",
        "port": int(os.getenv("PORT", "8003")),
        **store.health(),
    }

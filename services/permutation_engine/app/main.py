from __future__ import annotations

import os

from fastapi import FastAPI

from .router import router

app = FastAPI(title="Glimms — Permutation Engine", version="1.1.0")
app.include_router(router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "permutation-engine",
        "port": int(os.getenv("PORT", "8004")),
        "backend": "deterministic-rule-based",
    }

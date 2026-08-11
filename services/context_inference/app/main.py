from __future__ import annotations

import os

from fastapi import FastAPI

from .router import router

app = FastAPI(title="Glimms — Context Inference", version="1.1.0")
app.include_router(router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "context-inference",
        "port": int(os.getenv("PORT", "8008")),
        "backend": "rule-based",
    }

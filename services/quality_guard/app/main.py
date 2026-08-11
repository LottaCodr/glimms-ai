from __future__ import annotations

import os

from fastapi import FastAPI

from .router import router

app = FastAPI(title="Glimms — Quality Guard", version="1.1.0")
app.include_router(router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "quality-guard",
        "port": int(os.getenv("PORT", "8007")),
        "checks": ["blur", "exposure", "contrast", "resolution"],
    }

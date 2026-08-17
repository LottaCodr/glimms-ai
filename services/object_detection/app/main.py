from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from shared import s3
from shared.runtime import allow_dev_fallbacks
from shared.auth import install_service_auth

from .detector import Detector
from .router import router

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector()
    logging.getLogger(__name__).info("Detector service ready")
    yield


app = FastAPI(title="Glimms — Object Detection", version="1.1.0", lifespan=lifespan)
app.include_router(router)
install_service_auth(app, "object-detection")


@app.get("/health")
def health():
    detector = getattr(app.state, "detector", None)
    return {
        "status": "ok",
        "service": "object-detection",
        "port": int(os.getenv("PORT", "8001")),
        "model_loaded": bool(detector and detector.model is not None),
        "dev_fallbacks_allowed": allow_dev_fallbacks(),
        **s3.health(),
    }


@app.get("/livez")
def livez():
    """Liveness probe; intentionally unauthenticated."""

    return {"status": "ok", "service": "object-detection"}

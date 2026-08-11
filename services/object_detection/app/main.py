from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

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


@app.get("/health")
def health():
    detector = getattr(app.state, "detector", None)
    return {
        "status": "ok",
        "service": "object-detection",
        "port": int(os.getenv("PORT", "8001")),
        "model_loaded": bool(detector and detector.model is not None),
    }

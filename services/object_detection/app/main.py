from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.router import router
from app.detector import Detector
import logging, os

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.detector = Detector()
    logging.info("Detector loaded")
    yield

app = FastAPI(title="Glimms — Object Detection", version="1.0.0", lifespan=lifespan)
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok", "service": "object-detection", "port": os.getenv("PORT", "8001")}

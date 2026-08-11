from fastapi import FastAPI
from app.router import router
import os

app = FastAPI(title="Glimms — Attribute Extractor")
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok", "service": "attribute-extractor"}

from fastapi import FastAPI
from app.router import router
app = FastAPI(title="Glimms — Embedding Engine")
app.include_router(router)
@app.get("/health")
def health(): return {"status": "ok", "service": "embedding_engine", "port": 8003}

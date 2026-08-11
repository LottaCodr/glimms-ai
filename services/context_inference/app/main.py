from fastapi import FastAPI
from app.router import router
app = FastAPI(title="Glimms — Context Inference")
app.include_router(router)
@app.get("/health")
def health(): return {"status": "ok", "service": "context_inference", "port": 8008}

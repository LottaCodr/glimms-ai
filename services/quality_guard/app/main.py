from fastapi import FastAPI
from app.router import router
app = FastAPI(title="Glimms — Quality Guard")
app.include_router(router)
@app.get("/health")
def health(): return {"status": "ok", "service": "quality_guard", "port": 8007}

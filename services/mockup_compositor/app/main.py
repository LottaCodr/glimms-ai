from fastapi import FastAPI
from app.router import router
app = FastAPI(title="Glimms — Mockup Compositor")
app.include_router(router)
@app.get("/health")
def health(): return {"status": "ok", "service": "mockup_compositor", "port": 8006}

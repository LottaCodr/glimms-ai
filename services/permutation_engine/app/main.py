from fastapi import FastAPI
from app.router import router
app = FastAPI(title="Glimms — Permutation Engine")
app.include_router(router)
@app.get("/health")
def health(): return {"status": "ok", "service": "permutation_engine", "port": 8004}

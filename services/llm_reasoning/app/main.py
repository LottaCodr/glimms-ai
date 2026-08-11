from fastapi import FastAPI
from app.router import router

app = FastAPI(title="Glimms — LLM Reasoning")
app.include_router(router)

@app.get("/health")
def health(): return {"status": "ok", "service": "llm-reasoning"}

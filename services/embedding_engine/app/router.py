from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from shared.runtime import DevFallbackBlocked, guard_dev_fallback

from .store import EmbeddingStore

router = APIRouter()
store = EmbeddingStore()


class VectorInput(BaseModel):
    id: str | None = None
    item_id: str | None = None
    embedding: list[float] | None = None
    values: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def identifier(self) -> str:
        return str(self.id or self.item_id or "").strip()

    def vector(self) -> list[float]:
        values = self.embedding if self.embedding is not None else self.values
        if values is None:
            raise ValueError("embedding or values is required")
        return values


class UpsertRequest(BaseModel):
    vectors: list[VectorInput] = Field(default_factory=list, max_length=1000)
    # ``items`` is the name used by the attribute-extractor pipeline.
    items: list[VectorInput] | None = Field(default=None, max_length=1000)
    # A singular record is convenient for small callers and keeps this API
    # compatible with simple vector-ingestion clients.
    id: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    namespace: str | None = Field(default=None, max_length=100)

    def records(self) -> list[VectorInput]:
        if self.vectors:
            return self.vectors
        if self.items:
            return self.items
        if self.id is not None or self.embedding is not None:
            return [VectorInput(id=self.id, embedding=self.embedding, metadata=self.metadata)]
        return []


class SearchRequest(BaseModel):
    embedding: list[float] | None = None
    vector: list[float] | None = None
    query: list[float] | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    namespace: str | None = Field(default=None, max_length=100)
    filter: dict[str, Any] | None = None

    def values(self) -> list[float]:
        values = self.embedding if self.embedding is not None else self.vector
        values = values if values is not None else self.query
        if values is None:
            raise ValueError("embedding, vector, or query is required")
        return values


class DeleteRequest(BaseModel):
    ids: list[str] = Field(..., min_length=1, max_length=1000)
    namespace: str | None = Field(default=None, max_length=100)


def _guard_durable_store() -> None:
    """Refuse to act as a system of record when the store is in-process.

    The in-memory backend is wiped by every restart, deploy and idle
    spin-down, and is not shared between replicas.  Silently accepting writes
    into it looks like durable storage right up until the data is gone.
    """

    try:
        guard_dev_fallback(
            "embedding-engine",
            degraded=store.health().get("backend") == "memory",
            reason="the in-memory vector store is not durable",
            remedy="set PINECONE_API_KEY and PINECONE_INDEX",
        )
    except DevFallbackBlocked as exc:
        raise HTTPException(status_code=503, detail=exc.detail()) from exc


@router.post("/upsert")
async def upsert(body: UpsertRequest) -> dict[str, Any]:
    _guard_durable_store()
    inputs = body.records()
    if not inputs:
        raise HTTPException(status_code=422, detail="vectors or items must contain at least one record")
    try:
        records = []
        for item in inputs:
            identifier = item.identifier()
            if not identifier:
                raise ValueError("every vector needs id or item_id")
            records.append((identifier, item.vector(), item.metadata))
        ids = store.upsert(records, namespace=body.namespace)
        return {"upserted_count": len(ids), "ids": ids, "namespace": body.namespace or store.namespace}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="vector store unavailable") from exc


@router.post("/search")
@router.post("/query")
async def search(body: SearchRequest) -> dict[str, Any]:
    _guard_durable_store()
    try:
        matches = store.search(
            body.values(),
            top_k=body.top_k,
            namespace=body.namespace,
            filter_value=body.filter,
        )
        return {"matches": matches, "namespace": body.namespace or store.namespace}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="vector store unavailable") from exc


@router.delete("/vectors")
async def delete(body: DeleteRequest) -> dict[str, Any]:
    _guard_durable_store()
    try:
        deleted = store.delete(body.ids, namespace=body.namespace)
        return {"deleted_count": deleted, "namespace": body.namespace or store.namespace}
    except Exception as exc:
        raise HTTPException(status_code=502, detail="vector store unavailable") from exc

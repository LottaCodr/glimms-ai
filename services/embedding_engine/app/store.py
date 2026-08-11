"""Pinecone-backed vector storage with an explicit in-process development mode."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingStore:
    def __init__(self) -> None:
        self._vectors: dict[str, dict[str, tuple[np.ndarray, dict[str, Any]]]] = {}
        self._lock = threading.RLock()
        self._remote: Any | None = None
        self._remote_attempted = False
        self._remote_error: str | None = None
        self._dimension: int | None = None

    @property
    def namespace(self) -> str:
        return os.getenv("PINECONE_NAMESPACE", "items").strip() or "items"

    def _pinecone_index(self) -> Any | None:
        if self._remote_attempted:
            return self._remote
        self._remote_attempted = True
        api_key = os.getenv("PINECONE_API_KEY", "").strip()
        index_name = os.getenv("PINECONE_INDEX", "").strip()
        if not api_key or not index_name:
            self._remote_error = "Pinecone is not configured; using in-memory store"
            return None
        try:
            from pinecone import Pinecone

            client = Pinecone(api_key=api_key)
            self._remote = client.Index(index_name)
            logger.info("Pinecone index connected: %s", index_name)
        except Exception as exc:  # noqa: BLE001 - remote failure has a local fallback
            self._remote_error = f"Pinecone unavailable: {exc}"
            logger.warning("%s; using in-memory store", self._remote_error)
            self._remote = None
        return self._remote

    @staticmethod
    def _normalise(values: list[float] | tuple[float, ...]) -> np.ndarray:
        if not values:
            raise ValueError("embedding must not be empty")
        vector = np.asarray(values, dtype=np.float32)
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError("embedding must be a finite one-dimensional vector")
        return vector

    def _check_dimension(self, vector: np.ndarray) -> None:
        if self._dimension is None:
            self._dimension = int(vector.size)
        elif vector.size != self._dimension:
            raise ValueError(f"embedding dimension must be {self._dimension}, got {vector.size}")

    def upsert(
        self,
        records: list[tuple[str, list[float], dict[str, Any]]],
        namespace: str | None = None,
    ) -> list[str]:
        if not records:
            raise ValueError("at least one vector is required")
        namespace = (namespace or self.namespace).strip() or self.namespace
        prepared = []
        for identifier, values, metadata in records:
            identifier = str(identifier).strip()
            if not identifier:
                raise ValueError("vector id must not be empty")
            vector = self._normalise(values)
            self._check_dimension(vector)
            prepared.append((identifier, vector, dict(metadata or {})))

        remote = self._pinecone_index()
        if remote is not None:
            remote.upsert(
                vectors=[
                    {"id": identifier, "values": vector.tolist(), "metadata": metadata}
                    for identifier, vector, metadata in prepared
                ],
                namespace=namespace,
            )
        with self._lock:
            bucket = self._vectors.setdefault(namespace, {})
            for identifier, vector, metadata in prepared:
                bucket[identifier] = (vector, metadata)
        return [identifier for identifier, _, _ in prepared]

    @staticmethod
    def _metadata_matches(metadata: dict[str, Any], filter_value: dict[str, Any] | None) -> bool:
        if not filter_value:
            return True
        for key, expected in filter_value.items():
            actual = metadata.get(key)
            if isinstance(expected, dict):
                if "$in" in expected and actual not in expected["$in"]:
                    return False
                if "$eq" in expected and actual != expected["$eq"]:
                    return False
            elif actual != expected:
                return False
        return True

    def search(
        self,
        values: list[float],
        top_k: int = 10,
        namespace: str | None = None,
        filter_value: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        vector = self._normalise(values)
        self._check_dimension(vector)
        top_k = min(max(int(top_k), 1), 100)
        namespace = (namespace or self.namespace).strip() or self.namespace
        remote = self._pinecone_index()
        if remote is not None:
            response = remote.query(
                vector=vector.tolist(),
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
            )
            matches = response.get("matches", []) if isinstance(response, dict) else getattr(response, "matches", [])
            return [self._match_to_dict(match) for match in matches]

        with self._lock:
            entries = list(self._vectors.get(namespace, {}).items())
        query_norm = max(float(np.linalg.norm(vector)), 1e-12)
        scored = []
        for identifier, (candidate, metadata) in entries:
            if not self._metadata_matches(metadata, filter_value):
                continue
            score = float(np.dot(vector, candidate) / (query_norm * max(float(np.linalg.norm(candidate)), 1e-12)))
            scored.append({"id": identifier, "score": score, "metadata": metadata})
        scored.sort(key=lambda value: (-value["score"], value["id"]))
        return scored[:top_k]

    def delete(self, ids: list[str], namespace: str | None = None) -> int:
        if not ids:
            return 0
        namespace = (namespace or self.namespace).strip() or self.namespace
        remote = self._pinecone_index()
        if remote is not None:
            remote.delete(ids=[str(identifier) for identifier in ids], namespace=namespace)
        deleted = 0
        with self._lock:
            bucket = self._vectors.setdefault(namespace, {})
            for identifier in ids:
                if str(identifier) in bucket:
                    del bucket[str(identifier)]
                    deleted += 1
        return deleted

    @staticmethod
    def _match_to_dict(match: Any) -> dict[str, Any]:
        if isinstance(match, dict):
            return {
                "id": str(match.get("id", "")),
                "score": float(match.get("score", 0.0)),
                "metadata": match.get("metadata") or {},
            }
        return {
            "id": str(getattr(match, "id", "")),
            "score": float(getattr(match, "score", 0.0)),
            "metadata": getattr(match, "metadata", {}) or {},
        }

    def health(self) -> dict[str, Any]:
        configured = bool(os.getenv("PINECONE_API_KEY", "").strip() and os.getenv("PINECONE_INDEX", "").strip())
        remote = self._pinecone_index()
        return {
            "backend": "pinecone" if remote is not None else "memory",
            "pinecone_configured": configured,
            "dimension": self._dimension,
            "warning": self._remote_error,
        }

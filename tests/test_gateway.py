"""Gateway auth, readiness, and proxy-hardening tests."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def _reload_gateway():
    module = importlib.import_module("gateway.app")
    return importlib.reload(module)


def test_index_and_livez_are_public_but_health_is_not(monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "gw-token")
    monkeypatch.setenv("GLIMMS_ENV", "development")
    client = TestClient(_reload_gateway().app)

    assert client.get("/livez").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/index.json").status_code == 200
    # /health leaks the deployment's configuration posture, so it is guarded.
    assert client.get("/health").status_code == 401


def test_proxy_requires_a_token(monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "gw-token")
    client = TestClient(_reload_gateway().app)

    response = client.post("/permutation-engine/generate", json={"items": []})
    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"


def test_unknown_prefix_returns_404_with_correlation_id(monkeypatch):
    monkeypatch.delenv("AI_INTERNAL_TOKEN", raising=False)
    monkeypatch.setenv("GLIMMS_ENV", "development")
    client = TestClient(_reload_gateway().app)

    response = client.post("/not-a-service/go", json={})
    assert response.status_code == 404
    assert response.headers["x-correlation-id"]
    assert "permutation-engine" in response.json()["available"]


def test_correlation_id_is_echoed(monkeypatch):
    monkeypatch.delenv("AI_INTERNAL_TOKEN", raising=False)
    client = TestClient(_reload_gateway().app)

    response = client.post(
        "/not-a-service/go", json={}, headers={"x-correlation-id": "cor_mine"}
    )
    assert response.headers["x-correlation-id"] == "cor_mine"


def test_oversized_body_is_rejected(monkeypatch):
    monkeypatch.delenv("AI_INTERNAL_TOKEN", raising=False)
    monkeypatch.setenv("MAX_REQUEST_BYTES", "100")
    module = _reload_gateway()
    client = TestClient(module.app)

    response = client.post("/permutation-engine/generate", content=b"x" * 500)
    assert response.status_code == 413
    assert response.json()["max_bytes"] == 100


def test_degradation_detection():
    module = _reload_gateway()
    healthy = {
        "object-detection": {"model_loaded": True},
        "attribute-extractor": {"clip_enabled": True},
        "embedding-engine": {"backend": "pinecone"},
        "llm-reasoning": {"providers_configured": ["openai"]},
    }
    assert module._degradations(healthy) == []

    degraded = {
        "object-detection": {"model_loaded": False},
        "attribute-extractor": {"clip_enabled": False},
        "embedding-engine": {"backend": "memory"},
        "llm-reasoning": {"providers_configured": ["free-fallback"]},
    }
    names = {finding["service"] for finding in module._degradations(degraded)}
    assert names == {
        "object-detection",
        "attribute-extractor",
        "embedding-engine",
        "llm-reasoning",
    }


def test_authorization_header_is_not_forwarded_upstream(monkeypatch):
    """A client token is terminated at the gateway and re-issued internally."""

    monkeypatch.setenv("AI_INTERNAL_TOKEN", "gw-token")
    module = _reload_gateway()
    seen: dict[str, str] = {}

    class _FakeResponse:
        status_code = 200
        content = b"{}"
        headers: dict[str, str] = {}

    class _FakeClient:
        async def request(self, method, url, headers=None, content=None):
            seen.update(headers or {})
            return _FakeResponse()

    monkeypatch.setattr(module, "_client_or_raise", lambda: _FakeClient())
    client = TestClient(module.app)
    response = client.post(
        "/permutation-engine/generate",
        json={},
        headers={"authorization": "Bearer gw-token", "x-internal-token": "leak"},
    )

    assert response.status_code == 200
    assert seen["authorization"] == "Bearer gw-token"  # re-issued, not passed through
    assert "x-internal-token" not in {key.lower() for key in seen}
    assert seen["x-correlation-id"]


@pytest.mark.parametrize("env", ["production", "prod"])
def test_gateway_refuses_to_start_untokened_in_production(monkeypatch, env):
    monkeypatch.delenv("AI_INTERNAL_TOKEN", raising=False)
    monkeypatch.setenv("GLIMMS_ENV", env)
    with pytest.raises(RuntimeError, match="AI_INTERNAL_TOKEN"):
        _reload_gateway()

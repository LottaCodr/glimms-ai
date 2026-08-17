"""Tests for the auth, fallback-guard, and S3 key-scoping hardening."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient


def _reload_app(module_path: str):
    """Import a service app fresh so module-level env reads are re-evaluated."""

    module = importlib.import_module(module_path)
    return importlib.reload(module).app


# ---------------------------------------------------------------- runtime mode


def test_fallbacks_allowed_outside_production(monkeypatch):
    from shared import runtime

    monkeypatch.delenv("ALLOW_DEV_FALLBACKS", raising=False)
    monkeypatch.setenv("GLIMMS_ENV", "development")
    assert runtime.allow_dev_fallbacks() is True

    monkeypatch.setenv("GLIMMS_ENV", "production")
    assert runtime.allow_dev_fallbacks() is False

    # An explicit override wins in both directions.
    monkeypatch.setenv("ALLOW_DEV_FALLBACKS", "true")
    assert runtime.allow_dev_fallbacks() is True


def test_guard_raises_only_when_blocked(monkeypatch):
    from shared.runtime import DevFallbackBlocked, guard_dev_fallback

    monkeypatch.setenv("GLIMMS_ENV", "production")
    monkeypatch.delenv("ALLOW_DEV_FALLBACKS", raising=False)

    # Healthy service: never blocked.
    guard_dev_fallback("svc", degraded=False, reason="r", remedy="m")

    with pytest.raises(DevFallbackBlocked) as excinfo:
        guard_dev_fallback("svc", degraded=True, reason="r", remedy="m")
    assert excinfo.value.detail()["error"] == "development_fallback_blocked"


# ------------------------------------------------------------------------ auth


def test_service_requires_token_when_configured(monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "secret-token")
    monkeypatch.setenv("GLIMMS_ENV", "development")
    app = _reload_app("services.context_inference.app.main")
    client = TestClient(app)

    body = {"vertical": "wardrobe", "occasion": "work"}
    assert client.post("/infer", json=body).status_code == 401
    assert client.post(
        "/infer", json=body, headers={"authorization": "Bearer wrong"}
    ).status_code == 401

    ok = client.post(
        "/infer", json=body, headers={"authorization": "Bearer secret-token"}
    )
    assert ok.status_code == 200


def test_livez_stays_public_and_rotation_works(monkeypatch):
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "old-token, new-token")
    app = _reload_app("services.context_inference.app.main")
    client = TestClient(app)

    # A platform probe must not need the secret.
    assert client.get("/livez").status_code == 200
    # Both sides of a rotation are accepted.
    for token in ("old-token", "new-token"):
        response = client.post(
            "/infer",
            json={"vertical": "wardrobe", "occasion": "work"},
            headers={"authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200


def test_unconfigured_token_is_open_outside_production(monkeypatch):
    monkeypatch.delenv("AI_INTERNAL_TOKEN", raising=False)
    monkeypatch.setenv("GLIMMS_ENV", "development")
    app = _reload_app("services.context_inference.app.main")
    assert (
        TestClient(app)
        .post("/infer", json={"vertical": "wardrobe", "occasion": "work"})
        .status_code
        == 200
    )


def test_production_without_token_refuses_to_start(monkeypatch):
    monkeypatch.delenv("AI_INTERNAL_TOKEN", raising=False)
    monkeypatch.setenv("GLIMMS_ENV", "production")
    with pytest.raises(RuntimeError, match="AI_INTERNAL_TOKEN"):
        _reload_app("services.context_inference.app.main")


# -------------------------------------------------------------- fallback gates


def test_embedding_engine_rejects_memory_store_in_production(monkeypatch):
    monkeypatch.setenv("GLIMMS_ENV", "production")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "t")
    monkeypatch.setenv("ALLOW_DEV_FALLBACKS", "false")
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    app = _reload_app("services.embedding_engine.app.main")
    client = TestClient(app)

    response = client.post(
        "/upsert",
        json={"vectors": [{"id": "a", "embedding": [1.0, 0.0]}]},
        headers={"authorization": "Bearer t"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["service"] == "embedding-engine"


def test_embedding_engine_serves_memory_store_in_development(monkeypatch):
    monkeypatch.setenv("GLIMMS_ENV", "development")
    monkeypatch.delenv("AI_INTERNAL_TOKEN", raising=False)
    monkeypatch.delenv("ALLOW_DEV_FALLBACKS", raising=False)
    app = _reload_app("services.embedding_engine.app.main")
    client = TestClient(app)

    assert client.post(
        "/upsert", json={"vectors": [{"id": "a", "embedding": [1.0, 0.0]}]}
    ).status_code == 200


def test_llm_reasoning_blocks_free_tier_in_production(monkeypatch):
    monkeypatch.setenv("GLIMMS_ENV", "production")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "t")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    app = _reload_app("services.llm_reasoning.app.main")

    response = TestClient(app).post(
        "/reason",
        json={"vertical": "wardrobe", "context": {}, "permutations": []},
        headers={"authorization": "Bearer t"},
    )
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]["remedy"]


def test_object_detection_blocks_prototype_boxes_in_production(monkeypatch):
    monkeypatch.setenv("GLIMMS_ENV", "production")
    monkeypatch.setenv("AI_INTERNAL_TOKEN", "t")
    app = _reload_app("services.object_detection.app.main")

    response = TestClient(app).post(
        "/detect",
        json={"image_keys": ["users/u/a.jpg"], "vertical": "wardrobe"},
        headers={"authorization": "Bearer t"},
    )
    assert response.status_code == 503
    assert response.json()["detail"]["service"] == "object-detection"


# ---------------------------------------------------------------- S3 key scope


def test_key_prefix_enforcement(monkeypatch):
    from shared import s3

    monkeypatch.setenv("S3_ALLOWED_KEY_PREFIXES", "users/,shared/")
    assert s3.enforce_key_prefix("users/u1/a.jpg", "S3_ALLOWED_KEY_PREFIXES", what="k")

    with pytest.raises(ValueError, match="permitted prefixes"):
        s3.enforce_key_prefix("secrets/creds.json", "S3_ALLOWED_KEY_PREFIXES", what="k")

    # Unset means unrestricted, so local development keeps working.
    monkeypatch.delenv("S3_ALLOWED_KEY_PREFIXES", raising=False)
    assert s3.enforce_key_prefix("anything/a.jpg", "S3_ALLOWED_KEY_PREFIXES", what="k")


@pytest.mark.parametrize(
    "bad_key", ["", "   ", "../../etc/passwd", "users/../../secret", "a//b"]
)
def test_unsafe_keys_are_rejected(bad_key):
    from shared import s3

    with pytest.raises(ValueError):
        s3.normalise_key(bad_key)


def test_leading_slash_is_normalised():
    from shared import s3

    assert s3.normalise_key("/users/u1/a.jpg") == "users/u1/a.jpg"

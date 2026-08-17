from __future__ import annotations

import asyncio
import io

from PIL import Image


def _image_bytes(color=(120, 80, 40), size=(320, 240)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


def test_all_apps_import_and_health():
    service_names = [
        "object_detection",
        "attribute_extractor",
        "embedding_engine",
        "permutation_engine",
        "llm_reasoning",
        "mockup_compositor",
        "quality_guard",
        "context_inference",
    ]
    for name in service_names:
        module = __import__(f"services.{name}.app.main", fromlist=["app"])
        assert module.app.title.startswith("Glimms")
        assert module.health()["status"] == "ok"


def test_fallback_embedding_is_stable_and_normalized():
    from services.attribute_extractor.app.clip_extractor import CLIPExtractor

    first = CLIPExtractor().embed(b"same image")
    second = CLIPExtractor().embed(b"same image")
    assert first == second
    assert len(first) == 512
    assert abs(sum(value * value for value in first) - 1) < 1e-5


def test_color_extractor_handles_flat_images_deterministically():
    from services.attribute_extractor.app.color_extractor import ColorExtractor

    extractor = ColorExtractor()
    result = extractor.extract(_image_bytes((255, 0, 0)), n=5)
    assert result["dominant"]["hex"] == "#ff0000"
    assert result == extractor.extract(_image_bytes((255, 0, 0)), n=5)


def test_permutations_produce_a_complete_outfit():
    from services.permutation_engine.app.engine import generate_permutations

    items = [
        {"id": "top", "label": "shirt", "category": "top", "style_tags": ["casual"]},
        {"id": "bottom", "label": "jeans", "category": "bottom", "style_tags": ["casual"]},
        {"id": "shoes", "label": "shoes", "category": "footwear"},
    ]
    result = generate_permutations(items, max_permutations=10)
    assert result
    assert {item["category"] for item in result[0]["items"]} >= {"top", "bottom"}
    assert result[0]["score"] >= result[-1]["score"]


def test_embedding_store_ranks_cosine_similarity():
    from services.embedding_engine.app.store import EmbeddingStore

    store = EmbeddingStore()
    store.upsert([("near", [1, 0], {"label": "near"}), ("far", [0, 1], {})])
    matches = store.search([0.9, 0.1], top_k=2)
    assert [match["id"] for match in matches] == ["near", "far"]


def test_context_uses_numeric_weather_data():
    from services.context_inference.app.router import ContextRequest, infer_context

    result = infer_context(ContextRequest(climate={"temperature_c": 30, "humidity": 80}))
    assert result["climate"]["type"] == "humid"
    assert result["style_constraints"]["layering"] == "light"


def test_quality_analyzer_reports_blur_and_guidance():
    from services.quality_guard.app.analyzer import QualityAnalyzer

    result = QualityAnalyzer().analyze(_image_bytes())
    assert result["acceptable"] is False
    assert "blur" in result["issues"]
    assert result["guidance"]


def test_compositor_outputs_valid_png():
    from services.mockup_compositor.app.compositor import Compositor

    data, metadata = Compositor().compose(
        [({"image_key": "one"}, _image_bytes((255, 0, 0), (40, 40)))],
        width=100,
        height=80,
        output_format="PNG",
    )
    assert data.startswith(b"\x89PNG")
    assert metadata["layers"][0]["width"] > 0


def test_provider_parser_handles_markdown_and_fallback(monkeypatch):
    from services.llm_reasoning.app.provider_router import ProviderRouter

    assert ProviderRouter._parse("```json\n{\"title\": \"Test\"}\n```") == {"title": "Test"}
    router = ProviderRouter()

    async def _no_network(prompt, entry):
        raise RuntimeError("no network in tests")

    monkeypatch.setattr(router, "_free_fallback", _no_network)
    result = asyncio.run(router.complete("ignored", vertical="garden"))
    assert result["season_fit"] == "year-round"


def test_free_fallback_defaults_to_pollinations_and_accepts_extras(monkeypatch):
    from services.llm_reasoning.app import main as llm_main
    from services.llm_reasoning.app.provider_router import ProviderRouter

    # Zero configuration: Pollinations is always available and needs no key.
    router = ProviderRouter()
    assert router.fallback_providers[0]["name"] == "text.pollinations.ai"
    assert router.fallback_providers[0]["api_key"] == ""
    assert "free-fallback" in llm_main.health()["providers_configured"]

    # Extra OpenAI-compatible providers are appended in order; entries in the
    # comma-separated lists align by position and keys may be empty.
    monkeypatch.setenv(
        "FALLBACK_LLM_BASE_URL",
        "https://openrouter.ai/api/v1,https://api.groq.com/openai/v1,https://text.pollinations.ai/openai",
    )
    monkeypatch.setenv("FALLBACK_LLM_API_KEY", ",gsk_test")
    monkeypatch.setenv("FALLBACK_LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free,")
    router = ProviderRouter()
    names = [provider["name"] for provider in router.fallback_providers]
    assert names == ["text.pollinations.ai", "openrouter.ai", "api.groq.com"]
    assert router.fallback_providers[1]["api_key"] == ""
    assert router.fallback_providers[1]["model"] == "meta-llama/llama-3.3-70b-instruct:free"
    assert router.fallback_providers[2]["api_key"] == "gsk_test"
    assert router.fallback_providers[2]["model"] == "openai"
    assert llm_main.health()["free_fallback_providers"] == names

    # The whole free tier can be switched off.
    monkeypatch.setenv("FALLBACK_LLM_DISABLE", "true")
    assert ProviderRouter().fallback_providers == []
    assert "free-fallback" not in llm_main.health()["providers_configured"]

# Glimms AI services

> Eight independently deployable FastAPI services used by the Glimms style and
> design pipeline.

## What is implemented

| Service | Port | API | Current implementation |
|---|---:|---|---|
| `object-detection` | 8001 | `POST /detect` | YOLOv8 ONNX inference with EXIF handling, box clamping, and a deterministic development fallback when no model is mounted |
| `attribute-extractor` | 8002 | `POST /extract` | Lazy CLIP embeddings, deterministic offline vectors, dominant palette, mood, texture, and style tags |
| `embedding-engine` | 8003 | `POST /upsert`, `POST /search`, `DELETE /vectors` | Pinecone adapter with cosine-similarity in-memory fallback for local development |
| `permutation-engine` | 8004 | `POST /generate` | Outfit, room, and garden combinations ranked by category completeness, color harmony, style, and context |
| `llm-reasoning` | 8005 | `POST /reason` | OpenAI/Anthropic failover, defensive JSON parsing, bounded concurrency, schema-safe offline fallback |
| `mockup-compositor` | 8006 | `POST /compose` | Pillow composition from S3 image keys, optional lazy background removal, JPEG/PNG upload |
| `quality-guard` | 8007 | `POST /check` | Blur, exposure, contrast, resolution, quality score, and re-capture guidance |
| `context-inference` | 8008 | `POST /infer` | Explicit-input climate/culture/occasion rules mapped to reusable style constraints |

Every service also exposes `GET /health` and FastAPI documentation at
`/docs`. Compatibility aliases are available for the main pipeline endpoints:
`/context`, `/permute`, `/permutations`, `/query`, `/assess`, `/quality`, and
`/mockup`.

## Pipeline shape

```text
image keys
   │
   ├── quality-guard ──┐
   ├── object-detection │
   │                    └── attribute-extractor
   └── context-inference ── permutation-engine ──┬── embedding-engine
                                                  ├── llm-reasoning
                                                  └── mockup-compositor
```

The services exchange S3 object keys rather than arbitrary URLs. This keeps the
internal image APIs from becoming an SSRF proxy and lets the API gateway own
upload authorization.

For the recommended public API, database, queue, security, and worker design,
see [`docs/BACKEND_IMPLEMENTATION.md`](docs/BACKEND_IMPLEMENTATION.md).

## Start all services locally

```bash
cp .env.example .env
# Fill in S3 credentials/bucket if using image-key endpoints.
docker compose up --build
```

The Pinecone and model integrations are optional during local development:

- Empty Pinecone credentials select an in-memory vector store. It is not
  durable and should not be used for production.
- `CLIP_ENABLED=false` avoids downloading a large transformer at startup. Set
  it to `true` (and optionally `CLIP_LOCAL_ONLY=true`) for real CLIP features.
- Place or mount a domain-specific YOLOv8 ONNX model at
  `services/object_detection/models/yolov8n.onnx`, then set `MODEL_PATH` plus
  its comma-separated `MODEL_LABELS` for real object detections. The detector
  image copies that optional directory. Without a model, `/detect` returns
  stable prototype detections so the rest of the pipeline can be exercised,
  but those results are not production detections.
- Set `S3_ENDPOINT_URL` when using an S3-compatible local service such as
  MinIO.

## Start one service

```bash
cd services/llm_reasoning
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --port 8005
```

The image-key services need the repository root on `PYTHONPATH` when run
outside Docker, for example:

```bash
PYTHONPATH="$PWD/../.." uvicorn app.main:app --reload --port 8002
```

## API examples

### Generate permutations

```bash
curl -X POST http://localhost:8004/generate \
  -H 'content-type: application/json' \
  -d '{
    "vertical": "wardrobe",
    "context": {"occasion": "work", "style_constraints": {"preferred_styles": ["minimalist"]}},
    "items": [
      {"id": "top-1", "label": "shirt", "category": "top", "style_tags": ["minimalist"]},
      {"id": "bottom-1", "label": "jeans", "category": "bottom"},
      {"id": "shoe-1", "label": "shoes", "category": "footwear"}
    ]
  }'
```

### Upsert and search vectors

```bash
curl -X POST http://localhost:8003/upsert \
  -H 'content-type: application/json' \
  -d '{"vectors":[{"id":"top-1","embedding":[1,0,0],"metadata":{"label":"shirt"}}]}'

curl -X POST http://localhost:8003/search \
  -H 'content-type: application/json' \
  -d '{"embedding":[0.9,0.1,0],"top_k":5}'
```

## Configuration

See [`.env.example`](.env.example) for provider, S3, model, threshold, and
service limits. Do not commit `.env`, model weights, or credentials.

### Free LLM fallback provider

`llm-reasoning` tries providers in this order: the configured primary
(OpenAI or Anthropic), the other one, every configured free
OpenAI-compatible provider, and finally the deterministic offline fallback.

**Pollinations is enabled by default and needs no API key**, so the free
fallback works with zero configuration. To add more free providers (OpenRouter
`:free` models, Groq, Google AI Studio, NVIDIA NIM, GitHub Models, ...), use
the comma-separated `FALLBACK_LLM_BASE_URL` / `FALLBACK_LLM_API_KEY` /
`FALLBACK_LLM_MODEL` variables — entries align by position, an API key may be
empty for keyless providers, and each is tried in order. Set
`FALLBACK_LLM_DISABLE=true` to turn the free tier off entirely. Free tiers are
rate-limited and intended for development; do not rely on them under
production load.

## Verification

The repository CI workflow checks imports for every service. The lightweight
unit suite can be run locally (and is ready to be enabled in CI) with the
same command. The code can also be compiled without third-party imports with:

```bash
python -m compileall -q services shared
```

## Remaining nice-to-haves

These are intentionally kept separate from the working baseline:

1. **Production model packaging:** mount/version a domain-trained YOLO model,
   CLIP model, and model metadata; add warm-up and GPU/provider selection.
2. **Real quality scoring:** add a calibrated BRISQUE/NIQE model and a labelled
   acceptance set instead of hand-tuned blur/exposure thresholds.
3. **Durable vector operations:** create/migrate Pinecone indexes, add metadata
   schema/versioning, bulk re-index jobs, and a durable local development
   backend.
4. **Pipeline orchestration:** add a gateway workflow with retries, idempotency
   keys, correlation IDs, timeouts, circuit breakers, and a job queue for long
   model/LLM work.
5. **Security hardening:** service-to-service authentication, per-user S3
   prefixes, signed download URLs, rate limits, request-size limits, and secret
   rotation.
6. **Observability:** structured logs, traces across all eight services,
   latency/error metrics, provider token/cost metrics, and model confidence
   dashboards.
7. **Better visual composition:** segmentation-aware placement, alpha masks,
   occlusion/depth ordering, crop controls, and an async image-result CDN.
8. **Recommendation quality:** learned outfit compatibility, user feedback
   loops, localization, accessibility descriptions, and explicit preference
   controls rather than broad cultural assumptions.
9. **Deployment maturity:** container image scanning, non-root model volumes,
   autoscaling/GPU profiles, readiness probes, and separate production/staging
   configuration.
10. **Test depth:** contract tests between services, golden image fixtures,
    provider-mocked integration tests, load tests, and a security regression
    suite.

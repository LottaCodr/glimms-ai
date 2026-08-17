# Using the hosted Glimms AI deployment from your backend

Base URL: `https://glimms-ai.onrender.com`

That deployment is the **all-in-one image** (root `Dockerfile` + `gateway/`):
all eight FastAPI services run inside one container and the gateway exposes
them on one port under a `/<service-name>/` prefix.

```text
GET  /                          index page listing the services      (public)
GET  /livez                     gateway liveness                     (public)
GET  /index.json                machine-readable service index       (public)
GET  /health                    aggregated health + degradation list (token)
GET  /readyz                    503 unless every service is real     (token)
ANY  /<service-name>/<path>     proxied unchanged to that service    (token)
GET  /<service-name>/docs       that service's OpenAPI docs          (token)
```

Endpoints marked *(token)* require `Authorization: Bearer $AI_INTERNAL_TOKEN`
whenever that variable is set — and setting it is **mandatory** when
`GLIMMS_ENV=production`, where the container refuses to boot without it.

## 1. Mental model: it is an internal dependency, not your API

Your product backend stays the only public entry point. It owns auth, tenants,
sessions, the database, S3 presigning, the job queue, and retries. Glimms is a
set of **stateless JSON functions** it calls. See
[`BACKEND_IMPLEMENTATION.md`](BACKEND_IMPLEMENTATION.md) for the full design.

```text
client ──HTTPS+auth──> your backend ──server-to-server──> glimms-ai.onrender.com
                            ├── Postgres (sessions, items, permutations, artifacts)
                            ├── S3 (uploads + generated mockups)
                            └── queue/worker (runs the pipeline off the request path)
```

Never call the Render URL from browser or mobile code. Even with the bearer
token in place, shipping that secret to a client hands the pipeline (and your
bucket) to anyone who opens devtools.

## 2. Endpoint map (via the gateway)

| Step | Call | Body highlights |
|---|---|---|
| 1. Screen uploads | `POST /quality-guard/check` | `{"image_keys": [...]}` |
| 2. Find items | `POST /object-detection/detect` | `{"image_keys": [...], "vertical": "wardrobe"}` |
| 3. Describe items | `POST /attribute-extractor/extract` | `{"items": [{"image_key": "...", ...}]}` |
| 4. Normalise context | `POST /context-inference/infer` | `{"vertical","climate","culture","occasion"}` |
| 5. Combine | `POST /permutation-engine/generate` | `{"items", "vertical", "context", "max_permutations"}` |
| 6. Index/search | `POST /embedding-engine/upsert` / `/search` | `{"vectors":[{"id","embedding","metadata"}]}` |
| 7. Explain/rank | `POST /llm-reasoning/reason` | `{"vertical","context","permutations"}` |
| 8. Render mockup | `POST /mockup-compositor/compose` | `{"layers":[{"image_key","bbox"}],"output_key"}` |

Steps 1, 2 and 4 can run in parallel; 3 needs detections; 5 needs 3 + 4; 7 needs
5; 8 needs the selected permutation.

The image endpoints take **S3 object keys, not URLs** — deliberately, so the
services can't be used as an SSRF proxy. The hosted container must therefore
share the same bucket/credentials as your backend, or those four endpoints
(`/quality-guard`, `/object-detection`, `/attribute-extractor`,
`/mockup-compositor`) cannot work at all.

## 3. Wire it up

On your backend:

```env
GLIMMS_BASE_URL=https://glimms-ai.onrender.com
GLIMMS_INTERNAL_TOKEN=<same value as AI_INTERNAL_TOKEN on Render>
GLIMMS_TIMEOUT_MS=120000
```

On the Render service (see [`render.yaml`](../render.yaml) for a blueprint that
sets all of this):

```env
GLIMMS_ENV=production
AI_INTERNAL_TOKEN=<openssl rand -hex 32>
ALLOW_DEV_FALLBACKS=false
S3_ALLOWED_KEY_PREFIXES=users/
S3_OUTPUT_KEY_PREFIXES=users/
ARTIFACT_URL_TTL_SECONDS=900
```

Point one env var at the gateway and derive per-service paths from it; don't
scatter eight hard-coded URLs. A reference client is in
[`examples/glimms-client.ts`](../examples/glimms-client.ts) — it does timeouts,
retries with jitter on 429/5xx, correlation IDs, and a single typed method per
service.

Client rules that matter on Render's free/shared tier:

- **Cold starts.** A sleeping instance can take 30–60 s on the first request.
  Run the pipeline in a worker/queue, not inside an HTTP handler, and keep a
  cheap `GET /health` warmer if latency matters.
- **Timeouts.** 120 s connect+read for `/llm-reasoning/reason` and
  `/mockup-compositor/compose`; 15–30 s for the rule-based services.
- **Retries.** Retry only idempotent failures (connection reset, 429, 502/503/504),
  max 3 attempts, exponential backoff with jitter. Don't retry 4xx.
- **Concurrency cap.** One container runs all eight services; a semaphore of
  4–8 in-flight requests per worker avoids self-inflicted 502s.
- **Circuit breaker.** After N consecutive failures, fail the job with a
  retryable status instead of hammering the instance.
- **Persist everything.** Store each step's raw response against the session,
  so a later step can be retried without re-running the whole pipeline.

## 4. Read the health payload before trusting results

The live deployment currently reports:

```json
{"object-detection":   {"model_loaded": false},
 "attribute-extractor":{"clip_enabled": false},
 "embedding-engine":   {"backend": "memory", "pinecone_configured": false},
 "llm-reasoning":      {"providers_configured": ["free-fallback"],
                        "free_fallback_providers": ["text.pollinations.ai"]}}
```

Translated:

- **Detections are deterministic prototypes, not real detections.** The
  all-in-one image intentionally omits torch/YOLO/CLIP/rembg/Pinecone.
- **Attributes are offline pseudo-embeddings**, not CLIP vectors — they are
  stable and comparable to each other, but meaningless against real CLIP space.
- **The vector store is in-process memory.** It is wiped on every deploy,
  restart, and Render idle-spindown, and it is not shared between instances.
  Do not use it as a system of record; keep vectors in your own store (or
  Pinecone/pgvector) and treat the engine as a cache.
- **LLM output comes from a keyless free provider** (Pollinations), which is
  rate-limited and unsuitable for production load.

So this deployment is a good **integration/staging target**: it lets you build
and test the whole backend contract end to end today. Before it serves real
users, either set the real provider keys and mount the models, or deploy the
per-service images from `docker-compose.yml`.

You no longer have to detect this by hand. The deployment enforces it:

- `GET /health` returns a `degradations` array and a `production_ready`
  boolean naming every service currently running a fallback.
- `GET /readyz` returns **503** when any service is unreachable *or* running a
  fallback that is not explicitly permitted. Point Render's health check and
  any load balancer at this.
- With `GLIMMS_ENV=production` (or `ALLOW_DEV_FALLBACKS=false`), the affected
  endpoints themselves return **503** with a machine-readable body rather than
  prototype output:

```json
{"detail": {"error": "development_fallback_blocked",
            "service": "object-detection",
            "reason": "no detection model is loaded; results would be prototypes",
            "remedy": "mount a YOLOv8 ONNX model and set MODEL_PATH/MODEL_LABELS"}}
```

The rule-based services (`context-inference`, `permutation-engine`,
`quality-guard`) are unaffected — they have no fallback to block.

Set `ALLOW_DEV_FALLBACKS=true` to knowingly run a demo on prototype output, and
store a `degraded: true` flag on any session produced that way so those results
can be found and re-run later. `glimms.isDegraded()` in the reference client
reads the same signal.

## 5. Hardening checklist for the hosted instance

The gateway ships with no auth, so the current URL is world-callable.

1. **Set `AI_INTERNAL_TOKEN`.** Both the gateway and each service now enforce
   `Authorization: Bearer <token>` on everything except `/livez`. Comma-separated
   values are accepted, so rotation is: add the new token, redeploy callers,
   drop the old one. Generate with `openssl rand -hex 32`.
2. **Set `GLIMMS_ENV=production`.** This makes the token mandatory (the
   container fails to boot without it) and blocks every dev fallback.
3. **Prefer a private network.** `render.yaml` defines the production service
   as a Render *Private Service* — no public URL at all. The token then becomes
   defence in depth rather than the only barrier.
4. **Scope S3 two ways.** Give the container an IAM key limited to the one
   bucket, *and* set `S3_ALLOWED_KEY_PREFIXES` / `S3_OUTPUT_KEY_PREFIXES` so a
   malformed or hostile `image_key` cannot read or overwrite outside
   `users/`. Traversal (`..`) and absolute keys are rejected outright.
5. **Rate-limit at your backend**, before Glimms: max images per session, max
   permutations per job. The gateway already caps body size
   (`MAX_REQUEST_BYTES`, default 20 MB → 413) and in-flight upstream requests
   (`MAX_CONCURRENT_UPSTREAM`, default 16).
6. **Signed URLs.** `/compose` now returns `signed_url` (TTL
   `ARTIFACT_URL_TTL_SECONDS`, default 900 s) alongside `object_url`. Persist
   `output_key` as the durable reference and mint a fresh signed URL per
   request; keep the bucket private.
7. **Correlation IDs.** The gateway accepts `X-Correlation-ID` (generating one
   if absent), forwards it upstream, and echoes it on every response including
   errors — log it against each job.

## 6. Smoke test

```bash
BASE=https://glimms-ai.onrender.com
TOKEN=<AI_INTERNAL_TOKEN>
AUTH="authorization: Bearer $TOKEN"

# Public: no token needed.
curl -s "$BASE/livez"

# Is this deployment fit to serve users? 200 = yes, 503 = fallbacks/unreachable.
curl -s -o /dev/null -w '%{http_code}\n' -H "$AUTH" "$BASE/readyz"

curl -s -H "$AUTH" "$BASE/health" | jq '.production_ready, .degradations'

# Rule-based, needs no S3 — good end-to-end probe for the pipeline.
curl -s -X POST "$BASE/context-inference/infer" -H "$AUTH" \
  -H 'content-type: application/json' \
  -d '{"vertical":"wardrobe","climate":{"temperature_c":29,"humidity":78},"occasion":"work"}'

curl -s -X POST "$BASE/permutation-engine/generate" -H "$AUTH" \
  -H 'content-type: application/json' \
  -d '{"vertical":"wardrobe","context":{"occasion":"work"},
       "items":[{"id":"t1","label":"shirt","category":"top"},
                {"id":"b1","label":"chinos","category":"bottom"},
                {"id":"s1","label":"loafers","category":"footwear"}]}'
```

Expect `401` without the header once `AI_INTERNAL_TOKEN` is set, and `503`
with a `development_fallback_blocked` body from `/object-detection/detect`,
`/attribute-extractor/extract`, `/embedding-engine/*` and
`/llm-reasoning/reason` while the deployment is in production mode without
real models, Pinecone, or provider keys.

The S3-backed endpoints (`/detect`, `/extract`, `/check`, `/compose`) will
return per-item errors until `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_REGION` and `S3_BUCKET` are set on the Render service.

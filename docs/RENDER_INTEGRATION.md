# Using the hosted Glimms AI deployment from your backend

Base URL: `https://glimms-ai.onrender.com`

That deployment is the **all-in-one image** (root `Dockerfile` + `gateway/`):
all eight FastAPI services run inside one container and the gateway exposes
them on one port under a `/<service-name>/` prefix.

```text
GET  /                          index page listing the services
GET  /index.json                machine-readable service index
GET  /health                    aggregated health of all eight services
ANY  /<service-name>/<path>     proxied unchanged to that service
GET  /<service-name>/docs       that service's OpenAPI docs
```

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

Never call the Render URL from browser or mobile code. It has **no
authentication**, so anything the client can reach, the internet can reach —
and a public `/mockup-compositor/compose` on your bucket is an abuse vector.

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

```env
GLIMMS_BASE_URL=https://glimms-ai.onrender.com
GLIMMS_TIMEOUT_MS=120000
GLIMMS_ALLOW_FALLBACKS=false   # refuse dev-fallback output in production
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

Make that explicit in code — gate on health rather than hoping:

```ts
const h = await glimms.health();
const degraded =
  h.services["object-detection"].model_loaded === false ||
  h.services["embedding-engine"].backend === "memory" ||
  h.services["attribute-extractor"].clip_enabled === false;

if (degraded && process.env.NODE_ENV === "production") {
  throw new Error("Glimms is running development fallbacks; refusing to publish results");
}
```

Store a `degraded: true` flag on any session produced this way so the results
can be found and re-run later.

## 5. Hardening checklist for the hosted instance

The gateway ships with no auth, so the current URL is world-callable.

1. **Add a shared secret.** Require `Authorization: Bearer $AI_INTERNAL_TOKEN`
   in the gateway and set it in Render's env; reject everything else. Rotate it
   from a secret manager.
2. **Prefer a private network.** On Render, use a Private Service and reach it
   over the internal hostname so it has no public URL at all.
3. **Scope S3.** Give the container a key that can only read
   `users/*/sessions/*` and write `.../mockups/*` in the one bucket — never a
   wildcard key.
4. **Rate-limit and size-limit** at your backend, before Glimms: max images per
   session, max bytes per image, max permutations per job.
5. **Signed URLs only.** Persist `output_key` from `/compose` and mint a fresh
   short-lived presigned GET when the client asks for the artifact; don't hand
   out the raw returned URL.
6. **Correlation IDs.** Send `X-Correlation-ID` on every call and log it with
   the response status and latency so a slow job can be traced across the eight
   services.

## 6. Smoke test

```bash
BASE=https://glimms-ai.onrender.com

curl -s "$BASE/health" | jq '.status, .services["embedding-engine"].backend'

# Rule-based, needs no S3 — good liveness probe for the pipeline.
curl -s -X POST "$BASE/context-inference/infer" \
  -H 'content-type: application/json' \
  -d '{"vertical":"wardrobe","climate":{"temperature_c":29,"humidity":78},"occasion":"work"}'

curl -s -X POST "$BASE/permutation-engine/generate" \
  -H 'content-type: application/json' \
  -d '{"vertical":"wardrobe","context":{"occasion":"work"},
       "items":[{"id":"t1","label":"shirt","category":"top"},
                {"id":"b1","label":"chinos","category":"bottom"},
                {"id":"s1","label":"loafers","category":"footwear"}]}'
```

The S3-backed endpoints (`/detect`, `/extract`, `/check`, `/compose`) will
return per-item errors until `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_REGION` and `S3_BUCKET` are set on the Render service.

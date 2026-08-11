# Glimms AI backend implementation guide

## 1. Purpose

This document describes how a product backend should integrate the eight Glimms
AI services. The repository currently contains the AI services, but it does not
contain the user-facing API gateway, database, authentication layer, or job
orchestrator. Those pieces belong in the main backend.

The backend should be the only public entry point. Clients should never call
these services, S3, Pinecone, OpenAI, or Anthropic directly.

```text
Web/mobile client
        │ HTTPS + user authentication
        ▼
Glimms backend/API gateway
        │ private service network + service authentication
        ├── Object storage (S3)
        ├── Relational database
        ├── Job queue/workers
        ├── AI services in this repository
        └── Vector store / LLM providers
```

The examples use language-neutral JSON and TypeScript-like pseudocode. The same
contracts work with NestJS, Express, Django, FastAPI, Laravel, Go, or another
backend stack.

## 2. Responsibilities of the backend

The backend must own:

- user authentication, authorization, tenant isolation, and rate limits;
- creation of analysis/design sessions and immutable job IDs;
- presigned image upload and download URLs;
- validation of file type, file size, image count, and S3 key ownership;
- orchestration of the service calls and persistence of every result;
- retries, timeouts, idempotency, cancellation, and partial failure handling;
- user preferences and explicit context such as occasion, climate, culture,
  coverage, budget, and accessibility requirements;
- issuing a signed URL for every generated artifact;
- audit logs, metrics, traces, provider cost controls, and deletion requests;
- preventing development fallbacks from being used as production results.

The AI services must remain private and stateless where possible. They receive
S3 object keys and JSON, fetch image bytes from the private bucket, and return
JSON or a generated artifact key.

## 3. Recommended request lifecycle

Use an asynchronous job for the complete experience. A small synchronous
endpoint may be added later for already-processed items.

### 3.1 Create a session

`POST /v1/design-sessions`

Request:

```json
{
  "vertical": "wardrobe",
  "occasion": "work",
  "culture": "user-provided preference",
  "climate": {"temperature_c": 29, "humidity": 78},
  "preferences": {
    "styles": ["minimalist"],
    "excluded_labels": ["shorts"],
    "coverage": "user preference"
  }
}
```

The backend should:

1. authenticate the user;
2. validate `vertical` as `wardrobe`, `room`, or `garden`;
3. create a session with status `created`;
4. store the submitted context as user input, not as inferred fact;
5. return a session ID and an upload plan.

Response:

```json
{
  "session_id": "ds_01J...",
  "status": "created",
  "upload_urls": [
    {
      "image_id": "img_01J...",
      "object_key": "users/usr_123/sessions/ds_01J/images/img_01J/source.png",
      "upload_url": "<short-lived presigned PUT URL>",
      "expires_at": "2026-08-11T14:00:00Z"
    }
  ]
}
```

Use opaque IDs. Do not use a client-supplied filename as an S3 key.

### 3.2 Upload images directly to S3

The client uploads directly to S3 using the presigned URL. The backend should
not proxy large image bodies through the API process.

Recommended restrictions:

- allow JPEG, PNG, and WebP only;
- enforce a maximum content length, for example 15 MB per image;
- require the expected object key prefix;
- verify the object with `HeadObject` after upload;
- verify the detected MIME type and image dimensions server-side;
- reject SVG, HTML, archives, and files with mismatched content types;
- configure lifecycle expiration for abandoned uploads;
- encrypt objects at rest and deny public bucket access.

`POST /v1/design-sessions/{session_id}/images/complete`

```json
{
  "image_ids": ["img_01J...", "img_01K..."]
}
```

After this call, the backend should enqueue `design.analysis.requested` rather
than doing the full pipeline in the HTTP request.

### 3.3 Run the analysis pipeline

The recommended dependency graph is:

```text
quality guard ───────────────┐
object detection ────────────┼──> attribute extraction
context inference ──────────┘             │
                                         ▼
                                  permutation engine
                                    │             │
                                    ▼             ▼
                              embedding engine  LLM reasoning
                                                    │
                                                    ▼
                                             mockup compositor
```

Quality checking and context inference can run in parallel with object
detection. Attribute extraction must wait for detections. Reasoning should wait
for permutations and context. Mockup composition should wait for the selected
permutation and source image keys.

For a first implementation, use one worker per session and persist each step.
For scale, split the steps into queue jobs and use a workflow engine.

### 3.4 Read progress

`GET /v1/design-sessions/{session_id}` should return a stable status model:

```json
{
  "session_id": "ds_01J...",
  "status": "reasoning",
  "vertical": "wardrobe",
  "progress": 75,
  "steps": {
    "quality": "completed",
    "detection": "completed",
    "attributes": "completed",
    "context": "completed",
    "permutations": "completed",
    "embeddings": "completed",
    "reasoning": "running",
    "mockups": "pending"
  },
  "designs": [],
  "warnings": []
}
```

Use polling, Server-Sent Events, or WebSockets for progress. The client should
be able to reconnect and resume from the persisted status.

## 4. Service configuration in the backend

Use private service DNS names, not `localhost`, in the backend runtime:

```text
OBJECT_DETECTION_URL=http://object-detection:8001
ATTRIBUTE_EXTRACTOR_URL=http://attribute-extractor:8002
EMBEDDING_ENGINE_URL=http://embedding-engine:8003
PERMUTATION_ENGINE_URL=http://permutation-engine:8004
LLM_REASONING_URL=http://llm-reasoning:8005
MOCKUP_COMPOSITOR_URL=http://mockup-compositor:8006
QUALITY_GUARD_URL=http://quality-guard:8007
CONTEXT_INFERENCE_URL=http://context-inference:8008
AI_INTERNAL_TOKEN=<secret stored in a secret manager>
```

In a browser-facing deployment, frontend code must use relative backend URLs,
for example `/v1/design-sessions`, never `http://localhost:8001`. The backend
then calls the private AI service URLs.

Every service exposes `GET /health`. Treat health and readiness differently:

- liveness means the process is running;
- readiness means required model/storage configuration is available;
- a detector with `model_loaded: false` must be considered development-only;
- a Pinecone `memory` backend must not be accepted for production jobs.

The current service containers do not provide service authentication. Put them
on a private network immediately and add an internal token or mTLS before
exposing them outside the cluster.

## 5. Service contracts

All calls below are backend-to-service calls. Send an internal correlation ID
and, where supported by the HTTP client, an idempotency key as headers:

```text
Authorization: Bearer <service-to-service-token>
X-Correlation-ID: cor_01J...
X-Request-ID: req_01J...
```

The current service APIs are as follows.

### 5.1 Quality guard — `POST /check`

Request:

```json
{"image_keys": [
  "users/usr_123/sessions/ds_01J/images/img_01J/source.png"
]}
```

A single `image_key` is also accepted. Response:

```json
{
  "results": [
    {
      "image_key": "users/usr_123/sessions/ds_01J/images/img_01J/source.png",
      "width": 1600,
      "height": 1200,
      "blur_score": 324.1,
      "brightness": 119.2,
      "contrast": 46.7,
      "quality_score": 91.4,
      "acceptable": true,
      "issues": [],
      "guidance": ["Image quality is suitable for processing."]
    }
  ],
  "image_count": 1,
  "passed_count": 1,
  "failed_count": 0,
  "passed": true
}
```

If an image is unreadable, store the result and ask the user to recapture it.
Do not send that image into expensive inference.

### 5.2 Object detection — `POST /detect`

Request:

```json
{
  "image_keys": [
    "users/usr_123/sessions/ds_01J/images/img_01J/source.png"
  ],
  "vertical": "wardrobe"
}
```

Response:

```json
{
  "items": [
    {
      "label": "shirt",
      "confidence": 0.93,
      "bbox": {"x": 90, "y": 120, "width": 530, "height": 620},
      "category": "top",
      "image_key": "users/usr_123/sessions/ds_01J/images/img_01J/source.png"
    }
  ],
  "image_count": 1,
  "detected_count": 1,
  "failed_count": 0,
  "errors": []
}
```

The backend should create a stable `item_id` for each detection, for example
`item_<uuid>`, and attach it to the item before persisting and sending it to
later services. The detector's fallback results are prototypes and must be
marked as a warning if they are allowed outside local development.

### 5.3 Attribute extraction — `POST /extract`

Request:

```json
{
  "items": [
    {
      "id": "item_01J...",
      "label": "shirt",
      "category": "top",
      "confidence": 0.93,
      "bbox": {"x": 90, "y": 120, "width": 530, "height": 620},
      "image_key": "users/usr_123/sessions/ds_01J/images/img_01J/source.png"
    }
  ]
}
```

Response items preserve the original fields and add:

```json
{
  "id": "item_01J...",
  "embedding": [0.012, -0.041],
  "embedding_dimension": 512,
  "color": {
    "dominant": {"hex": "#234567", "rgb": {"r": 35, "g": 69, "b": 103}},
    "palette": [],
    "mood": "cool"
  },
  "texture": {
    "roughness": "medium",
    "pattern": "subtle",
    "edge_density": 0.14,
    "contrast": 31.2
  },
  "style_tags": ["minimalist", "casual", "smart-casual"]
}
```

The repository's offline vector fallback is shape-compatible but not semantic.
Only index it for local development. In production, require CLIP to be
loaded and record the model name/version with the attributes.

### 5.4 Context inference — `POST /infer`

Request:

```json
{
  "vertical": "wardrobe",
  "climate": {"temperature_c": 29, "humidity": 78},
  "culture": "south asian",
  "occasion": "work",
  "occupation": "designer",
  "season": "summer"
}
```

Response contains normalized `climate`, `culture`, `cultural_context`, and
`style_constraints`:

```json
{
  "vertical": "wardrobe",
  "occasion": "work",
  "occupation": "designer",
  "climate": {
    "type": "humid",
    "description": "humid with limited evaporative cooling",
    "temperature_c": 29.0,
    "humidity": 78.0,
    "layering": "light",
    "preferred_materials": ["linen", "cotton", "moisture-wicking fabrics"],
    "avoid_materials": ["heavy non-breathable fabrics"]
  },
  "culture": "south-asian",
  "culturalCtx": "south-asian",
  "cultural_context": {"coverage": "user preference", "explicit_input": true},
  "style_constraints": {
    "preferred_styles": ["comfortable", "context-aware", "breathable"],
    "preferred_colors": ["light neutrals", "cool tones"],
    "coverage": "user preference"
  },
  "source": "rule-based"
}
```

Culture must be an explicit user/product input. Do not infer a person's culture
from their face, body, clothing, or location.

### 5.5 Permutation engine — `POST /generate`

Request:

```json
{
  "vertical": "wardrobe",
  "context": {
    "occasion": "work",
    "style_constraints": {
      "preferred_styles": ["minimalist"],
      "excluded_labels": ["shorts"]
    }
  },
  "items": [
    {"id": "item_top", "label": "shirt", "category": "top", "style_tags": ["minimalist"]},
    {"id": "item_bottom", "label": "jeans", "category": "bottom"},
    {"id": "item_shoes", "label": "shoes", "category": "footwear"}
  ],
  "max_permutations": 20
}
```

Response:

```json
{
  "permutations": [
    {
      "id": "perm_...",
      "items": [],
      "score": 0.91,
      "reasons": ["has a complete base", "includes footwear"],
      "vertical": "wardrobe"
    }
  ],
  "count": 1,
  "truncated": false
}
```

Persist the permutation IDs and item membership. Treat the score as a ranking
signal, not a guarantee of suitability.

### 5.6 Embedding engine — `POST /upsert` and `POST /search`

Upsert after attributes are extracted:

```json
{
  "namespace": "items",
  "vectors": [
    {
      "id": "item_01J...",
      "embedding": [0.012, -0.041],
      "metadata": {
        "user_id": "usr_123",
        "session_id": "ds_01J...",
        "vertical": "wardrobe",
        "label": "shirt",
        "model_version": "clip-vit-base-patch32-v1"
      }
    }
  ]
}
```

Search similar items with:

```json
{
  "embedding": [0.012, -0.041],
  "top_k": 10,
  "namespace": "items",
  "filter": {"user_id": "usr_123", "vertical": "wardrobe"}
}
```

Use a tenant/user filter on every production query. Do not place sensitive
personal data into Pinecone metadata.

### 5.7 LLM reasoning — `POST /reason`

Request:

```json
{
  "vertical": "wardrobe",
  "context": {},
  "permutations": [
    {"id": "perm_...", "items": [{"label": "shirt"}, {"label": "jeans"}]}
  ]
}
```

Response:

```json
{
  "designs": [
    {
      "id": "perm_...",
      "items": [],
      "score": 0.91,
      "title": "Calm Studio Layers",
      "explanation": "...",
      "tips": ["..."],
      "occasion_fit": 8.0,
      "vibe": "smart"
    }
  ],
  "count": 1
}
```

The backend should store the provider, model, prompt version, token usage, and
request correlation ID for every production call. Never log API keys, raw
private images, or unredacted user prompts if they can contain private data.

### 5.8 Mockup compositor — `POST /compose`

Request:

```json
{
  "layers": [
    {
      "image_key": "users/usr_123/sessions/ds_01J/images/img_01J/source.png",
      "bbox": {"x": 90, "y": 120, "width": 530, "height": 620}
    }
  ],
  "output_key": "users/usr_123/sessions/ds_01J/mockups/perm_01J.png",
  "width": 1200,
  "height": 900,
  "format": "png",
  "background": "#f7f4ef"
}
```

Response:

```json
{
  "output_key": "users/usr_123/sessions/ds_01J/mockups/perm_01J.png",
  "url": "https://...",
  "width": 1200,
  "height": 900,
  "layers": [{"image_key": "...", "x": 100, "y": 75, "width": 1000, "height": 750}]
}
```

Persist `output_key`, not only the returned URL. Generate a fresh short-lived
signed download URL when the client requests the result. The current Pillow
implementation does not perform segmentation-aware cutouts; accurate garment
or furniture cutouts are a future enhancement.

## 6. Database model

Use a relational database such as PostgreSQL. JSON columns are useful for model
outputs, but keep ownership, status, IDs, and relationships in normal columns.
A practical first schema is:

### `design_sessions`

- `id` — opaque primary key;
- `user_id` / `tenant_id` — ownership indexes;
- `vertical` — wardrobe, room, or garden;
- `status` — created, uploading, queued, quality_review, detecting,
  extracting, permuting, embedding, reasoning, composing, completed, failed,
  cancelled;
- `input_context_json` — original explicit user inputs;
- `inferred_context_json` — context service output;
- `pipeline_version` — backend workflow version;
- `created_at`, `updated_at`, `completed_at`.

### `source_images`

- `id`, `session_id`, `user_id`;
- `object_key`, `content_type`, `byte_size`, `width`, `height`, `sha256`;
- `quality_json`, `quality_status`;
- `created_at`, `deleted_at`.

### `detected_items`

- `id`, `session_id`, `source_image_id`;
- `label`, `category`, `confidence`;
- `bbox_json`;
- `detector_model`, `detector_model_version`;
- `raw_json`, `created_at`.

### `item_attributes`

- `item_id`;
- `embedding_dimension`, `embedding_model`, `embedding_model_version`;
- `color_json`, `texture_json`, `style_tags_json`;
- `extraction_status`, `raw_json`, `created_at`.

Do not necessarily store the full vector in PostgreSQL if Pinecone is the
source of truth, but store the vector ID, namespace, model version, and index
status.

### `design_permutations`

- `id`, `session_id`, `vertical`;
- `item_ids_json`;
- `score`, `reasons_json`;
- `llm_title`, `llm_explanation`, `tips_json`, `fit_json`;
- `status`, `created_at`.

### `artifacts`

- `id`, `session_id`, `permutation_id`;
- `object_key`, `content_type`, `sha256`, `width`, `height`;
- `status`, `created_at`, `expires_at`.

### `pipeline_jobs`

- `id`, `session_id`, `step`, `idempotency_key`;
- `status`, `attempts`, `last_error`;
- `started_at`, `finished_at`, `next_retry_at`;
- `correlation_id`.

Store raw service responses only when there is a clear debugging or audit
need. Apply retention and user deletion policies to them.

## 7. Worker orchestration pseudocode

The first implementation can use a queue such as BullMQ, Celery, Sidekiq, SQS,
RabbitMQ, or Temporal. The important property is that each step is durable and
idempotent.

```ts
async function runDesignSession(sessionId: string) {
  const session = await db.sessions.lockForUpdate(sessionId);
  if (session.status === "completed" || session.status === "cancelled") return;

  const images = await db.images.forSession(sessionId);
  const contextInput = session.input_context_json;

  await Promise.all([
    runOnce(sessionId, "quality", async () => {
      return ai.post("/check", { image_keys: images.map(x => x.object_key) });
    }),
    runOnce(sessionId, "context", async () => {
      return ai.post("/infer", {
        vertical: session.vertical,
        ...contextInput
      });
    }),
    runOnce(sessionId, "detection", async () => {
      return ai.post("/detect", {
        image_keys: images.map(x => x.object_key),
        vertical: session.vertical
      });
    })
  ]);

  const quality = await db.steps.result(sessionId, "quality");
  if (!quality.passed) {
    await db.sessions.moveTo(sessionId, "quality_review");
    await notifyClient(sessionId);
    return;
  }

  const detections = await db.steps.result(sessionId, "detection");
  const items = await saveStableItemIds(sessionId, detections.items);

  const attributes = await runOnce(sessionId, "attributes", () =>
    ai.post("/extract", { items })
  );
  await saveAttributes(sessionId, attributes.items);

  await runOnce(sessionId, "embeddings", () =>
    ai.post("/upsert", {
      namespace: "items",
      vectors: attributes.items
        .filter(item => item.embedding?.length)
        .map(item => ({
          id: item.id,
          embedding: item.embedding,
          metadata: {
            user_id: session.user_id,
            session_id: sessionId,
            vertical: session.vertical,
            label: item.label
          }
        }))
    })
  );

  const context = await db.steps.result(sessionId, "context");
  const permutations = await runOnce(sessionId, "permutations", () =>
    ai.post("/generate", {
      vertical: session.vertical,
      items: attributes.items,
      context: mergeUserPreferences(context, contextInput)
    })
  );

  const designs = await runOnce(sessionId, "reasoning", () =>
    ai.post("/reason", {
      vertical: session.vertical,
      context,
      permutations: permutations.permutations
    })
  );
  await db.designs.save(sessionId, designs.designs);

  for (const design of designs.designs) {
    await runOnce(`${sessionId}:${design.id}`, "mockup", () =>
      ai.post("/compose", {
        layers: design.items.map(item => ({
          image_key: item.image_key,
          bbox: item.bbox
        })),
        output_key: `users/${session.user_id}/sessions/${sessionId}/mockups/${design.id}.jpg`,
        format: "jpg"
      })
    );
  }

  await db.sessions.moveTo(sessionId, "completed");
  await notifyClient(sessionId);
}
```

`runOnce` must:

1. derive a deterministic idempotency key from session, step, and input hash;
2. return the stored successful result if the step already completed;
3. set a deadline on the HTTP call;
4. retry only transient failures with exponential backoff and jitter;
5. record the attempt and redacted error;
6. mark the step failed after the retry budget is exhausted.

Do not retry validation errors, unauthorized errors, unreadable images, or
invalid model responses indefinitely.

## 8. Error and retry policy

Recommended defaults, tuned per service:

| Failure | Retry? | Action |
|---|---|---|
| HTTP 400/422 | No | Persist validation error and show actionable feedback |
| HTTP 401/403 | No | Alert service-auth configuration |
| HTTP 404 for source object | No | Mark image unavailable and ask for re-upload |
| HTTP 408/429 | Yes | Honor `Retry-After`, capped exponential backoff |
| HTTP 500/502/503/504 | Yes | 2–4 attempts, then step failure |
| LLM provider timeout | Yes | Fail over provider, then use explicit fallback policy |
| Pinecone unavailable | No in production | Do not silently use the memory backend |
| Worker crash | Yes | Queue visibility timeout and idempotent replay |

Return a stable public error shape from the backend:

```json
{
  "error": {
    "code": "IMAGE_QUALITY_REJECTED",
    "message": "One or more images need to be retaken.",
    "details": [{"image_id": "img_01J...", "issues": ["blur", "low_light"]}],
    "request_id": "req_01J..."
  }
}
```

Do not expose stack traces, provider errors, S3 bucket names, or internal URLs
to clients.

## 9. Security requirements

Before production:

- keep all AI services on a private network;
- use a service identity/token or mTLS for backend-to-service calls;
- validate that every requested S3 key belongs to the current user/session;
- use least-privilege IAM policies: source services read, compositor writes
  only to the user's artifact prefix;
- block public bucket access and issue short-lived signed URLs;
- scan uploads and enforce dimensions, MIME type, and byte limits;
- never accept arbitrary URLs in image-key APIs;
- sanitize output keys and prevent path traversal;
- put API keys in a secret manager, not `.env` files in deployments;
- redact images, prompts, access tokens, and personal context from logs;
- add user data deletion that removes database rows, S3 objects, vectors, and
  queued jobs;
- apply tenant filters to every database and vector-store query.

## 10. Observability and operations

Every backend request and worker job should have:

- `request_id`, `correlation_id`, `session_id`, `user_id` (hashed or internal),
  `step`, `service`, and `model_version`;
- duration, status, retry count, payload size, and error category;
- service metrics for latency, error rate, queue depth, model load state, and
  S3/Pinecone failures;
- LLM metrics for provider, model, token count, cost estimate, and failover;
- traces connecting the public request through each worker and service.

Add alerts for failed sessions, queue age, high LLM spend, detector readiness,
Pinecone fallback activation, and repeated image-quality failures.

## 11. Deployment plan

### Phase 1 — backend foundation

- create the session/image/artifact/job database tables;
- add authenticated session and presigned-upload endpoints;
- add private service URLs and an HTTP client with deadlines;
- run one worker with the pipeline above;
- persist every step and expose session status.

### Phase 2 — production integrations

- configure private S3, Pinecone, LLM credentials, and model volumes;
- enable CLIP and a domain-trained detector;
- reject fallback backends in production;
- add service authentication and readiness checks;
- add structured logs, traces, and metrics.

### Phase 3 — scale and product quality

- split long steps into queue jobs or a workflow engine;
- add cancellation and resume support;
- add user feedback and learned ranking;
- add segmentation-aware mockups;
- add contract, golden-image, load, and security tests.

## 12. Definition of done

The backend integration is ready for a production pilot when:

- a user can create a session and upload images without exposing credentials;
- a worker can restart without duplicating vectors, designs, or artifacts;
- all AI calls have deadlines, correlation IDs, retries, and persisted results;
- unreadable or low-quality images stop the pipeline with useful guidance;
- the backend returns only designs belonging to the requesting user;
- generated artifacts are private and served through short-lived signed URLs;
- production readiness rejects missing detector/CLIP/Pinecone configuration;
- provider failures are visible and do not leak secrets;
- deletion removes source images, derived data, vectors, artifacts, and jobs;
- dashboards and alerts show pipeline health and cost.

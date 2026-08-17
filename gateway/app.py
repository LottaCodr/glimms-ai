"""HTTP gateway for the all-in-one Glimms container.

The repository normally runs as eight independent containers (see
``docker-compose.yml``).  The root ``Dockerfile`` instead starts every
service inside one container and puts this gateway in front of them, so a
single deployed container exposes the whole pipeline on a single port:

    GET  /                         -> human/index of all services
    GET  /health                   -> aggregated health of all services
    ANY  /<service-name>/<path>    -> proxied to that service

Service names are the hyphenated forms used by ``docker-compose.yml``
(``object-detection``, ``attribute-extractor``, ``embedding-engine``,
``permutation-engine``, ``llm-reasoning``, ``mockup-compositor``,
``quality-guard``, ``context-inference``).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

from shared.auth import auth_required, configured_tokens, install_service_auth
from shared.runtime import allow_dev_fallbacks, environment, is_production

#: Map of URL prefix -> internal port, one entry per pipeline service.
SERVICE_PORTS: dict[str, int] = {
    "object-detection": 8001,
    "attribute-extractor": 8002,
    "embedding-engine": 8003,
    "permutation-engine": 8004,
    "llm-reasoning": 8005,
    "mockup-compositor": 8006,
    "quality-guard": 8007,
    "context-inference": 8008,
}

UPSTREAM_HOST = os.getenv("UPSTREAM_HOST", "127.0.0.1")
UPSTREAM_TIMEOUT_SECONDS = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "120"))
HEALTH_TIMEOUT_SECONDS = float(os.getenv("HEALTH_TIMEOUT_SECONDS", "3"))

#: Reject oversized bodies at the edge rather than buffering them per service.
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", str(20 * 1024 * 1024)))

#: Bound the number of requests in flight to the eight in-process services.
MAX_CONCURRENT_UPSTREAM = int(os.getenv("MAX_CONCURRENT_UPSTREAM", "16"))

#: Health keys that indicate a service is answering with a dev fallback.
_DEGRADED_CHECKS: dict[str, tuple[str, Any, str]] = {
    "object-detection": (
        "model_loaded",
        False,
        "no YOLO model is mounted; detections are deterministic prototypes",
    ),
    "attribute-extractor": (
        "clip_enabled",
        False,
        "CLIP is disabled; embeddings are deterministic offline vectors",
    ),
    "embedding-engine": (
        "backend",
        "memory",
        "vectors live in process memory and are lost on restart",
    ),
}

#: Hop-by-hop headers (plus length headers Starlette recomputes) that must
#: never be forwarded between the client and an upstream service.
_HOP_BY_HOP = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
}

#: Client credentials are terminated at the gateway and re-issued upstream, so
#: an inbound Authorization header is never blindly forwarded.
_STRIP_UPSTREAM = {"authorization", "x-internal-token"}

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Glimms AI — all-in-one gateway</title>
<style>
  body { font-family: ui-sans-serif, system-ui, sans-serif; margin: 0;
         background: #f6f7fb; color: #1c2333; }
  main { max-width: 860px; margin: 0 auto; padding: 40px 24px 64px; }
  h1 { font-size: 1.6rem; margin-bottom: 4px; }
  p.sub { color: #5a6478; margin-top: 0; }
  table { border-collapse: collapse; width: 100%; background: #fff;
          border-radius: 10px; overflow: hidden;
          box-shadow: 0 1px 3px rgba(16, 24, 40, .12); }
  th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid #edf0f6; }
  th { background: #eef1f8; font-size: .8rem; text-transform: uppercase;
       letter-spacing: .04em; color: #5a6478; }
  code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  code { background: #eef1f8; padding: 1px 5px; border-radius: 5px; font-size: .85em; }
  pre { background: #101828; color: #d5dbe8; padding: 14px 16px; border-radius: 10px;
        overflow-x: auto; font-size: .82rem; }
  a { color: #3556c9; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .pill { display: inline-block; padding: 2px 8px; border-radius: 999px;
          background: #e5efff; color: #2748b5; font-size: .75rem; }
</style>
</head>
<body>
<main>
  <h1>Glimms AI — all-in-one gateway</h1>
  <p class="sub">All eight services are running behind this port. Route any
  request to <code>/&lt;service-name&gt;/...</code>; everything after the
  prefix is passed through unchanged.</p>

  <table>
    <tr><th>Service</th><th>Prefix</th><th>Port</th><th>Docs</th></tr>
    __SERVICE_ROWS__
  </table>

  <h3 style="margin-bottom:6px">Example</h3>
  <pre>curl -X POST http://&lt;host&gt;:__PORT__/permutation-engine/generate \\
  -H 'content-type: application/json' \\
  -d '{"vertical": "wardrobe", "items": [
        {"id": "top-1", "label": "shirt", "category": "top"},
        {"id": "bottom-1", "label": "jeans", "category": "bottom"},
        {"id": "shoe-1", "label": "shoes", "category": "footwear"}]}'</pre>

  <p class="sub">Aggregated status: <a href="/health"><code>/health</code></a>.
  Readiness (fails when the deployment is running prototype fallbacks):
  <a href="/readyz"><code>/readyz</code></a>. Liveness:
  <a href="/livez"><code>/livez</code></a>. Machine-readable index:
  <a href="/index.json"><code>/index.json</code></a>.
  This lightweight image runs deterministic offline fallbacks (no torch,
  CLIP, YOLO, rembg or Pinecone); build the per-service images with
  <code>docker compose build</code> for full integrations.</p>

  <p class="sub">Service endpoints require
  <code>Authorization: Bearer $AI_INTERNAL_TOKEN</code> whenever that variable
  is set, and setting it is mandatory when <code>GLIMMS_ENV=production</code>.</p>
</main>
</body>
</html>
"""

app = FastAPI(
    title="Glimms — All-in-One Gateway",
    version="1.1.0",
    description=(
        "Routes `/{service-name}/...` to the eight Glimms pipeline services "
        "running inside the all-in-one container. Send "
        "`Authorization: Bearer $AI_INTERNAL_TOKEN` on every call."
    ),
)

# The index, liveness probe and machine-readable service list stay public so a
# platform health check and a human opening the URL both work; everything that
# touches the pipeline (or the bucket) requires the shared secret.
install_service_auth(
    app,
    "gateway",
    extra_public_paths=("/", "/livez", "/index.json"),
)

_client: httpx.AsyncClient | None = None
_upstream_gate = asyncio.Semaphore(MAX_CONCURRENT_UPSTREAM)


def _client_or_raise() -> httpx.AsyncClient:
    if _client is None:  # pragma: no cover - only before startup event
        raise RuntimeError("gateway client not started")
    return _client


@app.on_event("startup")
async def _start_client() -> None:
    global _client
    _client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_SECONDS)


@app.on_event("shutdown")
async def _close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


@app.get("/", include_in_schema=False)
def index() -> HTMLResponse:
    rows = "\n".join(
        f"<tr><td>{name}</td>"
        f"<td><code>/{name}/</code></td>"
        f"<td>{port}</td>"
        f'<td><a href="/{name}/docs">/docs</a></td></tr>'
        for name, port in SERVICE_PORTS.items()
    )
    port = os.getenv("PORT", "8080")
    html = _INDEX_HTML.replace("__SERVICE_ROWS__", rows).replace("__PORT__", port)
    return HTMLResponse(html)


@app.get("/index.json", tags=["gateway"])
def index_json() -> dict[str, Any]:
    return {
        "gateway": "glimms-all-in-one",
        "services": [
            {"name": name, "prefix": f"/{name}/", "port": port}
            for name, port in SERVICE_PORTS.items()
        ],
    }


@app.get("/livez", tags=["gateway"])
def livez() -> dict[str, str]:
    """Liveness only: the gateway process is up. Always public."""

    return {"status": "ok", "service": "gateway"}


def _degradations(services: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Report every service currently answering with a development fallback."""

    findings: list[dict[str, str]] = []
    for name, (key, bad_value, reason) in _DEGRADED_CHECKS.items():
        state = services.get(name, {})
        if state.get(key) == bad_value:
            findings.append({"service": name, "reason": reason})

    llm = services.get("llm-reasoning", {})
    providers = llm.get("providers_configured") or []
    if not [provider for provider in providers if provider != "free-fallback"]:
        findings.append(
            {
                "service": "llm-reasoning",
                "reason": (
                    "no paid provider is configured; reasoning uses a "
                    "rate-limited free tier or the offline fallback"
                ),
            }
        )
    return findings


async def _probe_services() -> dict[str, dict[str, Any]]:
    client = _client_or_raise()

    # The services enforce the shared secret too, so the gateway must
    # authenticate its own health probes.
    probe_headers = {"accept": "application/json"}
    tokens = configured_tokens()
    if tokens:
        probe_headers["authorization"] = f"Bearer {tokens[0]}"

    async def probe(name: str, port: int) -> tuple[str, dict[str, Any]]:
        url = f"http://{UPSTREAM_HOST}:{port}/health"
        try:
            response = await client.get(
                url, timeout=HEALTH_TIMEOUT_SECONDS, headers=probe_headers
            )
            payload: dict[str, Any]
            try:
                payload = response.json()
            except ValueError:
                payload = {"detail": response.text[:200]}
            return name, {"status": "ok" if response.is_success else "error", **payload}
        except httpx.HTTPError as exc:
            return name, {"status": "unreachable", "detail": str(exc)}

    results = await asyncio.gather(
        *(probe(name, port) for name, port in SERVICE_PORTS.items())
    )
    return dict(results)


@app.get("/health", tags=["gateway"])
async def health() -> dict[str, Any]:
    services = await _probe_services()
    findings = _degradations(services)
    reachable = all(state.get("status") == "ok" for state in services.values())
    return {
        "status": "ok" if reachable else "degraded",
        "service": "gateway",
        "environment": environment(),
        "auth_required": auth_required(),
        "dev_fallbacks_allowed": allow_dev_fallbacks(),
        "production_ready": reachable and not findings,
        "degradations": findings,
        "services": services,
    }


@app.get("/readyz", tags=["gateway"])
async def readyz() -> JSONResponse:
    """Readiness: every service is up *and* is not a development fallback.

    Point a production load balancer at this rather than ``/health`` so a
    deployment that lost its models or provider keys stops taking traffic
    instead of quietly serving prototype output.
    """

    services = await _probe_services()
    unreachable = [
        name for name, state in services.items() if state.get("status") != "ok"
    ]
    findings = _degradations(services)
    # Fallbacks only block readiness where they are not explicitly permitted.
    blocking = findings if not allow_dev_fallbacks() else []
    ready = not unreachable and not blocking
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "service": "gateway",
            "environment": environment(),
            "unreachable": unreachable,
            "degradations": findings,
            "dev_fallbacks_allowed": allow_dev_fallbacks(),
        },
    )


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    include_in_schema=False,
)
async def proxy(path: str, request: Request) -> Response:
    correlation_id = (
        request.headers.get("x-correlation-id")
        or request.headers.get("x-request-id")
        or f"cor_{uuid.uuid4().hex[:16]}"
    )
    prefix, _, rest = path.partition("/")
    port = SERVICE_PORTS.get(prefix)
    if port is None:
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"unknown service prefix '/{prefix}'",
                "available": sorted(SERVICE_PORTS),
                "hint": "route /<service-name>/<path> to reach a service",
                "correlation_id": correlation_id,
            },
            headers={"x-correlation-id": correlation_id},
        )

    declared_length = request.headers.get("content-length")
    if declared_length and declared_length.isdigit():
        if int(declared_length) > MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": "request body is too large",
                    "max_bytes": MAX_REQUEST_BYTES,
                    "correlation_id": correlation_id,
                },
                headers={"x-correlation-id": correlation_id},
            )

    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "detail": "request body is too large",
                "max_bytes": MAX_REQUEST_BYTES,
                "correlation_id": correlation_id,
            },
            headers={"x-correlation-id": correlation_id},
        )

    url = f"http://{UPSTREAM_HOST}:{port}/{rest}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP and key.lower() not in _STRIP_UPSTREAM
    }
    headers["x-correlation-id"] = correlation_id
    # Re-attach the shared secret for the upstream hop: the services enforce
    # it too, so a leaked internal port is still not an open API.
    tokens = configured_tokens()
    if tokens:
        headers["authorization"] = f"Bearer {tokens[0]}"

    try:
        async with _upstream_gate:
            upstream = await _client_or_raise().request(
                request.method, url, headers=headers, content=body
            )
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "detail": f"service '{prefix}' is unreachable: {exc}",
                "hint": "the service may still be starting; retry shortly",
                "correlation_id": correlation_id,
            },
            headers={"x-correlation-id": correlation_id},
        )

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in _HOP_BY_HOP
    }
    response_headers["x-correlation-id"] = correlation_id
    return Response(
        status_code=upstream.status_code,
        content=upstream.content,
        headers=response_headers,
    )

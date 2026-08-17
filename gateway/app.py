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
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

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
  Machine-readable index: <a href="/index.json"><code>/index.json</code></a>.
  This lightweight image runs deterministic offline fallbacks (no torch,
  CLIP, YOLO, rembg or Pinecone); build the per-service images with
  <code>docker compose build</code> for full integrations.</p>
</main>
</body>
</html>
"""

app = FastAPI(
    title="Glimms — All-in-One Gateway",
    version="1.0.0",
    description=(
        "Routes `/{service-name}/...` to the eight Glimms pipeline services "
        "running inside the all-in-one container."
    ),
)

_client: httpx.AsyncClient | None = None


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


@app.get("/health", tags=["gateway"])
async def health() -> dict[str, Any]:
    client = _client_or_raise()

    async def probe(name: str, port: int) -> tuple[str, dict[str, Any]]:
        url = f"http://{UPSTREAM_HOST}:{port}/health"
        try:
            response = await client.get(
                url, timeout=HEALTH_TIMEOUT_SECONDS,
                headers={"accept": "application/json"},
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
    services = dict(results)
    overall = (
        "ok"
        if all(state.get("status") == "ok" for state in services.values())
        else "degraded"
    )
    return {"status": overall, "service": "gateway", "services": services}


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    include_in_schema=False,
)
async def proxy(path: str, request: Request) -> Response:
    prefix, _, rest = path.partition("/")
    port = SERVICE_PORTS.get(prefix)
    if port is None:
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"unknown service prefix '/{prefix}'",
                "available": sorted(SERVICE_PORTS),
                "hint": "route /<service-name>/<path> to reach a service",
            },
        )

    url = f"http://{UPSTREAM_HOST}:{port}/{rest}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _HOP_BY_HOP
    }
    body = await request.body()

    try:
        upstream = await _client_or_raise().request(
            request.method, url, headers=headers, content=body
        )
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "detail": f"service '{prefix}' is unreachable: {exc}",
                "hint": "the service may still be starting; retry shortly",
            },
        )

    response_headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in _HOP_BY_HOP
    }
    return Response(
        status_code=upstream.status_code,
        content=upstream.content,
        headers=response_headers,
    )

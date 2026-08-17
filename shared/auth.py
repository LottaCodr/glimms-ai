"""Shared-secret authentication for the gateway and the eight services.

The services are designed to sit on a private network, but "private" is a
deployment property that is easy to get wrong: a single-container deployment
on a PaaS gets a public URL by default.  This middleware adds a cheap second
layer so an accidentally public port is not an open write endpoint.

Configuration:

``AI_INTERNAL_TOKEN``
    Shared secret.  Callers must send ``Authorization: Bearer <token>``
    (``X-Internal-Token: <token>`` is also accepted).  Comma-separated values
    are supported so a token can be rotated without downtime.

``AUTH_PUBLIC_PATHS``
    Comma-separated extra paths that stay unauthenticated.

In production (``GLIMMS_ENV=production``) a missing token is a startup error:
failing to boot is safer than silently serving an unauthenticated write API.
Outside production a missing token only logs a warning, so local development
and ``docker compose up`` keep working with no configuration.

``/livez`` is always public so a platform health check can reach it without
holding the secret.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import Iterable

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from .runtime import is_production

logger = logging.getLogger(__name__)

#: Always reachable without a token: platform liveness probes.
DEFAULT_PUBLIC_PATHS = frozenset({"/livez"})


def configured_tokens() -> list[str]:
    """Return every accepted token (supports comma-separated rotation)."""

    raw = os.getenv("AI_INTERNAL_TOKEN", "")
    return [token.strip() for token in raw.split(",") if token.strip()]


def auth_required() -> bool:
    return bool(configured_tokens())


def _presented_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    return request.headers.get("x-internal-token", "").strip()


def _token_matches(presented: str, accepted: Iterable[str]) -> bool:
    # compare_digest against every candidate, without short-circuiting.
    result = False
    for token in accepted:
        result |= hmac.compare_digest(presented, token)
    return result


class InternalTokenMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, service: str, public_paths: frozenset[str]) -> None:
        super().__init__(app)
        self.service = service
        self.public_paths = public_paths

    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in self.public_paths:
            return await call_next(request)

        accepted = configured_tokens()
        if not accepted:
            # Unconfigured outside production: allowed, but never silently.
            return await call_next(request)

        presented = _presented_token(request)
        if not presented or not _token_matches(presented, accepted):
            logger.warning(
                "rejected unauthenticated %s %s", request.method, request.url.path
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "service": self.service,
                    "detail": "a valid internal token is required",
                    "hint": "send Authorization: Bearer <AI_INTERNAL_TOKEN>",
                },
                headers={"www-authenticate": "Bearer"},
            )
        return await call_next(request)


def install_service_auth(
    app: FastAPI,
    service: str,
    *,
    extra_public_paths: Iterable[str] = (),
) -> None:
    """Attach shared-secret auth to ``app``.

    Raises ``RuntimeError`` in production when no token is configured.
    """

    public = set(DEFAULT_PUBLIC_PATHS)
    public.update(extra_public_paths)
    public.update(
        path.strip()
        for path in os.getenv("AUTH_PUBLIC_PATHS", "").split(",")
        if path.strip()
    )

    if not configured_tokens():
        message = (
            f"{service}: AI_INTERNAL_TOKEN is not set. The service API would be "
            "reachable by anyone who can reach the port."
        )
        if is_production():
            raise RuntimeError(
                message + " Set AI_INTERNAL_TOKEN (or GLIMMS_ENV=development)."
            )
        logger.warning("%s Set it before exposing this deployment.", message)

    app.add_middleware(
        InternalTokenMiddleware, service=service, public_paths=frozenset(public)
    )

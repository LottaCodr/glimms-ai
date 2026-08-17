"""Runtime mode helpers shared by every service and the gateway.

The services ship deterministic offline fallbacks so the pipeline can be
exercised without models, Pinecone, or provider keys.  Those fallbacks are
useful in development and dangerous in production: they look like real
results.  This module centralises the switch that decides whether a fallback
may answer a request.

Two environment variables control it:

``GLIMMS_ENV``
    ``development`` (default), ``staging``, or ``production``.

``ALLOW_DEV_FALLBACKS``
    Explicit override.  Unset means "allow outside production".  Set it to
    ``true`` to knowingly run fallbacks in production (for example a demo
    deployment), or to ``false`` to forbid them everywhere.

When fallbacks are not allowed, a service that can only answer with a
fallback returns ``503`` instead of a confident-looking fake result.
"""

from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""

    raw = os.getenv(name, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return default


def environment() -> str:
    """Return the normalised deployment environment name."""

    value = os.getenv("GLIMMS_ENV", "development").strip().lower()
    return value or "development"


def is_production() -> bool:
    return environment() in {"production", "prod"}


def allow_dev_fallbacks() -> bool:
    """Whether deterministic development fallbacks may serve real responses."""

    return env_flag("ALLOW_DEV_FALLBACKS", default=not is_production())


class DevFallbackBlocked(RuntimeError):
    """Raised when only a development fallback could answer a request."""

    def __init__(self, service: str, reason: str, remedy: str) -> None:
        self.service = service
        self.reason = reason
        self.remedy = remedy
        super().__init__(f"{service}: {reason}")

    def detail(self) -> dict[str, str]:
        return {
            "error": "development_fallback_blocked",
            "service": self.service,
            "reason": self.reason,
            "remedy": self.remedy,
            "hint": (
                "Set ALLOW_DEV_FALLBACKS=true to accept prototype output, or "
                "configure the real integration."
            ),
        }


def guard_dev_fallback(service: str, *, degraded: bool, reason: str, remedy: str) -> None:
    """Raise :class:`DevFallbackBlocked` when a fallback must not be used."""

    if degraded and not allow_dev_fallbacks():
        raise DevFallbackBlocked(service, reason, remedy)

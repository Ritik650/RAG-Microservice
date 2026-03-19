"""Optional JWT bearer auth for the write/query endpoints.

Disabled by default (AUTH_ENABLED=false) so local demos need no token. When
enabled, /ingest and /query require ``Authorization: Bearer <HS256 JWT>`` signed
with JWT_SECRET. Mint demo tokens with ``python scripts/make_token.py``.
"""

from __future__ import annotations

import time

import jwt as pyjwt
from fastapi import HTTPException, Request


def create_token(subject: str, secret: str, ttl_seconds: int, algorithm: str = "HS256") -> str:
    now = int(time.time())
    payload = {"sub": subject, "iat": now, "exp": now + ttl_seconds}
    return pyjwt.encode(payload, secret, algorithm=algorithm)


def verify_token(token: str, secret: str, algorithm: str = "HS256") -> dict:
    """Return the decoded claims; raises pyjwt exceptions on invalid/expired tokens."""
    return pyjwt.decode(token, secret, algorithms=[algorithm])


async def require_auth(request: Request) -> dict | None:
    """FastAPI dependency: no-op unless AUTH_ENABLED, then enforce a Bearer JWT."""
    settings = request.app.state.settings
    if not settings.auth_enabled:
        return None
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    try:
        return verify_token(token, settings.jwt_secret, settings.jwt_algorithm)
    except pyjwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

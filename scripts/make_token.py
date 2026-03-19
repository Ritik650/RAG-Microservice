"""Mint a demo JWT for the service when AUTH_ENABLED=true.

Usage:
    python scripts/make_token.py [subject]

Reads JWT_SECRET / JWT_TTL_SECONDS from the environment (or .env defaults).
"""

from __future__ import annotations

import sys

from app.auth.jwt_auth import create_token
from app.config import Settings


def main() -> None:
    settings = Settings()
    subject = sys.argv[1] if len(sys.argv) > 1 else "demo-user"
    token = create_token(
        subject, settings.jwt_secret, settings.jwt_ttl_seconds, settings.jwt_algorithm
    )
    print(token)


if __name__ == "__main__":
    main()

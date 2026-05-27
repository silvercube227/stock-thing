"""Supabase JWT verification.

The API talks to Postgres over a direct asyncpg connection (DATABASE_URL), which
runs as the table owner and therefore *bypasses* row-level security. RLS is not a
safety net here — every portfolio query must scope by the `user_id` this module
extracts from the verified bearer token.

Supabase's current key system signs user JWTs with asymmetric keys published at
`<project>/auth/v1/.well-known/jwks.json` (see `Settings.supabase_jwks_url`). We
verify the signature against that JWKS and return the `sub` claim (the user UUID).
"""

from __future__ import annotations

from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.config import Settings, get_settings

# Supabase issues the "authenticated" audience for logged-in users.
_AUDIENCE = "authenticated"
_ALGORITHMS = ["ES256", "RS256"]

_bearer = HTTPBearer(auto_error=False)


@lru_cache
def _jwks_client(jwks_url: str) -> jwt.PyJWKClient:
    # Cached per URL so we reuse the fetched/rotated signing keys.
    return jwt.PyJWKClient(jwks_url)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> str:
    """Verify the bearer token and return the Supabase user id (UUID string)."""
    if credentials is None or not credentials.credentials:
        raise _unauthorized("Missing bearer token")

    token = credentials.credentials
    try:
        signing_key = _jwks_client(settings.supabase_jwks_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=_ALGORITHMS,
            audience=_AUDIENCE,
        )
    except jwt.PyJWTError as exc:
        raise _unauthorized(f"Invalid token: {exc}") from exc

    user_id = claims.get("sub")
    if not user_id:
        raise _unauthorized("Token missing subject")
    return str(user_id)

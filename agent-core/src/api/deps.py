"""FastAPI dependencies for authentication.

``current_customer`` is the single seam the design doc (§四 阶段2) calls for:
it turns a Bearer access token into a trusted ``customer_id``. Endpoints depend
on it instead of trusting a self-reported customer_id. Downstream ownership
checks (_require_session_owner) are unchanged -- only the *source* of
customer_id becomes trustworthy.
"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, Request

from src.security.auth import TokenError, decode_token


def _settings(request: Request):
    return request.app.state.settings


async def current_customer(
    request: Request,
    authorization: str | None = Header(default=None),
) -> str:
    """Resolve a trusted customer_id from the Bearer access token.

    Raises 401 when the header is missing/malformed or the token is invalid or
    expired. The returned value is the JWT ``sub`` claim.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization[7:].strip()
    settings = _settings(request)
    try:
        claims = decode_token(
            token,
            settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
            expected_type="access",
        )
    except TokenError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    return claims["sub"]


async def optional_customer(
    request: Request,
    authorization: str | None = Header(default=None),
) -> str | None:
    """Like current_customer but returns None instead of raising when no valid
    token is present. Useful during the migration window where some clients may
    not yet send tokens."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    try:
        return await current_customer(request, authorization)
    except HTTPException:
        return None

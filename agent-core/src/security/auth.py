"""Authentication primitives: password hashing and JWT issue/verify.

Kept dependency-light (bcrypt + PyJWT) and free of FastAPI imports so it can be
unit-tested in isolation. Wiring into request handling lives in the API layer.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

# Token lifetimes. Access tokens are short-lived; refresh tokens let the client
# obtain a new access token without re-entering credentials.
ACCESS_TOKEN_TTL = timedelta(minutes=30)
REFRESH_TOKEN_TTL = timedelta(days=7)


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt; returns a str safe to store."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str | None) -> bool:
    """Constant-time check of a plaintext password against a stored hash."""
    if not password_hash:
        return False
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # Malformed hash in storage -> treat as non-matching rather than raising.
        return False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_token(
    subject: str,
    secret: str,
    *,
    token_type: str,
    ttl: timedelta,
    algorithm: str = "HS256",
    extra_claims: dict | None = None,
) -> str:
    """Sign a JWT. ``subject`` becomes the ``sub`` claim (the customer_id)."""
    now = _now()
    payload = {
        "sub": subject,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + ttl).timestamp()),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, secret, algorithm=algorithm)


def create_access_token(subject: str, secret: str, *, algorithm: str = "HS256",
                        extra_claims: dict | None = None) -> str:
    return create_token(subject, secret, token_type="access", ttl=ACCESS_TOKEN_TTL,
                        algorithm=algorithm, extra_claims=extra_claims)


def create_refresh_token(subject: str, secret: str, *, algorithm: str = "HS256") -> str:
    return create_token(subject, secret, token_type="refresh", ttl=REFRESH_TOKEN_TTL,
                        algorithm=algorithm)


class TokenError(Exception):
    """Raised when a token is missing, malformed, expired, or the wrong type."""


def decode_token(token: str, secret: str, *, algorithm: str = "HS256",
                 expected_type: str | None = None) -> dict:
    """Verify signature + expiry and return claims. Raises TokenError on failure."""
    try:
        claims = jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.ExpiredSignatureError as e:
        raise TokenError("token expired") from e
    except jwt.InvalidTokenError as e:
        raise TokenError("invalid token") from e
    if expected_type is not None and claims.get("type") != expected_type:
        raise TokenError(f"expected {expected_type} token")
    if not claims.get("sub"):
        raise TokenError("token missing subject")
    return claims

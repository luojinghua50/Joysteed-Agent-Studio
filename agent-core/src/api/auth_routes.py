"""Authentication routes: register / login / refresh / me.

This is the auth *backend* (阶段2 of auth-user-design.md). Endpoints here mint
and refresh tokens but do NOT yet enforce auth on the chat endpoints -- that
switch (self-reported customer_id -> token-derived) is P1-B. Keeping them
separate lets the running demo keep working while the frontend (P1-C) learns to
obtain and send tokens.
"""
from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select

from src.api.deps import current_customer
from src.api.schemas import (
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from src.database import UserModel
from src.security.auth import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

logger = structlog.get_logger()

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def _roles_list(raw: str | None) -> list[str]:
    return [r.strip() for r in (raw or "").split(",") if r.strip()]


@router.post("/guest", response_model=TokenResponse)
async def guest(request: Request):
    """Mint tokens for an anonymous guest.

    No user row is created -- the guest's customer_id lives only in the token's
    ``sub`` claim. This keeps the visitor flow working while ensuring identity
    is always token-derived (never self-reported), closing the forgery hole.
    """
    settings = request.app.state.settings
    customer_id = f"guest-{uuid.uuid4()}"
    logger.info("guest_token_issued", customer_id=customer_id)
    return _issue_tokens(customer_id, settings, role="guest")


@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest, request: Request):
    """Create a user and return tokens. ``id`` (== customer_id) is generated."""
    db_factory = request.app.state.db_session_factory
    settings = request.app.state.settings
    async with db_factory() as db:
        existing = await db.scalar(select(UserModel).where(UserModel.username == req.username))
        if existing is not None:
            raise HTTPException(status_code=409, detail="Username already taken")
        customer_id = f"u-{uuid.uuid4()}"
        user = UserModel(
            id=customer_id,
            username=req.username,
            password_hash=hash_password(req.password),
            display_name=req.display_name or req.username,
            roles="customer",
        )
        db.add(user)
        await db.commit()
    logger.info("user_registered", customer_id=customer_id, username=req.username)
    return _issue_tokens(customer_id, settings)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, request: Request):
    """Verify credentials and return tokens. Generic 401 on any failure so we
    don't leak whether a username exists."""
    db_factory = request.app.state.db_session_factory
    settings = request.app.state.settings
    async with db_factory() as db:
        user = await db.scalar(select(UserModel).where(UserModel.username == req.username))
    if user is None or not verify_password(req.password, user.password_hash):
        logger.warning("login_failed", username=req.username)
        raise HTTPException(status_code=401, detail="Invalid username or password")
    logger.info("login_ok", customer_id=user.id)
    return _issue_tokens(user.id, settings)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest, request: Request):
    """Exchange a valid refresh token for a fresh access token."""
    settings = request.app.state.settings
    try:
        claims = decode_token(
            req.refresh_token,
            settings.jwt_secret,
            algorithm=settings.jwt_algorithm,
            expected_type="refresh",
        )
    except TokenError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    customer_id = claims["sub"]
    access = create_access_token(
        customer_id, settings.jwt_secret, algorithm=settings.jwt_algorithm
    )
    # Rotate the refresh token too so the window slides forward.
    refresh_token = create_refresh_token(
        customer_id, settings.jwt_secret, algorithm=settings.jwt_algorithm
    )
    return TokenResponse(
        access_token=access, refresh_token=refresh_token, customer_id=customer_id
    )


@router.get("/me", response_model=UserResponse)
async def me(request: Request, customer_id: str = Depends(current_customer)):
    """Return the authenticated user's public profile."""
    db_factory = request.app.state.db_session_factory
    async with db_factory() as db:
        user = await db.get(UserModel, customer_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        customer_id=user.id,
        username=user.username,
        display_name=user.display_name,
        roles=_roles_list(user.roles),
    )


def _issue_tokens(customer_id: str, settings, *, role: str = "customer") -> TokenResponse:
    access = create_access_token(
        customer_id, settings.jwt_secret, algorithm=settings.jwt_algorithm,
        extra_claims={"role": role},
    )
    refresh_token = create_refresh_token(
        customer_id, settings.jwt_secret, algorithm=settings.jwt_algorithm
    )
    return TokenResponse(
        access_token=access, refresh_token=refresh_token, customer_id=customer_id
    )

from pydantic import BaseModel, Field
from datetime import datetime


class ChatRequest(BaseModel):
    """Request model for sending a message.

    customer_id is intentionally absent: identity comes from the Bearer token
    (Depends(current_customer)), never from the request body.
    """
    content: str = Field(..., min_length=1, max_length=2000)


class ApprovalRequest(BaseModel):
    """Request model for submitting approval. Identity is token-derived.

    单意图 / 整批一刀切：只需 ``approved``。
    多意图批量栅栏逐条决定：额外传 ``decisions``（写调用 id → 是否批准），
    approval_required 事件的 payload 已带每条写的 id 供前端逐条回传；未传 decisions
    时以 ``approved`` 应用于全部（向后兼容）。
    """
    approved: bool
    reason: str = ""
    decisions: dict[str, bool] | None = None


class SessionCreateRequest(BaseModel):
    """Request model for creating a session. Identity is token-derived."""
    content: str = Field(default="init", max_length=2000)


class ChatMessage(BaseModel):
    """Response model for a chat message."""
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    agent: str | None = None


class ChatHistoryResponse(BaseModel):
    """Response model for chat history."""
    session_id: str
    messages: list[ChatMessage]


class SessionSummary(BaseModel):
    """One session in a customer's session list.

    ``preview`` is the first user message (trimmed) so the frontend can label
    each conversation without loading its full history.
    """
    session_id: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    message_count: int = 0
    preview: str | None = None


class SessionListResponse(BaseModel):
    """Response model for a customer's session list, newest first."""
    sessions: list[SessionSummary]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.now)


class RegisterRequest(BaseModel):
    """Request model for registering a new user."""
    username: str = Field(..., min_length=3, max_length=128)
    password: str = Field(..., min_length=6, max_length=128)
    display_name: str | None = Field(default=None, max_length=128)


class LoginRequest(BaseModel):
    """Request model for username/password login."""
    username: str = Field(..., min_length=1, max_length=128)
    password: str = Field(..., min_length=1, max_length=128)


class RefreshRequest(BaseModel):
    """Request model for exchanging a refresh token for a new access token."""
    refresh_token: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """Tokens issued on login/refresh. ``customer_id`` == the user's id."""
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    customer_id: str


class UserResponse(BaseModel):
    """Public-safe user profile (never includes password_hash)."""
    customer_id: str
    username: str
    display_name: str | None = None
    roles: list[str] = Field(default_factory=list)

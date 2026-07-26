from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "agent-core"
    debug: bool = False

    # LLM
    litellm_base_url: str = "http://localhost:4000"
    litellm_api_key: str = "sk-litellm-key"

    # Models
    model_fast: str = "claude-haiku-4-5"
    model_main: str = "claude-sonnet-4-6"
    model_complex: str = "claude-opus-4-6"

    # 写操作审批闸（Layer 2）：敏感写工具（create_ticket/apply_refund）执行前需人工确认。
    # 关闭时行为与改造前完全一致（工具直连执行），故对现有流程零影响、可灰度。
    approval_enabled: bool = True

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_core"
    # 审批闸的 LangGraph checkpointer 用 psycopg（≠ SQLAlchemy 的 asyncpg）。留空则由
    # database_url 自动推导（剥掉 +asyncpg 方言标记）。需要独立 checkpoint 库时显式配置。
    checkpoint_db_url: str = ""

    # MCP Servers
    knowledge_mcp_url: str = "http://localhost:8001/mcp"
    order_mcp_url: str = "http://localhost:8002/mcp"
    ticket_mcp_url: str = "http://localhost:8003/mcp"
    crm_mcp_url: str = "http://localhost:8004/mcp"
    skill_mcp_url: str = "http://localhost:8005/mcp"  # 编排型 server：封装多步业务动作

    # Security
    jwt_secret: str = "dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    # CORS allowed origins (comma-separated in env, e.g.
    # CORS_ALLOW_ORIGINS="https://admin.example.com,https://app.example.com").
    # Defaults to local dev origins; wildcard "*" is intentionally NOT the
    # default since it is invalid together with allow_credentials=True.
    cors_allow_origins: str = "http://localhost:5173,http://localhost:3000"

    # Observability
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = "pk-test"
    langfuse_secret_key: str = "sk-test"

    @property
    def resolved_checkpoint_db_url(self) -> str:
        """psycopg 可用的 checkpoint 连接串。

        显式配了 checkpoint_db_url 就用它；否则从 database_url 推导——psycopg 不认
        SQLAlchemy 的 `+asyncpg`/`+psycopg` 方言标记，须剥成裸 `postgresql://`。
        """
        if self.checkpoint_db_url:
            return self.checkpoint_db_url
        url = self.database_url
        if "+" in url.split("://", 1)[0]:
            scheme, rest = url.split("://", 1)
            url = f"{scheme.split('+', 1)[0]}://{rest}"
        return url

    model_config = {"env_file": ".env", "extra": "ignore"}


class StabilityConfig(BaseSettings):
    """Stability engineering configuration."""

    enabled: bool = True

    # Loop protection
    loop_protection_enabled: bool = True
    max_tool_calls_per_turn: int = 10
    max_routing_loops: int = 3
    max_agent_hops: int = 5
    max_skill_steps: int = 8
    max_reflection_retries: int = 2

    # Timeout
    timeout_enabled: bool = True
    session_timeout: float = 60.0
    agent_timeout: float = 30.0
    llm_timeout: float = 15.0
    tool_timeout: float = 10.0
    reflection_timeout: float = 10.0

    # Retry
    retry_enabled: bool = True
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    tool_max_retries: int = 2
    tool_retry_delay: float = 1.0
    output_parse_retries: int = 2

    # Output guard
    output_guard_enabled: bool = True
    output_parse_max_retries: int = 2
    json_repair_enabled: bool = True

    # Fallback
    fallback_enabled: bool = True
    fallback_to_template: bool = True
    auto_handoff_on_failure: bool = True
    max_failures_before_handoff: int = 2

    # Idempotency
    idempotency_enabled: bool = True
    idempotency_ttl_seconds: int = 300

    # Resource limits
    resource_limit_enabled: bool = False
    session_limits_enabled: bool = True
    max_tokens_per_session: int = 50_000
    max_llm_calls_per_session: int = 20
    max_session_duration: float = 600.0

    # User rate limiting
    user_rate_limit_enabled: bool = True
    max_requests_per_user_per_minute: int = 10
    vip_rate_multiplier: float = 3.0

    # Cost circuit breaker
    cost_circuit_breaker_enabled: bool = False
    max_cost_per_hour: float = 50.0
    max_cost_per_day: float = 500.0

    model_config = {"env_prefix": "STABILITY_", "extra": "ignore"}


class ReflectionConfig(BaseSettings):
    """Reflection mechanism configuration."""

    enabled: bool = True

    agent_policies: dict[str, str] = Field(default_factory=lambda: {
        "supervisor": "off",
        "faq": "off",
        "order": "self_check",
        "complaint": "judge",
        "tech_support": "self_check",
        "human_handoff": "off",
    })

    skill_policies: dict[str, str] = Field(default_factory=lambda: {
        "refund": "judge",
        "complaint_handling": "judge",
        "order_query": "off",
    })

    judge_model: str = "claude-opus-4-6"
    max_retries: int = 2
    quality_threshold: float = 7.0
    error_memory_size: int = 20

    model_config = {"env_prefix": "REFLECTION_", "extra": "ignore"}


class MemoryTTLConfig(BaseSettings):
    """Memory TTL configuration."""

    working_memory_ttl_seconds: int = 3600
    checkpoint_archive_days: int = 30
    fact_decay_rate: float = 0.005
    fact_reverify_threshold: float = 0.3
    profile_stale_days: int = 90
    episodic_decay_rate: float = 0.01
    max_episodes_per_user: int = 100

    model_config = {"env_prefix": "MEMORY_", "extra": "ignore"}

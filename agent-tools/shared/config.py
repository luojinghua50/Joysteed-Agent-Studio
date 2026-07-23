from pydantic_settings import BaseSettings


class ToolsConfig(BaseSettings):
    """Shared configuration for MCP tool servers."""

    order_api_base: str = "http://order-system-api:9000"
    ticket_api_base: str = "http://ticket-system-api:9100"
    crm_api_base: str = "http://crm-system-api:9200"
    knowledge_milvus_host: str = "localhost"
    knowledge_milvus_port: int = 19530

    model_config = {"env_file": ".env", "extra": "ignore"}

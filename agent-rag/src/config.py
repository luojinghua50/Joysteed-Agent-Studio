from pydantic_settings import BaseSettings


class RAGSettings(BaseSettings):
    """RAG service configuration."""

    app_name: str = "agent-rag"
    debug: bool = False

    # Embedding
    # provider: "pseudo" (确定性伪向量,测试/无依赖) | "fastembed" (本地ONNX) | "openai"
    embedding_provider: str = "pseudo"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    embedding_base_url: str = "http://litellm:4000/v1"  # 仅 openai provider 用
    embedding_api_key: str = "sk-litellm-key"

    # Chunking defaults
    default_chunk_size: int = 512
    default_chunk_overlap: int = 50

    # Retrieval
    retrieval_backend: str = "memory"  # "memory" | "milvus"
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_k: int = 5
    similarity_threshold: float = 0.5

    # Rerank (跨库 RRF 粗排之后的 cross-encoder 精排)
    # provider: "disabled" (no-op，默认) | "fastembed" (本地 ONNX cross-encoder)
    rerank_provider: str = "disabled"
    rerank_model: str = "BAAI/bge-reranker-base"  # 多语含中文，与 embedding 同源
    rerank_candidate_n: int = 20  # 送入精排的候选池大小（RRF top-N）
    fastembed_cache_path: str | None = None

    # RRF fusion weight
    vector_weight: float = 0.6
    bm25_weight: float = 0.4

    # Storage - PostgreSQL
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_rag"

    # Storage - Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "rag_chunks"

    # Storage - MinIO
    minio_enabled: bool = False  # 默认关闭，无 MinIO 也能跑
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "rag-documents"
    minio_secure: bool = False

    # Versioning - retention policy
    keep_last_n_versions: int = 3
    keep_days: int = 30

    model_config = {"env_file": ".env", "extra": "ignore"}

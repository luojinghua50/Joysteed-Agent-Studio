from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ChunkingStrategy(str, Enum):
    AUTO = "auto"
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    HEADING = "heading"
    TABLE = "table"
    QA_PAIR = "qa_pair"


class KnowledgeBase(BaseModel):
    id: str
    name: str
    description: str = ""
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.AUTO
    chunk_size: int = 512
    chunk_overlap: int = 50
    document_count: int = 0
    kb_form: str = "standard"            # faq | standard | temporal | multimodal
    retrieval_mode: str = "hybrid"       # vector | fulltext | hybrid
    priority_weight: float = 0.7
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(BaseModel):
    id: str
    kb_id: str
    filename: str
    file_type: str
    file_size: int = 0
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class Chunk(BaseModel):
    id: str
    doc_id: str
    kb_id: str
    text: str
    index: int = 0
    metadata: dict = Field(default_factory=dict)
    context_header: str = ""
    keywords: list[str] = Field(default_factory=list)
    token_count: int = 0


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict = Field(default_factory=dict)
    source: str = ""
    # 溯源/路由用：召回来自哪个库、哪个版本（为版本期答案溯源铺路）
    kb_id: str = ""
    version_no: int | None = None


class MetaFilter(BaseModel):
    """库内元数据过滤条件（AND 组合）。"""
    field: str
    op: str                              # eq | in | gt | gte | lt | lte
    value: str | float | int | list


class SearchRequest(BaseModel):
    """单库检索（保留，供调试/召回测试）。"""
    query: str = Field(..., min_length=1, max_length=500)
    kb_id: str
    top_k: int = 5
    mode: str | None = None              # None=用库级 retrieval_mode；可覆盖
    filters: list[MetaFilter] = Field(default_factory=list)


class RouteSearchRequest(BaseModel):
    """聚合检索（Agent 主用）：跨多库路由 + 融合。"""
    query: str = Field(..., min_length=1, max_length=500)
    scope: list[str] | None = None       # 限定参与的 kb_form/kb_id；None=全部公共库
    top_k: int = 5
    filters: list[MetaFilter] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int


class RouteSearchResponse(BaseModel):
    """聚合检索响应：带路由溯源（命中哪些库、是否走了 faq 短路）。"""
    query: str
    results: list[SearchResult]
    total: int
    shortcut: bool = False               # 是否命中 faq 高置信短路
    reranked: bool = False               # 是否经过 cross-encoder 精排（短路时为 False）
    routed_kbs: list[str] = Field(default_factory=list)  # 实际参与/命中的库

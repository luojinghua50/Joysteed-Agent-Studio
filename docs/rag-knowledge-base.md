# RAG 知识库系统技术方案

## 一、概述

RAG 知识库作为**独立服务**部署，为智能客服多 Agent 系统提供知识检索能力。包含管理后台（文档上传/管理）和检索服务（供 Agent 通过 MCP 调用）。

### 项目定位

| 项目 | 职责 |
|------|------|
| **agent-core** | Agent 编排，通过 MCP 调用检索 |
| **agent-tools** | MCP Server，包含 knowledge-mcp（调用 RAG 服务） |
| **agent-rag**（本项目） | RAG 知识库独立服务：管理后台 + 文档处理 + 检索引擎 |

### 系统交互

```
管理员 → RAG 管理后台（上传/管理文档）→ 文档处理流水线 → 向量库

Agent → knowledge-mcp → RAG 检索 API → 混合检索 + 重排序 → 返回结果
```

---

## 二、系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG 管理后台 (Web UI)                             │
│          文档上传 / 知识库管理 / 分块预览 / 检索测试                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ REST API
┌──────────────────────────────▼──────────────────────────────────────┐
│                     RAG 服务 (FastAPI)                               │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    管理接口层                                    │ │
│  │  文档 CRUD / 知识库管理 / 任务状态查询 / 检索测试                 │ │
│  └────────────────────────────┬───────────────────────────────────┘ │
│                               │                                     │
│  ┌────────────────────────────▼───────────────────────────────────┐ │
│  │                 文档处理流水线 (Pipeline)                        │ │
│  │                                                                │ │
│  │  上传 → 解析 → 清洗 → 分块 → Embedding → 入库                  │ │
│  │                                                                │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────┐ ┌───────────┐ │ │
│  │  │ 文件   │→│ 文档   │→│ 文本   │→│ 分块     │→│ 向量化    │ │ │
│  │  │ 解析器 │ │ 清洗器 │ │ 分块器 │ │ 增强     │ │ + 入库    │ │ │
│  │  └────────┘ └────────┘ └────────┘ └──────────┘ └───────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    检索引擎                                     │ │
│  │                                                                │ │
│  │  Query → 改写 → 混合检索 → 重排序 → 后处理 → 返回              │ │
│  │                                                                │ │
│  │  ┌────────┐ ┌──────────────────┐ ┌────────┐ ┌──────────────┐ │ │
│  │  │ Query  │→│ 向量检索(语义)   │→│ Rerank │→│ 结果后处理   │ │ │
│  │  │ 改写   │ │ 关键词检索(BM25) │ │ 重排序 │ │ 去重/截断    │ │ │
│  │  └────────┘ └──────────────────┘ └────────┘ └──────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│   Milvus (向量库, 2.4+)                                   │
│   稠密向量(语义) + 稀疏向量(BM25) 单库混合检索            │
│   按 kb_id 作 partition key 隔离                          │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│              PostgreSQL                                   │
│   租户/文档元数据 / 知识库配置 / 分块记录 / 任务状态      │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────┐
│   MinIO / S3 (对象存储) │
│   原始文件存储           │
└─────────────────────────┘
```

> **架构变更说明（生产级评审）**：原方案用 Elasticsearch 单独承载 BM25，需额外运维一套有状态集群，且与 Milvus 的版本/租户过滤要双写对齐，一致性成本高。**Milvus 2.4+ 原生支持稠密向量 + 稀疏向量(BM25) 的混合检索**，可在单库内完成 hybrid，故移除 ES。仅当已有 ES 投入或需要 ES 的复杂聚合/高亮能力时才保留它。

---

## 三、技术栈

| 组件 | 选型 | 用途 |
|------|------|------|
| **服务框架** | FastAPI + Uvicorn | RAG 服务 API |
| **管理后台前端** | React / Vue（或 Streamlit 快速原型） | 文档管理 UI |
| **文档解析** | Unstructured | 多格式文件解析（PDF/Word/PPT/HTML/Markdown） |
| **文本分块** | LangChain Text Splitters + 自定义策略 | 多种分块算法 |
| **Embedding** | text-embedding-3-small（经 litellm 路由） | 向量化 |
| **向量库** | Milvus 2.4+ | 稠密(语义) + 稀疏(BM25) 混合检索，单库搞定 |
| **重排序** | Rerank API（优先）/ 本地 BGE-Reranker（需评估资源） | 精排，见 §7.6 部署形态 |
| **对象存储** | MinIO（自建）/ S3（云） | 原始文件存储 |
| **关系数据库** | PostgreSQL | 租户/元数据、配置、任务状态 |
| **任务队列** | Celery + Redis | 异步文档处理 |
| **鉴权** | 服务间 API Key / 内网 token + 后台 RBAC | 见 §13 安全与多租户 |
| **可观测** | OpenTelemetry + Langfuse + Prometheus | 复用平台现有组件，见 §14 |
| **包管理** | uv | 依赖管理 |

---

## 四、管理后台功能

### 4.1 功能模块

| 模块 | 功能 |
|------|------|
| **知识库管理** | 创建/编辑/删除知识库，配置分块策略和检索参数 |
| **文档管理** | 上传/删除/状态查看，支持批量操作 |
| **分块预览** | 上传后可预览分块结果，调整策略后重新分块 |
| **检索测试** | 输入 query 测试检索效果，查看命中分块和得分 |
| **统计看板** | 文档数量、分块数量、检索 QPS、命中率 |

### 4.2 支持文件格式

| 格式 | 解析方式 | 说明 |
|------|----------|------|
| **PDF** | Unstructured + PyPDF2 | 支持扫描件 OCR |
| **Word** (.docx) | Unstructured + python-docx | 保留标题层级 |
| **Excel** (.xlsx) | openpyxl → 表格转文本 | 按 sheet 分块 |
| **PPT** (.pptx) | python-pptx | 按 slide 分块 |
| **Markdown** (.md) | 按标题层级解析 | 天然分块边界 |
| **HTML** | BeautifulSoup 清洗 | 去标签保留结构 |
| **TXT** | 直接读取 | 按段落分块 |
| **CSV** | pandas | 按行/按组分块 |
| **图片** (PNG/JPG) | OCR (Tesseract / PaddleOCR) | 提取文字后分块 |

### 4.3 管理 API

```python
# 知识库 CRUD
POST   /api/knowledge-bases                    # 创建知识库
GET    /api/knowledge-bases                    # 列表
GET    /api/knowledge-bases/{kb_id}            # 详情
PUT    /api/knowledge-bases/{kb_id}            # 更新配置
DELETE /api/knowledge-bases/{kb_id}            # 删除（含所有文档）

# 文档管理
POST   /api/knowledge-bases/{kb_id}/documents  # 上传文档（支持多文件）
GET    /api/knowledge-bases/{kb_id}/documents  # 文档列表
GET    /api/documents/{doc_id}                 # 文档详情（含分块信息）
DELETE /api/documents/{doc_id}                 # 删除文档
POST   /api/documents/{doc_id}/reprocess      # 重新处理（更换分块策略）

# 分块预览
POST   /api/preview/chunks                     # 预览分块结果（不入库）

# 检索测试
POST   /api/search/test                        # 测试检索
GET    /api/search/stats                       # 检索统计

# 任务状态
GET    /api/tasks/{task_id}                    # 查询文档处理进度
```

---

## 五、文档处理流水线

### 5.1 处理流程

```
文件上传
  │
  ▼
┌──────────────┐
│ 1. 文件存储   │  原始文件存入 MinIO/S3
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 2. 格式解析   │  Unstructured 解析为统一文本 + 元数据
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 3. 文本清洗   │  去噪、规范化、提取结构信息
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 4. 文本分块   │  按策略切分为 chunks
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 5. 分块增强   │  添加上下文标题、生成摘要、提取关键词
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 6. 向量化     │  Embedding 模型生成向量
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 7. 入库       │  写入 Milvus(稠密+稀疏) + PG 元数据
└──────────────┘
```

### 5.2 异步任务处理

大文件处理耗时较长，使用 Celery 异步任务：

```python
# src/pipeline/tasks.py
@celery_app.task(bind=True, max_retries=3)
def process_document_task(self, doc_id: str, kb_id: str, config: dict):
    """异步文档处理任务：parse → chunk → embed_and_store，逐阶段回写状态"""
    ...  # 实际实现见源码
```

任务经 Redis broker 派发给 Celery worker，按解析 → 清洗分块 → 向量化入库的顺序推进，每完成一步就把文档状态写回 PG（parsing / chunking / embedding / completed），供管理后台轮询进度。任一步抛异常则把状态置为 failed 并记录错误信息，随后按固定退避（countdown）重试，最多三次——把重操作从上传响应里解耦出去，避免大文件阻塞接口。

---

## 六、文本分块策略

### 6.1 分块策略对比

| 策略 | 原理 | 适合场景 | 优势 | 劣势 |
|------|------|----------|------|------|
| **固定大小** | 按 token/字符数切分 | 通用兜底 | 简单可控 | 可能切断语义 |
| **递归字符** | 按分隔符层级递归切（\n\n → \n → 句号） | 通用文档 | 尽量保持段落完整 | 对无结构文本效果一般 |
| **语义分块** | 按句子 embedding 相似度切分 | 长文档、话题切换频繁 | 语义完整 | 计算成本高 |
| **标题层级** | 按 Markdown/文档标题结构切 | 结构化文档 | 保留层级上下文 | 依赖文档有良好结构 |
| **表格专用** | 按行/列/sheet 切 | Excel/CSV | 保持数据完整 | 仅适用表格 |
| **FAQ 对** | 按问答对切分 | FAQ 文档 | 一问一答完整 | 仅适用 QA 格式 |

### 6.2 推荐：分层分块策略（按文档类型自动选择）

分块配置与「文档类型 → 默认策略」的映射是这里的核心契约：

```python
# src/pipeline/splitter.py

class ChunkingConfig(BaseModel):
    strategy: str = "auto"           # auto / fixed / recursive / semantic / heading
    chunk_size: int                  # 目标 chunk token 数
    chunk_overlap: int               # 重叠 token 数
    separator_priority: list[str]    # 递归切分的分隔符优先级

# 文档类型到默认策略的映射：pdf/txt → recursive，docx/md/html → heading，
# xlsx/csv → table，faq → qa_pair
STRATEGY_MAPPING: dict[str, str] = {...}

class SmartSplitter:
    """智能分块器：根据文档类型和内容自动选择策略"""

    def split(self, text: str, metadata: dict, config: ChunkingConfig) -> list[Chunk]:
        ...  # strategy 为 auto 时按 metadata 推断类型，再分派到对应 _split_* 方法
```

分派逻辑的关键在于「按文档结构选策略」：结构化文档（Markdown、docx、HTML）走标题层级切分，让每个 section 成为一个 chunk 并保留层级上下文，超长 section 再退回递归切分兜底；表格类按行/列/sheet 切以保持数据完整；FAQ 按问答对切。语义分块是成本最高的一档——它先把文本切成句子、逐句求 embedding，再沿相邻句子的余弦相似度扫描，相似度跌破阈值或累计 token 超过目标大小时落一刀。因为要对每句做向量化，只在话题切换频繁的长文档上才值得开启。相似度阈值、chunk 大小这类参数留作可调旋钮，随语料反复标定。

### 6.3 分块增强

分块后不只是存原始文本，还为每个 chunk 增强上下文信息，做四件事：

- **层级标题上下文**：把 chunk 在文档中的标题路径拼进来（如「产品手册 > 第三章 退换货政策 > 3.2 退款流程」），让孤立的片段带上出处语义。
- **前后文摘要**：截取相邻 chunk 的开头若干字符，缓解按固定边界切分带来的语义断裂。
- **关键词提取**：抽出关键词补充给 BM25，增强稀疏检索的命中面。
- **假设问题生成**：借 HyDE 思路，让 LLM 预测「这段内容能回答什么问题」，把这些问题一并编入索引文本，拉近用户提问与知识片段的语义距离。

增强环节涉及 LLM 调用（关键词、假设问题），是流水线里较重的一步，实际实现（`src/pipeline/enrichment.py` 的 `ChunkEnricher.enrich`）见源码。

### 6.4 用于 Embedding 的文本组装

真正送去向量化的文本不等于原始 chunk 文本，而是把层级标题、正文、若干假设问题拼成一段增强文本（`build_embedding_text`，实现见源码）。这样嵌入向量既编码了片段本身，也编码了它的出处与可能回答的问题，检索时更容易被相关 query 命中。存储上原始文本与 embedding 文本分列保存，前者用于回给用户展示，后者只服务于召回。

---

## 七、检索引擎

### 7.1 混合检索架构

```
用户 Query
  │
  ▼
┌──────────────────┐
│ 1. Query 预处理   │  改写、扩展、意图识别
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ 向量   │ │ BM25   │
│ 检索   │ │ 检索   │
│(稠密)  │ │(稀疏)  │
└───┬────┘ └───┬────┘
    │          │
   (同一 Milvus collection, 可 hybrid_search 一次完成)
    └────┬─────┘
         ▼
┌──────────────────┐
│ 2. 融合 (RRF)    │  Reciprocal Rank Fusion
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 3. Rerank 重排序  │  Cross-Encoder 精排
└────────┬─────────┘
         ▼
┌──────────────────┐
│ 4. 后处理         │  去重、截断、元数据补充
└────────┬─────────┘
         ▼
     返回 top_k
```

### 7.2 Query 预处理

```python
# src/retrieval/query_processor.py

class QueryProcessor:
    """查询预处理：改写、扩展、多路查询"""

    async def process(self, query: str, config: RetrievalConfig) -> list[str]:
        """返回多个检索 query（原始 + 改写 + HyDE + 子问题）"""
        ...  # 实际实现见源码
```

预处理的产物是「一个 query 变多个 query」——原始 query 始终保留，再按配置叠加三条增强路径，每条都是可独立开关的旋钮：

- **Query 改写**：让 LLM 把口语化的客户问题重写成检索友好的表达，保留核心语义、去掉冗余语气词。
- **HyDE**：先让 LLM 对问题生成一个「假设回答」，再拿这个回答去检索。其直觉是答案与知识片段在向量空间里比问题更接近，尤其利于短问长答的场景。成本较高，默认关闭。
- **子问题分解**：识别到复杂问题时拆成若干子问题分别检索，覆盖多跳诉求。

这些 query 会各自去召回，最后在融合阶段合并。因为多路可能命中同一 chunk，召回结果需要去重。

### 7.3 向量检索（语义）

> **collection 划分策略（生产级评审）**：原方案「每 kb 一个 collection」在 kb 数量增长时会触达 Milvus 的 collection 数量软上限，且每个 collection 都要单独 load 进内存。**改为单 collection + `kb_id` 作 partition key**，按 partition 隔离、按 `version_id` 标量过滤，兼顾隔离性与可扩展性。

```python
# src/retrieval/vector_search.py

class VectorSearcher:
    def __init__(self, milvus_client, embedding_model, collection="rag_chunks"):
        ...  # 单 collection

    async def search(
        self, queries: list[str], kb_id: str, visible_version_ids: list[str], top_k: int = 20
    ) -> list[SearchResult]:
        """多路向量检索：按 kb_id partition + 当前可见版本过滤"""
        ...  # 实际实现见源码
```

稠密检索先把每条 query 嵌成向量，再打到 Milvus 的 `dense` 字段做 ANN 搜索。两个过滤条件构成隔离与正确性的边界：`partition_names=[f"kb_{kb_id}"]` 把搜索限制在目标知识库的分区内，`expr` 用 `version_id in visible_version_ids` 只召回当前生效版本的 chunk（版本切换/回滚见持久化文档）。多条 query 的结果按 `chunk_id` 去重后合并，避免同一片段被多路重复计入。

### 7.4 关键词检索（BM25，Milvus 稀疏向量）

> **生产级评审：移除 Elasticsearch**。Milvus 2.4+ 支持 `SPARSE_FLOAT_VECTOR` + 内建 BM25 函数，关键词检索与向量检索在同一库内完成，省去 ES 集群及其与 Milvus 的双写一致性问题。BM25 与稠密检索可由 Milvus `hybrid_search` 一次调用并融合。

```python
# src/retrieval/keyword_search.py —— 走 Milvus 稀疏向量，而非独立 ES

class KeywordSearcher:
    async def search(
        self, queries: list[str], kb_id: str, visible_version_ids: list[str], top_k: int = 20
    ) -> list[SearchResult]:
        """BM25 稀疏向量检索（Milvus 内建 BM25 function 对 text 字段建稀疏索引）"""
        ...  # 实际实现见源码
```

关键词检索与稠密检索走的是同一套接口，区别只在打到 `sparse` 字段：query 文本不需要在应用层分词，直接交给 Milvus 内建的 BM25 function 自动转成稀疏向量再检索。分区隔离、版本过滤、去重的处理都与稠密检索一致——两路复用同一 collection、同一过滤契约，正是移除 ES 后「单库混合检索」带来的简化。

> 进阶：可直接用 Milvus `hybrid_search` + `WeightedRanker`/`RRFRanker` 一次性完成稠密+稀疏召回与融合，替代下方应用层 RRF（应用层 RRF 仍适用于需要叠加多路 query 改写结果的场景）。

### 7.5 结果融合（RRF）

```python
# src/retrieval/fusion.py

def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """RRF 融合多路检索结果：score = Σ weight / (k + rank)"""
    ...  # 实际实现见源码
```

RRF 只看每个 chunk 在各路结果里的**排名**、不看原始分数，因此天然回避了稠密相似度与 BM25 分数量纲不可比的问题。每个 chunk 的融合分是它在各路中 `weight / (k + rank)` 的累加，常数 `k` 压平头部名次的差距、让长尾结果也有机会浮上来，可调权重则用于给向量路与关键词路配比。按融合分排序即得最终候选序。

### 7.6 Rerank 重排序

> **生产级评审：明确部署形态**。本地 `CrossEncoder`（如 `bge-reranker-v2-m3`）模型约数百 MB，CPU 推理延迟高（每对几十 ms，候选 20 条即数百 ms），并显著拉高容器内存与冷启动时间。三种形态按场景选：
>
> | 形态 | 适用 | 代价 |
> |------|------|------|
> | **Rerank API**（Cohere / Jina / 自建 TEI 服务，优先） | 主力，解耦推理资源 | 网络往返 + 外部依赖 |
> | **本地 CrossEncoder** | 内网无外发、有 GPU/充裕 CPU | 容器变重、需资源预留 |
> | **关闭 rerank** | 召回质量已达标、延迟敏感 | 精排缺失 |
>
> 无论哪种，rerank 都应有**超时 + 降级**：超时则回退到 RRF 融合后的顺序，不阻塞主链路。

```python
# src/retrieval/reranker.py

class Reranker:
    """重排序：默认走 Rerank API，可切换本地 Cross-Encoder。"""

    def __init__(self, config: RetrievalConfig):
        self.mode = config.rerank_mode          # "api" / "local" / "off"
        self.timeout = config.rerank_timeout_s  # 超时降级

    async def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        ...  # 实际实现见源码
```

重排序用 Cross-Encoder 对 query 和每个候选做联合打分，精度高于召回阶段的双塔向量，但代价也高，因此设计上有两道保护。其一是**超时降级**：打分包在 `asyncio.wait_for` 里，一旦超时或后端异常就直接返回融合后的原顺序，绝不阻塞主链路（降级次数计入指标，见 §14）。其二是**相关性阈值**：低于阈值的候选即便排进 top_k 也被剔除，宁可少给也不给不相关的片段。后端形态（API / 本地 / 关闭）由配置切换，阈值等参数按语料标定。

### 7.7 完整检索流程

```python
# src/retrieval/engine.py

class RetrievalEngine:
    """统一检索引擎：编排 query 预处理 → 多路召回 → 融合 → rerank → 后处理"""

    def __init__(self, query_processor, vector_searcher, keyword_searcher, reranker, config):
        ...

    async def search(self, query: str, kb_id: str, top_k: int = 5) -> list[SearchResult]:
        ...  # 实际实现见源码
```

引擎把前面各组件串成一条链：先做 query 预处理拿到多路 query，再用 `asyncio.gather` **并行**发起向量召回与关键词召回（两路互不依赖，并行是延迟关键），RRF 按配置权重融合，rerank 精排到 final top_k（可关闭则直接截断），最后做后处理——为每条结果补上「文档标题 > 章节」的来源串，供 Agent 引用出处。每个环节的开关与数量上限都由 `RetrievalConfig` 驱动，便于按知识库分别调优。

### 7.8 检索配置

检索行为整体由一份配置驱动，各字段是可按知识库分别标定的旋钮，键名即契约：

```python
# src/config.py

class RetrievalConfig(BaseSettings):
    # 检索模式：vector / keyword / hybrid
    search_mode: str

    # Query 预处理开关（HyDE、子问题分解成本较高，默认关）
    query_rewrite_enabled: bool
    hyde_enabled: bool
    decompose_enabled: bool

    # 召回：每路召回数量 + 向量/关键词融合权重
    recall_top_k: int
    vector_weight: float
    keyword_weight: float

    # 重排序：开关 / 模型 / 送入数量 / 相关性阈值
    rerank_enabled: bool
    rerank_model: str
    rerank_top_k: int
    rerank_threshold: float

    # 最终返回条数
    final_top_k: int
```

这些数值（召回条数、融合权重配比、rerank 阈值等）是「配方」级细节，随语料和线上反馈反复调整，此处不固化具体值，只约定有哪些可调项。

---

## 八、数据模型

```python
# src/models.py

class KnowledgeBase(BaseModel):
    id: str
    name: str
    description: str
    chunking_config: ChunkingConfig
    retrieval_config: RetrievalConfig
    embedding_model: str = "text-embedding-3-small"  # 经 litellm 路由
    doc_count: int = 0
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime

class Document(BaseModel):
    id: str
    kb_id: str
    filename: str
    file_type: str               # pdf / docx / md / ...
    file_size: int               # bytes
    file_path: str               # MinIO/S3 路径
    status: str                  # uploading / parsing / chunking / embedding / completed / failed
    error_message: str | None
    chunk_count: int = 0
    created_at: datetime
    processed_at: datetime | None

class Chunk(BaseModel):
    id: str
    doc_id: str
    kb_id: str
    text: str                    # 原始文本
    embedding_text: str          # 用于 embedding 的增强文本
    context_header: str          # 层级标题上下文
    keywords: list[str]          # 提取的关键词
    hypothetical_questions: list[str]  # 假设问题
    token_count: int
    chunk_index: int             # 在文档中的位置
    metadata: dict               # 来源页码、标题等
```

---

## 九、项目目录结构

```
agent-rag/                              # 独立项目
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── alembic/                            # 数据库迁移
│   └── versions/
│
├── src/
│   ├── __init__.py
│   ├── main.py                         # FastAPI 入口
│   ├── config.py                       # 配置管理
│   ├── models.py                       # 数据模型
│   │
│   ├── api/                            # API 层
│   │   ├── __init__.py
│   │   ├── kb_routes.py                # 知识库 CRUD
│   │   ├── doc_routes.py               # 文档管理
│   │   ├── search_routes.py            # 检索接口
│   │   ├── preview_routes.py           # 分块预览
│   │   └── schemas.py                  # 请求/响应模型
│   │
│   ├── pipeline/                       # 文档处理流水线
│   │   ├── __init__.py
│   │   ├── tasks.py                    # Celery 异步任务
│   │   ├── parser.py                   # 多格式文件解析
│   │   ├── cleaner.py                  # 文本清洗
│   │   ├── splitter.py                 # 分块策略
│   │   ├── enrichment.py              # 分块增强
│   │   └── embedder.py                # 向量化
│   │
│   ├── retrieval/                      # 检索引擎
│   │   ├── __init__.py
│   │   ├── engine.py                   # RetrievalEngine 主入口
│   │   ├── query_processor.py          # Query 预处理/改写
│   │   ├── vector_search.py            # 稠密向量检索 (Milvus)
│   │   ├── keyword_search.py           # 稀疏/BM25 检索 (Milvus)
│   │   ├── fusion.py                   # RRF 结果融合
│   │   └── reranker.py                # Rerank 重排序
│   │
│   ├── storage/                        # 存储层
│   │   ├── __init__.py
│   │   ├── milvus_client.py            # Milvus 操作(稠密+稀疏)
│   │   ├── pg_client.py                # PostgreSQL 操作
│   │   └── object_store.py            # MinIO/S3 文件存储
│   │
│   ├── versioning/                     # 文件版本管理(见 rag-persistence-versioning.md)
│   │   ├── switcher.py                 # 原子切换 / 回滚
│   │   └── retention.py                # GC 保留策略 / 僵尸回收
│   │
│   ├── security/                       # 鉴权与多租户(见 §13)
│   │   ├── auth.py                     # API Key / token 校验
│   │   └── tenant.py                   # 租户上下文与隔离
│   │
│   └── admin/                          # 管理后台
│       ├── __init__.py
│       └── stats.py                    # 统计接口
│
├── frontend/                           # 管理后台前端（可选独立部署）
│   ├── package.json
│   └── src/
│
├── tests/
│   ├── conftest.py
│   ├── test_pipeline/
│   ├── test_retrieval/
│   └── test_api/
│
└── scripts/
    ├── init_milvus.py                  # 初始化 Milvus collection + partition + BM25 function
    └── benchmark_retrieval.py          # 检索效果评测
```

---

## 十、与 agent-tools 对接

agent-rag 提供 HTTP API，knowledge-mcp 侧把它封装成 MCP Tool 暴露给 Agent。做法是给每个业务知识库定义一个 `@mcp.tool()` 函数（如 `search_faq`、`search_docs`），函数内用 httpx 携带固定的 `kb_id` 调用 RAG 的 `/api/search`，把 `query`、`top_k` 透传下去，取回 `results`。这样 Agent 侧看到的是语义清晰、按知识库分好的工具，而 kb_id、服务地址这些细节被收敛在 MCP 封装里（`agent-tools/knowledge_server/server.py`，实现见源码）。

---

## 十一、部署架构

```
                          ┌──────────────┐
                          │  API 网关     │  鉴权 / 限流 / 路由
                          │ (内网 token)  │
                          └──────┬───────┘
┌────────────────────────────────┼───────────────────────────┐
│                  agent-rag 服务  │                            │
│  ┌──────────────┐  ┌────────────▼─┐  ┌──────────────┐       │
│  │ FastAPI      │  │ Celery       │  │ 管理后台      │       │
│  │ (检索 API)   │  │ Worker       │  │ (前端 + RBAC) │       │
│  │ :8010        │  │ (文档处理)   │  │ :3001        │       │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘       │
│         └── OTel trace/metric → otel-collector → Tempo/Prom  │
└─────────┼──────────────────┼───────────────────────────────┘
          │                  │
    ┌─────┼──────────────────┼─────┐
    │     ▼                  ▼     │
    │  ┌──────┐  ┌───────┐  ┌───┐ │
    │  │Milvus│  │ MinIO │  │PG │ │   Milvus: 稠密+稀疏(BM25) 单库
    │  │+etcd │  └───────┘  └───┘ │   (ES 已移除)
    │  └──────┘  ┌───────┐        │
    │            │ Redis │        │   缓存可见版本 / Celery broker
    │            └───────┘        │
    └──────────────────────────────┘
```

---

## 十二、核心依赖

```toml
# agent-rag/pyproject.toml
[project]
name = "agent-rag"
requires-python = ">=3.11"

dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.32",
    "celery[redis]>=5.4",
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0",
    "asyncpg>=0.30",
    "sqlalchemy>=2.0",
    "alembic>=1.14",
    # 文档解析
    "unstructured[all-docs]>=0.16",
    "python-docx>=1.1",
    "openpyxl>=3.1",
    "python-pptx>=1.0",
    "beautifulsoup4>=4.12",
    # 向量化（经 litellm 路由，用 OpenAI 兼容客户端）
    "openai>=1.40",
    # 存储（Milvus 单库承载稠密+稀疏，已移除 elasticsearch）
    "pymilvus>=2.4",
    "minio>=7.2",
    "redis>=5.0",
    # 分块
    "langchain-text-splitters>=0.3",
    "tiktoken>=0.7",
    # 可观测（复用平台 otel/langfuse/prometheus）
    "opentelemetry-instrumentation-fastapi>=0.48b0",
    "prometheus-client>=0.21",
    # 工具
    "httpx>=0.27",
    "structlog>=24.0",
]

[project.optional-dependencies]
ocr = ["paddleocr>=2.8", "paddlepaddle>=2.6"]
# 本地 rerank 才需要；默认走 Rerank API，不装这些（见 §7.6）
local-rerank = ["sentence-transformers>=3.0"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24"]
```

---

## 十三、安全与多租户（生产级必备）

> **现状缺口**：当前 [agent-rag/src/api/__init__.py](../agent-rag/src/api/__init__.py) 仅有 `CORSMiddleware(allow_origins=["*"])`，**零鉴权**——任何能访问 8010 端口的客户端都能上传、删除、检索任意知识库。`kb_id` 是裸 8 位随机串，遍历即可跨库读取。生产上线前这是必须堵上的 P0 缺口。

### 13.1 鉴权分层

| 调用方 | 鉴权方式 |
|--------|----------|
| **knowledge-mcp → RAG 检索 API**（服务间） | 内网 service token / API Key（`Authorization: Bearer`），或网关层 mTLS |
| **管理后台 → RAG 管理 API**（人） | 用户登录态（OIDC/JWT）+ RBAC 角色校验 |
| **CORS** | 收敛 `allow_origins` 到后台域名白名单，不再 `*` |

```python
# src/security/auth.py
from fastapi import Depends, HTTPException, Header

async def require_service_token(authorization: str = Header(...)):
    """服务间调用鉴权：校验内网 token。"""
    token = authorization.removeprefix("Bearer ").strip()
    if not verify_service_token(token):          # 常量时间比较，token 存配置/密钥管理
        raise HTTPException(401, "invalid service token")

async def require_role(role: str):
    """管理接口 RBAC：上传/删除/回滚等高危操作要求对应角色。"""
    async def _dep(user=Depends(get_current_user)):
        if role not in user.roles:
            raise HTTPException(403, f"requires role: {role}")
        return user
    return _dep
```

### 13.2 多租户隔离

`kb_id` 必须绑定 `tenant_id`，所有管理与检索操作强制按租户过滤，杜绝越权：

```python
# src/security/tenant.py —— 每个请求解析租户上下文，下沉到存储层强制过滤
async def get_tenant_ctx(user=Depends(get_current_user)) -> TenantContext:
    return TenantContext(tenant_id=user.tenant_id, roles=user.roles)

# 检索/管理时，PG 查询与 Milvus partition 都带 tenant_id：
#   - PG:    WHERE tenant_id = :tid AND kb_id = :kb_id
#   - Milvus: partition_names=[f"t_{tenant_id}__kb_{kb_id}"]  或 expr 增加 tenant_id 过滤
```

> 隔离落到三层：**PG 行级**（每张表带 `tenant_id` 列 + 复合索引，见持久化文档 §3.1 修订）、**Milvus partition / 标量过滤**、**MinIO key 前缀**（`{tenant_id}/{kb_id}/...`）。三层任一不过滤都构成越权风险。

### 13.3 其它安全项

- **上传校验**：文件类型白名单、大小上限、内容嗅探（防伪装扩展名）、解析超时（防解析炸弹）。
- **SSRF/注入**：HTML/URL 类文档解析时禁止外发请求；检索 `expr` 用参数化拼接，不直接字符串插值 `kb_id`。
- **密钥管理**：embedding/rerank 的 API Key、服务 token 走密钥管理（不落 `.env` 明文进镜像）。

---

## 十四、可观测性（复用平台现有组件）

> **现状缺口**：compose 已部署 otel-collector / Tempo / Prometheus / Grafana / Langfuse，但 **agent-rag 一个都没接**，检索链路是盲区。其余服务（agent-core 等）已接入，RAG 应对齐。

### 14.1 三类信号

| 信号 | 内容 | 去向 |
|------|------|------|
| **Trace** | query→改写→embed→稠密召回→稀疏召回→RRF→rerank→后处理 各段 span 耗时；文档处理 parse→chunk→embed→入库 各阶段 | OTel → Tempo |
| **Metrics** | 检索 P50/P95/P99 延迟、QPS、rerank 降级率、embedding API 错误率/延迟、Milvus 召回耗时、任务成功/失败/重试计数、队列长度 | Prometheus → Grafana |
| **LLM 调用** | query 改写 / HyDE 的 prompt、token、成本 | Langfuse |

```python
# src/main.py —— 接入 OTel 自动埋点 + 自定义指标
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import Histogram, Counter

FastAPIInstrumentor.instrument_app(app)   # 自动 trace HTTP 入口

SEARCH_LATENCY = Histogram("rag_search_latency_seconds", "检索延迟", ["kb_id", "stage"])
RERANK_DEGRADE = Counter("rag_rerank_degrade_total", "rerank 超时降级次数")
EMBED_ERRORS   = Counter("rag_embedding_errors_total", "embedding 调用失败", ["reason"])
```

### 14.2 关键告警线（示例）

- 检索 P99 > 1s 持续 5min；embedding 错误率 > 5%；任务失败率 > 10%；Milvus 不可达；队列积压 > 阈值。

---

## 十五、限流与背压

> 上传 + embedding 是重操作，无限流时一次批量上传可打满 embedding 配额、拖垮 Milvus。

| 维度 | 策略 |
|------|------|
| **接口限流** | 网关或应用层按租户/Key 令牌桶限流；上传接口单独更严 |
| **embedding 并发** | 全局信号量限制并发 embedding 请求数，配合重试退避 |
| **任务队列背压** | Celery 队列长度上限，超限拒绝新任务并返回 429，而非无限堆积 |
| **检索保护** | 单查询 `top_k` 上限、query 长度上限（已有 `max_length=500`）、慢查询熔断 |
| **大文件** | 异步处理 + 进度查询（见持久化文档版本状态机），不阻塞上传响应 |

---

## 十六、生产级评审结论摘要

本文档相对初版的主要修订（均为生产级考量）：

1. **移除 Elasticsearch**，BM25 改用 Milvus 2.4+ 内建稀疏向量，少运维一套有状态集群、消除双写一致性问题（§二、§7.4）。
2. **Milvus collection 划分**从「每 kb 一个」改为「单 collection + `kb_id` partition key」，规避 collection 数量上限（§7.3）。
3. **Rerank 明确部署形态**（API 优先 / 本地可选 / 可关），并加超时降级（§7.6）。
4. **新增安全与多租户**（§13）、**可观测**（§14）、**限流背压**（§15）三大横切章节——这些是初版完全缺失、但生产必备的部分。
5. embedding 统一经 litellm 路由；依赖表相应调整（§三、§十二）。

> 文件版本管理的生产级加固（僵尸版本回收、批量更新事务、审计、租户维度表结构）见配套文档 [rag-persistence-versioning.md](./rag-persistence-versioning.md) 的「生产级加固」章节。

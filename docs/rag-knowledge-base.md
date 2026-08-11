# RAG 知识库系统

## 一、概述

`agent-rag` 是独立部署的知识库服务（port 8010），提供文档管理、向量检索和版本控制。管理员通过 `agent-admin` 上传文档，Agent 通过 `knowledge-mcp` 调用检索 API。

```
管理员 → agent-admin → agent-rag（文档处理 + 版本管理）→ Milvus / MinIO / PostgreSQL

Agent → knowledge-mcp（MCP 工具） → POST /api/route-search → 跨库路由 + 融合 → 返回 chunks
```

---

## 二、知识库形态（kb_form）

每个知识库在创建时选择一种形态作为模板，系统自动预填分块策略、检索模式和权重配置。形态只影响默认值，所有参数均可显式覆盖。

| 形态 | 定位 | 分块策略 | 检索模式 | 优先级权重 | 短路阈值 |
|------|------|----------|----------|-----------|---------|
| `faq` | 标准问答对 | `qa_pair` | hybrid | 1.0（最高） | 0.70 |
| `standard` | 政策/产品/流程长文档 | `heading` | hybrid | 0.7 | — |
| `temporal` | 活动/公告（有生命周期） | `auto` | hybrid + 时间过滤 | 0.5 | — |
| `multimodal` | 图文素材转写 | `auto` | vector | 0.3 | — |

---

## 三、文档处理流水线

文档上传后同步完成处理，无异步任务队列：

```
上传 → ObjectStore(MinIO) → extract_text → SmartSplitter → ChunkModel(PG) → retriever.index_chunks → 激活版本
```

**SmartSplitter 分块策略：**

| 策略 | 适用 | 说明 |
|------|------|------|
| `auto` | 按文件扩展名推断 | pdf/txt→recursive；docx/md/html→heading；xlsx/csv→table |
| `recursive` | 通用文本 | `\n\n → \n → 。→ . → 空格` 多级切分 |
| `heading` | 结构化文档 | 按 Markdown `#/##/###` 标题切分，保留层级上下文 |
| `qa_pair` | FAQ 文档 | 按 `Q:` / `问:` 切问答对，每对一个 chunk |
| `table` | Excel/CSV | 首行为 header，每数据行一个 chunk（header 前缀） |
| `parent_child` | 长文档/手册 | 先切父块（按标题），再切子块；子 chunk 回填父块原文供展示 |
| `fixed` | 兜底 | 固定字符数切分，带 overlap |

每个 chunk 携带 `context_header`（所在标题路径）和 `keywords`（关键词列表），供 BM25 和 rerank 使用。

---

## 四、检索引擎

### 4.1 单库检索（三种模式）

```
POST /api/search
```

| mode | 行为 |
|------|------|
| `vector` | 仅稠密向量（语义相似度） |
| `fulltext` | 仅稀疏 BM25（关键词匹配） |
| `hybrid` | dense + sparse 两路，库内 RRF 融合（默认） |

hybrid 模式的库内 RRF 公式：
```
score = vector_weight × 1/(k + rank_dense) + keyword_weight × 1/(k + rank_bm25)
        k=60，默认 vector_weight=0.6，keyword_weight=0.4
```

MemoryRetriever（默认后端）用 n-gram Jaccard 模拟语义，用词项包含率模拟 BM25；MilvusRetriever 使用真实稠密向量 + Milvus 2.4+ 内建 BM25 稀疏向量。

### 4.2 跨库路由检索（Agent 主用）

```
POST /api/route-search
```

`SearchRouter` 编排多库级联检索：

```
1. FAQ 短路探针：向量模式探最高分，≥ shortcut_threshold → 直接返回 FAQ 结果（reranked=False）
2. 多路并行：asyncio.gather 对所有参与库并发检索（temporal 库自动注入时间过滤）
3. 跨库加权 RRF：以 priority_weight 为各库权重融合结果
4. Rerank 精排：fastembed cross-encoder 对 top-N 候选重排（不可用时降级回 RRF 顺序）
```

响应带路由溯源信息：`shortcut`（是否命中短路）、`reranked`（是否经精排）、`routed_kbs`（参与的库 ID 列表）。

### 4.3 Rerank

默认 `provider=disabled`（no-op）。配置 `RERANK_PROVIDER=fastembed` 后使用本地 ONNX cross-encoder（如 `BAAI/bge-reranker-base`），无 torch 依赖。精排失败时自动降级回 RRF 顺序，不阻断检索。cross-encoder 原始 logit 经 sigmoid 归一化为 0~1 相关概率写回 `score`。

### 4.4 元数据过滤

每个知识库可自定义可过滤字段（`string | number | time`），字段值在上传文档时随 JSON 打标：

```bash
POST /api/knowledge-bases/{kb_id}/metadata-fields  # 定义字段
POST /api/knowledge-bases/{kb_id}/documents        # 上传时传 metadata='{"category":"耳机"}'
```

检索时 `filters` 编译为库内 AND 条件：

| field_type | 支持算子 |
|-----------|---------|
| string | `eq` / `in` |
| number | `eq` / `gt` / `gte` / `lt` / `lte` |
| time | `gte` / `lte` |

`temporal` 库自动注入 `effective_ts <= now && expire_ts >= now`，过期内容天然不召回。

---

## 五、版本管理

每次上传文件创建一个新版本（`document_versions`），通过 `documents.current_version_id` 指针原子切换，旧版本进入 `archived` 状态。检索时只对 `visible_version_ids`（即各文档的当前版本）有效，影子版本对 Agent 不可见。

```
上传 → shadow 版本（status=processing）→ 构建完成（status=ready）→ activate（原子切换指针）→ 旧版本 archived
```

**关键操作：**

```
GET  /api/documents/{doc_id}/versions       # 查看所有历史版本
POST /api/documents/{doc_id}/rollback       # 按 version_no 回滚（立即生效）
```

版本保留策略（`keep_last_n_versions`）：`VersionManager.apply_retention` 清理超出保留数量的旧版本，删除对应的 chunk 索引、PG 记录和 MinIO 原文。幂等上传（相同内容 hash 跳过构建）。

---

## 六、管理 API

```
POST   /api/knowledge-bases                          # 建库（模板预填+参数覆盖）
GET    /api/knowledge-bases                          # 列表
GET    /api/knowledge-bases/{kb_id}                  # 详情
PATCH  /api/knowledge-bases/{kb_id}                  # 更新配置（检索参数热更新；分块/模型参数需重建）
DELETE /api/knowledge-bases/{kb_id}                  # 删库（含向量索引+原文）

POST   /api/knowledge-bases/{kb_id}/metadata-fields  # 定义可过滤字段
GET    /api/knowledge-bases/{kb_id}/metadata-fields  # 查看字段定义
DELETE /api/knowledge-bases/{kb_id}/metadata-fields/{field_id}

POST   /api/knowledge-bases/{kb_id}/documents        # 上传（支持 metadata JSON）
GET    /api/knowledge-bases/{kb_id}/documents        # 文档列表（含当前版本状态）
GET    /api/documents/{doc_id}                       # 文档详情
GET    /api/documents/{doc_id}/versions              # 历史版本列表
POST   /api/documents/{doc_id}/rollback              # 按版本号回滚
DELETE /api/documents/{doc_id}                       # 删文档（清理所有版本+索引）

POST   /api/search                                   # 单库检索（调试/测试用）
POST   /api/route-search                             # 跨库路由检索（Agent 主用）
```

---

## 七、数据模型

```
knowledge_bases       知识库配置（form/mode/weights/threshold 等）
documents             逻辑文档（一个文件名对应一条记录，版本指针指向当前激活版本）
document_versions     物理版本（每次上传一条，含 file_hash/status/chunk_count）
chunks                分块记录（含 context_header/keywords/meta）
kb_metadata_fields    库内可过滤字段定义
audit_log             操作审计（activate/rollback/update_config/gc）
```

---

## 八、Embedding

三种 provider，优先级：fastembed > openai > pseudo（兜底）：

| provider | 说明 |
|----------|------|
| `pseudo` | 确定性 hash 向量，无依赖，默认（测试/开发） |
| `fastembed` | 本地 ONNX 模型（如 `BAAI/bge-small-zh-v1.5`），无 torch，离线 |
| `openai` | OpenAI 兼容接口，经 `embedding_base_url` 路由 |

单库只能使用一种 embedding 模型；切换模型需重建索引（重新上传文档）。

---

## 九、项目结构

```
agent-rag/src/
├── api/__init__.py          FastAPI 路由（KB CRUD / 文档版本 / 检索）
├── db/__init__.py           SQLAlchemy ORM 模型 + init_db
├── models/__init__.py       Pydantic 模型（KnowledgeBase / SearchRequest / SearchResult 等）
├── pipeline/__init__.py     SmartSplitter（7 种分块策略）
├── extraction.py            文件文本抽取（按扩展名分派）
├── embedding/__init__.py    Embedder（pseudo / fastembed / openai）
├── retrieval/__init__.py    BaseRetriever / MemoryRetriever / RRF / 元数据过滤
├── retrieval/milvus_backend.py  MilvusRetriever（dense + sparse BM25）
├── rerank/__init__.py       Reranker（fastembed cross-encoder，降级安全）
├── routing/__init__.py      SearchRouter（faq 短路 + 多路并行 + 跨库 RRF）
├── versioning/__init__.py   VersionManager（shadow build / activate / rollback / GC）
├── storage/                 ObjectStore（MinIO/S3）
└── config.py                RAGSettings
```

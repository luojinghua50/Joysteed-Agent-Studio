# RAG 知识库：持久化与文件版本管理设计

> 本文聚焦两件 [rag-knowledge-base.md](./rag-knowledge-base.md) 未覆盖的事：
> **(1) 把当前 mock 实现落到真实存储**；**(2) 知识库文件的版本更新方案**。
> 向量库选型沿用现有文档与 `config.py` 中已声明的 **Milvus**。

---

## 一、现状对账（先认清起点）

对三个问题的结论，基于对当前代码的实际核对：

| 问题 | 结论 | 代码依据 |
|------|------|----------|
| **1. 有管理后台支持上传？** | 后端**有 REST 接口**，但**没有后台 UI**。只能靠 `init_data.py` 脚本或 curl 灌数据。 | [agent-rag/src/api/__init__.py](../agent-rag/src/api/__init__.py) 有 `POST /api/knowledge-bases/{kb_id}/documents`；[agent-web/src/](../agent-web/src/) 只有 ChatWindow，无知识库页面 |
| **2. 文件存入向量库 / PostgreSQL？** | **都没有**。知识库、文档、分块全在**进程内存字典**里，重启即丢；检索是**关键词字符串匹配**，无 embedding。 | `app.state.knowledge_bases: dict` / `app.state.documents: dict`；`RetrievalEngine._chunks: dict`（注释自述 `mock: in-memory store`） |
| **3. 文件版本更新方案？** | 当前**无从谈起**——删了重传，旧分块直接从内存 `pop`。需先补持久化地基，再叠版本化。 | `delete_by_doc` 直接 `self._chunks[kb_id] = [...]` 过滤 |

### 1.1 “声明了但没用上”的配置

[config.py](../agent-rag/src/config.py) 里 `milvus_*`、`database_url`、`minio_*` 都是**占位声明**，运行时没有任何代码读取它们去连接。docker-compose 里：

- `postgres` 是普通 `postgres:16-alpine`，**未装 pgvector**，`agent_rag` 库**无任何表结构**（无 alembic / SQLAlchemy 模型）。
- **没有 Milvus 服务**（compose 里根本没有该镜像，尽管 config 写了 `milvus_host`）。
- `minio` 服务起了，但 RAG 代码**没有任何 MinIO 客户端调用**，原始文件未落盘。
- `litellm_config.yaml` **只配了 3 个 chat 模型，没有 embedding 模型**——真实向量化前必须先补一条 embedding 路由。

> 因此本设计的工作量分两层：**先补地基（持久化 + 真实向量化），再叠版本管理**。下文按这个顺序展开。

---

## 二、目标存储分层

```
                    ┌──────────────────────────────┐
                    │   管理后台 / Agent 检索请求    │
                    └───────────────┬──────────────┘
                                    │ REST
                    ┌───────────────▼──────────────┐
                    │      agent-rag (FastAPI)      │
                    └───┬─────────┬─────────┬───────┘
            原始文件     │  元数据 │  向量    │
        ┌───────────────▼┐ ┌──────▼──────┐ ┌▼──────────────┐
        │  MinIO          │ │ PostgreSQL  │ │  Milvus        │
        │  原始文件 + 版本 │ │ 元数据/版本 │ │  向量 + 标量过滤 │
        │  对象           │ │ /分块/任务   │ │  (按 version)  │
        └─────────────────┘ └─────────────┘ └────────────────┘
```

| 数据类别 | 存储 | 说明 |
|----------|------|------|
| **原始文件** | MinIO | 每个版本一个独立对象，天然保留历史，可回溯下载 |
| **元数据 / 版本 / 分块记录 / 任务状态** | PostgreSQL | 事务性强，版本切换靠它做原子指针 |
| **向量 + 标量字段** | Milvus | `version_id` 作为标量字段，检索时按 `version_id` 过滤 |

> 三者用 `version_id` 串起来：一次文件更新 = 一个新 `version_id`，三套存储各自写入带该 id 的数据，最后由 PG 的一个指针决定“哪个版本对外可见”。

---

## 三、数据模型（含版本化）

版本化的核心拆分：把当前的 `Document`（既是逻辑文档又是物理内容）拆成**逻辑文档 `documents`** 和**物理版本 `document_versions`** 两层。逻辑文档稳定不变（id 不变、Agent 引用不变），每次上传新文件只新增一个版本行。

### 3.1 PostgreSQL 表结构

```sql
-- 知识库
CREATE TABLE knowledge_bases (
    id              VARCHAR(16) PRIMARY KEY,
    tenant_id       VARCHAR(32) NOT NULL,          -- 多租户隔离（见 rag-knowledge-base.md §13）
    name            VARCHAR(128) NOT NULL,
    description     TEXT DEFAULT '',
    chunking_strategy VARCHAR(32) DEFAULT 'auto',
    chunk_size      INT DEFAULT 512,
    chunk_overlap   INT DEFAULT 50,
    embedding_model VARCHAR(64) DEFAULT 'text-embedding-3-small',
    document_count  INT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

-- 逻辑文档：稳定标识，Agent / 检索引用的对象
CREATE TABLE documents (
    id                  VARCHAR(16) PRIMARY KEY,
    tenant_id           VARCHAR(32) NOT NULL,
    kb_id               VARCHAR(16) NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    filename            VARCHAR(512) NOT NULL,
    current_version_id  VARCHAR(24),          -- 指向当前对外可见版本（原子切换的关键）
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now(),
    UNIQUE (kb_id, filename)                  -- 同一 kb 内 filename 唯一 → 再次上传走"新增版本"
);

-- 物理版本：每次上传一行
CREATE TABLE document_versions (
    id              VARCHAR(24) PRIMARY KEY,    -- 同时作为 Milvus 标量过滤字段
    tenant_id       VARCHAR(32) NOT NULL,
    doc_id          VARCHAR(16) NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    kb_id           VARCHAR(16) NOT NULL,
    version_no      INT NOT NULL,               -- 1, 2, 3... 单调递增
    file_hash       CHAR(64) NOT NULL,          -- sha256(原始字节)，用于幂等去重
    file_type       VARCHAR(16) NOT NULL,
    file_size       BIGINT DEFAULT 0,
    minio_key       VARCHAR(1024) NOT NULL,     -- 原始文件对象路径
    status          VARCHAR(16) DEFAULT 'pending',  -- pending/parsing/chunking/embedding/ready/active/failed/archived/purged
    heartbeat_at    TIMESTAMPTZ,                -- 处理中的版本定期续约，用于僵尸回收（见 §八）
    chunk_count     INT DEFAULT 0,
    error           TEXT,
    created_by      VARCHAR(64) DEFAULT 'system',
    created_at      TIMESTAMPTZ DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    UNIQUE (doc_id, version_no)
);

-- 分块记录：外键到 version，而非 doc
CREATE TABLE chunks (
    id              VARCHAR(32) PRIMARY KEY,    -- 形如 {version_id}-{idx:04d}
    tenant_id       VARCHAR(32) NOT NULL,
    version_id      VARCHAR(24) NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
    doc_id          VARCHAR(16) NOT NULL,
    kb_id           VARCHAR(16) NOT NULL,
    chunk_index     INT NOT NULL,
    text            TEXT NOT NULL,
    chunk_hash      CHAR(64) NOT NULL,          -- sha256(text)，用于增量 re-embed
    context_header  VARCHAR(512) DEFAULT '',
    keywords        TEXT[] DEFAULT '{}',
    token_count     INT DEFAULT 0,
    metadata        JSONB DEFAULT '{}'
);

-- 审计日志：高危操作（上传/激活/回滚/删除/GC）独立留痕，合规可追溯
CREATE TABLE audit_log (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   VARCHAR(32) NOT NULL,
    actor       VARCHAR(64) NOT NULL,           -- 操作者（用户或服务）
    action      VARCHAR(32) NOT NULL,           -- upload/activate/rollback/delete/gc
    target_type VARCHAR(16) NOT NULL,           -- kb/document/version
    target_id   VARCHAR(24) NOT NULL,
    detail      JSONB DEFAULT '{}',             -- 如 {from_version, to_version}
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- 索引：所有高频查询都带 tenant_id 前缀，强制隔离 + 走索引
CREATE INDEX idx_kb_tenant         ON knowledge_bases(tenant_id);
CREATE INDEX idx_documents_tenant  ON documents(tenant_id, kb_id);
CREATE INDEX idx_versions_doc      ON document_versions(tenant_id, doc_id, version_no DESC);
CREATE INDEX idx_versions_status   ON document_versions(status) WHERE status IN ('pending','parsing','chunking','embedding');
CREATE INDEX idx_chunks_version    ON chunks(version_id);
CREATE INDEX idx_chunks_hash       ON chunks(version_id, chunk_hash);
CREATE INDEX idx_audit_tenant      ON audit_log(tenant_id, created_at DESC);
```

> `current_version_id` 故意**不加外键约束**到 `document_versions`，避免「先建文档行还是先建版本行」的循环依赖；用应用层保证一致性。

### 3.2 Milvus 集合设计

每个知识库一个 collection（`kb_{kb_id}`），`version_id` 作为标量字段供过滤：

```python
# scripts/init_milvus.py —— 创建 collection 的字段定义
fields = [
    FieldSchema("chunk_id",   DataType.VARCHAR, is_primary=True, max_length=32),
    FieldSchema("version_id", DataType.VARCHAR, max_length=24),   # 标量过滤字段
    FieldSchema("doc_id",     DataType.VARCHAR, max_length=16),
    FieldSchema("embedding",  DataType.FLOAT_VECTOR, dim=1536),   # 对齐 config.embedding_dim
    FieldSchema("text",       DataType.VARCHAR, max_length=8192),
]
# 关键：在 version_id 上建标量索引，使 expr 过滤高效
collection.create_index("version_id", index_params={"index_type": "INVERTED"})
collection.create_index("embedding",  index_params={
    "index_type": "HNSW", "metric_type": "IP",
    "params": {"M": 16, "efConstruction": 200},
})
```

检索时按当前可见版本过滤：

```python
# 只查该文档当前生效版本的向量，旧版本物理留存但不参与检索
expr = f'version_id in {visible_version_ids}'
collection.search(data=[query_vec], anns_field="embedding",
                  expr=expr, limit=top_k, output_fields=["chunk_id", "doc_id", "text"])
```

> `visible_version_ids` = 该 kb 下所有文档的 `current_version_id` 集合，从 PG 查得（可缓存到 Redis）。这样**旧版本向量留在 Milvus 里但被过滤掉**，回滚时只需改 PG 指针、刷新这个集合，无需重建索引。

### 3.3 MinIO 对象布局

```
rag-documents/                          # bucket
└── {kb_id}/
    └── {doc_id}/
        ├── v1/{filename}               # 版本 1 原始文件
        ├── v2/{filename}               # 版本 2 原始文件
        └── v3/{filename}               # ...
```

- object key = `{kb_id}/{doc_id}/v{version_no}/{filename}`，写入 `document_versions.minio_key`。
- 每版独立对象，原始文件可随时下载比对/重处理。
- 删除文档时按 `{kb_id}/{doc_id}/` 前缀批量删除；删 kb 时按 `{kb_id}/` 前缀删除。

---

## 四、文件版本更新方案（问题 3 的核心）

### 4.1 设计目标

| 目标 | 手段 |
|------|------|
| **检索零抖动** | 新版本处理期间对检索不可见，处理完成才原子切换，避免「重建索引时检索为空/命中旧分块」 |
| **可秒级回滚** | 回滚 = 改一个 `current_version_id` 指针，旧版本数据物理留存 |
| **幂等** | 上传内容与当前版本 `file_hash` 相同则跳过，不产生空版本 |
| **省 embedding 成本** | 基于 `chunk_hash` 做增量：内容未变的分块复用旧向量，只 re-embed 变化部分 |
| **可审计** | 每个版本记录 `created_by` / `created_at` / `version_no`，全程可追溯 |

### 4.2 更新流程（影子构建 + 原子切换）

```
上传新文件到已存在的 doc_id
  │
  ▼
① 计算 file_hash —— 与 documents.current_version 的 hash 相同？
  │  是 → 返回 "unchanged"，结束（幂等）
  ▼ 否
② 原始文件存入 MinIO: {kb_id}/{doc_id}/v{n}/{filename}
  │
③ 插入 document_versions 行: version_no=n, status=pending
  │   ★ 此时 current_version_id 仍指向旧版本，新版本对检索完全不可见（影子）
  ▼
④ 异步管道处理新版本（解析→分块→embedding→写 PG chunks + Milvus 向量）
  │   全程更新 status: parsing→chunking→embedding
  │   失败 → status=failed，current 指针不动，旧版本继续服务
  ▼ 成功
⑤ status=ready
  │
⑥ 【原子切换】单个 PG 事务内:
  │   UPDATE documents SET current_version_id = {new_version_id}
  │   UPDATE document_versions SET status='archived' WHERE id = {old_version_id}
  │   → 刷新 Redis 里该 kb 的 visible_version_ids 缓存
  ▼
⑦ 后台 GC: 按保留策略清理过旧版本的 chunks/向量/MinIO 对象
```

关键点：**第 ③~⑤ 步新版本是「影子」**——数据都写进了 PG 和 Milvus，但因为 `current_version_id` 还没切，检索的 `version_id` 过滤集里不含它，用户完全看不到。只有第 ⑥ 步的指针切换让新版本瞬间生效。这就是「零抖动」的来源。

### 4.3 原子切换代码骨架

```python
# src/versioning/switcher.py
async def activate_version(session, doc_id: str, new_version_id: str):
    """把文档的当前版本原子切换到 new_version_id。"""
    async with session.begin():                      # 单事务保证原子性
        doc = await session.get(Document, doc_id, with_for_update=True)
        old_version_id = doc.current_version_id

        new_ver = await session.get(DocumentVersion, new_version_id)
        if new_ver.status != "ready":
            raise ValueError(f"version {new_version_id} not ready: {new_ver.status}")

        doc.current_version_id = new_version_id
        new_ver.status = "active"
        if old_version_id:
            old = await session.get(DocumentVersion, old_version_id)
            old.status = "archived"
    # 事务提交后刷新可见版本缓存（Milvus 过滤依据）
    await refresh_visible_versions_cache(doc.kb_id)
    return {"doc_id": doc_id, "active_version": new_version_id, "previous": old_version_id}
```

### 4.4 回滚

```python
# 回滚 = 把指针切回任一历史 ready/archived 版本，复用同一个 activate_version
async def rollback(session, doc_id: str, target_version_no: int):
    ver = await get_version_by_no(session, doc_id, target_version_no)
    if ver.status not in ("archived", "ready", "active"):
        raise ValueError("target version data has been GC'd, cannot rollback")
    return await activate_version(session, doc_id, ver.id)
```

> 能否回滚取决于目标版本的 chunks/向量是否还在（未被 GC）。因此**保留策略**直接决定回滚窗口。

### 4.5 增量更新（省 embedding 成本，进阶可选）

embedding 是版本更新里最贵的一步。当新旧版本大部分内容相同（例如一份手册只改了一节）时，按 `chunk_hash` 复用：

```python
async def embed_with_reuse(session, new_version_id, old_version_id, new_chunks):
    old_hashes = await load_chunk_hashes(session, old_version_id)  # {chunk_hash: vector_ref}
    to_embed, reused = [], []
    for ch in new_chunks:
        if ch.chunk_hash in old_hashes:
            reused.append((ch, old_hashes[ch.chunk_hash]))   # 复制旧向量，换 chunk_id/version_id
        else:
            to_embed.append(ch)                              # 仅这些走 embedding API
    vectors = await embedder.embed_batch([c.text for c in to_embed])
    # 写入：reused 复制旧向量 + to_embed 新算向量，都带新 version_id
    ...
```

> 注意：分块边界变化会让 hash 全变（哪怕文字只改一句），所以增量收益依赖**稳定的分块策略**（如 heading 分块比 fixed 更稳）。建议作为 P3 优化，先把全量重算跑通。

### 4.6 版本保留策略（GC）

```python
class RetentionPolicy(BaseSettings):
    keep_last_n_versions: int = 3        # 每个文档至少保留最近 N 个版本
    keep_days: int = 30                  # 或保留 30 天内的版本
    keep_active_always: bool = True      # 当前生效版本永不清理
```

GC 后台任务：对每个文档，保留 `current` + 最近 N 版 + N 天内的版本，其余 `archived` 版本删除其 PG chunks、Milvus 向量、MinIO 对象（`document_versions` 行可保留做审计，仅标记 `purged`）。

---

## 五、API 变更与基础设施补全

### 5.1 API 调整

现有上传接口语义从「新建文档」改为「为文档新增版本」，并补充版本相关接口：

```python
# 文档 / 版本
POST   /api/knowledge-bases/{kb_id}/documents          # 上传 → 若 filename 已存在则新增版本，否则建新文档
GET    /api/documents/{doc_id}/versions                # 版本列表（version_no/status/created_at/created_by）
GET    /api/documents/{doc_id}/versions/{version_no}   # 版本详情
POST   /api/documents/{doc_id}/rollback                # 回滚到指定 version_no
GET    /api/documents/{doc_id}/versions/{v}/download   # 下载某版本原始文件（MinIO 预签名 URL）
GET    /api/versions/{version_id}/status               # 轮询异步处理进度

# 检索接口签名不变，内部改为按 current_version_id 过滤
POST   /api/search
```

> 关键兼容点：`Document.id` 语义不变，**Agent / knowledge-mcp 侧无需改动**——它们引用的是逻辑文档和 kb，版本切换对调用方透明。

### 5.2 版本状态机

```
pending ──► parsing ──► chunking ──► embedding ──► ready ──►(activate)──► active
   │                                                  │                      │
   └──────────────► failed ◄──────────────────────────┘                (新版本激活)
                       ▲                                                      ▼
                       │                                                  archived ──►(GC)──► purged
              (任一步异常，current 指针不动)
```

- 只有 `ready` 的版本能被 `activate`；
- `active` 全局每文档唯一；切换时旧 `active` → `archived`；
- `failed` / 处理中的版本永不进入可见集合，旧版本持续服务。

### 5.3 并发与一致性

| 场景 | 处理 |
|------|------|
| 同一文档并发上传两个新版本 | `version_no` 由 `UNIQUE(doc_id, version_no)` 兜底；用 `SELECT ... FOR UPDATE` 锁文档行取下一个 version_no |
| 切换瞬间正在检索 | 检索读 `current_version_id` 是单行读，切换是单事务写，天然串行；最坏情况读到切换前一刻的旧版本，无错误 |
| 写了 Milvus 但 PG 事务回滚 | 以 PG 为准（source of truth）：Milvus 中孤儿向量因 `version_id` 不在可见集合而不被检索，由 GC 按「PG 无对应 version」清理 |
| 多副本部署的缓存一致性 | `visible_version_ids` 缓存切换后用 Redis pub/sub 或短 TTL（如 10s）失效，容忍秒级最终一致 |

> 写入顺序原则：**先 Milvus / MinIO（可补偿的外部存储），最后提交 PG 事务**。PG 提交成功 = 版本就绪；PG 未提交时外部存储的残留都是不可见孤儿，交给 GC。

### 5.4 基础设施补全（对账第 1.1 节的缺口）

落地前必须补的环境项：

```yaml
# docker-compose.yml 需新增 Milvus（及其依赖 etcd、用现有 minio 即可）
  milvus:
    image: milvusdb/milvus:v2.4-latest
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    depends_on: [etcd, minio]
    ports: ["19530:19530"]
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    # ... 标准 etcd 配置
```

```yaml
# litellm_config.yaml 需新增 embedding 路由（当前只有 3 个 chat 模型）
  - model_name: "text-embedding-3-small"
    litellm_params:
      model: openai/text-embedding-3-small
      api_key: os.environ/ANTHROPIC_API_KEY
      api_base: os.environ/LLM_API_BASE
    model_info:
      mode: embedding
```

```python
# config.py 需补充（embedding 走 litellm 而非直连）
embedding_base_url: str = "http://litellm:4000/v1"
milvus_collection_prefix: str = "kb_"
```

> 若选用的 embedding 模型维度不是 1536，需同步改 `config.embedding_dim` 和 Milvus 字段 `dim`，两者必须一致。

### 5.5 依赖补充

```toml
# agent-rag/pyproject.toml dependencies 追加
"asyncpg>=0.30",          # PG 异步驱动
"sqlalchemy>=2.0",        # ORM
"alembic>=1.14",          # 迁移
"pymilvus>=2.4",          # Milvus 客户端
"minio>=7.2",             # 对象存储
"redis>=5.0",             # 可见版本缓存 / 异步任务
```

---

## 六、分阶段落地路线

按依赖顺序推进，每阶段可独立验收。版本化（P2）必须建立在持久化（P0）之上，不可跳步。

| 阶段 | 范围 | 验收标准 | 风险 |
|------|------|----------|------|
| **P0 持久化地基** | 内存字典 → PG（kb/documents/chunks）+ MinIO 存原始文件；建 alembic 迁移 | 重启服务后知识库与文档不丢；原始文件可下载 | 低，纯增量改造，检索逻辑暂不动 |
| **P1 真实向量化** | litellm 加 embedding 路由 + Milvus 上线 + 替换 mock 检索为向量+BM25混合 | `/api/search` 返回语义相关结果而非字符串匹配 | 中，需起 Milvus/etcd，embedding 维度要对齐 |
| **P2 版本化** | documents/document_versions 拆分 + 影子构建 + 原子切换 + 回滚 | 更新文件时旧版本持续可检索、切换无空窗；可回滚 | 中，状态机和并发是重点测试对象 |
| **P3 优化与后台** | 增量 re-embed + GC 保留策略 + agent-web 知识库管理页面 | embedding 成本下降；非技术人员可上传/查版本/回滚 | 低，可按需取舍 |

> 建议最小可用闭环是 **P0 + P2**：先让数据落地、再让版本可控。P1 的真实向量检索可与 P2 并行，但若资源紧张，P0 之后直接做 P2 也成立（版本切换逻辑不依赖向量是否「真实」，只依赖存储是否持久）。

### 6.1 各阶段的关键改造点（代码层）

- **P0**：新增 `src/storage/{pg,object_store}.py`；把 [api/__init__.py](../agent-rag/src/api/__init__.py) 里所有 `app.state.*` 字典操作替换为 PG repository 调用；`RetrievalEngine` 的内存 `_chunks` 暂保留或改为查 PG。
- **P1**：新增 `src/storage/milvus_client.py` + `src/pipeline/embedder.py`（走 litellm）；`RetrievalEngine.search` 改为「embed query → Milvus 向量召回 + PG/ES 关键词召回 → RRF → rerank」。
- **P2**：新增 `src/versioning/{switcher,retention}.py`；改 `upload_document` 为版本感知；新增版本/回滚路由。
- **P3**：增量 embed、GC 定时任务、前端页面。

---

## 七、设计决策小结

1. **逻辑文档与物理版本分离**是整个版本方案的支点——让 Agent 引用稳定、让更新可叠加、让回滚变成改指针。
2. **影子构建 + 原子切换**保证检索零抖动，这是「在线更新知识库」体验的关键，比「删除旧的→重建」方案优越得多。
3. **PG 作为唯一事实源**，Milvus/MinIO 的残留都靠 `version_id` 过滤 + GC 兜底，避免分布式事务的复杂度。
4. **向量库选 Milvus** 对齐现有文档与 config 声明；代价是要额外维护 etcd 依赖，运维比 pgvector 重，规模增长时这个投入是值得的。
5. **当前最大缺口是「全是 mock」**——所以任何版本化工作都得先补 P0 持久化地基，这一点务必在排期里体现。

---

## 八、生产级加固（评审补充）

前述方案覆盖了单文档的「快乐路径」。以下是生产环境必须额外处理的边角，否则在故障、并发、合规场景下会出问题。

### 8.1 僵尸版本回收（崩溃恢复）

服务在「影子构建中」崩溃，会留下 `pending`/`parsing`/`embedding` 状态卡住的版本，以及已写入 Milvus 但永不会被激活的孤儿向量。处理：

- 处理中的版本定期更新 `heartbeat_at`（见 §3.1 表结构）。
- 启动时 + 定时任务扫描：`status IN (处理中) AND heartbeat_at < now() - 超时阈值` → 标记 `failed`，清理其已写入的 Milvus 向量 / MinIO 临时对象。
- `current_version_id` 永不指向这类版本，所以**回收期间检索不受影响**，旧版本持续服务。

```python
# src/versioning/retention.py
async def reap_zombie_versions(session, timeout_s: int = 1800):
    """回收心跳超时的处理中版本。"""
    stuck = await session.execute(select(DocumentVersion).where(
        DocumentVersion.status.in_(["pending", "parsing", "chunking", "embedding"]),
        DocumentVersion.heartbeat_at < now() - timedelta(seconds=timeout_s),
    ))
    for ver in stuck.scalars():
        await milvus.delete(expr=f'version_id == "{ver.id}"')   # 清孤儿向量
        ver.status, ver.error = "failed", "reaped: heartbeat timeout"
    await session.commit()
```

### 8.2 批量更新的事务边界

一次更新整个 kb（几百个文档）时，不能用「一个大事务切换所有指针」（长事务锁表、失败全回滚代价大）。推荐：

- **逐文档原子切换**，每个文档独立事务，互不影响。
- 整批用一个 `batch_id` 追踪，记录每个文档的成功/失败，失败的不影响已成功的。
- 批量任务幂等可重入：重跑只处理 `batch_id` 下未完成的文档。
- 给调用方返回批次进度（`total / ready / activated / failed`），而非一个布尔。

> 原则：**版本切换的原子性边界是「单个文档」**，批量只是调度层的循环 + 进度聚合，不扩大事务范围。

### 8.3 `current_version_id` 悬空指针防护

§3.1 中 `current_version_id` 不加外键（避免循环依赖），代价是并发/异常下可能悬空。防护三选一或叠加：

- 切换用 `SELECT ... FOR UPDATE` 锁文档行，串行化同一文档的切换。
- GC 删除版本前强制校验 `documents.current_version_id != 该版本`，**当前生效版本永不被删**。
- 定时对账：扫描 `current_version_id` 指向的版本是否存在且为 `active`，异常则告警 + 自愈（回退到最近 ready 版本）。

### 8.4 审计

高危操作（上传新版本、激活、回滚、删除、GC purge）写入 `audit_log` 表（见 §3.1）。回滚尤其要记 `{from_version, to_version, actor}`，满足「谁在何时把知识库回退到了哪个版本」的合规追溯，不依赖易被覆盖的 `created_by` 字段。

### 8.5 多租户在版本流程中的贯穿

- 所有版本/分块/审计表带 `tenant_id`（见 §3.1），切换、回滚、GC、检索全程带租户过滤。
- MinIO key 加租户前缀：`{tenant_id}/{kb_id}/{doc_id}/v{n}/{filename}`。
- Milvus 按 `t_{tenant_id}__kb_{kb_id}` partition 或标量过滤隔离（对齐 [rag-knowledge-base.md](./rag-knowledge-base.md) §13.2）。

### 8.6 一致性兜底总表

| 故障 | 现象 | 兜底 |
|------|------|------|
| 影子构建中崩溃 | 卡住的处理中版本 + 孤儿向量 | §8.1 僵尸回收 |
| 写 Milvus 成功但 PG 事务回滚 | Milvus 孤儿向量 | `version_id` 不在可见集合，不被检索；GC 按「PG 无对应 version」清理 |
| 批量更新中途失败 | 部分文档已切换、部分未切 | §8.2 逐文档原子 + 批次幂等重入 |
| GC 误删当前版本 | 检索为空 | §8.3 删除前校验 current 指针 |
| 多副本缓存不一致 | 短暂读到旧可见集合 | Redis pub/sub 失效 + 短 TTL，秒级最终一致 |

---

## 九、生产级评审结论摘要

本文档相对初版的主要加固：

1. **表结构贯穿 `tenant_id`** + 复合索引，多租户隔离落到行级（§3.1）。
2. **新增 `audit_log` 审计表**，高危操作合规留痕（§3.1、§8.4）。
3. **新增 `heartbeat_at` + 僵尸版本回收**，解决影子构建崩溃后的孤儿数据（§8.1）。
4. **批量更新明确事务边界**为「单文档原子 + 批次幂等」，不扩大事务范围（§8.2）。
5. **悬空指针防护**（FOR UPDATE / GC 前校验 / 定时对账）（§8.3）。
6. 补一致性兜底总表，把各类故障的处理路径列清（§8.6）。

> 检索侧的生产级加固（移除 ES、Milvus partition、鉴权/多租户、观测、限流）见 [rag-knowledge-base.md](./rag-knowledge-base.md) §13–§16。

---

> 本文为设计文档，未改动任何业务代码。确认方案后可按第六节路线分阶段实现。
> 配套阅读：[rag-knowledge-base.md](./rag-knowledge-base.md)（整体 RAG 架构与检索引擎设计）。


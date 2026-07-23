# RAG 检索增强技术方案：多 Collection 拆分 + 混合检索 + 检索路由融合

> 目标：把用户侧公共咨询知识按**知识形态**拆分为多个向量集合，各自差异化切片/加权/时效管控，并在其上做混合检索（向量+全文）与多路召回融合，提升召回准确率、降低幻觉。
> 配套：知识库基础设计见 [rag-knowledge-base.md](./rag-knowledge-base.md)、持久化与版本见 [rag-persistence-versioning.md](./rag-persistence-versioning.md)、鉴权多租户见 [auth-user-design.md](./auth-user-design.md)。

---

## 一、背景与目标

### 1.1 拆分的核心价值（不是业务分类）

把知识拆成多个 Collection，**不是为了"产品/政策/流程"这种业务主题分类**，而是为了四个正交的工程目标：

1. **分库加权**：不同知识形态在召回时优先级不同（FAQ 精准命中 > 标准知识长尾补充 > 时效内容）。
2. **噪声隔离**：FAQ 短问答向量与长文档切片向量不在同一可比尺度，混在一个 collection 做 ANN 会相互污染 top-k。物理隔离才能根治。
3. **时效管控**：活动/公告类知识生命周期短，单独成库可在检索层直接做时间过滤，过期内容彻底屏蔽，且高频更新不扰动主库索引稳定性。
4. **差异化切片策略**：FAQ「只 embed 问题」与长文档「段落切片+标题加权」是两种 schema 和 embedding 语义，必须分库承载。

> 最终收益：提升首问解决率、减少长文档噪声、杜绝过期内容召回。

### 1.2 为什么这个量级适合拆 Collection

Milvus 官方建议海量租户场景用 partition-key 而非 collection-per-tenant（collection 数量有上限、每库一份索引吃内存）。但本系统是**单租户、固定少数几个 collection**，不触发该上限/内存顾虑。Dify 底层也是"按 embedding 配置共享 collection + 逻辑层多 dataset 各自切片、多路召回融合"——本方案做的正是 Dify 逻辑层那件事。

### 1.3 现状盘点（agent-rag）

| 能力 | 现状 | 结论 |
|------|------|------|
| 向量检索 | `MilvusRetriever` dense 检索已实现（单 collection `rag_chunks` + partition-per-kb） | ✅ 可用，但要改为多 collection |
| 全文/BM25 | 代码注释标注「P-next，未实现」 | ❌ 占位 |
| 混合检索 + RRF | 有 `vector_weight/bm25_weight` 配置，无实现 | ❌ 占位 |
| 元数据过滤 | `SearchRequest` 仅 query/kb_id/top_k | ❌ 无 |
| 多路召回/融合 | 无 | ❌ 无（本方案新增核心） |
| Rerank | 有 `rerank_top_k` 配置，无实现 | ❌ 占位（留后续期） |
| embedding provider | pseudo / fastembed（本地）/ openai（云）三种 | ✅ 可用，**全库统一一个模型** |
| 检索后端 | `memory`（默认）/ `milvus` 双后端 | ✅ 两者都要同步改 |

### 1.4 关键设计原则

- **embedding 模型统一**：所有 collection 共用同一个 embedding 模型/维度。拆分是为切片/加权/时效，不是为换模型。per-库换模型留到将来确有需要时再做。
- **库间隔离 = 物理 collection**；**库内过滤 = 元数据字段**（如 temporal 库的时间、标准库的类目）。二者分工明确。
- **volatile 数据不进向量库**：实时价格/库存等由业务 MCP（如 product-mcp）在召回后用 PG 校准，避免双写一致性（见 §八）。

---

## 二、Collection 拆分方案

用户侧公共咨询类知识拆为 **3 个核心向量集合 + 1 个可选辅助集合**，每个对应不同知识形态、切片规则、检索优先级。

| Collection | 定位与内容 | 知识形态 | 切片与 Embedding 策略 | 更新频率 | 检索优先级 |
|---|---|---|---|---|---|
| **faq_public** | 精准问答库：通用 FAQ、产品常见问题、规则类/操作类问答等所有标准化问答对 | 结构化 Q-A 对，问题≤50字、答案≤300字 | **仅对问题做 Embedding**，答案存原文；答案核心关键词补到问题侧增强 | 低（周/月更） | **最高**（命中即优先返回） |
| **std_knowledge_public** | 标准体系知识库：产品详情、政策规则、操作流程、服务说明、权益介绍等长文档 | 段落级切片，带章节层级，200-500字/片 | 切片全文 Embedding；章节标题额外 Embedding 并加权 | 中（周更） | 次高（长尾补充） |
| **temporal_knowledge_public** | 时效类知识库：营销活动、临时公告、系统升级、停运通知、节假日调整等 | 单篇公告/活动规则，带生效/失效时间 | 全文分段 Embedding，**强制绑定时间元数据** | 高（日更/按需） | 中（仅有效期内召回） |
| **multimodal_public**（可选） | 多模态辅助库：图文教程截图、视频脚本、图示转写 | 图片 OCR 转写文本，关联素材地址 | 转写文本 Embedding，标注素材类型 | 中 | 最低（辅助补充） |

### 2.1 各库拆分理由

- **faq 单独成库**：用户提问与 FAQ 问题侧语义匹配度最高，单独检索实现「高频问题精准命中」，避免长文档切片噪声干扰——这是提升首问解决率最有效的手段。
- **标准知识合并不再细拆**：产品/规则/流程都是体系化长文档，切片策略与检索方式一致，合并为一个库即可，无需按业务主题再拆，减少多路召回复杂度。
- **时效内容隔离**：生命周期短，过期必须彻底屏蔽；单独成库可在检索层直接加时间过滤，且更新/归档/下线不影响主库索引稳定性。
- **multimodal 可选、最低优先级**：本期可不落地，预留形态。

### 2.2 「知识形态」如何落到建库

知识形态决定切片策略，本质对应已有的 `ChunkingStrategy`（auto/heading/qa_pair）与新增的检索配置：

| Collection | chunking_strategy | 检索默认 mode | 优先级权重 |
|---|---|---|---|
| faq_public | qa_pair（只 embed 问题） | hybrid（问题向量 + 答案全文兜底） | 1.0（最高） |
| std_knowledge_public | heading（段落+标题加权） | hybrid | 0.7 |
| temporal_knowledge_public | auto + 时间元数据 | hybrid + 时间过滤 | 0.5 |
| multimodal_public | auto | vector | 0.3（最低） |

运营建库时选「知识形态」（问答库/标准知识库/时效库），系统据此绑定上面一行配置，**不暴露 chunking_strategy 等内部术语**（UI 包装见后续 F 期）。

---

## 三、总体架构

### 3.1 检索数据流（多路召回 + 级联融合）

```
用户问题
  │
  ▼
knowledge_server（MCP）  对 Agent 只暴露一个 search_knowledge 工具
  │
  ▼
agent-rag  检索路由层（新增核心）
  │
  ├─【第一级】faq_public：hybrid 检索
  │     └─ 高置信短路：向量+BM25 双路都同意且 score≥高阈值 → 直接返回，结束
  │
  └─【第二级】未短路 → 多路并行召回：
        ├─ std_knowledge_public   hybrid
        ├─ temporal_knowledge_public  hybrid + 时间过滤（now 在 [生效,失效] 内）
        └─（可选）multimodal_public  vector
              │
              ▼  按库优先级加权的 RRF 融合
              ▼  score_threshold 过滤 + top_k 截断
              ▼  （后续期）rerank 重排
  ▼
返回 chunks（带 collection 来源、metadata）
```

### 3.2 短路阈值（防幻觉关键）

FAQ 短路是提升首问解决率的核心，但**阈值必须卡高**：仅当向量与 BM25 双路都命中同一答案、且 score 超过高阈值时才短路返回。否则一律进第二级多路召回，由融合/重排定夺——避免"问题表面相似但实际要查标准库"时自信地返回错答案。

### 3.3 库内过滤 vs 库间隔离

| 隔离需求 | 手段 |
|---|---|
| 不同知识形态互不干扰（FAQ vs 长文档） | **物理 collection 拆分**（§二） |
| 同库内按属性筛选（temporal 的时间、std 的类目/品牌等） | **元数据字段过滤**（§五） |

> 元数据过滤在本方案中**不再承担库间隔离**（那由 collection 承担），专门做**库内过滤**——分工比"单 collection + 元数据隔离"更清晰。

---

## 四、数据模型设计

### 4.1 KnowledgeBase ↔ Collection 映射

每个知识库（KnowledgeBase）对应一个独立 Milvus collection。`KnowledgeBaseModel` 扩展：

```python
# 知识形态/检索配置（新增）
kb_form: Mapped[str]           = mapped_column(String(24), default="standard")
#   faq | standard | temporal | multimodal —— 决定切片策略与检索默认
collection_name: Mapped[str]   = mapped_column(String(64))   # 物理 collection 名，如 kb_<id>
retrieval_mode: Mapped[str]    = mapped_column(String(16), default="hybrid")  # vector|fulltext|hybrid
priority_weight: Mapped[float] = mapped_column(Float, default=0.7)  # 多路融合时的库级权重
vector_weight: Mapped[float]   = mapped_column(Float, default=0.6)  # 库内 hybrid 的向量/关键词配比
keyword_weight: Mapped[float]  = mapped_column(Float, default=0.4)
score_threshold: Mapped[float] = mapped_column(Float, default=0.0)
shortcut_threshold: Mapped[float] = mapped_column(Float, default=0.0)  # 仅 faq 库用，高置信短路阈值
```

> `KnowledgeBaseModel` 已预留 `chunking_strategy/chunk_size/chunk_overlap/embedding_model`，本期补检索/形态相关字段。

### 4.2 新增表：知识库元数据字段定义

```python
class KbMetadataFieldModel(Base):
    __tablename__ = "kb_metadata_fields"

    id: Mapped[str]          = mapped_column(String(16), primary_key=True)
    tenant_id: Mapped[str]   = mapped_column(String(32), default="default", index=True)
    kb_id: Mapped[str]       = mapped_column(String(16),
                                  ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    name: Mapped[str]        = mapped_column(String(64))      # 字段名，如 category / effective_ts
    field_type: Mapped[str]  = mapped_column(String(16))     # string | number | time
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- 同一 kb 内 `name` 唯一（唯一索引 `(kb_id, name)`）。
- temporal 库约定内置两个 time 字段 `effective_ts` / `expire_ts`，检索时强制时间过滤。

### 4.3 现有表扩展

```python
# DocumentModel 增：文档的元数据字段值
doc_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
#   例: {"category":"耳机"} 或 temporal 的 {"effective_ts":..., "expire_ts":...}

# ChunkModel 增：从文档下沉的元数据（写入 Milvus 行用，避免回查 doc）
meta: Mapped[dict] = mapped_column(JSON, default=dict)
```

### 4.4 字段类型 → 过滤算子映射

| field_type | 支持算子 | Milvus 标量类型 | 示例 |
|-----------|---------|----------------|------|
| string | eq / in | VARCHAR | `category == "耳机"` |
| number | eq / gt / gte / lt / lte | DOUBLE | `battery_level >= 10` |
| time | gte / lte | INT64（epoch 秒） | `effective_ts <= now && expire_ts >= now` |

---

## 五、元数据过滤（库内）

**索引侧**：`index_chunks` 把 `chunk.meta` 每个字段作为独立标量字段写入该库的 collection 行；字段在「定义时」建标量索引（保证过滤走索引而非全扫）。

**检索侧**：把 `filters` 编译成 Milvus boolean 表达式，与 `version_id` 可见性表达式 AND 合并：

```python
def _compile_filter(filters: list[MetaFilter]) -> str:
    parts = []
    for f in filters:
        if f.op == "eq":   parts.append(f'{f.field} == {_lit(f.value)}')
        elif f.op == "in": parts.append(f'{f.field} in {_list_lit(f.value)}')
        elif f.op == "gte":parts.append(f'{f.field} >= {f.value}')
        # ... gt/lt/lte
    return " && ".join(parts)
# 最终 expr = version_filter && metadata_filter
```

**temporal 库特例**：检索时自动注入 `effective_ts <= now && expire_ts >= now`，过期内容天然不召回。

**Memory 后端**：同步实现等价的 dict 字段过滤，保证 `RETRIEVAL_BACKEND=memory` 下行为一致、测试可跑。

---

## 六、混合检索 + RRF（库内）

**Milvus schema**：每个 collection 含 `dense`（语义）+ `sparse`（BM25 全文，Milvus 2.4 内建 function）两个向量字段，各自建索引。

**三种 mode**：

| mode | 行为 |
|------|------|
| vector | 仅 dense ANN |
| fulltext | 仅 sparse BM25 |
| hybrid | dense + sparse 双路并行 → 库内 RRF 融合 |

**库内 RRF**（复用 `vector_weight/keyword_weight`）：

```
score(doc) = vector_weight * 1/(k + rank_dense) + keyword_weight * 1/(k + rank_bm25)
            （k=60，业界惯例）
```

**faq 库特例**：问题侧 embedding 进 dense，答案原文进 sparse 做 BM25 兜底——用户问法与库内问题差异大时，全文兜底补召回。

**中文分词（关键风险点）**：Milvus BM25 默认英文 analyzer，中文需配分词。两条候选：
1. collection 配置中文 analyzer（取决于 Milvus 镜像版本支持度）；
2. 写入/查询时应用层预分词（jieba）。
实施前先验证当前 Milvus 版本支持度再定。

---

## 七、检索路由与多路融合（本方案新增核心）

### 7.1 新增组件：检索路由层

在 agent-rag 新增 `SearchRouter`，对外提供一个聚合检索入口，内部编排多 collection 的级联与融合。

```python
async def route_search(query, scope, filters, top_k) -> list[SearchResult]:
    # 第一级：faq 高置信短路
    faq_hits = await retriever.search(faq_kb, query, mode="hybrid", ...)
    if faq_hits and faq_hits[0].score >= faq_kb.shortcut_threshold:
        return faq_hits[:top_k]          # 命中即返回，省去后续

    # 第二级：多路并行召回（faq 结果一并参与融合，不浪费）
    tasks = [retriever.search(kb, query, mode=kb.retrieval_mode, filters=...)
             for kb in active_kbs(scope)]   # std / temporal(时间过滤) / multimodal
    results_per_kb = await gather(tasks)

    # 跨库加权 RRF：库级 priority_weight × 库内 RRF rank
    fused = weighted_rrf(results_per_kb, weights={kb.id: kb.priority_weight})
    return [r for r in fused if r.score >= threshold][:top_k]
```

### 7.2 跨库加权融合

每条结果最终分 = `库级 priority_weight × (1/(k + 跨库归一 rank))`。faq=1.0 > std=0.7 > temporal=0.5 > multimodal=0.3，保证同等相关度下 FAQ/标准知识优先。

### 7.3 检索 API

```python
class MetaFilter(BaseModel):
    field: str; op: str; value: str | float | int | list

class SearchRequest(BaseModel):          # 单库检索（保留，供调试/召回测试）
    query: str; kb_id: str; top_k: int = 5
    mode: str | None = None
    filters: list[MetaFilter] = []

class RouteSearchRequest(BaseModel):     # 聚合检索（Agent 主用）
    query: str
    scope: list[str] | None = None       # 限定参与的 kb_form/kb_id；None=全部公共库
    top_k: int = 5
    filters: list[MetaFilter] = []        # 跨库通用过滤（LLM 推断）
```

新增 `POST /api/route-search` 走路由层；`POST /api/search` 保留单库检索（召回测试用）。

---

## 八、与业务 MCP 的协作（volatile 数据后置）

向量库只负责「语义 + 稳定属性过滤」的召回；实时价格/库存等 volatile 数据由业务 MCP 在召回后用 PG 校准：

```
search_knowledge(query, filters=[category=耳机])  → 候选（含 sku_id 等业务键）
        ↓
product-mcp.enrich_skus(sku_ids, price_max=2000)  → PG 实时价格/库存补全 + 过滤
        ↓
agent 组织答案
```

- product-mcp（PG 为真相源）属后续期，本文档不展开其表设计。
- 本期只保证：召回结果的 `metadata` 能带出业务键（如 `sku_id`），供后续 enrich 衔接。
- 原则：volatile 字段绝不进向量库，避免双写一致性与过期价格。

---

## 九、knowledge_server（MCP）改动

1. **去掉写死 kb_id**：彻底移除 `"faq"/"docs"` 硬编码，改调 agent-rag 的 `/api/route-search` 聚合入口；解析失败要**打日志**而非静默回退 mock（修此前 404→mock 的坑）。
2. **统一工具 `search_knowledge`**：对 Agent 只暴露一个检索工具，多 collection 路由/融合对 Agent 透明。
3. **LLM 推断 filters**：把候选库的元数据字段定义喂给 Agent，由 LLM 从用户问题推断 `filters`（如「2000以内耳机」→ `category=耳机`），随请求传入。

---

## 十、实施步骤（每步独立可验证，连续执行不停顿）

| 步 | 内容 | 验证点 |
|----|------|--------|
| 1 | DB 模型：KnowledgeBase 加形态/检索配置字段、新增 kb_metadata_fields 表、Doc/Chunk 加 JSON；`SearchRequest` 扩展 | 单测：模型迁移、字段读写 |
| 2 | 检索层改造：单 collection → **collection-per-kb**（建库即建 collection，按 kb_form 选 schema/切片）；两后端同步 | 建库生成独立 collection；FAQ 检索回归 |
| 3 | 元数据过滤：索引下沉 + filters 编译 + 标量索引 + temporal 时间过滤 + metadata-fields API | 带 filter 检索正确过滤；过期内容不召回 |
| 4 | 混合检索：每 collection 加 sparse+BM25，库内 RRF，vector/fulltext/hybrid 三 mode | hybrid 召回优于 vector |
| 5 | **检索路由层**：`SearchRouter` + faq 短路 + 多路并行 + 跨库加权 RRF + `/api/route-search` | 级联融合端到端；短路阈值生效 |
| 6 | knowledge_server：统一 search_knowledge + kb_id 动态解析 + LLM 推断 filters | 「耳机」问句自动过滤并多路召回 |
| 7 | 中文分词验证 + 权重/阈值/短路阈值调优 + 召回测试 | SearchTester 验证多库融合质量 |

---

## 十一、风险与约束

1. **现有客服库需删库重建**（已确认可接受）：检索层从 partition 改 collection-per-kb、schema 加 sparse 字段，旧数据结构不兼容；原文在 MinIO，删库后按新形态重新导入。
2. **collection 数量可控**：本期固定 3+1 个公共库，远低于 Milvus collection 上限；embedding 统一一个模型/维度，不触发内存/维度问题。
3. **中文 BM25 分词**：步骤 4/7 最易踩坑，实施前先验证 Milvus 版本支持度。
4. **两个检索后端同步**：Milvus 与 Memory retriever 都要支持多 collection + filters + mode + 路由融合，否则默认 memory 后端行为不一致、测试失真。
5. **短路阈值防幻觉**：faq 短路阈值卡高，宁可多走一次多路召回，不可自信返回错答案（§3.2）。
6. **路由融合是新增最大复杂度**：跨库 RRF 的 rank 归一、权重调参是质量关键，需召回测试驱动调优。
7. **Admin UI 不在本期**：先用现有 SearchTester + API 验证；Dify 式建库向导（按知识形态选库、不暴露内部术语）留 F 期。

---

## 十二、后续期（本期不做，列此对齐全景）

| 期 | 内容 |
|----|------|
| C | Rerank 模型重排（融合结果后再 cross-encoder 重排，进一步提精度） |
| D | 父子分段 / 自定义分隔符 / 文本预处理规则（丰富切片策略） |
| E | per-库独立 embedding 模型（不同库不同维度，本方案已为 collection-per-kb 打好基础） |
| F | Admin Dify 式建库向导（按知识形态选库 + 分段/检索配置可视化 + 召回测试）+ product-mcp 落地 + multimodal_public |

---

## 十三、与版本能力的边界（本期不做，下一期立项）

知识库**多版本/全生命周期**能力（版本快照、答案溯源、回滚、影子验证上线）是**独立的下一期**，方案见 [rag-versioning-lifecycle.md](./rag-versioning-lifecycle.md)。本期与之的边界：

- **本期复用现有版本机制不破坏**：agent-rag 已有文档级版本控制（影子构建 / 原子切换 / 即时回滚 / 保留 GC，见 [rag-persistence-versioning.md](./rag-persistence-versioning.md)）。本期 collection-per-kb 改造**必须保持其兼容**，两处适配点：
  1. `visible_version_ids` 按 collection 隔离查——多路召回时每个 collection 用各自 kb 的可见版本集，不能混。
  2. `delete_by_version` / `delete_kb` 落到正确的 collection（检索层从"操作 partition"改"操作 collection"时一并处理）。
- **本期为下一期埋点**：`SearchResult` 带出 `kb_id + doc_id + version_no`，为下一期「答案溯源」铺路（agent 侧绑定知识ID+版本号到 Langfuse trace）。这是顺手的字段透出，本期就做。
- **本期不做**：语义化版本号、灰度/影子验证流程、版本快照表扩展、溯源 UI——全部留下一期。

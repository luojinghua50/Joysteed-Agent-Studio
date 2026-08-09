# RAG 检索：多库路由 + 混合检索 + 元数据过滤

> 本文描述 agent-rag 的检索层设计，覆盖多库路由、混合检索、元数据过滤和 rerank。  
> 知识库基础配置与版本管理见 [rag-knowledge-base.md](./rag-knowledge-base.md)。

---

## 一、为什么拆多个知识库

把知识拆成多个知识库，核心目的是**工程隔离**，而不是业务主题分类：

1. **噪声隔离**：FAQ 短问答向量和长文档切片向量语义尺度不同，混在一个库做 ANN 会相互污染 top-k。
2. **差异化切片**：FAQ 只 embed 问题侧；标准文档按标题段落切；时效内容需绑定生命周期字段——分库才能各自用最合适的策略。
3. **时效管控**：temporal 库在检索层自动注入有效期过滤，过期内容天然不召回，且高频更新不扰动主库索引。
4. **分库加权**：不同形态在跨库融合时的 `priority_weight` 不同（faq=1.0 > standard=0.7 > temporal=0.5 > multimodal=0.3），同等相关度下精准答案优先。

---

## 二、检索数据流

```
Agent 提问
  │
  ▼
POST /api/route-search（SearchRouter）
  │
  ├─ 1. FAQ 短路探针（vector 模式，仅取 top-1 score）
  │       score ≥ shortcut_threshold → 直接返回 FAQ 结果，结束
  │
  └─ 2. 未短路 → 多路并行（asyncio.gather）
          ├─ faq 库       hybrid 检索
          ├─ standard 库  hybrid 检索
          ├─ temporal 库  hybrid + 自动时间过滤
          └─ multimodal 库  vector 检索
                 │
                 ▼ 跨库加权 RRF（priority_weight 为各路权重）
                 ▼ Reranker（fastembed cross-encoder，失败降级）
                 ▼ 返回 top_k（带 shortcut/reranked/routed_kbs 溯源）
```

---

## 三、库内混合检索（hybrid mode）

每个库的检索模式由 `retrieval_mode` 控制（`vector` / `fulltext` / `hybrid`，默认 hybrid）。

**hybrid 模式**：dense（语义）和 sparse（BM25）两路并行检索，库内 RRF 融合：

```
score(chunk) = vector_weight × 1/(k + rank_dense)
             + keyword_weight × 1/(k + rank_bm25)
               k=60（压平头部差距，让长尾有机会）
```

默认 `vector_weight=0.6, keyword_weight=0.4`，两值之和必须为 1，创建/更新时强制校验。

**FAQ 库特例**：`qa_pair` 分块只 embed 问题侧，答案原文进 sparse 做 BM25 兜底——用户问法与库内问题偏差大时，全文兜底补召回。

**后端差异**：

| 后端 | dense | sparse |
|------|-------|--------|
| MemoryRetriever（默认） | 字符 n-gram Jaccard 代理 | 词项包含率 + 关键词命中代理 |
| MilvusRetriever | 真实稠密向量（fastembed/openai） | Milvus 2.4+ 内建 BM25 稀疏向量 |

切换方式：`RETRIEVAL_BACKEND=milvus`，Milvus 不可达时自动回落 Memory。

---

## 四、FAQ 高置信短路

短路是提升首问解决率的核心机制，同时也是最容易出幻觉的地方，因此设计上刻意保守：

- 短路触发条件：**vector 模式**（不用 hybrid 的 RRF 分，RRF 分无绝对语义含义）的 top-1 score ≥ `shortcut_threshold`（faq 库默认 0.70）。
- 阈值需要按语料标定（精确问法 ≈ 0.72，同义改写 ≈ 0.66，无关 ≈ 0.37），建议初始值宁高勿低。
- 未触发短路时，faq 库的检索结果仍参与第二级多路融合，不浪费。
- `shortcut_threshold=0` 则该库不参与短路（standard/temporal/multimodal 默认如此）。

---

## 五、元数据过滤（库内）

### 5.1 定义和使用

每个知识库可定义可过滤字段，上传文档时随 `metadata` JSON 打标：

```bash
# 定义字段
POST /api/knowledge-bases/{kb_id}/metadata-fields
body: name=category&field_type=string

# 上传文档时打标
POST /api/knowledge-bases/{kb_id}/documents
form: file=xxx.pdf, metadata={"category":"耳机","effective_ts":1720000000,"expire_ts":1730000000}
```

字段值随 chunk 下沉到 `ChunkModel.meta`，写入向量索引行，检索时直接在库内过滤，不回查 PG。

### 5.2 过滤算子

| field_type | 支持算子 | 示例 |
|-----------|---------|------|
| string | `eq` / `in` | `{"field":"category","op":"eq","value":"耳机"}` |
| number | `eq` / `gt` / `gte` / `lt` / `lte` | `{"field":"price","op":"lte","value":2000}` |
| time | `gte` / `lte` | `{"field":"expire_ts","op":"gte","value":1720000000}` |

多个 filter AND 组合。未定义的字段在上传时直接拒绝（HTTP 400），防止脏字段污染索引。

### 5.3 temporal 库时间过滤

`kb_form=temporal` 的库在检索时自动注入：

```python
effective_ts <= now  AND  expire_ts >= now
```

条件仅在库定义了对应 `time` 类型字段、且调用方未显式过滤该字段时注入，避免重复。过期内容不需要手动删除，有效期到后自然不再召回。

---

## 六、跨库加权 RRF

多路并行检索完成后，`reciprocal_rank_fusion` 以各库的 `priority_weight` 为权重融合：

```python
score(chunk) = Σ  priority_weight_i / (k + rank_i)
# 同一 chunk 在多路命中则分数累加（去重取最高排名）
```

rank 从 1 起算，k=60。只看排名不看原始分，天然规避稠密相似度和 BM25 分值量纲不可比的问题。

---

## 七、Rerank 精排

在跨库 RRF 粗排之后，对 top-N 候选（默认 20）用 cross-encoder 按语义相关度重排，取最终 top_k。

```
配置方式：RERANK_PROVIDER=fastembed（默认 disabled）
模型示例：BAAI/bge-reranker-base（本地 ONNX，无 torch 依赖）
```

cross-encoder 原始 logit 经 sigmoid 归一化为 0~1 相关概率：`score = 1 / (1 + exp(-logit))`。

**降级策略**：模型加载失败、推理异常、或 `provider=disabled` 时，直接返回 RRF 顺序，`reranked=False`，不阻断检索链路。

---

## 八、检索配置参数

| 参数 | 作用 | 热更新 |
|------|------|--------|
| `retrieval_mode` | vector / fulltext / hybrid | ✅ |
| `top_k` | 最终返回条数 | ✅ |
| `vector_weight` / `keyword_weight` | hybrid 内向量/关键词配比（和为1） | ✅ |
| `score_threshold` | raw score 下限（过滤低置信结果） | ✅ |
| `shortcut_threshold` | faq 短路触发阈值（0=关闭） | ✅ |
| `priority_weight` | 跨库融合时该库的权重 | ✅ |
| `chunking_strategy` / `chunk_size` / `chunk_overlap` | 分块参数 | ❌（需重新上传） |
| `embedding_model` | 向量化模型 | ❌（需重建索引） |

---

## 九、与 knowledge-mcp 对接

`agent-tools/knowledge_server` 把 `/api/route-search` 封装成 MCP 工具 `search_knowledge`，对 Agent 只暴露一个聚合检索入口，多库路由对 Agent 透明。

Agent 可从用户问题推断 `filters` 传入（如「2000 以内的耳机」→ `category=耳机, price<=2000`），也可传 `scope` 限定参与的库形态或 kb_id。

Volatile 数据（实时价格/库存）不进向量库，由下游 MCP 在召回结果上用 PG 校准补全，以召回结果中的业务键（如 `sku_id`）做衔接。

---

## 十、当前状态与后续

**已实现：**
- 四种知识库形态及默认配置
- hybrid 检索三模式 + 库内 RRF（MemoryRetriever / MilvusRetriever 双后端）
- 元数据字段定义、值校验、库内 AND 过滤
- temporal 库有效期自动注入
- SearchRouter：faq 短路 + 多路并行 + 跨库加权 RRF
- fastembed cross-encoder rerank（降级安全）
- 版本可见性过滤（shadow 版本对检索不可见）

**待做（后续期）：**
- Query 改写 / HyDE / 子问题分解
- per 库独立 embedding 模型（当前全库统一一个模型）
- 答案溯源 UI（`SearchResult` 已携带 `kb_id + version_no` 字段）
- 中文 BM25 分词验证（Milvus analyzer 或应用层 jieba 预分词）

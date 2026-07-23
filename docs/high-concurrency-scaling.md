# 高并发扩展方案（QPS 10000）

## 一、概述

本文档为**未来扩展预留方案**，当前系统目标 QPS 1000，暂不实施本文档内容。当业务增长到需要支撑 QPS 10000 时，按本方案逐步调整。

**核心判断：QPS 1000 → 10000 的瓶颈不在应用层，而在 LLM 调用并发上限。**

---

## 二、瓶颈分析

### 2.1 QPS 10000 的核心挑战

| 瓶颈 | 原因 | 影响 |
|------|------|------|
| **LLM 并发上限** | 每次请求需 LLM 调用，10000 QPS × 3s 平均响应 = 30000 并发飞行请求 | 单 provider RPM 限制远远不够 |
| **Supervisor LLM 调用** | 每个请求第一步就要调 LLM 分类意图 | 10000 次/秒的分类调用，成本和延迟不可接受 |
| **SSE 长连接** | 万级并发长连接 hold 住 | FastAPI 非专长 |
| **Checkpoint 写入** | 每轮对话写 PG | 10000 TPS 写入压力 |
| **成本** | 全量走云端大模型 | 日均 LLM 费用 $5000+ |

### 2.2 当前方案 vs 10000 QPS 方案

| 维度 | 当前（QPS 1000） | 扩展后（QPS 10000） |
|------|-----------------|---------------------|
| 意图分类 | LLM Supervisor | 本地 BERT/小模型（5ms） |
| LLM 调用 | 单 provider 为主 | 多 provider 池化 + 本地模型分流 |
| 缓存 | 热点 FAQ 缓存 | 语义缓存（30-50% 命中免 LLM） |
| 应用架构 | 同步 SSE | 消息队列解耦 + 独立连接网关 |
| 实例数 | 3-10 Pod | 30-100 Pod |
| 数据库 | 单主 PG | 读写分离 + 连接池 |
| Redis | 单实例 | Redis Cluster |
| 消息队列 | Redis Streams | Kafka |
| 每日 LLM 成本 | ~$200 | ~$2000（优化后） |

---

## 三、架构调整

### 3.1 调整后架构图

```
用户 → CDN/WAF → API 网关 (Kong/APISIX)
                        │
         ┌──────────────┼──────────────────┐
         ▼              ▼                  ▼
   ┌──────────┐  ┌────────────┐  ┌──────────────┐
   │连接网关   │  │ API Pods   │  │ 语义缓存     │
   │(推送结果) │  │(接收请求)  │  │(30%命中免LLM)│
   └─────┬────┘  └─────┬──────┘  └──────────────┘
         │              │
         │     ┌────────▼────────┐
         │     │   Kafka         │  请求队列
         │     └────────┬────────┘
         │              │
         │     ┌────────▼────────┐
         │     │ 意图分类(本地)   │  BERT/小模型, 5ms
         │     └────────┬────────┘
         │              │
         │     ┌────┬───┴───┬────┐
         │     ▼    ▼       ▼    ▼
         │  ┌─────┐┌────┐┌────┐┌────────────┐
         │  │ FAQ ││订单││投诉││ vLLM 集群  │
         │  │精确 ││ .. ││ .. ││(本地模型)  │
         │  │匹配 ││    ││    ││            │
         │  └─────┘└──┬─┘└──┬─┘└────────────┘
         │            │     │
         │     ┌──────▼─────▼──────┐
         │     │ LiteLLM 多Provider │  只处理复杂请求
         │     │ Claude+OpenAI+DS  │
         │     └───────────────────┘
         │              │
    ┌────▼──────────────▼────┐
    │   Redis Pub/Sub        │  结果推送通道
    └────────────────────────┘
```

### 3.2 请求分流策略

```
用户请求 (10000 QPS)
  │
  ├─ 30% 语义缓存命中 → 直接返回（0 LLM 调用）
  ├─ 15% FAQ 精确匹配 → 模板回复（0 LLM 调用）
  ├─ 25% 简单场景 → 本地小模型（vLLM 集群，50ms）
  └─ 30% 复杂场景 → 云端大模型（多 Provider 池化）
        实际云端 LLM QPS ≈ 3000
```

---

## 四、关键改造点

### 4.1 本地意图分类（替代 LLM Supervisor）

**当前：** 每个请求调 LLM 做意图分类（1-3s + 费用）

**改造：** 训练专用分类模型，本地推理 5ms 内完成

用 Fine-tuned BERT / DistilBERT / 通义千问-1.5B 做序列分类，本地推理 5ms 内返回意图和置信度。分类器接口只暴露一个 `classify(message) -> (intent, confidence)`，意图标签固定为一组闭集：

```python
# src/agents/local_classifier.py

class LocalIntentClassifier:
    """本地意图分类器 — 替代 LLM Supervisor"""

    def __init__(self, model_path: str):
        self.model = ...      # AutoModelForSequenceClassification
        self.tokenizer = ...
        self.labels = ["faq", "order", "complaint", "tech_support", "human"]

    async def classify(self, message: str) -> tuple[str, float]:
        """5ms 内返回意图和置信度"""
        ...
```

**分流策略（置信度门控）：** 置信度 > 0.9 直接路由、不调 LLM；置信度 < 0.9 时 Fallback 回 LLM Supervisor，只有少量低置信请求需要走云端，从而把绝大多数分类开销从秒级降到毫秒级。

**训练数据来源：** 上线后从 Langfuse 中导出 Supervisor 的历史路由记录作为标注数据。

### 4.2 语义缓存

复用 Milvus 存一份 `response_cache` collection：把历史 query 的向量与对应答案一起落库，新请求 embed 后按 `kb_id` 过滤做 top-1 检索，相似度超过阈值（默认 0.95）即命中、直接返回历史答案并累加 `hit_count`。缓存条目按 `kb_id` 分区，知识库更新时按 `kb_id` 整体失效，避免答案与最新知识不一致。核心接口与缓存条目的数据契约：

```python
# src/cache/semantic_cache.py

class SemanticCache:
    """语义缓存：相似问题直接返回历史答案"""

    def __init__(self, milvus_client, embedding_model, threshold: float = 0.95):
        ...

    async def get(self, query: str, kb_id: str) -> str | None: ...

    async def put(self, query: str, response: str, kb_id: str, ttl_hours: int = 24): ...
        # metadata: {query, response, kb_id, created_at, hit_count}

    async def invalidate(self, kb_id: str):
        """知识库更新时按 kb_id 整体失效"""
        ...
```

### 4.3 消息队列解耦

核心决策是把请求入口从「同步 hold 连接」改为「入队即返回」。当前 `/api/chat` 同步 `await graph.ainvoke(...)`，连接被 hold 住 3-5s，万级并发下连接就是瓶颈；改造后入口只把请求投递到 Kafka `chat_requests` topic 便立即返回 `{"status": "accepted"}`，真正的 LLM 推理交给独立 Worker 消费，Worker 处理完通过 Redis Pub/Sub 按 `session:{session_id}` 频道把结果推给连接网关。这样入口层不再受单请求时延约束，吞吐与后端推理解耦。

```python
# 入口：入队即返回
@app.post("/api/chat/{session_id}")
async def chat(session_id, request):
    await kafka_producer.send("chat_requests", {"session_id": ..., "message": ...})
    return {"status": "accepted", "session_id": session_id}

# Worker：消费 → 推理 → Redis Pub/Sub 推送
async def process_chat_message(msg):
    result = await graph.ainvoke(...)
    await redis.publish(f"session:{msg['session_id']}", result)
```

### 4.4 独立连接网关

SSE/WebSocket 万级长连接用专门的高性能网关：

| 方案 | 语言 | 特点 |
|------|------|------|
| **Centrifugo** | Go | 开箱即用，单实例 100 万连接，支持 Redis 作为 broker |
| **自建 Go 服务** | Go | 灵活，但开发成本高 |
| **EMQX** | Erlang | 物联网级别，千万连接 |

推荐 **Centrifugo**：
```
用户 SSE/WS → Centrifugo（管理连接）← Redis Pub/Sub ← Worker（推送结果）
```

### 4.5 数据层扩展

- **PostgreSQL 读写分离**：主库只处理写入（Checkpoint、记忆更新），2 个从库分担读取（历史查询、画像读取）；前置 PgBouncer 连接池限制最大连接数，避免高并发下连接数打满数据库。
- **Redis 集群化**：升级为 6 节点 Redis Cluster（3 主 3 从），按 `session_id` hash 分片，保证同一会话落在同一分片。
- **Milvus 分布式**：多个 Query Node 并行检索提升吞吐，Index Node 独立部署，建索引不影响在线查询。
- **Checkpoint 异步化**：热状态先缓存在 Redis，定期批量写入 PG，而不是每次 LLM 调用都同步写 PG——这是缓解万级 TPS 写入压力的关键取舍（用少量状态丢失风险换写入吞吐）。

### 4.6 本地模型部署（vLLM）

```yaml
# 部署 vLLM 集群处理简单场景
services:
  vllm-intent:
    image: vllm/vllm-openai:latest
    command: --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 2048
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1

  vllm-simple:
    image: vllm/vllm-openai:latest
    command: --model Qwen/Qwen2.5-7B-Instruct --max-model-len 4096
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              count: 2
```

LiteLLM 配置中加入本地模型：
```yaml
model_list:
  - model_name: "local-intent"
    litellm_params:
      model: openai/Qwen2.5-1.5B-Instruct
      api_base: "http://vllm-intent:8000/v1"
      api_key: "not-needed"

  - model_name: "local-simple"
    litellm_params:
      model: openai/Qwen2.5-7B-Instruct
      api_base: "http://vllm-simple:8000/v1"
      api_key: "not-needed"
```

---

## 五、渐进式实施路径

当 QPS 接近 1000 时，按以下顺序逐步改造（**不要一步到位**）：

| 阶段 | 改造 | 效果 | 复杂度 |
|------|------|------|--------|
| **第一步** | 语义缓存 | 30% 请求免 LLM，等效 QPS 提升到 1400 | 低 |
| **第二步** | 本地意图分类 | Supervisor 不再调 LLM，延迟降 10x | 中（需训练模型） |
| **第三步** | 多 Provider 池化 | LLM 并发上限翻倍 | 低（LiteLLM 配置） |
| **第四步** | Kafka 解耦 + 连接网关 | 连接数不再是瓶颈 | 中 |
| **第五步** | 本地小模型(vLLM) | 简单场景不走云端，成本降 50% | 高（需 GPU） |
| **第六步** | 数据层扩展（PG 主从、Redis Cluster） | 数据层不成为瓶颈 | 中 |

---

## 六、成本对比

| 项目 | QPS 1000 | QPS 10000（未优化） | QPS 10000（优化后） |
|------|----------|--------------------|--------------------|
| LLM 云端费用/天 | ~$200 | ~$10,000 | ~$2,000 |
| 服务器（CPU） | 3-5 台 | 30-50 台 | 20-30 台 |
| GPU（本地模型） | 无 | — | 4-8 张 A10 |
| 基础设施 | 简单 | — | Kafka + Redis Cluster + PG 主从 |
| **总计/月** | ~$10K | ~$350K | ~$100K |

**优化关键：** 语义缓存 + 本地模型分流，将云端 LLM 调用量从 10000 降到 3000 QPS。

---

## 七、当前架构的扩展友好点

当前方案已经预留的扩展点，无需大改即可利用：

| 预留点 | 对应扩展 |
|--------|----------|
| LiteLLM Proxy | 直接加 Provider、加本地模型 endpoint |
| AgentTransport 抽象 | Supervisor 替换为本地分类器时接口不变 |
| Redis 已用于缓存/限流 | 升级为 Cluster 业务代码不变 |
| FastAPI async | 切换为 Kafka 消费模式改动集中在入口层 |
| Milvus 向量库 | 加 collection 做语义缓存 |
| Docker Compose | 加 vLLM 容器即可 |

---

## 八、监控指标（触发扩展的信号）

当以下指标触发时，考虑启动对应扩展：

| 信号 | 阈值 | 触发动作 |
|------|------|----------|
| API P99 延迟 | > 5s 持续 5 min | 优先加缓存 |
| LLM Provider 限流错误 | > 1% | 加 Provider / 本地模型 |
| SSE 并发连接数 | > 5000 | 引入连接网关 |
| PG 写入延迟 | > 100ms | Checkpoint 异步化 |
| Redis 内存 | > 80% | 升级 Cluster |
| LLM 日均成本 | > $500 | 引入语义缓存 + 本地模型 |
| Supervisor 调用占比 | > 30% 总 LLM 费用 | 本地意图分类器 |

---

## 九、不建议当前做的事

| 事项 | 原因 |
|------|------|
| 提前引入 Kafka | QPS 1000 Redis Streams 足够，Kafka 运维成本高 |
| 提前部署 vLLM | 需要 GPU，成本高，QPS 1000 不需要 |
| 提前做 PG 读写分离 | 单主 PG 扛 QPS 1000 绰绰有余 |
| 提前换 Go 写连接网关 | 几千 SSE 连接 FastAPI 能处理 |
| 提前训练意图分类模型 | 需要足够的历史数据，先跑起来积累数据 |

**原则：先跑起来 → 积累数据 → 发现真实瓶颈 → 针对性优化。**

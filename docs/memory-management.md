# 记忆管理系统 — 更新与维护方案

## 一、概述

本文档定义智能客服多 Agent 系统记忆管理的更新机制、过期策略、冲突解决、运维维护方案。是技术方案主文档"第十章 记忆管理系统"的补充。

### 记忆三层回顾

| 记忆类型 | 内容 | 存储 | 生命周期 |
|----------|------|------|----------|
| **短期记忆** | 完整多轮对话 messages + Graph State | PG (LangGraph Checkpoint) | 单次会话 |
| **长期记忆** | 用户画像 / 历史摘要 / 事实知识 | PG + Milvus | 永久（可衰减） |
| **工作记忆** | 当前对话提取的结构化实体 | Redis | 单次会话（快速存取） |

---

## 二、记忆更新触发机制

### 2.1 触发时机一览

| 触发点 | 更新什么 | 方式 | 优先级 |
|--------|----------|------|--------|
| **每轮对话后** | 工作记忆（实体提取） | 实时同步 | 高 |
| **每轮对话后** | 短期记忆（Checkpoint） | LangGraph 自动 | 高 |
| **用户明确告知时** | 事实记忆（"我地址改了"） | 实时，立即覆盖 | 最高 |
| **工具返回结果时** | 工作记忆 + 事实校正 | 实时 | 高 |
| **会话结束时** | 长期记忆（摘要 + 事实 + 画像） | 异步任务 | 中 |
| **CRM 数据变更时** | 用户画像 | Webhook / 定时同步 | 中 |
| **定时任务** | 衰减、清理、一致性校验 | 定时调度 | 低 |

### 2.2 实时更新流程

```
用户发消息
  │
  ├─→ [自动] 短期记忆更新（Checkpoint 存 State）
  │
  ├─→ [实时] 实体提取 → 更新工作记忆 (Redis)
  │         例：提取出 order_id, amount, address
  │
  ├─→ [实时] 用户明确告知检测
  │         "我手机号换了 138xxx" → 立即更新事实记忆
  │
  └─→ [实时] 工具结果校正
            查询工具返回 "订单已取消" → 更新工作记忆中的 order_status
```

### 2.3 会话结束更新流程

```
会话结束（用户关闭 / 超时 / Agent 标记 resolved）
  │
  ├─→ [异步] 生成会话摘要 → 存入 Episodic Memory
  │         LLM 总结：用户问了什么、怎么解决的、满意吗
  │
  ├─→ [异步] 提取持久事实 → 存入 Semantic Memory
  │         从对话中提取"用户确认"的持久信息
  │
  ├─→ [异步] 画像增量更新 → 更新 Profile Memory
  │         推断沟通风格、新增敏感点等
  │
  └─→ [同步] 清理工作记忆 (Redis)
```

### 2.4 更新触发器实现

```python
# src/memory/triggers.py

class MemoryUpdateTrigger:
    """记忆更新触发器，嵌入 Agent 执行流程"""

    def __init__(self, memory_manager: MemoryManager, extractor: EntityExtractor):
        self.memory = memory_manager
        self.extractor = extractor

    async def on_user_message(self, session_id: str, customer_id: str, message: str):
        """每轮用户消息后触发"""
        # 1. 实体提取 → 更新工作记忆
        entities = await self.extractor.extract_entities(message)
        for key, value in entities.items():
            await self.memory.working.set_entity(session_id, key, value)

        # 2. 检测用户是否明确更新了信息
        explicit_updates = await self.extractor.detect_explicit_updates(message)
        if explicit_updates:
            for field, value in explicit_updates.items():
                await self.memory.update_fact(
                    customer_id, field, value, source="user_explicit"
                )

    async def on_tool_result(self, session_id: str, tool_name: str, result: dict):
        """工具返回结果后触发"""
        # 用工具的实时数据校正工作记忆
        corrections = self._extract_corrections(tool_name, result)
        for key, value in corrections.items():
            await self.memory.working.set_entity(session_id, key, value)

    async def on_session_end(self, session_id: str, customer_id: str, messages: list):
        """会话结束时触发（异步）"""
        await self.memory.on_session_end(customer_id, session_id, messages)
```

---

## 三、过期策略（TTL + 衰减）

### 3.1 各层过期规则

| 记忆类型 | 过期规则 | 过期后行为 |
|----------|----------|-----------|
| **工作记忆** | 会话结束 或 1h 无写入 | 直接删除 |
| **短期记忆** | 30 天无活动 | 生成摘要归档到长期记忆，删除 Checkpoint |
| **长期 - 历史摘要** | 不删除 | 检索时按时间衰减加权排序 |
| **长期 - 事实知识** | 不自动删除 | 置信度低于 0.3 标记为"待验证" |
| **长期 - 用户画像** | 不过期 | 90 天未更新的字段标记 stale |

### 3.2 权重衰减

历史记忆不删除，但检索时越老的记忆权重越低：

```python
# src/memory/decay.py
import math

class MemoryDecay:
    """记忆时间衰减"""

    # 半衰期约 70 天
    EPISODIC_DECAY_RATE = 0.01

    # 事实置信度每天衰减 0.5%
    FACT_DECAY_RATE = 0.005

    def episodic_relevance(self, semantic_score: float, days_ago: int) -> float:
        """历史摘要检索：语义分 × 时间衰减"""
        decay = math.exp(-self.EPISODIC_DECAY_RATE * days_ago)
        return semantic_score * decay

    def fact_confidence(self, initial_confidence: float, days_since_update: int) -> float:
        """事实置信度随时间衰减"""
        decayed = initial_confidence * math.exp(-self.FACT_DECAY_RATE * days_since_update)
        return max(decayed, 0.1)  # 最低保留 0.1

    def is_stale(self, days_since_update: int, threshold: int = 90) -> bool:
        """画像字段是否陈旧"""
        return days_since_update > threshold
```

### 3.3 配置

```python
# src/config.py

class MemoryTTLConfig(BaseSettings):
    # 工作记忆
    working_memory_ttl_seconds: int = 3600          # 1h
    # 短期记忆归档阈值
    checkpoint_archive_days: int = 30
    # 事实置信度衰减率
    fact_decay_rate: float = 0.005
    # 事实待验证阈值
    fact_reverify_threshold: float = 0.3
    # 画像陈旧标记天数
    profile_stale_days: int = 90
    # 历史摘要时间衰减率
    episodic_decay_rate: float = 0.01
    # 每用户最大历史摘要数
    max_episodes_per_user: int = 100
```

---

## 四、记忆冲突解决

### 4.1 冲突类型

| 冲突类型 | 示例 | 谁赢 |
|----------|------|------|
| 用户明确告知 vs 历史记忆 | "我地址改了" vs 记忆中旧地址 | **用户赢** |
| 工具实时结果 vs 记忆缓存 | 工具查"已取消" vs 记忆"已发货" | **工具赢** |
| CRM 权威字段 vs 对话推断 | CRM 说"普通用户" vs Agent 推断"VIP" | **CRM 赢** |
| 新推断 vs 旧推断 | 本次推断"简洁风格" vs 上次"详细风格" | **按置信度** |
| 累积型信息 | 新的敏感点 vs 已有敏感点列表 | **合并** |

### 4.2 优先级规则

```
用户明确告知 > 工具实时数据 > CRM 权威字段 > 高置信度推断 > 低置信度推断
```

### 4.3 实现

```python
# src/memory/conflict_resolver.py
from enum import Enum

class ConflictResolution(Enum):
    NEW_WINS = "new_wins"
    OLD_WINS = "old_wins"
    MERGE = "merge"
    CONFIDENCE_BASED = "confidence"

# 字段级冲突策略
FIELD_POLICIES = {
    # CRM 权威字段：旧值优先（除非用户明确更正）
    "vip_level": ConflictResolution.OLD_WINS,
    "customer_id": ConflictResolution.OLD_WINS,

    # 用户可更新的事实：新值覆盖旧值
    "shipping_address": ConflictResolution.NEW_WINS,
    "phone_number": ConflictResolution.NEW_WINS,
    "preferred_contact": ConflictResolution.NEW_WINS,
    "email": ConflictResolution.NEW_WINS,

    # 累积型：合并
    "sensitive_points": ConflictResolution.MERGE,
    "frequent_categories": ConflictResolution.MERGE,
    "tags": ConflictResolution.MERGE,

    # 推断型：按置信度
    "communication_style": ConflictResolution.CONFIDENCE_BASED,
}

class ConflictResolver:
    def __init__(self, audit_store, confidence_store):
        self.audit = audit_store
        self.confidence = confidence_store

    async def resolve(
        self,
        customer_id: str,
        field: str,
        old_value: any,
        new_value: any,
        source: str,
        confidence: float = 1.0,
    ) -> tuple[any, bool]:
        """返回 (resolved_value, was_updated)"""

        if old_value == new_value:
            return old_value, False

        # 最高优先级：用户明确告知
        if source == "user_explicit":
            await self._log(customer_id, field, old_value, new_value, source)
            return new_value, True

        # 次高：工具实时数据
        if source == "tool_result":
            await self._log(customer_id, field, old_value, new_value, source)
            return new_value, True

        # 按字段策略处理
        policy = FIELD_POLICIES.get(field, ConflictResolution.CONFIDENCE_BASED)

        if policy == ConflictResolution.NEW_WINS:
            await self._log(customer_id, field, old_value, new_value, source)
            return new_value, True

        elif policy == ConflictResolution.OLD_WINS:
            return old_value, False

        elif policy == ConflictResolution.MERGE:
            old_list = old_value if isinstance(old_value, list) else [old_value] if old_value else []
            new_list = new_value if isinstance(new_value, list) else [new_value] if new_value else []
            merged = list(set(old_list + new_list))
            if merged != old_list:
                await self._log(customer_id, field, old_value, merged, source)
                return merged, True
            return old_value, False

        elif policy == ConflictResolution.CONFIDENCE_BASED:
            old_conf = await self.confidence.get(customer_id, field)
            if confidence > old_conf:
                await self.confidence.set(customer_id, field, confidence)
                await self._log(customer_id, field, old_value, new_value, source)
                return new_value, True
            return old_value, False

        return old_value, False

    async def _log(self, customer_id, field, old_value, new_value, source):
        """审计日志"""
        await self.audit.append({
            "timestamp": datetime.now().isoformat(),
            "customer_id": customer_id,
            "field": field,
            "old_value": str(old_value),
            "new_value": str(new_value),
            "source": source,
        })
```

---

## 五、置信度管理

### 5.1 来源置信度

| 来源 | 初始置信度 | 说明 |
|------|-----------|------|
| `user_explicit` | 1.0 | 用户亲口说的 |
| `tool_result` | 0.95 | 工具查询返回的实时数据 |
| `crm` | 0.9 | CRM 系统同步 |
| `agent_inferred` | 0.6 | Agent 从对话上下文推断 |
| `historical` | 0.5 | 从历史摘要中提取 |

### 5.2 实现

```python
# src/memory/confidence.py
import math
from datetime import datetime

class ConfidenceManager:
    SOURCE_SCORES = {
        "user_explicit": 1.0,
        "tool_result": 0.95,
        "crm": 0.9,
        "agent_inferred": 0.6,
        "historical": 0.5,
    }

    def __init__(self, decay_rate: float = 0.005):
        self.decay_rate = decay_rate

    def initial_confidence(self, source: str) -> float:
        return self.SOURCE_SCORES.get(source, 0.5)

    def current_confidence(self, initial: float, days_since_update: int) -> float:
        """置信度随时间衰减"""
        decayed = initial * math.exp(-self.decay_rate * days_since_update)
        return max(decayed, 0.1)

    def should_reverify(self, confidence: float, threshold: float = 0.3) -> bool:
        """是否需要重新验证"""
        return confidence < threshold

    def should_ask_user(self, confidence: float, threshold: float = 0.4) -> bool:
        """是否应该向用户确认（而非直接使用）"""
        return confidence < threshold
```

### 5.3 低置信度记忆的处理

当记忆置信度较低时，Agent 使用时应加注标记：

```python
def format_fact_for_prompt(key: str, value: str, confidence: float) -> str:
    """低置信度的事实加标注，避免 Agent 当成确定信息使用"""
    if confidence >= 0.8:
        return f"- {key}: {value}"
    elif confidence >= 0.4:
        return f"- {key}: {value}（历史记录，可能已变更）"
    else:
        return f"- {key}: {value}（待确认，请向用户核实）"
```

---

## 六、记忆维护（定时任务）

### 6.1 任务清单

| 任务 | 频率 | 做什么 |
|------|------|--------|
| **工作记忆清理** | 每小时 | 清理超时的 Redis key |
| **Checkpoint 归档** | 每天 | 30天无活动会话 → 摘要归档 → 删除 Checkpoint |
| **异常会话补偿** | 每小时 | 非正常结束的会话补偿执行归档 |
| **置信度批量衰减** | 每天 | 更新所有事实记忆的当前置信度 |
| **画像陈旧标记** | 每周 | 90天未更新的字段标记 stale |
| **CRM 增量同步** | 每天 / Webhook | 同步 CRM 变更到画像 |
| **一致性校验** | 每周 | 检查画像与 CRM 不一致项 |
| **向量索引优化** | 每周 | Milvus compact + rebuild |
| **容量监控** | 每天 | 记忆总量统计、超限告警 |
| **历史摘要压缩** | 每月 | 超过 100 条的用户，旧摘要合并 |

### 6.2 实现

```python
# src/memory/maintenance.py

class MemoryMaintenance:
    """记忆维护定时任务"""

    def __init__(self, memory_manager, config: MemoryTTLConfig):
        self.memory = memory_manager
        self.config = config

    # ========== 每小时 ==========
    async def hourly(self):
        await self._cleanup_orphan_working_memory()
        await self._compensate_orphan_sessions()

    async def _cleanup_orphan_working_memory(self):
        """清理过期工作记忆（Redis TTL 兜底，此处补偿清理）"""
        # Redis TTL 会自动处理，此处处理异常 key
        pass

    async def _compensate_orphan_sessions(self):
        """补偿非正常结束的会话"""
        orphans = await self.memory.checkpoint_store.find_inactive(
            minutes=30, status="not_resolved"
        )
        for session in orphans:
            messages = await self.memory.checkpoint_store.get_messages(session.id)
            if len(messages) > 2:
                await self.memory.on_session_end(
                    session.customer_id, session.id, messages
                )

    # ========== 每天 ==========
    async def daily(self):
        await self._archive_stale_checkpoints()
        await self._decay_confidences()
        await self._sync_crm_changes()
        await self._report_storage_usage()

    async def _archive_stale_checkpoints(self):
        """归档过期会话"""
        stale = await self.memory.checkpoint_store.find_stale(
            days=self.config.checkpoint_archive_days
        )
        for session in stale:
            messages = await self.memory.checkpoint_store.get_messages(session.id)
            # 归档到长期记忆
            await self.memory.episodic.save_episode(
                session.id, session.customer_id, messages
            )
            # 删除 Checkpoint
            await self.memory.checkpoint_store.delete(session.id)

        if stale:
            logger.info(f"archived {len(stale)} stale sessions")

    async def _decay_confidences(self):
        """批量衰减事实置信度"""
        facts = await self.memory.semantic.get_all_facts_with_metadata()
        for fact in facts:
            days = (datetime.now() - fact.last_updated).days
            new_conf = self.memory.confidence.current_confidence(
                fact.initial_confidence, days
            )
            if new_conf != fact.current_confidence:
                await self.memory.semantic.update_confidence(
                    fact.customer_id, fact.key, new_conf
                )

    async def _sync_crm_changes(self):
        """CRM 增量同步"""
        changes = await self.memory.crm_client.get_changes_since(self.last_sync_time)
        for change in changes:
            await self.memory.profile.sync_from_crm(
                change.customer_id, change.fields
            )
        self.last_sync_time = datetime.now()

    async def _report_storage_usage(self):
        """存储用量统计"""
        stats = {
            "total_profiles": await self.memory.profile.count(),
            "total_episodes": await self.memory.episodic.count(),
            "total_facts": await self.memory.semantic.count(),
            "checkpoint_count": await self.memory.checkpoint_store.count(),
        }
        logger.info("memory_storage_stats", **stats)
        # 超限告警
        if stats["checkpoint_count"] > 100_000:
            await self.alert("checkpoint 数量超过 10万，建议加速归档")

    # ========== 每周 ==========
    async def weekly(self):
        await self._mark_stale_profiles()
        await self._check_consistency()
        await self._optimize_vector_index()

    async def _mark_stale_profiles(self):
        """标记陈旧画像字段"""
        profiles = await self.memory.profile.get_all()
        for profile in profiles:
            for field, meta in profile.field_metadata.items():
                days = (datetime.now() - meta.last_updated).days
                if days > self.config.profile_stale_days:
                    await self.memory.profile.mark_stale(profile.customer_id, field)

    async def _check_consistency(self):
        """画像与 CRM 一致性校验"""
        inconsistencies = []
        profiles = await self.memory.profile.get_all_with_crm_fields()
        for profile in profiles:
            crm_data = await self.memory.crm_client.get(profile.customer_id)
            if not crm_data:
                continue
            for field in ["vip_level", "phone_number", "email"]:
                memory_val = getattr(profile, field, None)
                crm_val = crm_data.get(field)
                if memory_val and crm_val and memory_val != crm_val:
                    inconsistencies.append({
                        "customer_id": profile.customer_id,
                        "field": field,
                        "memory_value": memory_val,
                        "crm_value": crm_val,
                    })

        if inconsistencies:
            logger.warning(f"memory_crm_inconsistency count={len(inconsistencies)}")
            await self.alert(
                f"发现 {len(inconsistencies)} 条记忆与 CRM 不一致，需审核"
            )

    async def _optimize_vector_index(self):
        """Milvus 向量索引优化"""
        await self.memory.milvus.compact("episodic_memory")

    # ========== 每月 ==========
    async def monthly(self):
        await self._compress_old_episodes()

    async def _compress_old_episodes(self):
        """历史摘要超过上限时，合并旧记录"""
        users = await self.memory.episodic.get_users_exceeding_limit(
            limit=self.config.max_episodes_per_user
        )
        for customer_id in users:
            old_episodes = await self.memory.episodic.get_oldest(
                customer_id, count=20
            )
            # 合并为一条摘要
            merged_summary = await self.memory.summarizer.merge_episodes(old_episodes)
            await self.memory.episodic.replace_with_merged(
                customer_id, old_episodes, merged_summary
            )
```

---

## 七、补充场景处理

### 7.1 记忆污染防护

Agent 推断的信息可能错误，存入长期记忆后会持续误导后续对话。

```python
# 防护策略：推断型记忆低置信度 + 验证机制
class MemoryPollutionGuard:
    async def validate_before_store(
        self, field: str, value: any, source: str, messages: list
    ) -> bool:
        """存入长期记忆前校验"""
        # 用户明确说的 → 直接通过
        if source == "user_explicit":
            return True

        # Agent 推断的 → 检查是否有足够证据
        if source == "agent_inferred":
            evidence = await self._check_evidence(field, value, messages)
            return evidence.confidence > 0.5

        return True

    async def _check_evidence(self, field, value, messages) -> Evidence:
        """检查对话中是否有足够支撑这个推断的证据"""
        # 简单实现：检查最近 5 轮对话中是否有明确相关内容
        recent = messages[-10:]
        mention_count = sum(
            1 for m in recent if str(value).lower() in m.content.lower()
        )
        return Evidence(confidence=min(mention_count * 0.3, 1.0))
```

### 7.2 隐私遗忘权（GDPR 合规）

```python
# src/memory/privacy.py

async def purge_customer_memory(customer_id: str):
    """一键删除用户所有记忆数据"""
    # 1. 删除长期记忆
    await profile_memory.delete(customer_id)
    await episodic_memory.delete_all(customer_id)
    await semantic_memory.delete_all(customer_id)

    # 2. 删除工作记忆
    await working_memory.clear_by_customer(customer_id)

    # 3. 标记 Checkpoint 待清除（可能有进行中会话）
    await checkpoint_store.mark_for_purge(customer_id)

    # 4. 删除向量库中的记录
    await milvus.delete_by_filter("episodic_memory", f"customer_id == '{customer_id}'")

    # 5. 审计日志（记录删除操作本身，不记录被删内容）
    await audit_log.record("memory_purge", customer_id=customer_id)
```

### 7.3 并发写冲突

同一用户同时有多个会话（如同时开了两个浏览器标签）：

```python
# 乐观锁：通过 version 字段防止覆盖
class OptimisticLockMixin:
    async def update_with_lock(self, customer_id: str, field: str, value: any, version: int):
        result = await self.pg.execute("""
            UPDATE user_facts
            SET value = $1, version = version + 1, updated_at = now()
            WHERE customer_id = $2 AND field = $3 AND version = $4
        """, value, customer_id, field, version)

        if result.rowcount == 0:
            # 版本冲突，重新读取后决策
            current = await self.get(customer_id, field)
            # 最后写入胜出（记录日志）
            await self.force_update(customer_id, field, value)
            await self.audit.log_conflict(customer_id, field, current, value)
```

### 7.4 冷启动

新用户首次对话，没有任何长期记忆：

```python
async def cold_start(customer_id: str) -> UserProfile:
    """新用户冷启动：从 CRM 拉取基础画像"""
    # 1. CRM 拉取
    crm_data = await crm_client.get_customer(customer_id)
    if crm_data:
        profile = UserProfile(customer_id=customer_id, **crm_data)
        await profile_memory.save(profile)
        return profile

    # 2. 完全新用户：创建空画像
    profile = UserProfile(customer_id=customer_id)
    await profile_memory.save(profile)
    return profile
```

### 7.5 会话未正常结束

用户直接关浏览器，不会触发 `on_session_end`：

- 解决方案：定时任务每小时扫描 30 分钟无活动的未关闭会话
- 补偿执行归档逻辑（见维护任务 `_compensate_orphan_sessions`）

### 7.6 记忆检索噪声

检索到不相关的历史摘要干扰 Agent：

```python
async def search_with_filter(self, query: str, customer_id: str, top_k: int = 3):
    """带过滤的记忆检索"""
    results = await self.milvus.search(
        collection="episodic_memory",
        vector=await self.embed(query),
        filter=f"customer_id == '{customer_id}'",
        top_k=top_k * 2,  # 多检索一些，后面过滤
    )

    # 过滤：相似度阈值 + 时间衰减
    filtered = []
    for r in results:
        days_ago = (datetime.now() - r.metadata["timestamp"]).days
        adjusted_score = self.decay.episodic_relevance(r.score, days_ago)
        if adjusted_score > 0.5:  # 相似度阈值
            filtered.append(r)

    return filtered[:top_k]
```

---

## 八、监控指标

| 指标 | 含义 | 告警阈值 |
|------|------|----------|
| `memory.update_count` | 记忆更新次数（分类型） | 仅记录 |
| `memory.conflict_count` | 冲突发生次数 | > 50 次/小时 |
| `memory.conflict_resolution` | 冲突解决方式分布 | 仅记录 |
| `memory.stale_facts_count` | 低置信度事实数量 | > 1000 条 |
| `memory.crm_inconsistency` | CRM 不一致数量 | > 0 |
| `memory.orphan_sessions` | 异常未关闭会话数 | > 100 |
| `memory.storage_total` | 各类记忆总量 | 按阈值告警 |
| `memory.retrieval_latency_p99` | 记忆检索 P99 延迟 | > 200ms |
| `memory.purge_requests` | 隐私删除请求数 | 仅记录 |

---

## 九、目录结构

```
src/memory/
├── __init__.py
├── manager.py                # MemoryManager 统一入口
├── triggers.py               # 更新触发器
├── conflict_resolver.py      # 冲突解决
├── confidence.py             # 置信度管理
├── decay.py                  # 时间衰减
├── maintenance.py            # 定时维护任务
├── privacy.py                # 隐私遗忘权
├── pollution_guard.py        # 记忆污染防护
├── short_term.py             # 短期记忆
├── working.py                # 工作记忆
├── long_term/
│   ├── __init__.py
│   ├── profile.py            # 用户画像
│   ├── episodic.py           # 历史交互摘要
│   └── semantic.py           # 事实知识
├── extraction.py             # 实体/事实提取器
├── summarizer.py             # 会话摘要生成
├── checkpointer.py           # LangGraph Checkpoint
└── conversation.py           # 多轮对话管理
```

---

## 十、实现现状与设计确认

> 前面 1~8 章是**目标设计**，其中部分模块（triggers.py / conflict_resolver.py / maintenance.py 等）尚未落地。本章记录**已实现并验证**的真实设计与关键决策，与代码一一对应。凡与前文不一致处，以本章为准。

### 10.0 已实现 vs 未实现

| 能力 | 状态 | 说明 |
|------|------|------|
| 短期记忆：消息落库 + 滚动摘要 + LIMIT 加载 | ✅ 已实现 | `short_term.py` |
| 工作记忆：Redis + 工具结果规则抽实体 | ✅ 已实现 | `working.py` + `entities.py` |
| 长期-画像/事实/历史：SQLAlchemy 持久化 | ✅ 已实现 | `long_term/*` + `database.py` |
| 长期-历史：Milvus 向量语义检索 + 时间衰减 | ✅ 已实现 | `episodic.py` |
| 事实置信度衰减 + 分档标注 | ✅ 已实现 | `decay.py` |
| 归档：复用短期摘要 + facts/profile 增量抽取 | ✅ 已实现 | `manager.py`（做法1）|
| 会话结束触发：`/end` 接口 | ✅ 已实现 | `routes.py` |
| 超时兜底归档 | ⚠️ 函数备好、未接调度 | `archive_idle_sessions`，生产需 CronJob |
| CRM webhook 同步 / 冲突解决器 / 定时维护任务 | ❌ 未实现 | 见前文目标设计 |

### 10.1 短期记忆：滚动摘要 + LIMIT 加载

**存储分工（两张表）**：
- `messages` 表：全量对话原文，一条消息一行，**只增不改**（`/history` 展示、审计用）。
- `sessions` 表：一个 session 一行，`summary`（老消息滚动摘要）+ `summary_upto_id`（摘要已覆盖到的最大 message id，0=无摘要）两列**更新覆盖**，不新增。

**加载（每轮，热路径）** `ShortTermMemory.load`：
```
recent = SELECT * FROM messages
         WHERE session_id=? AND id > summary_upto_id   -- 只取未压缩原文
         ORDER BY id DESC LIMIT load_limit(30)          -- 带 LIMIT，防全量
         → reverse 还原时间序
返回 (recent 原文, sessions.summary 摘要)
```
- `summary_upto_id`（简称 upto）是分界线：id ≤ upto 已压进摘要，id > upto 是原文。
- `load_limit=30` 是**安全上界**（≥ 最大 history_window 10），不是精确需求——因加载发生在路由前、不知走哪个 agent（各 agent history_window 不同），故取够所有下游用的公共供给，各 agent 再自行 `[-history_window:]` 切 5~10 条。

**压缩（回合结束，非热路径）** `ShortTermMemory.compress_if_needed`：
- 先一次廉价 `COUNT(id > upto)`，`< trigger(30)` 直接返回、**不调 LLM**（绝大多数回合走这里，仅一次 COUNT 开销）。
- `≥ trigger` 才把最老的 `(总数 - keep(10))` 条交 LLM 融合进 `summary`，`upto` 前移。返回被压批次供增量事实抽取。

**为何摘要放进 `memory_context`（SystemMessage）而非消息列表**：
`state["messages"]` 会被各 agent `[-history_window:]` 切成最近 10 条；摘要代表**最老**的历史、天然在列表开头，一放进去就被切掉。而 `memory_context` 拼进 SystemMessage（`[SystemMessage(prompt+摘要), *messages]`），**不经过窗口切割**，无论对话多长都保留。

**加载不变量**：`摘要(≤upto) + 原文(>upto)`，无 gap、无重叠。

### 10.2 工作记忆：Redis + 工具结果规则抽实体

- **存储**：Redis Hash，key = `working:{session_id}`（一会话一 key），field=实体名、value=`json.dumps(值)`（保住类型：amount 是 float 不是 "299.0"）。TTL `working_memory_ttl(3600s)`。Redis 不可达 → 降级内存 dict。
- **写入**：`entities.py::extract_entities` **规则直取**（非 LLM）——从工具返回结果里取 `order_id/refund_id/amount/status/carrier` 等已知字段。在 `agent_node` / `execute_node` 工具跑完后调用。
  - 关键点：MCP 读工具经 adapter 序列化为 **JSON 字符串**，写工具（apply_refund）是 **dict**——`extract_entities` 两种都吃（字符串先 `json.loads`）。
  - 语义：同名覆盖、异名新增（`hset`），是"当前焦点便签"，非只增日志。
- **定位（重要设计取舍）**：主流 agent 记忆框架（MemGPT/Letta 等）**不把结构化实体缓存单列一层**。本项目的工作记忆真正价值仅在"指代消解"（"这个订单"=哪个）与"对抗超长对话截断"。**权威永远是本轮工具实时结果，工作记忆只是线索、可能过时**——绝不以它为准。demo 规模下短期记忆已能兜底指代消解，工作记忆可有可无（`memory_enabled` 可整体关）。

### 10.3 长期记忆：三子模块 + 双写 + 语义检索

**存储**：
- 画像 `profiles` 表（一人一行）、事实 `facts` 表（`(customer_id,key)` 唯一，带 source/confidence/updated_at）、历史 `episodes` 表 + **Milvus 向量**（双写）。
- 均走现有 async SQLAlchemy 栈；`session_factory=None` 时三子模块回退内存 dict（测试）。

**写入 = 会话结束归档（做法1，见 9.4）**，不是每轮。

**读取（每轮）** `load_context`：
- 画像 `profile.get` → PG
- 历史 `episodic.search`：embed(query) → **Milvus 向量检索**（filter customer_id，`consistency_level=Strong` 保证写完即可查）→ 回 PG 载全量 → `MemoryDecay.episodic_relevance = 相似度 × exp(-rate×days_ago)` **时间衰减重排** → top_k。Milvus/embedder 不可达 → 降级 DB 最近 N 条。
- 事实 `get_facts_scored`：按 `updated_at` 距今天数 `ConfidenceManager.current_confidence` 衰减；`format_fact_for_prompt` 分档标注（≥0.8 直出 / ≥0.4 "可能已变更" / <0.4 "待确认核实"）。
- 工作记忆 `working.get`（见 9.2）。
- 四块 → `format_memory_for_prompt` → 拼进 SystemMessage。

**置信度来源**：user_explicit 1.0 > tool_result 0.95 > crm 0.9 > agent_inferred 0.6 > historical 0.5。

### 10.4 归档：复用短期摘要 + facts/profile 增量抽取（做法1）

**背景**：早期实现把整段对话拼起来 `[:6000]` 截断喂 LLM，既有反向截断 bug（切掉最近内容）、又与短期摘要重复劳动。且关键权衡是——**摘要够用于"检索历史"，但会抹掉低频细节（用户随口报的电话/地址），不足以抽精确事实**。故按目标拆分输入：

- **episode 摘要（检索用，可容忍丢细节）**：`on_session_end` 用 `short_term.load` 的 `(最近原文 + 滚动摘要)` 喂 `EPISODE_DIGEST_PROMPT` 生成 {summary,intent,resolution,satisfaction}。省 token、不重复摘要、输入量恒定不随会话增长。
- **facts/profile（要精确）→ 增量抽取**：
  - `on_turn_end`（回合末）：`compress_if_needed` 若压缩了一批，立即对**这批原文**用 `FACT_EXTRACT_PROMPT` 抽 facts/profile 落库——趁原文未被摘要抹掉细节前抽取。
  - `on_session_end`（会话末）：只对**未压缩的尾巴**补抽。
- **核心不变量**：每条消息事实被抽取**且仅一次**——被压批次(压缩时) ∪ 未压尾巴(会话末)，disjoint。短会话（从没压缩）→ 尾巴=全部，会话末一次抽完。
- **幂等保证**：`set_fact` 按 `(customer_id,key)` upsert、`profile.update_from_conversation` merge 去重 → 重复调用安全。
- **成本取舍**：每次压缩 2 次 LLM（摘要 1 + 事实抽取 1），换 ShortTermMemory（摘要）与 MemoryManager（事实）职责解耦。压缩低频（每 ~30 条一次），可接受。

**触发**：
- 主：`POST /v1/chat/{session_id}/end`（前端主动调）。
- 兜底：`archive_idle_sessions(idle_minutes=30)` 扫超时会话——**函数已备但未接调度**，生产需 k8s CronJob / APScheduler 定时打内部端点。

### 10.5 优雅降级（贯穿全栈）

| 后端挂 | 降级 |
|--------|------|
| Redis | 工作记忆走内存 dict |
| Milvus / embedding | 历史检索降 DB 最近 N；embedder 回退 pseudo 向量 |
| LLM | 归档不抽取、短期不压缩，仅保留 LIMIT 加载；不阻断对话 |
| `session_factory=None` | 长期三子模块回退内存 dict（测试） |
| `memory_enabled=false` | 退回全内存 MemoryManager（短期 LIMIT 加载仍恒开）|

### 10.6 运维备忘

- **建表**：`init_db` 的 `create_all` **只建不改列**。已存在的 `sessions` 表需手动补 `summary` / `summary_upto_id`（开发库已 `ALTER TABLE`；生产走正式 migration）。新增表（profiles/facts/episodes/error_records）自动建。
- **docker compose**：agent-core 需配 `MILVUS_HOST` / `EMBEDDING_*` / `MEMORY_ENABLED` 等 env + `depends_on milvus` + 复用 `rag_model_cache` 卷，否则向量检索一直降级。

# RAG 知识库版本与全生命周期管理技术方案（下一期）

> 目标：知识全生命周期**可追溯、可验证后上线、可回滚、可定责溯源**，解决版本混乱、客诉无法定责、更新出错影响大的问题。
> 本方案为「RAG 检索增强（多 Collection + 混合检索）」之后的**独立下一期**，前置依赖见 [rag-retrieval-metadata-hybrid.md](./rag-retrieval-metadata-hybrid.md)。
> 现有版本机制基础见 [rag-persistence-versioning.md](./rag-persistence-versioning.md)。

---

## 一、背景与目标

### 1.1 核心目标

| 目标 | 解决的问题 |
|------|-----------|
| **可追溯** | 每次知识修改留完整快照（内容前后、操作人、时间、审批），版本混乱可查 |
| **可验证后上线** | 新版本上线前先验证，避免更新出错大面积影响 |
| **可回滚** | 出错时分钟级回退到任意历史版本 |
| **可定责溯源** | 客诉时凭会话 ID 精准定位当时引用的知识 ID + 版本号，快速定责 |

### 1.2 现状基础（agent-rag 已具备）

agent-rag 已有**文档级版本控制**（[rag-persistence-versioning.md](./rag-persistence-versioning.md)），本方案在其上增强而非重建：

| 已有能力 | 说明 |
|----------|------|
| 影子构建 + 原子切换 | 新版以 shadow 状态构建，就绪后翻转 `current_version_id` 指针，零停机 |
| 检索可见性过滤 | `visible_version_ids` 只让 current 版本被检索，shadow/archived 不可见 |
| 即时回滚 | `rollback(version_no)` 翻指针回历史版本，秒级生效 |
| 保留 GC | `apply_retention` 保留 current + 最近 N 版，其余 purge |
| 审计 | activate/rollback/gc 写 `_audit` |

### 1.3 分层范围（按成熟度/成本分层交付）

| 层级 | 内容 | 成本 | 本期 |
|------|------|------|------|
| **L1 增强既有** | 语义化版本号、版本快照永久保留、答案溯源、回滚原因记录 | 低 | ✅ 做 |
| **L2 影子验证上线** | 复用 shadow 机制：新版构建→后台召回测试验证→activate 全量。替代"流量灰度" | 低-中 | ✅ 做 |
| **L3 真·流量灰度** | 按用户群/渠道/坐席组路由分流、自动 A/B 效果对比、Milvus 别名双写切换 | 很高 | ⏸ 后置，独立立项 |

> **关键决策**：本方案用 **L2 影子验证替代流量灰度**。真灰度（L3）代价远超边际价值，且依赖完整用户体系 + 埋点平台（与现有"无登录访客 token"架构冲突），后置为独立项目。

---

## 二、L1：版本快照增强

### 2.1 语义化版本号

现有 `DocumentVersionModel.version_no` 是单调整数，扩展为 `major.minor`：

```python
# DocumentVersionModel 增
major: Mapped[int] = mapped_column(Integer, default=1)
minor: Mapped[int] = mapped_column(Integer, default=0)
# version_no 保留作全局单调序（内部排序/GC 用），major.minor 作业务语义版本
```

- **升主版本（major）**：规则、政策重大修订（运营在上传时显式标记 `bump=major`）。
- **升次版本（minor）**：文案优化、错漏修正（默认）。

### 2.2 快照永久保留 vs 向量 GC（成本权衡）

「永不覆盖旧版本」与现有 `apply_retention`（GC 老版本向量）的冲突，按**冷热分离**解决：

| 数据 | 策略 | 理由 |
|------|------|------|
| 版本元数据快照（关系库：内容摘要、操作人、时间、审批、major.minor） | **永久保留** | 便宜，满足追溯/审计 |
| 原文（MinIO） | **永久保留** | 回滚老版本时可重建向量 |
| 向量索引（Milvus） | **按保留策略 GC** | 贵；回滚老版本是低频操作，需要时从原文重建 |

> 回滚到已 GC 向量的老版本时：从 MinIO 原文重新切片+embedding 重建该版本向量，再 activate。用"原文永久 + 向量可重建"换"向量永久"的存储成本。

### 2.3 新增表：版本快照（追溯/审批）

```python
class KnowledgeVersionSnapshotModel(Base):
    __tablename__ = "knowledge_version_snapshots"

    id: Mapped[str]           = mapped_column(String(24), primary_key=True)
    tenant_id: Mapped[str]    = mapped_column(String(32), default="default", index=True)
    doc_id: Mapped[str]       = mapped_column(String(16), index=True)
    version_id: Mapped[str]   = mapped_column(String(24), index=True)  # 关联 DocumentVersionModel
    major: Mapped[int]        = mapped_column(Integer)
    minor: Mapped[int]        = mapped_column(Integer)
    content_summary: Mapped[str] = mapped_column(Text, default="")     # 修改前后摘要/diff 摘要
    actor: Mapped[str]        = mapped_column(String(64))
    approval_status: Mapped[str] = mapped_column(String(16), default="none")  # none|pending|approved|rejected
    approver: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
```

- 审批流第一版可做**轻量**：`approval_status` 默认 `none`（无需审批直接生效）；需要审批的库再开 `pending→approved` 流转。重型审批工作流后置。

### 2.4 L1：答案溯源（最高性价比）

LLM 每次答复绑定引用的「知识 ID + 版本号」，凭会话 ID 可回溯当时用的版本，客诉时快速定责。

**实现链路**（多处都已基本具备，本期把字段串起来）：

```
检索返回 SearchResult  → 带 kb_id + doc_id + version_no（major.minor）
        ↓
agent 把引用的 [知识ID:版本号] 记入：
   ① 会话消息的 metadata（持久化，凭 session_id 可查）
   ② Langfuse trace（已有全链路追踪，挂上溯源标签）
        ↓
前端答复可展示引用来源；后台凭 session_id 定位当时知识版本
```

- `SearchResult` 在检索期已透出 `kb_id/doc_id/version_no`（见检索方案 §十三），本期 agent 侧消费。
- 复用现有 [Langfuse 全链路追踪]，溯源信息作为 span 属性，无需新建追踪设施。

---

## 三、L2：影子验证上线（替代流量灰度）

### 3.1 思路

复用现有 shadow 机制，把「上线前验证」做成显式流程，拿到灰度 90% 的价值（上线前可验证、不影响线上、可回滚），却几乎零新增基础设施——**不需要流量路由、不需要用户分群、不需要 A/B 统计平台**。

### 3.2 流程

```
1. 上传新版本 → shadow 构建（status=ready，current 指针未动，线上仍走旧版）
2. 管理后台「待验证版本」列表展示该 shadow 版本
3. 运营用召回测试（SearchTester）对 shadow 版本验证：
     - 输入典型问题，检索 mode 指向该 shadow version_id
     - 对比新旧版本召回结果、答案质量
4. 验证通过 → 点「上线」→ activate（原子翻指针，全量生效）
5. 不通过 → 丢弃 shadow 版本（不影响线上）
```

### 3.3 需要的小改动

| 改动 | 说明 |
|------|------|
| 检索支持指定 version_id | 召回测试可定向查某个 shadow 版本（现有 `visible_version_ids` 过滤改为可显式传入待验证版本） |
| 管理后台「待验证版本」视图 | 列 shadow 版本 + 召回测试入口 + 上线/丢弃按钮（F 期 UI） |
| activate 入口暴露到 API/后台 | 现有 `activate()` 已实现，补 REST 入口 + 审计 |

> 与真灰度的区别：影子验证是「**全量切换前的人工验证**」，不分流线上流量；真灰度是「**部分线上流量先用新版**」。前者零流量治理成本，后者需要完整分群+埋点。

---

## 四、L1：一键回滚增强

现有 `rollback(version_no)` 已具备核心能力，本期补：

- **回滚原因 + 影响范围记录**：扩 `_audit`，记 `reason`、影响的 doc 范围。
- **缓存刷新**：若检索结果有缓存层，回滚后失效相关缓存（无缓存则不适用）。
- **管理后台一键按钮**：版本列表 → 选历史版本 → 一键回滚（F 期 UI）。
- **向量已 GC 的版本**：回滚时先从 MinIO 原文重建该版本向量，再 activate（见 §2.2）。

---

## 五、L3：真·流量灰度（后置，独立立项）

**本期不做**，列此说明真实代价与前置依赖，避免低估：

| 子能力 | 前置依赖 | 代价 |
|--------|---------|------|
| 按用户群/渠道/坐席组路由分流 | **完整用户体系**（现为无登录访客 token，无稳定分群身份） | 高 |
| 自动 A/B 效果对比（解决率/满意度/转人工率） | **会话结果埋点 + 标注 + 统计显著性平台** | 很高 |
| Milvus 别名双写切换 | 检索层引入别名间接层 + 临时 collection 双写 | 中-高 |

> L3 应作为独立项目立项，前置是「用户体系」+「客服数据埋点平台」两块基建。在它们就绪前，L2 影子验证已满足"更新不出错、可回退"的核心诉求。

---

## 六、实施步骤（连续执行，不停顿）

| 步 | 内容 | 验证点 |
|----|------|--------|
| 1 | DocumentVersionModel 加 major.minor + 上传时 bump 标记；版本快照表 | 升版本号正确；快照落库 |
| 2 | 快照永久保留 + 向量按策略 GC + 回滚时原文重建向量 | GC 后回滚老版本可重建 |
| 3 | 答案溯源：SearchResult 透出版本号 → agent 绑定到会话 + Langfuse | 凭 session_id 查到当时知识版本 |
| 4 | 影子验证：检索支持指定 version_id + activate REST 入口 + 审计 | 召回测试可验 shadow 版本，上线/丢弃生效 |
| 5 | 回滚增强：原因/影响记录 + 缓存刷新 | 回滚留痕完整 |
| 6 | 管理后台 UI：待验证版本、版本历史、一键回滚、溯源展示 | 后台全流程可操作（与 F 期 UI 合并） |

---

## 七、风险与约束

1. **依赖检索期完成**：本方案建立在检索期的 collection-per-kb + SearchResult 版本号透出之上，需检索期先落地。
2. **向量重建成本**：回滚已 GC 版本要重跑 embedding，大文档有延迟；可异步 + 进度提示。
3. **审批流先轻量**：第一版 `approval_status` 默认无审批，重型审批工作流（多级审批、通知）后置。
4. **L3 严禁混入**：流量灰度的前置（用户体系/埋点）未就绪前不启动，避免做半套。
5. **溯源依赖 Langfuse**：答案溯源复用现有追踪；若 Langfuse 不可用需有降级（至少落会话 metadata）。

# Prompt 工程管理方案

## 一、概述

Prompt 是多 Agent 系统的"灵魂"——Supervisor 路由准确率、Agent 回复质量、反思评估效果全部依赖 prompt 质量。本文档定义 prompt 的管理、版本控制、评估、发布策略。

---

## 二、Prompt 分类

| 类型 | 用途 | 示例 | 变更频率 |
|------|------|------|----------|
| **System Prompt** | Agent 角色定义和行为约束 | Supervisor 路由 prompt、Order Agent 角色 prompt | 低（周级） |
| **Skill Prompt** | Skill 内部步骤指引 | 退款校验提示、投诉安抚话术 | 中（天级） |
| **Judge Prompt** | 反思评估标准 | 质量评分维度、合规检查规则 | 低 |
| **Tool Description** | 工具描述（给 LLM 看） | `search_faq` 的 description | 低 |
| **Dynamic Prompt** | 运行时拼装（记忆注入、错误记忆） | 用户画像片段、历史错误禁止规则 | 实时 |

---

## 三、Prompt 注册表（Registry）

### 3.1 设计原则

- 所有 prompt 集中管理，不散落在代码各处
- 每个 prompt 有唯一 ID、版本号、元数据
- 支持按环境加载不同版本（dev/staging/prod）
- 变更有审计日志

### 3.2 存储方案

```
方案 A：文件系统（推荐 MVP 阶段）
  prompts/
  ├── supervisor/
  │   ├── system.md          # 当前版本
  │   └── system.v1.md       # 历史版本
  ├── agents/
  │   ├── faq_system.md
  │   ├── order_system.md
  │   └── complaint_system.md
  └── reflection/
      └── judge.md

方案 B：数据库（推荐生产阶段）
  PostgreSQL prompt_templates 表
  + Redis 缓存热加载
  + 管理后台 UI 编辑
```

### 3.3 数据模型

```python
# src/prompts/models.py

class PromptTemplate(BaseModel):
    id: str                          # "supervisor_system"
    name: str                        # "Supervisor 路由 Prompt"
    category: str                    # "system" / "skill" / "judge" / "tool_desc"
    agent: str | None                # 归属 Agent
    content: str                     # prompt 内容（支持 Jinja2 变量）
    version: int                     # 版本号
    variables: list[str]             # 模板变量列表
    metadata: dict                   # 标签、作者、备注
    status: str                      # "draft" / "active" / "deprecated"
    created_at: datetime
    updated_at: datetime

class PromptVersion(BaseModel):
    prompt_id: str
    version: int
    content: str
    change_reason: str               # 变更原因
    author: str
    eval_score: float | None         # 评估分数
    created_at: datetime
```

### 3.4 Prompt Registry 实现

```python
# src/prompts/registry.py

class PromptRegistry:
    """Prompt 注册表：集中加载和管理所有 prompt"""

    def __init__(self, store: PromptStore, cache: Redis):
        self.store = store
        self.cache = cache

    async def get(self, prompt_id: str, variables: dict | None = None) -> str:
        """获取 prompt（带缓存 + 变量渲染）"""
        # 1. 尝试缓存
        cached = await self.cache.get(f"prompt:{prompt_id}")
        if cached:
            template = cached
        else:
            # 2. 从 store 加载当前 active 版本
            prompt = await self.store.get_active(prompt_id)
            template = prompt.content
            await self.cache.setex(f"prompt:{prompt_id}", 300, template)

        # 3. 渲染变量
        if variables:
            template = self._render(template, variables)
        return template

    async def update(self, prompt_id: str, content: str, reason: str, author: str):
        """更新 prompt（自动创建新版本）"""
        current = await self.store.get_active(prompt_id)
        new_version = current.version + 1

        # 保存新版本
        await self.store.create_version(PromptVersion(
            prompt_id=prompt_id,
            version=new_version,
            content=content,
            change_reason=reason,
            author=author,
        ))

        # 更新 active 版本
        await self.store.set_active_version(prompt_id, new_version)

        # 清缓存
        await self.cache.delete(f"prompt:{prompt_id}")

    async def rollback(self, prompt_id: str, target_version: int):
        """回滚到指定版本"""
        await self.store.set_active_version(prompt_id, target_version)
        await self.cache.delete(f"prompt:{prompt_id}")

    def _render(self, template: str, variables: dict) -> str:
        """Jinja2 变量渲染"""
        from jinja2 import Template
        return Template(template).render(**variables)
```

---

## 四、核心 Prompt 设计

### 4.1 Supervisor 路由 Prompt

```markdown
# prompts/supervisor/system.md

你是智能客服路由器。根据用户消息判断意图并选择最合适的处理 Agent。

## 路由规则
- **faq**：产品功能、使用方法、政策咨询等常见问题
- **order**：订单相关（查询、修改、退款、取消、物流追踪）
- **complaint**：投诉、不满、要求赔偿、情绪激动
- **tech_support**：产品故障、技术问题、报错信息、操作异常
- **human**：用户明确要求人工、涉及隐私敏感问题、你无法判断意图

## 判断要点
1. 优先识别用户的核心诉求，忽略寒暄
2. 如果消息包含情绪词（"太差了"/"气死了"/"投诉"）→ complaint
3. 包含订单号或明确提到订单操作 → order
4. 用户在追问上一轮问题 → 保持当前 Agent 不切换
5. 无法确定 → human（宁可转人工，不要错误路由）

## 输出格式
返回 JSON：{"intent": "xxx", "confidence": 0.9, "reason": "一句话解释"}
```

### 4.2 Agent System Prompt 模板

```markdown
# prompts/agents/order_system.md

你是专业的订单客服，负责处理用户的订单相关问题。

## 身份
- 你是 {{company_name}} 的订单客服专员
- 态度友好专业，表达简洁清晰

## 能力范围
- 可以做：查询订单、查物流、申请退款、修改地址、催发货
- 不能做：修改价格、跳过审批直接退款、承诺不确定的时间

## 工作规范
1. 操作前先确认用户身份和订单信息
2. 退款需告知用户预计到账时间
3. 超出能力范围的问题，说明原因并建议转人工
4. 不编造信息，工具未返回的数据不要自己猜

{{#if memory_context}}
## 用户信息
{{memory_context}}
{{/if}}

{{#if error_context}}
## 注意事项
{{error_context}}
{{/if}}
```

### 4.3 Judge 评估 Prompt

```markdown
# prompts/reflection/judge.md

你是客服质量评估专家。严格评估以下 Agent 回复。

## 评估维度（每项 0-10 分）
1. **准确性**：是否基于工具返回的真实数据，有无编造
2. **完整性**：是否回答了用户的全部问题
3. **安全性**：有无承诺超出权限的操作、泄露内部信息
4. **合规性**：是否礼貌专业，有无推诿或敷衍
5. **可操作性**：是否给出明确的下一步指引

## 严重问题（直接判 FAIL）
- 编造数据（工具没返回的信息）
- 承诺无法做到的事
- 泄露内部系统信息
- 对用户不礼貌

## 输入
- 用户问题：{{user_message}}
- 工具调用及结果：{{tool_results}}
- Agent 回复：{{agent_response}}

## 输出 JSON
{"score": 8.5, "passed": true, "issues": [], "suggestion": ""}
```

---

## 五、Prompt 版本控制与发布

### 5.1 发布流程

```
编辑 Prompt (draft)
  │
  ▼
本地评测（自动化评估 + 人工抽查）
  │
  ├─ 评测通过 → 发布到 staging
  │                 │
  │                 ▼
  │            staging 验证（真实流量灰度 5%）
  │                 │
  │                 ├─ 指标正常 → 发布到 production
  │                 │
  │                 └─ 指标下降 → 回滚
  │
  └─ 评测不通过 → 继续修改
```

### 5.2 灰度发布（A/B Testing）

```python
# src/prompts/ab_testing.py

class PromptABTest(BaseModel):
    id: str
    prompt_id: str
    variant_a: int          # 版本号 A（当前版本）
    variant_b: int          # 版本号 B（新版本）
    traffic_ratio: float    # B 的流量比例（0.05 = 5%）
    metrics: dict           # 跟踪指标
    status: str             # "running" / "concluded"
    start_time: datetime
    end_time: datetime | None

class PromptABRouter:
    async def get_variant(self, prompt_id: str, session_id: str) -> int:
        """根据 session_id 决定使用哪个版本"""
        test = await self.get_active_test(prompt_id)
        if not test:
            return await self.registry.get_active_version(prompt_id)

        # 基于 session_id hash 分流，保证同一会话用同一版本
        hash_val = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
        if (hash_val % 100) / 100 < test.traffic_ratio:
            return test.variant_b
        return test.variant_a
```

---

## 六、Prompt 评估

### 6.1 自动化评估

| 评估维度 | 方法 | 工具 |
|----------|------|------|
| **意图准确率** | 标注测试集 + 批量跑 Supervisor | pytest + golden dataset |
| **回复质量** | Judge prompt 自动评分 | Langfuse Evaluation |
| **格式合规** | 检查输出是否符合 JSON Schema | Pydantic 校验 |
| **安全合规** | 检查是否触发禁止项 | 规则引擎 |
| **对比评估** | A/B 两版本回复让 LLM 打分 | Pairwise comparison |

### 6.2 评估数据集

```python
# tests/eval/golden_dataset.py

ROUTING_TEST_CASES = [
    {"input": "我的订单到哪了", "expected_intent": "order", "tags": ["物流"]},
    {"input": "你们产品太垃圾了 退钱", "expected_intent": "complaint", "tags": ["情绪"]},
    {"input": "怎么绑定银行卡", "expected_intent": "faq", "tags": ["操作指引"]},
    {"input": "我要找你们经理", "expected_intent": "human", "tags": ["转人工"]},
    # ... 200+ cases
]

async def eval_routing_accuracy(prompt_version: int) -> float:
    """评估路由 prompt 的意图准确率"""
    correct = 0
    for case in ROUTING_TEST_CASES:
        result = await supervisor.route(case["input"], prompt_version=prompt_version)
        if result["intent"] == case["expected_intent"]:
            correct += 1
    return correct / len(ROUTING_TEST_CASES)
```

### 6.3 评估阈值

| 指标 | 发布门槛 | 回滚触发 |
|------|----------|----------|
| 路由准确率 | >= 95% | < 90% |
| 回复质量均分 | >= 7.5 / 10 | < 6.5 |
| 格式合规率 | >= 99% | < 95% |
| 安全合规率 | 100% | < 100% |

---

## 七、Token 预算管理

每个 Agent 的 prompt 有 token 预算，防止注入记忆后 prompt 过长：

```python
# src/prompts/budget.py

TOKEN_BUDGETS = {
    "supervisor_system": 800,      # Supervisor 简洁
    "order_system": 1500,          # Agent prompt + 记忆
    "complaint_system": 2000,      # 投诉场景需要更多上下文
    "judge": 1000,                 # 评估 prompt
}

class PromptBudget:
    def fit_to_budget(self, prompt_id: str, base_prompt: str, dynamic_parts: dict) -> str:
        """确保最终 prompt 不超 token 预算"""
        budget = TOKEN_BUDGETS.get(prompt_id, 1500)
        base_tokens = count_tokens(base_prompt)
        remaining = budget - base_tokens

        # 动态部分按优先级截断
        priority_order = ["error_context", "memory_context", "history_context"]
        assembled = base_prompt
        for key in priority_order:
            if key in dynamic_parts:
                part = dynamic_parts[key]
                part_tokens = count_tokens(part)
                if part_tokens <= remaining:
                    assembled += f"\n\n{part}"
                    remaining -= part_tokens
                else:
                    # 截断
                    truncated = truncate_to_tokens(part, remaining - 50)
                    assembled += f"\n\n{truncated}\n[...已截断]"
                    break
        return assembled
```

---

## 八、目录结构

```
src/prompts/
├── __init__.py
├── registry.py              # Prompt 注册表
├── models.py                # 数据模型
├── store.py                 # 持久化（PG / 文件）
├── budget.py                # Token 预算管理
├── ab_testing.py            # A/B 测试
└── evaluator.py             # 自动化评估

prompts/                     # Prompt 模板文件（版本控制）
├── supervisor/
│   └── system.md
├── agents/
│   ├── faq_system.md
│   ├── order_system.md
│   ├── complaint_system.md
│   ├── tech_support_system.md
│   └── human_handoff_system.md
├── skills/
│   ├── refund_guide.md
│   └── complaint_handling.md
└── reflection/
    ├── judge.md
    └── self_check.md

tests/eval/
├── golden_dataset.py        # 标注测试集
├── eval_routing.py          # 路由评估
├── eval_quality.py          # 回复质量评估
└── eval_safety.py           # 安全合规评估
```

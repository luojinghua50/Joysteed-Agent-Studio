# 模型稳定性输出工程化方案

## 一、概述

本文档定义智能客服多 Agent 系统的稳定性工程方案，确保模型输出可控、系统不死循环、故障可恢复、成本不失控。

### 7 大稳定性策略

```
┌─────────────────────────────────────────────────────────────────┐
│                    稳定性工程化方案                                │
├─────────────────────────────────────────────────────────────────┤
│ 1. 循环保护（防死循环）                                           │
│ 2. 超时控制（多层级超时）                                         │
│ 3. 重试策略（Tool / LLM / MCP 分层重试）                         │
│ 4. 输出格式保障（结构化输出 + 校验 + 修复）                        │
│ 5. 降级兜底（Fallback 链 + 模板回复）                             │
│ 6. 幂等与去重（防止重复操作）                                      │
│ 7. 资源限制（Token / 并发 / 成本熔断）                            │
└─────────────────────────────────────────────────────────────────┘
```

### 开关设计原则

所有保护机制均支持开关，分三层粒度：

```
全局总开关 (enabled)
 └── 模块开关（loop_protection_enabled / timeout_enabled / ...）
      └── 细项开关 + 参数（每个具体策略可独立关闭/调参）
```

---

## 二、统一配置

```python
# src/config.py

class StabilityConfig(BaseSettings):
    """稳定性工程配置 — 所有开关和参数"""

    # ====== 全局总开关 ======
    enabled: bool = True  # False = 关闭所有保护（仅开发环境）

    # ====== 1. 循环保护 ======
    loop_protection_enabled: bool = True
    max_tool_calls_per_turn: int = 10
    max_routing_loops: int = 3
    max_agent_hops: int = 5
    max_skill_steps: int = 8
    max_reflection_retries: int = 2

    # ====== 2. 超时控制 ======
    timeout_enabled: bool = True
    session_timeout: float = 60.0
    agent_timeout: float = 30.0
    llm_timeout: float = 15.0
    tool_timeout: float = 10.0
    reflection_timeout: float = 10.0

    # ====== 3. 重试策略 ======
    retry_enabled: bool = True
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    tool_max_retries: int = 2
    tool_retry_delay: float = 1.0
    output_parse_retries: int = 2

    # ====== 4. 输出格式保障 ======
    output_guard_enabled: bool = True
    output_parse_max_retries: int = 2
    json_repair_enabled: bool = True

    # ====== 5. 降级兜底 ======
    fallback_enabled: bool = True
    fallback_to_template: bool = True
    auto_handoff_on_failure: bool = True
    max_failures_before_handoff: int = 2

    # ====== 6. 幂等去重 ======
    idempotency_enabled: bool = True
    idempotency_ttl_seconds: int = 300

    # ====== 7. 资源限制 ======
    resource_limit_enabled: bool = True

    # 单会话限制
    session_limits_enabled: bool = True
    max_tokens_per_session: int = 50_000
    max_llm_calls_per_session: int = 20
    max_session_duration: float = 600.0

    # 用户级限流
    user_rate_limit_enabled: bool = True
    max_requests_per_user_per_minute: int = 10
    vip_rate_multiplier: float = 3.0  # VIP 用户倍率

    # 全局成本熔断
    cost_circuit_breaker_enabled: bool = True
    max_cost_per_hour: float = 50.0
    max_cost_per_day: float = 500.0
```

### 多环境配置

```yaml
# .env.dev — 开发环境：宽松
STABILITY_ENABLED=true
RESOURCE_LIMIT_ENABLED=false
TIMEOUT_ENABLED=true
SESSION_TIMEOUT=300
COST_CIRCUIT_BREAKER_ENABLED=false

# .env.staging — 测试环境：接近生产
STABILITY_ENABLED=true
RESOURCE_LIMIT_ENABLED=true
MAX_COST_PER_HOUR=10
USER_RATE_LIMIT_ENABLED=false

# .env.prod — 生产环境：全部启用
STABILITY_ENABLED=true
RESOURCE_LIMIT_ENABLED=true
COST_CIRCUIT_BREAKER_ENABLED=true
MAX_COST_PER_HOUR=50
MAX_COST_PER_DAY=500
```

---

## 三、循环保护（防死循环）

### 3.1 防护点一览

| 防护点 | 限制 | 触发后行为 |
|--------|------|-----------|
| Tool Calling Loop | 单轮最大 **10 次** | 强制结束，用已有信息生成回复 |
| Supervisor 路由循环 | 同一会话最多重路由 **3 次** | 转人工 |
| 反思重试 | 最多 **2 次** | 接受当前结果输出 |
| Agent 间跳转 | 同一会话最大跳转 **5 次** | 强制结束，提示用户 |
| SubGraph Skill | 内部节点最大执行 **8 步** | 中止 Skill，返回部分结果 |

### 3.2 实现

`LoopGuard` 是嵌入每个执行单元的计数器式保护器：上半部分是各类循环的上限阈值（工具调用、路由重入、Agent 跳转、Skill 步数、反思重试），下半部分是运行时累计的计数字段。

```python
# src/guardrails/loop_protection.py

@dataclass
class LoopGuard:
    """循环保护器，嵌入每个执行单元"""
    # ── 阈值（可配置）──
    max_tool_calls: int = 10
    max_routing_loops: int = 3
    max_reflection_retries: int = 2
    max_agent_hops: int = 5
    max_skill_steps: int = 8

    # ── 运行时计数（init=False）──
    tool_call_count: int = field(default=0, init=False)
    routing_count: int = field(default=0, init=False)
    agent_hop_count: int = field(default=0, init=False)

    def check_tool_call(self) -> bool: ...       # 计数+1，未超限返回 True
    def check_routing(self) -> bool: ...
    def check_agent_hop(self) -> bool: ...
    def should_force_end(self) -> tuple[bool, str]: ...  # 任一计数越界即 (True, 原因)
    def reset(self): ...                          # 每轮开始清零
```

各 `check_*` 方法在对应事件发生时自增计数并返回是否仍在阈值内；`should_force_end` 供执行流程在每步后查询是否有任一维度越界（返回越界原因用于日志与降级路由）；`reset` 在每轮对话开始时清零，避免跨轮累加。

---

## 四、超时控制（多层级）

### 4.1 超时层级

```
┌─ 全局会话超时 (60s) ──────────────────────────────────────────┐
│  ┌─ 单 Agent 超时 (30s) ───────────────────────────────────┐  │
│  │  ┌─ 单次 LLM 调用超时 (15s) ─────────────────────────┐  │  │
│  │  │                                                    │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │  ┌─ 单次 Tool/MCP 调用超时 (10s) ────────────────────┐  │  │
│  │  │                                                    │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 4.2 实现

四层超时阈值集中在一个配置对象里，执行侧则统一收敛到一个基于 `asyncio.wait_for` 的包装器：

```python
# src/guardrails/timeout.py

@dataclass
class TimeoutConfig:
    session_timeout: float = 60.0
    agent_timeout: float = 30.0
    llm_timeout: float = 15.0
    tool_timeout: float = 10.0
    reflection_timeout: float = 10.0

class AgentTimeoutError(Exception): ...

# 通用超时包装器：超时后有 fallback 则走降级，否则抛 AgentTimeoutError
async def with_timeout(coro, timeout: float, fallback_fn=None): ...
```

关键设计在于超时不是「抛异常了事」，而是超时即降级：`with_timeout` 捕获 `asyncio.TimeoutError` 后，若传入了 `fallback_fn` 则执行降级分支，否则才抛出 `AgentTimeoutError` 交由上层处理。Agent 级超时的降级返回一条按意图选择的安抚话术并置 `resolved=False`，把「卡住」转成「一次可解释的失败 + 转人工引导」，而不是让请求悬挂到会话超时。

---

## 五、重试策略（分层）

### 5.1 重试分层

| 层 | 重试对象 | 策略 | 最大次数 | 退避 |
|----|----------|------|----------|------|
| **LLM 调用** | 5xx / 超时 / 限流 | 指数退避 + Fallback 模型 | 3 次 | 1s → 2s → 4s |
| **MCP Tool** | MCP Server 返回错误 / 超时 | 固定间隔重试 | 2 次 | 1s |
| **业务工具** | 业务系统返回可重试错误 | 区分可重试/不可重试 | 2 次 | 0.5s |
| **输出解析** | LLM 输出格式错误 | 带错误提示重新调用 | 2 次 | 0 |

### 5.2 错误分类

重试的前提是先把异常分成三类，让「重试 / 直接放弃 / 切降级」三条路径互斥且明确：

```python
# src/guardrails/retry.py

class RetryCategory(Enum):
    RETRYABLE = "retryable"          # 可重试：超时、5xx、限流
    NON_RETRYABLE = "non_retryable"  # 不可重试：4xx、业务逻辑错误
    FALLBACK = "fallback"            # 需要降级：模型不可用

def classify_error(error: Exception) -> RetryCategory: ...
```

分类规则：限流、超时、5xx 归为可重试；模型不可用归为需降级（触发 Fallback 模型）；其余（含 4xx、业务逻辑错误）一律不可重试，直接上抛。这样避免了对不可重试错误做无谓退避、也避免了对可降级错误反复重试。

### 5.3 重试执行器

执行器把「分类 + 退避 + 降级」串成一个通用函数，业务侧只需提供待执行的协程和降级分支：

```python
# src/guardrails/retry.py

async def retry_with_backoff(
    fn,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    fallback_fn=None,
): ...

async def call_mcp_tool_with_retry(client, tool_name: str, args: dict, config): ...
```

`retry_with_backoff` 每次失败先调 `classify_error`：不可重试立即上抛（不浪费退避时间），可降级且有 `fallback_fn` 则直接切降级，其余按 `base_delay * 2^attempt`（封顶 `max_delay`）指数退避后再试，全部耗尽才抛最后一次错误。`call_mcp_tool_with_retry` 是它在 MCP 场景的封装：把 `client.call_tool` 包上 `tool_timeout` 超时，再交给退避执行器，降级分支返回一条结构化的「工具暂不可用」结果，让 Agent 能带着这个信号继续生成回复而非崩溃。

---

## 六、输出格式保障

### 6.1 策略层次

| 策略 | 说明 | 触发条件 |
|------|------|----------|
| **Structured Output** | 模型原生结构化输出（强制 JSON Schema） | 默认使用 |
| **输出校验** | Pydantic 模型校验返回结构 | 结构化输出后自动校验 |
| **修复解析** | JSON 格式小错误自动修复 | 校验失败时 |
| **重试带提示** | 把错误信息反馈给 LLM 重新生成 | 修复失败时 |
| **Fallback 解析** | 退化为自由文本提取 | 所有结构化手段失败时 |
| **内容合规检查** | 不能承诺做不到的事、不能泄露敏感信息 | 最终输出前 |

### 6.2 实现

`OutputGuard` 暴露两个能力：带重试的结构化解析，和不依赖 LLM 的内容合规检查。

```python
# src/guardrails/output.py

class OutputParseError(Exception): ...

class OutputGuard:
    """输出格式保障"""

    def __init__(self, config: StabilityConfig): ...

    # 带重试的结构化输出：解析失败→带错误提示回灌重试→JSON 修复兜底
    async def parse_with_retry(self, llm, messages: list, schema: type[BaseModel]) -> BaseModel: ...

    # 规则式修复常见 JSON 格式问题（markdown 包裹/单引号/末尾逗号）
    def _attempt_json_repair(self, raw: str, schema: type[BaseModel]) -> BaseModel: ...

    # 纯规则引擎的内容合规检查，返回 (是否通过, 违规原因)
    async def content_compliance_check(self, response: str, state: dict) -> tuple[bool, str]: ...
```

`parse_with_retry` 的设计要点是「把失败反馈回灌给模型」：结构化输出校验失败（`ValidationError` / `JSONDecodeError`）时，不是简单重试，而是把上一次的原始输出和具体错误拼进消息再让 LLM 重生成，逼它对齐 Schema；重试仍失败才退到 `_attempt_json_repair` 做规则级修复（剥离 markdown 代码块包裹、单引号转双引号、去掉末尾多余逗号），修复不成才抛 `OutputParseError`。

`content_compliance_check` 刻意用规则引擎而非再调一次 LLM，避免在输出前引入额外延迟和不确定性。它拦三类问题：无依据的时间承诺（如「保证 X 小时内」）、泄露内部系统信息（traceback、Exception 等）、以及「拒绝回答却不给替代方案」。任一命中即返回失败与拼接后的违规原因。

---

## 七、降级兜底策略

### 7.1 降级链

```
正常流程失败时的降级链：

LLM 主模型调用失败
  → LiteLLM Fallback 模型自动切换
    → 模板回复（基于意图匹配预置模板）
      → 转人工

Agent 执行异常
  → 重试一次
    → 返回安全的通用回复 + 转人工建议
      → 直接转人工

MCP Tool 调用失败
  → 重试 2 次
    → 返回 "该服务暂时不可用" 给 Agent
      → Agent 用已有信息生成回复
        → 降级到模板回复
```

### 7.2 兜底模板

兜底模板按意图分桶，每个意图给一条既安抚又指明下一步（重试 / 帮助中心 / 转人工）的话术，`default` 兜住未匹配的意图：

```python
# src/guardrails/fallback.py

FALLBACK_TEMPLATES = {
    "faq": "...",           # 引导查帮助中心 / 转人工
    "order": "...",         # 订单系统繁忙，稍后重试 / 转人工
    "complaint": "...",     # 致歉 + 已记录 + 转专属客服
    "tech_support": "...",  # 系统繁忙 + 技术支持热线
    "default": "...",       # 兜底：直接转人工
}
```

### 7.3 降级处理器

降级处理器把「记日志 → 选模板 → 判是否转人工 → 组装回复」收敛成统一入口：

```python
# src/guardrails/fallback.py

class FallbackHandler:
    def __init__(self, config: StabilityConfig): ...

    # 统一故障降级入口：返回带模板回复 + resolved/转人工路由的 state 增量
    async def handle_failure(self, state: CustomerState, error: Exception, context: str) -> dict: ...

    # 是否转人工的判定
    def _should_handoff(self, error: Exception, state: dict) -> bool: ...

    # 结构化记录故障到可观测性系统
    async def _log_failure(self, state: dict, error: Exception, context: str): ...
```

`handle_failure` 先落一条结构化故障日志（保留 session/intent/agent/错误类型等，便于事后归因），再按意图取模板；若判定需转人工且开启了 `auto_handoff_on_failure`，追加转接话术并把 `current_agent` 改写为 `human_handoff`、`resolved` 置否，让图的路由据此走向人工节点。

转人工的判定有两条触发线：连续失败次数达到 `max_failures_before_handoff`（避免在同一故障上反复空转），以及投诉意图一旦失败必须转人工（这类场景对体验最敏感，不容降级话术兜底）。

---

## 八、幂等与去重

### 8.1 设计目标

防止 Tool Calling Loop 中因重试或循环导致重复执行副作用操作（如重复退款、重复创建工单）。

### 8.2 受保护的操作

| 工具 | 风险 | 保护方式 |
|------|------|----------|
| apply_refund | 重复退款 | 幂等键 = session_id + order_id + amount |
| cancel_order | 重复取消 | 幂等键 = session_id + order_id |
| create_ticket | 重复创建工单 | 幂等键 = session_id + 用户描述 hash |
| modify_order | 重复修改 | 幂等键 = session_id + order_id + 修改内容 hash |

### 8.3 实现

幂等保护只作用于一份显式的副作用工具白名单，键由「会话 + 工具名 + 排序后的参数」哈希而成，结果缓存在 Redis 带 TTL：

```python
# src/guardrails/idempotency.py

class IdempotencyGuard:
    """防止同一会话中重复执行副作用操作"""

    PROTECTED_TOOLS = {"apply_refund", "cancel_order", "create_ticket", "modify_order"}

    def __init__(self, redis_client, config: StabilityConfig): ...

    # 幂等键 = sha256(session_id:tool_name:sorted(args))
    def _make_key(self, session_id: str, tool_name: str, args: dict) -> str: ...

    # 返回 (is_duplicate, cached_result)：命中则直接回放上次结果
    async def check_and_mark(self, session_id: str, tool_name: str, args: dict) -> tuple[bool, dict | None]: ...

    # 执行成功后写入结果，供 TTL 窗口内的后续调用去重
    async def store_result(self, session_id: str, tool_name: str, args: dict, result: dict): ...
```

设计取舍有三点：一是只保护白名单内的副作用工具，只读查询不进幂等路径以免无谓开销；二是幂等键对参数做 `sort_keys` 后哈希，保证参数顺序无关、同一语义的调用命中同一键；三是用 TTL（默认 300s）而非永久去重，把保护范围限定在「同一会话的短时重试/循环」窗口内，避免跨会话或长期误判。`check_and_mark` 命中缓存即回放上次结果，让重复的退款/建单调用变成幂等读。

---

## 九、资源限制（成本熔断）

### 9.1 限制层级

| 层级 | 限制项 | 默认阈值 | 触发后行为 |
|------|--------|----------|-----------|
| **会话级** | Token 消耗 | 50,000 tokens | 强制结束，模板回复 |
| | LLM 调用次数 | 20 次 | 强制结束 |
| | 会话持续时间 | 10 分钟 | 提示新开会话 |
| **用户级** | 每分钟请求数 | 10 次（VIP ×3） | 限流，返回提示 |
| **全局** | 每小时成本 | $50 | 全局降级到缓存/模板 |
| | 每日成本 | $500 | 全局降级 + 告警 |

### 9.2 实现

资源限制以 Redis 计数器为后端，统一入口 `check_all` 按「用户级 → 会话级 → 全局」顺序短路检查，任一超限即带原因拒绝：

```python
# src/guardrails/resource_limit.py

class ResourceGuard:
    def __init__(self, config: StabilityConfig, redis_client): ...

    # 统一资源检查入口，返回 (allowed, reason)
    async def check_all(self, session_id: str, user_id: str, is_vip: bool = False) -> tuple[bool, str]: ...

    async def _check_user_rate(self, user_id: str, is_vip: bool) -> bool: ...      # rate_limit
    async def _check_session_limits(self, session_id: str) -> bool: ...            # session_limit
    async def _check_global_cost(self) -> bool: ...                               # cost_limit

    # 每次 LLM 调用后记账（token / 调用次数 / 全局成本）
    async def record_usage(self, session_id: str, tokens: int, cost: float): ...
```

三层检查各有机制：用户级限流用 `INCR` + 首次 `EXPIRE 60s` 实现滑动分钟窗口，VIP 用户按 `vip_rate_multiplier` 放大阈值；会话级同时卡 LLM 调用次数和累计 token，任一到顶即拒；全局成本熔断读小时/日累计成本对比阈值。`record_usage` 用一个 Redis pipeline 原子地累加会话 token、会话调用次数与全局小时/日成本，并各自设 TTL（会话与小时窗口 3600s、日窗口 86400s），让计数随时间窗口自然滚动过期。写路径与检查路径分离，保证检查是纯读、开销可控。

---

## 十、统一 Guardrail 引擎

### 10.1 集成到执行流程

所有保护机制通过 `GuardrailEngine` 统一包裹 Agent 执行：

`GuardrailEngine` 在构造时组合前面所有保护器，对外只暴露一个 `execute_safe` 入口：

```python
# src/guardrails/engine.py

class GuardrailEngine:
    """统一的稳定性保障引擎"""

    def __init__(self, config: StabilityConfig, redis_client):
        # 组合：LoopGuard / ResourceGuard / FallbackHandler / IdempotencyGuard / OutputGuard
        ...

    # 安全执行 Agent，所有保护措施在此集成
    async def execute_safe(
        self, agent_executor, state: CustomerState,
        session_id: str, user_id: str, is_vip: bool = False,
    ) -> dict: ...

    async def _execute_with_guards(self, agent_executor, state, session_id): ...  # 内层：循环保护
    def _resource_limit_response(self, reason: str) -> dict: ...                   # 资源超限话术
```

保护措施的嵌套顺序是这套引擎的核心设计：

- 最外层是全局开关，`enabled=False` 时直接裸执行（仅开发环境）；
- 第一道是资源检查，`check_all` 拒绝就立刻用对应原因的话术返回，避免为已超限的请求付出任何 LLM 成本；
- 第二道用 `asyncio.wait_for` 套一层会话级总超时，兜住内部任何环节的整体挂起；
- 最内层 `_execute_with_guards` 在每轮开始 `reset` 循环计数器，执行后查 `should_force_end`，命中即走降级；
- 最外围用 `try/except` 把会话超时和未预期异常都收口到 `FallbackHandler`，任何漏网错误都转成一次可解释的降级回复而非直接 500。

这样「资源 → 超时 → 循环 → 兜底」由外向内层层收敛，每一层都有明确的失败出口。

### 10.2 在 API 层接入

API 层不直接调用编排图，而是把图作为 `agent_executor` 交给 `execute_safe`，再对返回结果做 SSE 流式下发：

```python
# src/api/routes.py

@app.post("/api/chat/{session_id}")
async def send_message(session_id: str, request: ChatRequest):
    # 取 user_id / is_vip → 调 guardrail_engine.execute_safe(agent_executor=graph, ...)
    # → 对 result.messages 做 SSE 流式下发，返回 StreamingResponse
    ...
```

关键在于所有稳定性保护都收在 `execute_safe` 一层，路由函数本身不感知循环、超时、限流等细节，只负责透传会话上下文（`thread_id`、`user_id`、VIP 标识）和把结果转成流式事件。

---

## 十一、项目目录结构

```
src/guardrails/                    # 稳定性工程模块
├── __init__.py
├── engine.py                      # GuardrailEngine 统一入口
├── loop_protection.py             # 循环保护
├── timeout.py                     # 超时控制
├── retry.py                       # 重试策略
├── output.py                      # 输出格式保障
├── fallback.py                    # 降级兜底
├── idempotency.py                 # 幂等去重
└── resource_limit.py              # 资源限制
```

---

## 十二、监控与告警

稳定性相关指标应接入 Grafana 看板：

| 指标 | 含义 | 告警阈值 |
|------|------|----------|
| `guardrail.loop_triggered` | 循环保护触发次数 | > 10 次/分钟 |
| `guardrail.timeout_count` | 超时次数 | > 5 次/分钟 |
| `guardrail.retry_count` | 重试次数 | > 20 次/分钟 |
| `guardrail.fallback_triggered` | 降级触发次数 | > 5 次/分钟 |
| `guardrail.idempotency_hit` | 幂等命中（重复操作拦截） | 仅记录 |
| `guardrail.rate_limit_rejected` | 限流拒绝次数 | > 50 次/分钟 |
| `guardrail.cost_circuit_open` | 成本熔断打开 | 立即告警 |
| `guardrail.handoff_triggered` | 故障转人工次数 | > 10 次/分钟 |

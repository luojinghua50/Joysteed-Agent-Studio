# Token 消耗优化方案

> 面向 agent-studio 项目，分两个维度：Claude Code CLI 使用侧 + 项目代码侧。
> 生成日期：2026-06-28

---

## 维度一：降低 Claude Code CLI 本身的 token 消耗

核心原理：CLI 每次提问的成本 = 系统提示 + 工具定义 + **整个对话历史** + 本次新增内容，且**每一轮都重发**。省钱的关键就是不让无关上下文长期堆在窗口里。

按性价比排序：

1. **任务切换就 `/clear`**（最大杠杆）
   CLI 默认把整段会话历史随每条消息重发。修完一个任务又开新话题时，前面的日志、文件内容仍在窗口里反复计费。独立任务做完立即 `/clear`，成本回落到只剩当前任务。

2. **善用 prompt caching（自动生效、基本免费）**
   系统提示和工具定义默认走 5 分钟 TTL 缓存，命中部分约按 10% 计价。含义：**连续对话远比隔很久再问便宜**。把相关问题在同一 session 里连着问完，别问一句走开半小时再回来（缓存过期就全价重算）。

3. **检索类脏活交给 subagent**
   fan-out 搜索（扫目录、找调用点、读一堆文件）若直接在主对话做，文件内容会永久占用主窗口。交给 Explore / general-purpose subagent，它在自己上下文里读完只带回结论，主窗口保持干净，后续每轮都省。

4. **精准引用文件**
   "看整个 src 目录" vs "看 `executor.py`" 差几十倍 token。指名文件、必要时指名行号区间。

5. **模型分级**
   简单问答 / 格式调整用便宜档，复杂架构设计才上 Opus。用 `/model` 切换。

6. **长任务靠自动压缩，别手动重复贴上下文**
   窗口快满时 CLI 会自动压缩历史，无需把之前的结论再粘一遍。

**一句话经验**：一个任务一个干净 session，连续问完，检索外包给 subagent。

---

## 维度二：项目代码侧降低 token 消耗

调用链路：`ChatOpenAI` → litellm proxy → `anthropic/` provider → 上游。
已审阅文件：`agent-core/src/agents/graph.py`、`generic.py`、`tools/executor.py`、`litellm_config.yaml`。

### 已经做对的（保持）

- **模型分级路由**：haiku / sonnet / opus 按 agent 分配，fallback 链已配。
- **历史截断**：`history_window` 按 AgentSpec 截断，未无脑全量传。
- **多意图省传**：dispatch 扇出用改写后的 `_sub_query` 单条消息，不带全历史。

### 漏掉的最大杠杆：全链路零 prompt caching

`grep cache_control / ephemeral` 在 agent-core 里**一处都没有**。每次 LLM 调用，system prompt + 全部工具 JSON schema 都按全价重算。且被工具循环**放大**——`tools/executor.py` 每趟 `ainvoke` 都重发 `[SystemMessage, *history]` + 全套 tools，一次提问触发 3 轮工具调用 = system prompt 与工具定义全价计费 4 次。

---

## 改造清单（按 ROI 优先级）

### 1. 开启 Anthropic prompt caching（预计省 50–90% 输入 token）★最高优先级

litellm 走 `anthropic/` provider，原生支持 `cache_control`。把每个 agent 的 system prompt 和工具定义标记为可缓存，5 分钟内的重复调用（尤其工具循环的 2–4 趟）缓存部分只按约 10% 计价。

**注意坑点**：agent-core 用的是 `ChatOpenAI` 指向 litellm（`graph.py:25`），不是 Anthropic 原生 SDK。要让 `cache_control` 透传，有两条路：
- (a) 在 message 里带 `cache_control` 字段，由 litellm 翻译给上游；
- (b) 在 litellm_config 里对这几个模型开 caching 透传。

`ChatOpenAI → litellm → anthropic` 这层翻译是否保留 `cache_control` **必须先实测**，不能拍脑袋。

**验证方法**：发一个带 `cache_control` 的请求，检查上游返回的 `usage` 是否含 `cache_creation_input_tokens` / `cache_read_input_tokens`，确认链路通了再改 agent-core 代码。

### 2. system prompt 结构稳定化，动态内容后置

`generic.py:53-59` 现在把 `memory_context`、交接规则拼到 system prompt 后面。prompt caching 是前缀匹配——只要开头稳定，缓存能命中到第一个变化点。

- 把**固定的交接规则**放最前面，和基础 prompt 一起；
- 把**每轮都变的 `memory_context`** 放最末尾，让缓存边界尽量靠后。

### 3. 工具 schema 瘦身 + 缓存

`bind_tools(tools)` 每趟都发全部工具的完整 JSON schema：
- 工具描述和参数 schema 写精简（大段 description 是纯成本）；
- 工具定义随 system prompt 一起进缓存块（同第 1 点）。

### 4. 工具循环受益于 caching（无需改循环逻辑）

`tools/executor.py` 的循环每趟重发整个 `working_messages`。开了 caching 后，前缀（system + tools + 早期消息）命中缓存，只有新追加的 `ToolMessage` 全价——这正是 caching 对 agent 循环最大的价值点。第 1 点落地后此处成本自动大幅下降。

### 5. 关闭生产环境 verbose 日志

`litellm_config.yaml` 里 `set_verbose: true` + `log_responses: true` 不直接耗 API token，但会把完整请求/响应（含长 prompt）写日志，量大时是存储与排查成本。生产建议关掉或降级。

### 6. 中长期：supervisor 路由降级

`supervisor` 用 `model_main` 做纯路由分类（`graph.py:44`）。若只是意图分类，可考虑先用 haiku / 规则 / 小模型过一遍，只在不确定时上主力模型。需看 supervisor prompt 复杂度才能定，属第二阶段。

---

## 落地路线

1. **先验证 caching 透传**：在 litellm 链路发带 `cache_control` 的请求，看 `usage` 字段是否出现 cache 命中/创建。
2. 链路确认后落地第 1–3 点（80% 的收益所在）。
3. 第 4 点随第 1 点自动生效。
4. 第 5 点随手做。
5. 第 6 点列入第二阶段评估。

**收益预估**：第 1–4 点落地后，含工具循环的请求输入 token 预计下降 50–90%（取决于 system prompt + 工具 schema 占比，以及工具循环轮数）。

"""工作记忆实体抽取：从工具返回的结构化结果里直取关键实体（规则，非 LLM）。

设计原则（决策 C）：能从工具结果拿到的绝不让 LLM 猜——工具返回的 order_id/amount
等是权威数据，规则直取比 LLM 抽取更准、免费、零延迟。仅覆盖已知工具字段；
自由文本里的实体（"我地址改成X"）留给会话结束时的 LLM 抽取（事实/画像）。

输入 tool_results 形如 {tool_name: raw_result}（executor return_transcript/return_results
产出）。raw_result 有两种形态：
- 读工具经 MCP adapter（mcp_adapter.py）序列化为 **JSON 字符串**（query_order/track_shipping）；
- 写工具在 execute_node 路径拿到的是**原始 dict**（apply_refund）。
本函数两者都吃：字符串先尝试 json.loads 还原成 dict。
"""
import json

# 认可的工具结果字段 → 工作记忆实体键。只收结构化、稳定、对后续对话有用的键。
_ENTITY_KEYS = (
    "order_id",
    "refund_id",
    "amount",
    "status",
    "order_status",
    "carrier",
    "eta",
    "product",
)


def _as_dict(result):
    """把工具结果规整成 dict：已是 dict 直接用；JSON 字符串尝试解析；其余返回 None。"""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        s = result.strip()
        if s.startswith("{"):
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, dict) else None
            except (json.JSONDecodeError, ValueError):
                return None
    return None


def extract_entities(tool_results: dict | None) -> dict:
    """从 {tool_name: raw_result} 提取工作记忆实体。

    - result 规整为 dict（dict 直用 / JSON 字符串解析）；非结构化或含 error 键的跳过。
    - 多个工具返回同名键时后者覆盖前者（同一轮内后执行的更新）。
    - 返回 {entity_key: value}；无可提取时返回空 dict。
    """
    entities: dict = {}
    for _tool, raw in (tool_results or {}).items():
        result = _as_dict(raw)
        if result is None or result.get("error"):
            continue
        for key in _ENTITY_KEYS:
            if key in result and result[key] is not None:
                entities[key] = result[key]
    return entities

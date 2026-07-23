"""会话内点对点交接（Level 2 handoff）的目标提取。

Agent 在回复里用 `[HANDOFF:目标]` 标记交接意图，本模块只负责从文本中
提取目标，并校验它在该 Agent 的白名单（AgentSpec.can_handoff_to）内。
落地（计数/封顶/审计）由不调 LLM 的哑路由 chokepoint（router.py）完成。
"""
import re

_PATTERN = re.compile(r"\[HANDOFF:(\w+)\]")


def detect_handoff(text: str, allowed: list[str]) -> str | None:
    """从 Agent 回复中提取交接目标，只接受 spec 白名单内的目标。"""
    m = _PATTERN.search(text or "")
    if m and m.group(1) in (allowed or []):
        return m.group(1)
    return None

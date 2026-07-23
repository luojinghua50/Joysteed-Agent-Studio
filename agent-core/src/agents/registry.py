"""Agent 定义数据化（agents-as-data）：从 config/agents.yaml 加载。

新增业务 Agent = 在 agents.yaml 加一段，无需改任何 .py。

工具授权按最小权限原则，支持两种粒度（在 yaml 的 tools 列表里混用）：
  - "server:<name>"  授权该 MCP server 上的【全部】工具（含未来新增，零改动）
  - "tool:<name>"    只授权这一个具体工具（敏感操作用细粒度）
解析后落到 AgentSpec.server_grants / tool_grants 两个集合，由
tools/mcp_adapter.py 在运行时结合 MCP 动态发现的工具清单解析为实际工具。
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "agents.yaml"


@dataclass(frozen=True)
class AgentSpec:
    name: str                       # 节点名 / intent 标签
    prompt_id: str                  # PromptRegistry 中的 prompt ID（不内联 prompt 文本）
    server_grants: frozenset[str] = frozenset()  # 整 server 授权（"server:x"）
    tool_grants: frozenset[str] = frozenset()    # 细粒度工具授权（"tool:y"）
    model_key: str = "model_main"   # Settings 上的模型字段名，决定分级模型
    history_window: int = 10        # 喂给 LLM 的历史消息条数（FAQ 用 5）
    can_handoff_to: list[str] = field(default_factory=list)  # 允许交接的目标 Agent
    reflection: str = "off"         # off / self_check / judge，与 ReflectionConfig 对齐


def _parse_grants(raw_tools: list[str]) -> tuple[frozenset[str], frozenset[str]]:
    """把 yaml 的 ["server:order", "tool:create_ticket"] 拆成 (servers, tools)。"""
    servers, tools = set(), set()
    for entry in raw_tools or []:
        if entry.startswith("server:"):
            servers.add(entry[len("server:"):])
        elif entry.startswith("tool:"):
            tools.add(entry[len("tool:"):])
        else:
            # 裸工具名（向后兼容）：当作细粒度工具授权
            tools.add(entry)
    return frozenset(servers), frozenset(tools)


def _spec_from_dict(name: str, data: dict) -> AgentSpec:
    servers, tools = _parse_grants(data.get("tools", []))
    return AgentSpec(
        name=name,
        prompt_id=data["prompt_id"],
        server_grants=servers,
        tool_grants=tools,
        model_key=data.get("model_key", "model_main"),
        history_window=data.get("history_window", 10),
        can_handoff_to=list(data.get("can_handoff_to", [])),
        reflection=data.get("reflection", "off"),
    )


def load_registry(path: Path | str | None = None) -> dict[str, AgentSpec]:
    """从 YAML 加载 Agent 注册表。文件缺失/为空时返回空表（由调用方决定兜底）。"""
    cfg_path = Path(path) if path else _CONFIG_PATH
    if not cfg_path.is_file():
        return {}
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    agents = raw.get("agents", {}) or {}
    return {name: _spec_from_dict(name, data) for name, data in agents.items()}


# 模块级单例：进程启动时加载一次。新增 agent 改 yaml 即可，无需改本文件。
AGENT_REGISTRY: dict[str, AgentSpec] = load_registry()

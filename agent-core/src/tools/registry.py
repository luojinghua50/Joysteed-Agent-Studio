"""Tool registry: maps tools to MCP servers.

工具→Agent 的归属（旧 AGENT_TOOLS）已迁移到 src/agents/registry.py 的
AgentSpec.tools，registry 成为单一事实源；此处只保留工具→MCP server 的归属。
"""

TOOL_SERVER_MAP: dict[str, str] = {
    "search_knowledge": "knowledge",
    "get_knowledge_filters": "knowledge",
    "search_faq": "knowledge",
    "search_docs": "knowledge",
    "query_order": "order",
    "modify_order": "order",
    "apply_refund": "order",
    "check_refund_eligibility": "order",
    "track_shipping": "order",
    "urge_shipping": "order",
    "create_ticket": "ticket",
    "query_ticket": "ticket",
    "list_tickets": "ticket",
    "reassign_ticket": "ticket",
    "add_ticket_comment": "ticket",
    "apply_compensation": "ticket",
    "get_customer_info": "crm",
    "update_customer_tag": "crm",
}

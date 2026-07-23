from langchain_core.messages import AIMessage
from src.agents.state import CustomerState


async def human_handoff_node(state: CustomerState) -> dict:
    """Human Handoff Agent: transfers conversation to human support."""
    return {
        "messages": [AIMessage(content=(
            "正在为您转接人工客服，请稍候...\n"
            "预计等待时间 1-3 分钟，请不要关闭对话窗口。"
        ))],
        "resolved": True,
        "current_agent": "human_handoff",
    }

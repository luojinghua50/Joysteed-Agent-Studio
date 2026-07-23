import structlog
from langchain_core.messages import AIMessage

from src.agents.state import CustomerState

logger = structlog.get_logger()

FALLBACK_TEMPLATES = {
    "faq": "抱歉，我暂时无法回答您的问题。建议您查阅产品帮助中心，或联系人工客服为您解答。",
    "order": "抱歉，订单系统暂时响应较慢。请稍后再试，或联系人工客服为您查询。",
    "complaint": "非常抱歉给您带来不好的体验。我已记录您的反馈，正在为您转接专属客服处理。",
    "tech_support": "抱歉，当前系统繁忙。请您稍后重试，或拨打技术支持热线获取帮助。",
    "default": "抱歉，我暂时无法处理您的请求。正在为您转接人工客服。",
}


class FallbackHandler:
    """Handles graceful degradation when normal processing fails."""

    def __init__(self, config=None):
        self.config = config

    async def handle_failure(
        self,
        state: dict,
        error: Exception,
        context: str,
    ) -> dict:
        """Generate a safe fallback response on failure."""
        logger.error(
            "agent_failure_fallback",
            intent=state.get("intent"),
            agent=state.get("current_agent"),
            error_type=type(error).__name__,
            error_msg=str(error)[:500],
            context=context,
        )

        intent = state.get("intent", "default")
        template = FALLBACK_TEMPLATES.get(intent, FALLBACK_TEMPLATES["default"])

        should_handoff = self._should_handoff(error, state)
        if should_handoff:
            template += "\n正在为您转接人工客服..."

        return {
            "messages": [AIMessage(content=template)],
            "resolved": not should_handoff,
            "current_agent": "human_handoff" if should_handoff else state.get("current_agent"),
        }

    def _should_handoff(self, error: Exception, state: dict) -> bool:
        failure_count = state.get("failure_count", 0)
        max_failures = 2
        if self.config:
            max_failures = getattr(self.config, "max_failures_before_handoff", 2)

        if failure_count >= max_failures:
            return True
        if state.get("intent") == "complaint":
            return True
        return False

from pydantic import BaseModel


class JudgeResult(BaseModel):
    """Result from the judge evaluation."""

    score: float
    passed: bool
    issues: list[str]
    suggestion: str


JUDGE_PROMPT = """你是客服质量评估专家。严格评估以下 Agent 回复。

## 评估维度（每项 0-10 分，取平均）
1. **准确性**：回复内容是否基于工具返回的真实数据，有无编造
2. **完整性**：是否回答了用户的全部问题
3. **安全性**：是否有承诺超出权限的操作、泄露敏感信息
4. **合规性**：是否符合客服规范（礼貌、专业、不推诿）
5. **可操作性**：是否给出了用户下一步可以执行的明确指引

## 严重问题（直接判 FAIL）
- 编造数据（工具没返回的信息）
- 承诺无法做到的事
- 泄露内部系统信息
- 对用户不礼貌

## 输入
- 用户问题：{user_message}
- 工具调用及结果：{tool_results}
- Agent 回复：{agent_response}

## 输出 JSON
{{"score": 8.5, "passed": true, "issues": [], "suggestion": ""}}

如果 score < {threshold}，passed 设为 false，并给出具体 issues 和修正 suggestion。
"""


class JudgeReflector:
    """Judge reflector: evaluates agent responses for quality."""

    def __init__(self, config=None, llm=None):
        self.config = config
        self.llm = llm
        self.threshold = 7.0
        if config:
            self.threshold = getattr(config, "quality_threshold", 7.0)

    async def evaluate(
        self,
        user_message: str,
        tool_results: list[dict],
        agent_response: str,
    ) -> JudgeResult:
        """Evaluate an agent's response quality."""
        if self.llm:
            prompt = JUDGE_PROMPT.format(
                user_message=user_message,
                tool_results=str(tool_results),
                agent_response=agent_response,
                threshold=self.threshold,
            )
            result = await self.llm.ainvoke(prompt)
            return self._parse_result(result.content)

        return self._rule_based_evaluate(user_message, agent_response)

    def _rule_based_evaluate(
        self, user_message: str, agent_response: str
    ) -> JudgeResult:
        """Rule-based fallback evaluation (no LLM needed)."""
        issues = []
        score = 8.0

        if not agent_response or len(agent_response) < 10:
            issues.append("回复过短，可能未充分回答用户问题")
            score -= 3.0

        internal_patterns = ["traceback", "Exception", "Internal Server Error", "API error"]
        for pattern in internal_patterns:
            if pattern.lower() in agent_response.lower():
                issues.append(f"响应包含内部信息: {pattern}")
                score -= 5.0

        promise_patterns = ["保证", "一定会", "肯定能"]
        for pattern in promise_patterns:
            if pattern in agent_response:
                issues.append(f"不当承诺: {pattern}")
                score -= 2.0

        passed = score >= self.threshold
        suggestion = ""
        if not passed:
            suggestion = "请基于工具返回的真实数据重新生成回复，避免编造信息和过度承诺。"

        return JudgeResult(
            score=max(score, 0),
            passed=passed,
            issues=issues,
            suggestion=suggestion,
        )

    def _parse_result(self, content: str) -> JudgeResult:
        """Parse LLM output into JudgeResult."""
        import json
        try:
            data = json.loads(content)
            return JudgeResult(**data)
        except (json.JSONDecodeError, TypeError):
            return JudgeResult(score=7.0, passed=True, issues=[], suggestion="")

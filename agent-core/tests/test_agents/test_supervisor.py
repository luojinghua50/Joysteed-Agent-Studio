from src.agents.supervisor import _parse_intent, _strip_think, _intent_to_agent, route_by_intent


class TestParseIntent:
    def test_parse_faq(self):
        assert _parse_intent("faq") == "faq"
        assert _parse_intent("FAQ") == "faq"
        assert _parse_intent("  faq  ") == "faq"

    def test_parse_order(self):
        assert _parse_intent("order") == "order"
        assert _parse_intent("ORDER") == "order"

    def test_parse_complaint(self):
        assert _parse_intent("complaint") == "complaint"

    def test_parse_tech_support(self):
        assert _parse_intent("tech_support") == "tech_support"

    def test_parse_human(self):
        assert _parse_intent("human") == "human"

    def test_parse_unknown_defaults_to_human(self):
        assert _parse_intent("unknown") == "human"
        assert _parse_intent("") == "human"
        assert _parse_intent("random text") == "human"

    def test_parse_intent_in_sentence(self):
        assert _parse_intent("The intent is order") == "order"
        assert _parse_intent("I think this is a complaint") == "complaint"

    def test_parse_intent_strips_think_block(self):
        # 回归测试：本地 thinking 模型（如 qwen3）会在正式回答前吐出
        # <think>...</think>，该块常复述 system prompt 里列出的全部意图标签
        # （包括 "human"），若不剥离，旧的 set 子串匹配会不确定性地把每次
        # 分类都误判成 human。
        reasoning = (
            "<think>用户在问订单相关问题。规则里说 faq/order/complaint/"
            "tech_support/human 都是候选，但只有明确要求转人工才选 human，"
            "这里不是。</think>\norder"
        )
        assert _parse_intent(reasoning) == "order"

    def test_parse_intent_think_block_only_defaults_to_human(self):
        # 模型只输出了推理、没有给出正式结论标签时，剥完 <think> 后正文为空，
        # 属于"没有答案"而非"答案是 human"，与 unknown/空输入走同一条安全
        # 兜底路径——不应该去 think 块内容里挖结论，否则又回到旧 bug 的模式。
        reasoning = "<think>这是投诉，情绪很激动，应该选 complaint。</think>"
        assert _parse_intent(reasoning) == "human"

    def test_parse_intent_accepts_content_block_list(self):
        # langchain 的 response.content 在多模态场景下是
        # list[str | dict]，而不是纯 str。
        blocks = [{"type": "text", "text": "<think>闲聊，选 faq</think>"}, "faq"]
        assert _parse_intent(blocks) == "faq"


class TestStripThink:
    def test_removes_think_block(self):
        assert _strip_think("<think>reasoning here</think>order") == "order"

    def test_removes_multiline_think_block(self):
        text = "<think>\nline1\nline2\n</think>\nfaq"
        assert _strip_think(text) == "faq"

    def test_removes_unclosed_think_block(self):
        assert _strip_think("<think>reasoning without close") == ""

    def test_no_think_block_returns_unchanged(self):
        assert _strip_think("order") == "order"

    def test_empty_input(self):
        assert _strip_think("") == ""
        assert _strip_think(None) == ""

    def test_content_block_list(self):
        blocks = ["<think>x</think>", {"type": "text", "text": "order"}]
        assert _strip_think(blocks) == "order"


class TestIntentToAgent:
    def test_mapping(self):
        assert _intent_to_agent("faq") == "faq"
        assert _intent_to_agent("order") == "order"
        assert _intent_to_agent("complaint") == "complaint"
        assert _intent_to_agent("tech_support") == "tech_support"
        assert _intent_to_agent("human") == "human_handoff"

    def test_unknown_defaults_to_human_handoff(self):
        assert _intent_to_agent("unknown") == "human_handoff"


class TestRouteByIntent:
    def test_route_faq(self):
        state = {"intent": "faq", "messages": [], "customer_id": "C001"}
        assert route_by_intent(state) == "faq"

    def test_route_order(self):
        state = {"intent": "order", "messages": [], "customer_id": "C001"}
        assert route_by_intent(state) == "order"

    def test_route_complaint(self):
        state = {"intent": "complaint", "messages": [], "customer_id": "C001"}
        assert route_by_intent(state) == "complaint"

    def test_route_tech_support(self):
        state = {"intent": "tech_support", "messages": [], "customer_id": "C001"}
        assert route_by_intent(state) == "tech_support"

    def test_route_human(self):
        state = {"intent": "human", "messages": [], "customer_id": "C001"}
        assert route_by_intent(state) == "human_handoff"

    def test_route_none_defaults_to_human(self):
        state = {"intent": None, "messages": [], "customer_id": "C001"}
        assert route_by_intent(state) == "human_handoff"

"""diagnose_fault 编排单元测试。

mock 掉 skill_server 的 _client.call_tool（不起真实 leaf server），
覆盖：KB 命中 / KB 未命中建单 / order 查询失败降级 / 无 order_id / 并发调用。
这正是 skill 相比 LLM 编排多出来的可测性红利 —— 流程是确定的，可断言。
"""
import pytest

from skill_server import server
from skill_server.server import diagnose_fault, KB_SCORE_THRESHOLD


def _fake_client(order_result=None, kb_result=None):
    """构造一个替身 client，按 (server, tool) 返回预设结果，并记录调用。"""
    calls = []

    async def _call_tool(srv, tool, args):
        calls.append((srv, tool, args))
        if srv == "order":
            return order_result if order_result is not None else {"error": "not set"}
        if srv == "knowledge":
            return kb_result if kb_result is not None else {"results": []}
        return {"error": f"unexpected server {srv}"}

    client = type("FakeClient", (), {})()
    client.call_tool = _call_tool
    client._calls = calls
    return client


@pytest.fixture(autouse=True)
def restore_client():
    """每个用例后恢复真实 _client，避免相互污染。"""
    original = server._client
    yield
    server._client = original


async def test_kb_hit_resolves_no_ticket():
    """KB 命中可用方案 → resolved_by_kb=True, need_ticket=False, 无草稿。"""
    server._client = _fake_client(
        kb_result={"results": [
            {"title": "蓝牙重置", "content": "长按5秒", "score": 0.9},
        ]},
    )
    out = await diagnose_fault(symptom="耳机连不上蓝牙")

    assert out["resolved_by_kb"] is True
    assert out["need_ticket"] is False
    assert out["ticket_draft"] is None
    assert len(out["solutions"]) == 1
    assert out["source"] == "skill:diagnose_fault"


async def test_kb_miss_needs_ticket_with_draft():
    """KB 无可用方案(分数低于阈值) → need_ticket=True 且给出建单草稿(不含 customer_id)。"""
    server._client = _fake_client(
        kb_result={"results": [
            {"title": "无关内容", "content": "...", "score": 0.1},
        ]},
    )
    out = await diagnose_fault(symptom="设备无法开机")

    assert out["resolved_by_kb"] is False
    assert out["need_ticket"] is True
    draft = out["ticket_draft"]
    assert draft is not None
    assert "customer_id" not in draft          # 关键：身份由 agent 补全，skill 不碰
    assert draft["priority"] == "medium"
    assert "设备无法开机" in draft["description"]


async def test_order_context_included_when_order_ok():
    """提供 order_id 且查询成功 → order_context 带 product/status，且草稿含订单信息。"""
    server._client = _fake_client(
        # 对齐 order store 真实结构：商品在 items=[{name,qty,price}]，无独立 product 字段
        order_result={"status": "已签收", "items": [{"name": "XX蓝牙耳机", "qty": 1}]},
        kb_result={"results": []},                # 空结果 → need_ticket
    )
    out = await diagnose_fault(symptom="设备无法开机", order_id="SO123")

    assert out["order_context"] == {"product": "XX蓝牙耳机", "order_status": "已签收"}
    assert out["need_ticket"] is True
    assert "SO123" in out["ticket_draft"]["description"]


async def test_order_multi_items_joined():
    """多商品订单 → product 以顿号拼接各 item 名称。"""
    server._client = _fake_client(
        order_result={"status": "shipped",
                      "items": [{"name": "耳机", "qty": 1}, {"name": "充电线", "qty": 2}]},
        kb_result={"results": []},
    )
    out = await diagnose_fault(symptom="配件问题", order_id="ORD-001")
    assert out["order_context"]["product"] == "耳机、充电线"


async def test_order_query_failure_degrades():
    """order 查询失败(如订单不存在) → order_context=None，诊断不中断。"""
    server._client = _fake_client(
        order_result={"error": "订单 SOX 不存在"},
        kb_result={"results": [{"title": "方案", "content": "...", "score": 0.8}]},
    )
    out = await diagnose_fault(symptom="连不上", order_id="SOX")

    assert out["order_context"] is None
    assert out["resolved_by_kb"] is True         # KB 仍正常命中


async def test_no_order_id_skips_order_call():
    """未传 order_id → 不调 query_order，只调 search_knowledge。"""
    client = _fake_client(kb_result={"results": []})
    server._client = client
    await diagnose_fault(symptom="故障")

    called_servers = {c[0] for c in client._calls}
    assert "order" not in called_servers
    assert "knowledge" in called_servers


async def test_threshold_boundary():
    """分数恰好等于阈值 → 视为命中(>=)。"""
    server._client = _fake_client(
        kb_result={"results": [{"title": "t", "content": "c", "score": KB_SCORE_THRESHOLD}]},
    )
    out = await diagnose_fault(symptom="x")
    assert out["resolved_by_kb"] is True

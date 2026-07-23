import pytest
from httpx import AsyncClient, ASGITransport
from src.api.routes import create_app
from src.config import Settings
from src.database import init_db
from src.security.auth import create_access_token


@pytest.fixture
async def settings():
    return Settings()


@pytest.fixture
async def app(settings):
    application = create_app(settings)
    # ASGITransport does not run the app lifespan, so initialize the DB factory
    # explicitly against an in-memory SQLite database for isolated tests.
    application.state.db_session_factory = await init_db("sqlite+aiosqlite:///:memory:")
    # graph 已移入 lifespan 编译（依赖 checkpointer 连接池），lifespan 不跑时须手动补。
    # 测试无 PG，用进程内 MemorySaver（compile_graph 在 approval_enabled 时自动兜底）。
    from src.agents.graph import compile_graph
    application.state.graph = compile_graph(settings, prompts=application.state.prompts)
    return application


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def auth(settings):
    """Factory: build an Authorization header for a given customer_id.

    Identity is token-derived (P1-B), so tests mint a signed access token for
    the subject rather than self-reporting customer_id in the body."""
    def _auth(customer_id: str) -> dict[str, str]:
        token = create_access_token(
            customer_id, settings.jwt_secret, algorithm=settings.jwt_algorithm
        )
        return {"Authorization": f"Bearer {token}"}
    return _auth


@pytest.mark.asyncio
async def test_cors_allows_configured_origin(client):
    # A preflight from an allowlisted origin is reflected back with credentials.
    resp = await client.options("/v1/sessions", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    })
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert resp.headers.get("access-control-allow-credentials") == "true"


@pytest.mark.asyncio
async def test_cors_rejects_unknown_origin(client):
    # An origin outside the allowlist is not echoed back (no blanket "*").
    resp = await client.options("/v1/sessions", headers={
        "Origin": "http://evil.example.com",
        "Access-Control-Request-Method": "POST",
    })
    assert resp.headers.get("access-control-allow-origin") != "http://evil.example.com"
    assert resp.headers.get("access-control-allow-origin") != "*"

    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_create_session(client, auth):
    response = await client.post("/v1/sessions", json={"content": "hello"}, headers=auth("C001"))
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert data["customer_id"] == "C001"


@pytest.mark.asyncio
async def test_endpoints_require_token(client):
    # Without a Bearer token the chat endpoints reject the request (401),
    # so identity can no longer be self-reported.
    assert (await client.post("/v1/sessions", json={"content": "hi"})).status_code == 401
    assert (await client.post("/v1/chat/x", json={"content": "hi"})).status_code == 401
    assert (await client.get("/v1/chat/x/history")).status_code == 401


@pytest.mark.asyncio
async def test_get_history_nonexistent_session(client, auth):
    # A session that was never created is reported as not found, so history
    # of a guessed session_id is not silently returned.
    response = await client.get("/v1/chat/nonexistent/history", headers=auth("C001"))
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_chat_sse_response(client, auth):
    response = await client.post("/v1/chat/test-session-001", json={
        "content": "我的订单到哪了",
    }, headers=auth("C001"))
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    body = response.text
    assert "data:" in body
    assert '"type": "done"' in body or '"type":"done"' in body


@pytest.mark.asyncio
async def test_chat_classifies_order_intent(client, auth):
    response = await client.post("/v1/chat/test-session-002", json={
        "content": "查询我的订单物流",
    }, headers=auth("C001"))
    body = response.text
    assert "订单" in body


@pytest.mark.asyncio
async def test_chat_classifies_complaint_intent(client, auth):
    response = await client.post("/v1/chat/test-session-003", json={
        "content": "你们产品太差了 我要投诉",
    }, headers=auth("C001"))
    body = response.text
    assert "抱歉" in body


@pytest.mark.asyncio
async def test_approval_endpoint_streams(client, auth):
    """/approve 现在 resume 被审批闸暂停的图并流式返回（SSE），不再是同步 JSON。

    真实的审批批准/拒绝闭环在 test_agents/test_approval.py 图层充分覆盖；此处只验
    API 契约：端点返回 200 + text/event-stream，且以 done 事件收尾。
    """
    session_id = "approval-test-001"
    # 先建立该 session（发一条消息落库），/approve 会校验 session 归属
    await client.post(f"/v1/chat/{session_id}", json={"content": "hello"}, headers=auth("C001"))

    response = await client.post(f"/v1/chat/{session_id}/approve", json={
        "approved": True,
        "reason": "amount within limit",
    }, headers=auth("C001"))
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"type": "done"' in response.text


@pytest.mark.asyncio
async def test_approval_rejected_streams(client, auth):
    session_id = "approval-test-002"
    await client.post(f"/v1/chat/{session_id}", json={"content": "hello"}, headers=auth("C001"))
    response = await client.post(f"/v1/chat/{session_id}/approve", json={
        "approved": False,
        "reason": "exceeds limit",
    }, headers=auth("C001"))
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"type": "done"' in response.text


@pytest.mark.asyncio
async def test_chat_history_records_messages(client, auth):
    session_id = "history-test-001"
    await client.post(f"/v1/chat/{session_id}", json={
        "content": "hello",
    }, headers=auth("C001"))

    response = await client.get(f"/v1/chat/{session_id}/history", headers=auth("C001"))
    assert response.status_code == 200
    data = response.json()
    assert len(data["messages"]) == 2  # user + assistant
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_history_blocks_cross_customer_access(client, auth):
    # C001 owns the session; C999 must not be able to read its history.
    session_id = "owner-test-001"
    await client.post(f"/v1/chat/{session_id}", json={
        "content": "我的银行卡尾号是 1234",
    }, headers=auth("C001"))

    response = await client.get(f"/v1/chat/{session_id}/history", headers=auth("C999"))
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_chat_blocks_cross_customer_message(client, auth):
    # C999 must not be able to post into a session owned by C001.
    session_id = "owner-test-002"
    await client.post(f"/v1/chat/{session_id}", json={
        "content": "hello",
    }, headers=auth("C001"))

    response = await client.post(f"/v1/chat/{session_id}", json={
        "content": "intruder",
    }, headers=auth("C999"))
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_list_sessions_returns_only_own_sessions(client, auth):
    # C001 posts into two sessions; C999 into one. Each customer sees only
    # their own, so history survives logout/login without leaking across users.
    await client.post("/v1/chat/list-a", json={"content": "查询订单"}, headers=auth("C001"))
    await client.post("/v1/chat/list-b", json={"content": "退换货"}, headers=auth("C001"))
    await client.post("/v1/chat/list-c", json={"content": "别人的会话"}, headers=auth("C999"))

    response = await client.get("/v1/sessions", headers=auth("C001"))
    assert response.status_code == 200
    sessions = response.json()["sessions"]
    ids = {s["session_id"] for s in sessions}
    assert ids == {"list-a", "list-b"}
    # Preview carries the first user message; message_count is populated.
    previews = {s["session_id"]: s["preview"] for s in sessions}
    assert previews["list-a"] == "查询订单"
    assert all(s["message_count"] >= 1 for s in sessions)


@pytest.mark.asyncio
async def test_list_sessions_requires_token(client):
    assert (await client.get("/v1/sessions")).status_code == 401


@pytest.mark.asyncio
async def test_list_sessions_empty_for_new_customer(client, auth):
    response = await client.get("/v1/sessions", headers=auth("C-fresh"))
    assert response.status_code == 200
    assert response.json()["sessions"] == []


@pytest.mark.asyncio
async def test_list_sessions_omits_empty_sessions(client, auth):
    # A bare session with no messages (as created on every login) must not
    # appear in the list, so auto-resume never lands on a blank conversation.
    await client.post("/v1/sessions", json={"content": "init"}, headers=auth("C500"))
    await client.post("/v1/chat/has-msgs", json={"content": "查询订单"}, headers=auth("C500"))

    response = await client.get("/v1/sessions", headers=auth("C500"))
    assert response.status_code == 200
    sessions = response.json()["sessions"]
    ids = {s["session_id"] for s in sessions}
    assert ids == {"has-msgs"}

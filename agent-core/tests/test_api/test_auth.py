import pytest
from httpx import AsyncClient, ASGITransport
from src.api.routes import create_app
from src.config import Settings
from src.database import init_db


@pytest.fixture
async def app():
    settings = Settings()
    application = create_app(settings)
    application.state.db_session_factory = await init_db("sqlite+aiosqlite:///:memory:")
    # graph 已移入 lifespan 编译；lifespan 在 ASGITransport 下不跑，须手动补。
    from src.agents.graph import compile_graph
    application.state.graph = compile_graph(settings, prompts=application.state.prompts)
    return application


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_register_returns_tokens(client):
    resp = await client.post("/v1/auth/register", json={
        "username": "alice", "password": "secret123", "display_name": "Alice",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["access_token"]
    assert data["refresh_token"]
    assert data["token_type"] == "bearer"
    assert data["customer_id"].startswith("u-")


@pytest.mark.asyncio
async def test_register_duplicate_username_conflicts(client):
    await client.post("/v1/auth/register", json={"username": "bob", "password": "secret123"})
    resp = await client.post("/v1/auth/register", json={"username": "bob", "password": "other123"})
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_login_ok_and_wrong_password(client):
    await client.post("/v1/auth/register", json={"username": "carol", "password": "secret123"})

    ok = await client.post("/v1/auth/login", json={"username": "carol", "password": "secret123"})
    assert ok.status_code == 200
    assert ok.json()["access_token"]

    bad = await client.post("/v1/auth/login", json={"username": "carol", "password": "WRONG"})
    assert bad.status_code == 401


@pytest.mark.asyncio
async def test_login_unknown_user_is_401(client):
    resp = await client.post("/v1/auth/login", json={"username": "ghost", "password": "secret123"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_requires_token(client):
    resp = await client.get("/v1/auth/me")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_me_returns_profile_with_token(client):
    reg = await client.post("/v1/auth/register", json={
        "username": "dave", "password": "secret123", "display_name": "Dave",
    })
    token = reg.json()["access_token"]
    customer_id = reg.json()["customer_id"]

    resp = await client.get("/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["customer_id"] == customer_id
    assert data["username"] == "dave"
    assert data["display_name"] == "Dave"
    assert "customer" in data["roles"]


@pytest.mark.asyncio
async def test_me_rejects_garbage_token(client):
    resp = await client.get("/v1/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_refresh_issues_new_access_token(client):
    reg = await client.post("/v1/auth/register", json={"username": "erin", "password": "secret123"})
    refresh_token = reg.json()["refresh_token"]

    resp = await client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert resp.status_code == 200
    assert resp.json()["access_token"]


@pytest.mark.asyncio
async def test_guest_token_works_on_chat_endpoints(client):
    # A guest token (no user row) is a valid, unforgeable identity: it can
    # create and own a session.
    guest = await client.post("/v1/auth/guest")
    assert guest.status_code == 200
    data = guest.json()
    assert data["customer_id"].startswith("guest-")
    headers = {"Authorization": f"Bearer {data['access_token']}"}

    resp = await client.post("/v1/sessions", json={"content": "init"}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["customer_id"] == data["customer_id"]


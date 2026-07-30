import json
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import select, func

from src.api.schemas import (
    ChatRequest, ApprovalRequest, SessionCreateRequest, HealthResponse,
    ChatHistoryResponse, SessionListResponse,
)
from src.api.deps import current_customer
from src.agents.graph import compile_graph
from src.config import Settings
from src.database import init_db, SessionModel, MessageModel
from src.memory.manager import MemoryManager, format_memory_for_prompt
from src.memory.working import WorkingMemory
from src.observability import setup_langfuse, setup_opentelemetry

logger = structlog.get_logger()

REQUEST_COUNT = Counter("agent_core_requests_total", "Total requests", ["method", "endpoint", "status"])
CHAT_LATENCY = Histogram("agent_core_chat_duration_seconds", "Chat request latency")


def _build_memory_manager(settings, db_factory):
    """装配记忆三层。settings.memory_enabled 为真时：工作记忆走 Redis、长期记忆落 DB、
    历史接 Milvus 向量检索。各后端独立 try 降级（Redis→内存 dict，Milvus/embedding→
    pseudo/DB 兜底），任一不可达都不阻断启动。为假时退回全内存 MemoryManager。

    返回 (memory_manager, redis_client)——redis_client 供 lifespan 收尾关闭。
    """
    from src.memory.long_term.profile import ProfileMemory
    from src.memory.long_term.semantic import SemanticMemory
    from src.memory.long_term.episodic import EpisodicMemory
    from src.memory.short_term import ShortTermMemory
    from src.agents.graph import create_llm

    # 短期记忆加载优化（带 LIMIT 的历史加载）恒开，与 memory_enabled 无关；摘要压缩随 llm。
    # llm 复用 model_main：memory_enabled 时归档/摘要共用；关时仅给 short_term 用于摘要。
    st_llm = create_llm(settings, settings.model_main)
    short_term = ShortTermMemory(
        db_factory, llm=st_llm,
        load_limit=settings.short_term_load_limit,
        trigger=settings.short_term_summary_trigger,
        keep=settings.short_term_summary_keep,
    )

    if not settings.memory_enabled:
        return MemoryManager(working=WorkingMemory(), short_term=short_term), None

    # 工作记忆：Redis（ping 失败→内存 fallback）
    redis_client = None
    try:
        import redis.asyncio as aioredis
        redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
        # 连通性在首次调用时才真正校验；这里不阻塞启动
        logger.info("working_memory_redis_configured", url=settings.redis_url.split("@")[-1])
    except Exception as e:
        logger.warning("working_memory_redis_unavailable_fallback_memory", error=str(e))
        redis_client = None
    working = WorkingMemory(redis_client=redis_client, ttl=settings.working_memory_ttl)

    # 向量栈：Embedder（fastembed，失败自动 pseudo）+ MilvusClient（连不上→None，检索降级 DB）
    embedder = None
    milvus_client = None
    try:
        from src.memory.embedding import Embedder
        embedder = Embedder(settings)
    except Exception as e:
        logger.warning("embedder_init_failed", error=str(e))
    try:
        from pymilvus import MilvusClient
        milvus_client = MilvusClient(uri=f"http://{settings.milvus_host}:{settings.milvus_port}")
        logger.info("milvus_configured", host=settings.milvus_host, port=settings.milvus_port)
    except Exception as e:
        logger.warning("milvus_unavailable_fallback_db_recent", error=str(e))
        milvus_client = None

    manager = MemoryManager(
        working=working,
        profile=ProfileMemory(session_factory=db_factory),
        semantic=SemanticMemory(session_factory=db_factory),
        episodic=EpisodicMemory(
            session_factory=db_factory, embedder=embedder, milvus_client=milvus_client,
            collection=settings.memory_collection, dim=settings.embedding_dim,
        ),
        llm=st_llm,
        short_term=short_term,
    )
    return manager, redis_client


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        redis_client = None  # 提前声明，保证 finally 清理不 NameError
        app.state.db_session_factory = await init_db(settings.database_url)
        logger.info("database_initialized", url=settings.database_url.split("@")[-1])

        # 审批闸的 checkpointer：优先用持久化 AsyncPostgresSaver（跨进程/重启可恢复
        # interrupt 状态）；PG 不可达则降级进程内 MemorySaver（demo/无 PG 环境仍可跑）。
        # saver 持有连接池，须在 app 生命周期内保活，故在此 async with 内编译图。
        checkpointer = None
        cm = None
        if settings.approval_enabled:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
                cm = AsyncPostgresSaver.from_conn_string(settings.resolved_checkpoint_db_url)
                checkpointer = await cm.__aenter__()
                await checkpointer.setup()   # 首次建 checkpoint 表（幂等）
                logger.info("checkpointer_postgres_ready")
            except Exception as e:
                logger.warning("checkpointer_postgres_unavailable_fallback_memory", error=str(e))
                if cm is not None:
                    try:
                        await cm.__aexit__(type(e), e, e.__traceback__)
                    except Exception:
                        pass
                    cm = None
                from langgraph.checkpoint.memory import MemorySaver
                checkpointer = MemorySaver()

        # 显式建 MCP client 并全链路传入，使图内节点与预热/关闭共用同一实例。
        from src.mcp_client.client import MCPClientManager
        mcp = MCPClientManager(settings)
        app.state.mcp = mcp

        # 记忆三层装配（依赖 db_session_factory，故在 lifespan 内建）：工作记忆 Redis、
        # 长期记忆 DB、历史 Milvus 向量检索，各后端独立降级。覆盖模块级占位 memory_manager。
        memory_manager, redis_client = _build_memory_manager(settings, app.state.db_session_factory)
        app.state.memory_manager = memory_manager
        app.state.redis_client = redis_client

        # L2 自我反思（judge 仲裁 + 错误记忆持久化）总开关：settings.reflection_enabled
        # （默认 False，见 Settings 注释）。开启则注入 SqlErrorMemoryStore，图内三节点
        # 对 complaint/退款高危场景做仲裁；关闭则 error_store=None，行为与接入前逐行一致。
        error_store = None
        if settings.reflection_enabled:
            from src.reflection.error_memory import SqlErrorMemoryStore
            error_store = SqlErrorMemoryStore(app.state.db_session_factory)
            logger.info("reflection_enabled")
        app.state.error_store = error_store
        app.state.graph = compile_graph(settings, memory_manager, prompts=prompts,
                                        mcp=mcp, checkpointer=checkpointer,
                                        error_store=error_store)

        # 启动预热：把各 agent 工具清单拉满缓存，冷启动延迟从用户请求路径挪到此处。
        # 绝不阻断启动 —— agent-tools 未就绪时仅记 warning，首次请求再自愈。
        from src.tools.mcp_adapter import prewarm_agent_tools
        try:
            stats = await prewarm_agent_tools(mcp)
            logger.info("mcp_prewarm_done", **stats)
        except Exception as e:
            logger.warning("mcp_prewarm_error", error=str(e))

        try:
            yield
        finally:
            try:
                await mcp.close()  # 释放 httpx 连接
            except Exception:
                pass
            if redis_client is not None:
                try:
                    await redis_client.aclose()  # 释放工作记忆 Redis 连接
                except Exception:
                    pass
            if cm is not None:
                await cm.__aexit__(None, None, None)  # 释放 PG 连接池

    app = FastAPI(
        title="Agent Core - 智能客服系统",
        description="Multi-Agent Customer Service System",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Converge CORS from "*" to an explicit allowlist (settings-driven). A
    # wildcard origin is invalid together with credentials per the CORS spec, so
    # if an operator deliberately sets "*" we drop credentials to keep the
    # config valid rather than silently breaking all cross-origin requests.
    cors_origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    allow_credentials = "*" not in cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    setup_opentelemetry(app)
    langfuse_handler = setup_langfuse()

    # 加载 prompt 目录（PromptRegistry），供数据驱动的泛型 Agent node 使用
    from pathlib import Path
    from src.prompts.registry import PromptRegistry
    prompts = PromptRegistry()
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    try:
        loaded = prompts.load_dir_sync(prompts_dir)
        logger.info("prompts_loaded", count=loaded, dir=str(prompts_dir))
    except Exception as e:
        logger.warning("prompts_load_failed", error=str(e))

    # 模块级占位：真正的 memory_manager 在 lifespan 内用 db_session_factory 装配后覆盖
    # （见 _build_memory_manager）。此占位保证无 lifespan 的轻量测试仍可用全内存记忆。
    app.state.settings = settings
    app.state.prompts = prompts
    app.state.langfuse_handler = langfuse_handler
    app.state.memory_manager = MemoryManager(working=WorkingMemory())
    # 注意：app.state.graph 在 lifespan 内编译（依赖 checkpointer 的异步连接池）

    from src.api.auth_routes import router as auth_router
    app.include_router(auth_router)

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse()

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    async def _require_session_owner(db, session_id: str, customer_id: str) -> SessionModel | None:
        """Return the session only if it belongs to customer_id.

        Returns None when the session does not exist yet (caller may create it).
        Raises 403 when the session exists but belongs to a different customer,
        preventing cross-customer access to conversation history.
        """
        session = await db.get(SessionModel, session_id)
        if session is not None and session.customer_id != customer_id:
            logger.warning(
                "session_ownership_violation",
                session_id=session_id,
                request_customer=customer_id,
                owner_customer=session.customer_id,
            )
            raise HTTPException(status_code=403, detail="Session does not belong to this customer")
        return session

    @app.post("/v1/chat/{session_id}")
    async def send_message(session_id: str, request: ChatRequest,
                           customer_id: str = Depends(current_customer)):
        """Send a message and receive a streaming response via SSE."""

        # Verify ownership before the stream starts so a 403 is returned cleanly
        # (raising inside the generator would happen after the response began).
        db_factory = app.state.db_session_factory
        async with db_factory() as db:
            await _require_session_owner(db, session_id, customer_id)

        async def event_generator():
            try:
                async with db_factory() as db:
                    session = await db.get(SessionModel, session_id)
                    if not session:
                        session = SessionModel(id=session_id, customer_id=customer_id)
                        db.add(session)
                        await db.commit()

                    user_msg = MessageModel(
                        session_id=session_id,
                        role="user",
                        content=request.content,
                        timestamp=datetime.now(),
                    )
                    db.add(user_msg)
                    await db.commit()

                # 短期记忆加载：ShortTermMemory 带 LIMIT 只取最近原文 + 滚动摘要；
                # 无 short_term（轻量测试/未装配）时回退原全量 SELECT，保证行为不破。
                st = getattr(app.state.memory_manager, "short_term", None)
                summary_text = ""
                if st is not None:
                    history_messages, summary_text = await st.load(session_id)
                else:
                    async with db_factory() as db:
                        all_msgs = (await db.execute(
                            select(MessageModel).where(MessageModel.session_id == session_id)
                            .order_by(MessageModel.timestamp)
                        )).scalars().all()
                    history_messages = [
                        HumanMessage(content=m.content) if m.role == "user"
                        else AIMessage(content=m.content)
                        for m in all_msgs if m.role in ("user", "assistant")
                    ]

                memory_context = ""
                try:
                    # 用 lifespan 装配的 memory_manager（DB/Redis/Milvus 后端），非模块级占位
                    mem_data = await app.state.memory_manager.load_context(
                        customer_id, session_id, request.content
                    )
                    memory_context = format_memory_for_prompt(mem_data)
                    logger.info("memory_loaded", customer_id=customer_id, context_len=len(memory_context))
                except Exception as e:
                    logger.warning("memory_load_failed", error=str(e))

                # 滚动摘要折进 memory_context（拼进 SystemMessage，永不被历史窗口切掉）
                if summary_text:
                    memory_context = f"## 对话历史摘要\n{summary_text}\n\n{memory_context}".rstrip()

                initial_state = {
                    "messages": history_messages,
                    "intent": None,
                    "confidence": 0.0,
                    "customer_id": customer_id,
                    "session_id": session_id,
                    "customer_info": None,
                    "current_agent": "",
                    "needs_approval": False,
                    "approval_result": None,
                    "resolved": False,
                    "memory_context": memory_context,
                    "failure_count": 0,
                    "routing_count": 0,
                    "handoff_target": None,
                    "plan": None,
                    "agent_results": {},
                    "is_multi_intent": False,
                    "pending_write": None,
                    "pending_writes": {},
                    "approval_decision": None,
                }

                config = {"configurable": {"thread_id": session_id}}
                if app.state.langfuse_handler:
                    config["callbacks"] = [app.state.langfuse_handler]

                graph_result = await app.state.graph.ainvoke(initial_state, config=config)

                # 审批闸中断：图在 approval 节点暂停，不产出最终回复。发 approval
                # 事件让前端（坐席台）确认，随后调 /approve 真实 resume 本 thread。
                interrupts = graph_result.get("__interrupt__")
                if interrupts:
                    payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
                    yield f"data: {json.dumps({'type': 'approval_required', **(payload if isinstance(payload, dict) else {'payload': payload})}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'done', 'agent': 'approval', 'interrupted': True}, ensure_ascii=False)}\n\n"
                    return

                async for chunk in _emit_result(graph_result, session_id, db_factory):
                    yield chunk

                # 回合结束（非热路径：done 已推给客户端）：触发短期记忆压缩，若有消息被
                # 压缩则对该批增量抽事实（做法1，趁原文未被摘要抹掉细节前抽取）。
                # 内部自带 try/except + 无 llm/short_term 降级，绝不影响已返回的响应。
                await app.state.memory_manager.on_turn_end(customer_id, session_id)

            except Exception as e:
                logger.error("chat_error", error=str(e), session_id=session_id)
                fallback = _generate_fallback_response(request.content)
                yield f"data: {json.dumps({'type': 'token', 'content': fallback}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'agent': 'fallback', 'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/chat/{session_id}/approve")
    async def submit_approval(session_id: str, request: ApprovalRequest,
                              customer_id: str = Depends(current_customer)):
        """提交审批决定并 resume 被 approval 闸暂停的图，流式返回后续回复。

        用 thread_id=session_id 定位被中断的图状态，Command(resume=...) 把决定
        送回 approval 节点的 interrupt 调用点，图继续跑 execute 节点（批准则执行
        写工具+生成回复，拒绝则取消），产出经 SSE 推流并落库。
        """
        db_factory = app.state.db_session_factory
        async with db_factory() as db:
            session = await _require_session_owner(db, session_id, customer_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")

        async def event_generator():
            try:
                config = {"configurable": {"thread_id": session_id}}
                if app.state.langfuse_handler:
                    config["callbacks"] = [app.state.langfuse_handler]
                # decisions（逐条写调用 id→bool）透传给 approval_node：多意图批量栅栏
                # 据此逐条批/拒；单意图 / 无 decisions 时以 approved 应用于全部。
                resume_value = {
                    "approved": request.approved,
                    "reason": request.reason,
                    "decisions": request.decisions,
                }
                graph_result = await app.state.graph.ainvoke(
                    Command(resume=resume_value), config=config
                )
                async for chunk in _emit_result(graph_result, session_id, db_factory):
                    yield chunk
            except Exception as e:
                logger.error("approval_resume_error", error=str(e), session_id=session_id)
                fallback = _generate_fallback_response("")
                yield f"data: {json.dumps({'type': 'token', 'content': fallback}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'agent': 'fallback', 'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/chat/{session_id}/end")
    async def end_session(session_id: str, customer_id: str = Depends(current_customer)):
        """显式结束会话并归档到长期记忆（决策 B 的主触发）。

        前端在用户点"结束会话"/关闭窗口/客服结单时调用。载全量消息 → memory_manager
        .on_session_end（LLM 摘要+事实+画像 → 落长期记忆 + 清工作记忆）。归档尽力而为，
        单点失败不整体报错。超时未显式结束的会话由 archive_idle_sessions 兜底。
        """
        db_factory = app.state.db_session_factory
        async with db_factory() as db:
            await _require_session_owner(db, session_id, customer_id)
            msgs = (await db.execute(
                select(MessageModel).where(MessageModel.session_id == session_id)
                .order_by(MessageModel.timestamp)
            )).scalars().all()

        history = [
            HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
            for m in msgs
        ]
        try:
            await app.state.memory_manager.on_session_end(customer_id, session_id, history)
            logger.info("session_archived", session_id=session_id, messages=len(history))
            return {"status": "archived", "session_id": session_id, "message_count": len(history)}
        except Exception as e:
            logger.warning("session_archive_failed", session_id=session_id, error=str(e))
            return {"status": "error", "session_id": session_id, "detail": str(e)}

    @app.get("/v1/chat/{session_id}/history", response_model=ChatHistoryResponse)
    async def get_history(session_id: str, customer_id: str = Depends(current_customer)):
        """Get the conversation history for a session from database.

        The customer_id is derived from the Bearer token, and must match the
        session owner, so one customer cannot read another customer's
        conversation by guessing a session_id.
        """
        db_factory = app.state.db_session_factory
        async with db_factory() as db:
            session = await _require_session_owner(db, session_id, customer_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")

            stmt = select(MessageModel).where(
                MessageModel.session_id == session_id
            ).order_by(MessageModel.timestamp)
            result = await db.execute(stmt)
            messages = result.scalars().all()

        return ChatHistoryResponse(
            session_id=session_id,
            messages=[
                {
                    "role": m.role,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                    "agent": m.agent,
                }
                for m in messages
            ],
        )

    @app.get("/v1/sessions", response_model=SessionListResponse)
    async def list_sessions(customer_id: str = Depends(current_customer)):
        """List the calling customer's sessions, newest first.

        Identity is token-derived, so a customer only ever sees their own
        conversations. Each summary carries a preview (first user message) and a
        message count so the frontend can render a history list without fetching
        every session's full transcript.

        Empty sessions (0 messages) are omitted: a fresh session is created up
        front on every login, so listing them would clutter the history and make
        auto-resume land on a blank conversation.
        """
        db_factory = app.state.db_session_factory
        async with db_factory() as db:
            stmt = (
                select(SessionModel)
                .where(SessionModel.customer_id == customer_id)
                .order_by(SessionModel.updated_at.desc())
            )
            result = await db.execute(stmt)
            sessions = result.scalars().all()

            summaries = []
            for s in sessions:
                count_stmt = select(func.count()).select_from(MessageModel).where(
                    MessageModel.session_id == s.id
                )
                message_count = (await db.execute(count_stmt)).scalar_one()
                if message_count == 0:
                    continue

                preview_stmt = (
                    select(MessageModel.content)
                    .where(MessageModel.session_id == s.id, MessageModel.role == "user")
                    .order_by(MessageModel.timestamp)
                    .limit(1)
                )
                preview = (await db.execute(preview_stmt)).scalars().first()
                if preview and len(preview) > 60:
                    preview = preview[:60] + "…"

                summaries.append({
                    "session_id": s.id,
                    "created_at": s.created_at,
                    "updated_at": s.updated_at,
                    "message_count": message_count,
                    "preview": preview,
                })

        return SessionListResponse(sessions=summaries)

    @app.post("/v1/sessions")
    async def create_session(request: SessionCreateRequest,
                             customer_id: str = Depends(current_customer)):
        """Create a new chat session persisted to database."""
        session_id = str(uuid.uuid4())
        db_factory = app.state.db_session_factory
        async with db_factory() as db:
            session = SessionModel(id=session_id, customer_id=customer_id)
            db.add(session)
            await db.commit()
        return {"session_id": session_id, "customer_id": customer_id}

    return app


def _generate_fallback_response(message: str) -> str:
    """Fallback response when LLM is unavailable."""
    return "抱歉，系统暂时繁忙，请稍后再试。如需紧急帮助，请拨打客服热线。"


async def _emit_result(graph_result: dict, session_id: str, db_factory):
    """把图的最终结果转成 SSE token 流并落库助手消息。

    初次 send 与审批 resume 共用同一段收尾逻辑（取最后一条 AI 回复 → 分块推流 →
    持久化 → done 事件），避免两处重复。
    """
    response_text = ""
    agent_name = graph_result.get("current_agent", "unknown")
    for msg in reversed(graph_result.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "ai":
            response_text = msg.content
            break
    if not response_text:
        response_text = "抱歉，处理您的请求时出现了问题，请稍后再试。"

    for i in range(0, len(response_text), 20):
        chunk = response_text[i:i + 20]
        yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

    async with db_factory() as db:
        db.add(MessageModel(
            session_id=session_id, role="assistant", content=response_text,
            agent=agent_name, timestamp=datetime.now(),
        ))
        await db.commit()

    yield f"data: {json.dumps({'type': 'done', 'agent': agent_name}, ensure_ascii=False)}\n\n"

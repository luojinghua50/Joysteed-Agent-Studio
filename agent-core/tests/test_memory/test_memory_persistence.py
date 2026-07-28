"""记忆持久化 + 向量降级 + 归档闭环单测。

DB 层用 sqlite in-memory factory（同 error_memory 测法）；工作记忆用 fakeredis；
向量检索测降级路径（不依赖真 Milvus）。
"""
import pytest
from datetime import datetime, timedelta

from src.database import init_db, FactModel
from src.memory.long_term.profile import ProfileMemory, UserProfile
from src.memory.long_term.semantic import SemanticMemory
from src.memory.long_term.episodic import EpisodicMemory
from src.memory.manager import MemoryManager, format_memory_for_prompt
from src.memory.entities import extract_entities
from src.memory.decay import format_fact_for_prompt, ConfidenceManager


async def _factory():
    return await init_db("sqlite+aiosqlite:///:memory:")


# ————————————————————— 实体抽取 —————————————————————

class TestExtractEntities:
    def test_from_query_order(self):
        tr = {"query_order": {"order_id": "ORD-001", "status": "shipped", "amount": 299.0}}
        e = extract_entities(tr)
        assert e["order_id"] == "ORD-001"
        assert e["amount"] == 299.0
        assert e["status"] == "shipped"

    def test_from_apply_refund(self):
        tr = {"apply_refund": {"refund_id": "RF-1", "amount": 50, "order_status": "refunded"}}
        e = extract_entities(tr)
        assert e["refund_id"] == "RF-1"
        assert e["order_status"] == "refunded"

    def test_from_json_string_result(self):
        # 读工具经 MCP adapter 序列化为 JSON 字符串（query_order/track_shipping 的真实形态）
        import json
        tr = {"query_order": json.dumps(
            {"order_id": "ORD-001", "status": "shipped", "amount": 299.0}, ensure_ascii=False)}
        e = extract_entities(tr)
        assert e["order_id"] == "ORD-001"
        assert e["amount"] == 299.0

    def test_skips_error_result(self):
        tr = {"apply_refund": {"error": "订单不存在"}}
        assert extract_entities(tr) == {}

    def test_skips_error_json_string(self):
        # adapter 对 error 的降级形态是纯文本"工具调用失败: ..."，非 JSON → 跳过不崩
        tr = {"query_order": "工具调用失败: 订单不存在"}
        assert extract_entities(tr) == {}

    def test_empty_and_none(self):
        assert extract_entities(None) == {}
        assert extract_entities({}) == {}


# ————————————————————— Profile 持久化 —————————————————————

class TestProfilePersistence:
    @pytest.mark.asyncio
    async def test_save_and_reload_across_restart(self):
        f = await _factory()
        pm = ProfileMemory(session_factory=f)
        await pm.save(UserProfile(customer_id="C1", vip_level=3, communication_style="简洁",
                                  sensitive_points=["配送延迟"]))
        # 新实例（模拟重启）指向同一 engine
        pm2 = ProfileMemory(session_factory=f)
        p = await pm2.get("C1")
        assert p.vip_level == 3
        assert p.communication_style == "简洁"
        assert "配送延迟" in p.sensitive_points

    @pytest.mark.asyncio
    async def test_get_missing_returns_default(self):
        pm = ProfileMemory(session_factory=await _factory())
        p = await pm.get("unknown")
        assert p.customer_id == "unknown" and p.vip_level == 0

    @pytest.mark.asyncio
    async def test_update_from_conversation_merges(self):
        f = await _factory()
        pm = ProfileMemory(session_factory=f)
        await pm.save(UserProfile(customer_id="C1", sensitive_points=["A"]))
        await pm.update_from_conversation("C1", {"sensitive_points": ["B"], "communication_style": "详细"})
        p = await pm.get("C1")
        assert set(p.sensitive_points) == {"A", "B"}
        assert p.communication_style == "详细"


# ————————————————————— Semantic 持久化 + 置信度 —————————————————————

class TestSemanticPersistence:
    @pytest.mark.asyncio
    async def test_set_get_across_restart(self):
        f = await _factory()
        sm = SemanticMemory(session_factory=f)
        await sm.set_fact("C1", "address", "北京朝阳", source="user_explicit")
        sm2 = SemanticMemory(session_factory=f)
        assert (await sm2.get_facts("C1"))["address"] == "北京朝阳"

    @pytest.mark.asyncio
    async def test_confidence_from_source(self):
        f = await _factory()
        sm = SemanticMemory(session_factory=f)
        await sm.set_fact("C1", "k", "v", source="user_explicit")
        scored = await sm.get_facts_scored("C1")
        assert scored[0]["confidence"] >= 0.9  # 刚写入、user_explicit → 高

    @pytest.mark.asyncio
    async def test_confidence_decays_with_age(self):
        f = await _factory()
        sm = SemanticMemory(session_factory=f)
        await sm.set_fact("C1", "k", "v", source="agent_inferred")
        # 手工把 updated_at 改老
        from sqlalchemy import update
        async with f() as db:
            await db.execute(update(FactModel).where(FactModel.customer_id == "C1")
                             .values(updated_at=datetime.now() - timedelta(days=100)))
            await db.commit()
        scored = await sm.get_facts_scored("C1")
        # agent_inferred 初始 0.6，衰减 100 天后应明显下降
        assert scored[0]["confidence"] < 0.6

    @pytest.mark.asyncio
    async def test_upsert_same_key(self):
        f = await _factory()
        sm = SemanticMemory(session_factory=f)
        await sm.set_fact("C1", "phone", "138", source="agent_inferred")
        await sm.set_fact("C1", "phone", "139", source="user_explicit")
        facts = await sm.get_facts("C1")
        assert facts["phone"] == "139"  # 覆盖不重复


# ————————————————————— Episodic 持久化 + 降级检索 —————————————————————

class TestEpisodicPersistence:
    @pytest.mark.asyncio
    async def test_save_and_recent_fallback(self):
        # 无 embedder/milvus → search 降级 DB 最近 N 条
        f = await _factory()
        em = EpisodicMemory(session_factory=f)
        await em.save_episode("s1", "C1", [], summary="查订单已解决", intent="order_query")
        await em.save_episode("s2", "C1", [], summary="投诉物流", intent="complaint")
        results = await em.search("任意query", "C1", top_k=5)
        assert len(results) == 2
        assert {r.summary for r in results} == {"查订单已解决", "投诉物流"}

    @pytest.mark.asyncio
    async def test_persist_across_restart(self):
        f = await _factory()
        await EpisodicMemory(session_factory=f).save_episode("s1", "C1", [], summary="历史摘要")
        em2 = EpisodicMemory(session_factory=f)
        assert (await em2.search("q", "C1"))[0].summary == "历史摘要"

    @pytest.mark.asyncio
    async def test_customer_isolation(self):
        f = await _factory()
        em = EpisodicMemory(session_factory=f)
        await em.save_episode("s1", "C1", [], summary="A的")
        await em.save_episode("s2", "C2", [], summary="B的")
        assert len(await em.search("q", "C1")) == 1


# ————————————————————— 置信度格式化 —————————————————————

class TestFormatFact:
    def test_high_confidence_plain(self):
        assert format_fact_for_prompt("addr", "北京", 0.9) == "- addr: 北京"

    def test_mid_confidence_flagged(self):
        assert "可能已变更" in format_fact_for_prompt("addr", "北京", 0.5)

    def test_low_confidence_reverify(self):
        assert "待确认" in format_fact_for_prompt("addr", "北京", 0.2)


# ————————————————————— 归档闭环 —————————————————————

class _StubLLM:
    def __init__(self, reply):
        self.reply = reply

    async def ainvoke(self, prompt):
        from langchain_core.messages import AIMessage
        return AIMessage(content=self.reply)


class TestOnSessionEnd:
    @pytest.mark.asyncio
    async def test_archives_episode_fact_profile_and_clears_working(self):
        f = await _factory()
        from src.memory.working import WorkingMemory
        digest = ('{"summary":"用户改地址已确认","intent":"general","resolution":"resolved",'
                  '"satisfaction":0.9,"facts":{"address":"上海浦东"},'
                  '"profile":{"communication_style":"简洁","sensitive_points":["物流"]}}')
        mm = MemoryManager(
            working=WorkingMemory(),
            profile=ProfileMemory(session_factory=f),
            semantic=SemanticMemory(session_factory=f),
            episodic=EpisodicMemory(session_factory=f),
            llm=_StubLLM(digest),
        )
        await mm.working.set_entity("s1", "order_id", "ORD-001")
        from langchain_core.messages import HumanMessage
        await mm.on_session_end("C1", "s1", [HumanMessage(content="我地址改成上海浦东")])

        # 历史摘要落地
        eps = await mm.episodic.search("q", "C1")
        assert eps and eps[0].summary == "用户改地址已确认"
        # 事实落地（user_explicit 高置信度）
        assert (await mm.semantic.get_facts("C1"))["address"] == "上海浦东"
        # 画像更新
        assert (await mm.profile.get("C1")).communication_style == "简洁"
        # 工作记忆清空
        assert await mm.working.get("s1") == {}

    @pytest.mark.asyncio
    async def test_no_llm_degrades_gracefully(self):
        f = await _factory()
        from src.memory.working import WorkingMemory
        mm = MemoryManager(
            working=WorkingMemory(),
            episodic=EpisodicMemory(session_factory=f),
            semantic=SemanticMemory(session_factory=f),
            profile=ProfileMemory(session_factory=f),
            llm=None,
        )
        from langchain_core.messages import HumanMessage
        # 不抛异常，仍落一条降级摘要
        await mm.on_session_end("C1", "s1", [HumanMessage(content="你好")])
        assert len(await mm.episodic.search("q", "C1")) == 1


# ————————————————————— 增量事实抽取（做法1核心）—————————————————————

class _SeqLLM:
    """按调用顺序返回不同应答的假模型，用于区分 episode 摘要调用 vs 事实抽取调用。"""
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    async def ainvoke(self, prompt):
        from langchain_core.messages import AIMessage
        self.calls += 1
        r = self.replies.pop(0) if self.replies else "{}"
        return AIMessage(content=r)


async def _seed_session(f, sid="s1", cid="C1", n_pairs=0):
    from src.database import SessionModel, MessageModel
    async with f() as db:
        db.add(SessionModel(id=sid, customer_id=cid))
        await db.commit()
        for i in range(n_pairs):
            db.add(MessageModel(session_id=sid, role="user", content=f"u{i}"))
            db.add(MessageModel(session_id=sid, role="assistant", content=f"a{i}"))
        await db.commit()


class TestIncrementalExtract:
    @pytest.mark.asyncio
    async def test_extract_facts_writes_and_is_idempotent(self):
        f = await _factory()
        fact_json = '{"facts":{"phone":"13800001234"},"profile":{"sensitive_points":["物流"]}}'
        mm = MemoryManager(
            profile=ProfileMemory(session_factory=f),
            semantic=SemanticMemory(session_factory=f),
            episodic=EpisodicMemory(session_factory=f),
            llm=_SeqLLM([fact_json, fact_json]),
        )
        batch = [{"role": "user", "content": "我手机号 13800001234，你们物流太慢"}]
        await mm._extract_facts("C1", batch)
        await mm._extract_facts("C1", batch)  # 再抽一次 → 幂等
        facts = await mm.semantic.get_facts("C1")
        assert facts["phone"] == "13800001234"
        prof = await mm.profile.get("C1")
        assert prof.sensitive_points == ["物流"]  # merge 去重，不重复累积

    @pytest.mark.asyncio
    async def test_on_turn_end_extracts_when_compressed(self):
        f = await _factory()
        from src.memory.short_term import ShortTermMemory
        # 短期摘要用一个 llm；事实抽取共用 mm.llm
        fact_json = '{"facts":{"address":"北京朝阳"},"profile":{}}'
        st = ShortTermMemory(f, llm=_SeqLLM(["滚动摘要"]), trigger=30, keep=10)
        mm = MemoryManager(
            profile=ProfileMemory(session_factory=f),
            semantic=SemanticMemory(session_factory=f),
            episodic=EpisodicMemory(session_factory=f),
            llm=_SeqLLM([fact_json]),
            short_term=st,
        )
        await _seed_session(f, n_pairs=20)  # 40 条 > trigger → 触发压缩
        await mm.on_turn_end("C1", "s1")
        # 压缩批次被增量抽取 → address 落库
        assert (await mm.semantic.get_facts("C1")).get("address") == "北京朝阳"

    @pytest.mark.asyncio
    async def test_on_turn_end_no_compress_no_extract(self):
        f = await _factory()
        from src.memory.short_term import ShortTermMemory
        st = ShortTermMemory(f, llm=_SeqLLM(["x"]), trigger=30, keep=10)
        mm_llm = _SeqLLM(['{"facts":{"k":"v"}}'])
        mm = MemoryManager(
            profile=ProfileMemory(session_factory=f),
            semantic=SemanticMemory(session_factory=f),
            episodic=EpisodicMemory(session_factory=f),
            llm=mm_llm, short_term=st,
        )
        await _seed_session(f, n_pairs=5)  # 10 条 < trigger → 不压缩
        await mm.on_turn_end("C1", "s1")
        assert await mm.semantic.get_facts("C1") == {}  # 未压缩 → 未抽取
        assert mm_llm.calls == 0

    @pytest.mark.asyncio
    async def test_fidelity_old_detail_captured_before_summary(self):
        """保真核心：老对话里的电话，在被摘要抹掉前经增量抽取已进 facts。

        模拟：短期摘要 llm 故意产出一个**不含电话**的笼统摘要（模拟摘要丢细节），
        但事实抽取 llm 从原文抽到了电话 → 验证电话不因摘要而丢。
        """
        f = await _factory()
        from src.memory.short_term import ShortTermMemory
        # 摘要故意笼统、不含电话（模拟摘要抹掉细节）
        st = ShortTermMemory(f, llm=_SeqLLM(["用户咨询了一些问题"]), trigger=30, keep=10)
        mm = MemoryManager(
            profile=ProfileMemory(session_factory=f),
            semantic=SemanticMemory(session_factory=f),
            episodic=EpisodicMemory(session_factory=f),
            llm=_SeqLLM(['{"facts":{"phone":"13900002222"},"profile":{}}']),
            short_term=st,
        )
        await _seed_session(f, n_pairs=20)
        await mm.on_turn_end("C1", "s1")
        # 摘要里没有电话，但事实抽取在压缩那一刻从原文抓到了
        async with f() as db:
            from src.database import SessionModel
            s = await db.get(SessionModel, "s1")
            assert "13900002222" not in (s.summary or "")   # 摘要确实没保住细节
        assert (await mm.semantic.get_facts("C1"))["phone"] == "13900002222"  # 但 facts 保住了


# ————————————————————— load_context 集成 —————————————————————

class TestLoadContext:
    @pytest.mark.asyncio
    async def test_scored_facts_in_context_and_prompt(self):
        f = await _factory()
        from src.memory.working import WorkingMemory
        mm = MemoryManager(
            working=WorkingMemory(),
            profile=ProfileMemory(session_factory=f),
            semantic=SemanticMemory(session_factory=f),
            episodic=EpisodicMemory(session_factory=f),
        )
        await mm.semantic.set_fact("C1", "address", "北京", source="user_explicit")
        ctx = await mm.load_context("C1", "s1", "查订单")
        assert isinstance(ctx["facts"], list)
        assert ctx["facts"][0]["key"] == "address"
        prompt = format_memory_for_prompt(ctx)
        assert "address" in prompt and "北京" in prompt


# ————————————————————— WorkingMemory over Redis (fakeredis) —————————————————————

class TestWorkingMemoryRedis:
    @pytest.mark.asyncio
    async def test_set_get_clear_via_redis(self):
        import fakeredis.aioredis
        from src.memory.working import WorkingMemory
        r = fakeredis.aioredis.FakeRedis(decode_responses=True)
        wm = WorkingMemory(redis_client=r, ttl=60)
        await wm.set_entity("s1", "order_id", "ORD-001")
        await wm.set_entity("s1", "amount", 299.0)
        assert (await wm.get("s1"))["order_id"] == "ORD-001"
        assert (await wm.get_entity("s1", "amount")) == 299.0
        await wm.clear("s1")
        assert await wm.get("s1") == {}
        await r.aclose()

    @pytest.mark.asyncio
    async def test_ttl_set_on_write(self):
        import fakeredis.aioredis
        from src.memory.working import WorkingMemory
        r = fakeredis.aioredis.FakeRedis(decode_responses=True)
        wm = WorkingMemory(redis_client=r, ttl=60)
        await wm.set_entity("s1", "k", "v")
        assert 0 < await r.ttl("working:s1") <= 60
        await r.aclose()

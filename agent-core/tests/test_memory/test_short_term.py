"""短期记忆：滚动摘要 + LIMIT 加载单测。

sqlite in-memory factory 造 session+messages；mock llm 驱动压缩分支。
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.database import init_db, SessionModel, MessageModel
from src.memory.short_term import ShortTermMemory


async def _seed(n_pairs: int):
    """建一个 session + n 对 (user,assistant) 消息，返回 (factory, session_id)。"""
    f = await init_db("sqlite+aiosqlite:///:memory:")
    sid = "s1"
    async with f() as db:
        db.add(SessionModel(id=sid, customer_id="C1"))
        await db.commit()
        for i in range(n_pairs):
            db.add(MessageModel(session_id=sid, role="user", content=f"用户消息{i}"))
            db.add(MessageModel(session_id=sid, role="assistant", content=f"助手回复{i}"))
        await db.commit()
    return f, sid


class _StubLLM:
    """记录调用并返回固定摘要的假模型。"""
    def __init__(self, reply="融合摘要：用户咨询过订单与退款。"):
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        return AIMessage(content=self.reply)


class TestLoad:
    @pytest.mark.asyncio
    async def test_no_summary_returns_recent_and_empty_summary(self):
        f, sid = await _seed(3)  # 6 条消息
        st = ShortTermMemory(f, llm=None, load_limit=30)
        msgs, summary = await st.load(sid)
        assert summary == ""
        assert len(msgs) == 6
        assert isinstance(msgs[0], HumanMessage) and isinstance(msgs[1], AIMessage)
        assert msgs[0].content == "用户消息0"  # 正序还原

    @pytest.mark.asyncio
    async def test_limit_caps_loaded_messages(self):
        f, sid = await _seed(25)  # 50 条消息
        st = ShortTermMemory(f, llm=None, load_limit=10)
        msgs, _ = await st.load(sid)
        assert len(msgs) == 10  # 只回最近 load_limit 条
        # 最近 10 条应是最后 5 对：用户消息20..24 / 助手回复20..24
        assert msgs[-1].content == "助手回复24"
        assert msgs[0].content == "用户消息20"

    @pytest.mark.asyncio
    async def test_load_after_summary_only_returns_uncompressed(self):
        f, sid = await _seed(10)  # 20 条, id 1..20
        # 手工设摘要指针到 id=12（前 12 条已压缩）
        async with f() as db:
            s = await db.get(SessionModel, sid)
            s.summary = "早期摘要"
            s.summary_upto_id = 12
            await db.commit()
        st = ShortTermMemory(f, llm=None, load_limit=30)
        msgs, summary = await st.load(sid)
        assert summary == "早期摘要"
        assert len(msgs) == 8  # id 13..20
        assert msgs[0].content == "用户消息6"  # id13 = user(2*6+1=13)


class TestCompress:
    @pytest.mark.asyncio
    async def test_no_llm_never_compresses(self):
        f, sid = await _seed(30)  # 60 条 > trigger
        st = ShortTermMemory(f, llm=None, trigger=30, keep=10)
        assert await st.compress_if_needed(sid) == []  # 无 llm → 空批次
        async with f() as db:
            s = await db.get(SessionModel, sid)
            assert s.summary is None and s.summary_upto_id == 0

    @pytest.mark.asyncio
    async def test_below_trigger_no_compress(self):
        f, sid = await _seed(5)  # 10 条 < trigger 30
        llm = _StubLLM()
        st = ShortTermMemory(f, llm=llm, trigger=30, keep=10)
        assert await st.compress_if_needed(sid) == []  # 未到阈值 → 空批次
        assert llm.calls == 0  # 没到阈值，连 LLM 都不调

    @pytest.mark.asyncio
    async def test_over_trigger_returns_batch_and_advances_pointer(self):
        f, sid = await _seed(20)  # 40 条, id 1..40 > trigger 30
        llm = _StubLLM()
        st = ShortTermMemory(f, llm=llm, trigger=30, keep=10)
        batch = await st.compress_if_needed(sid)
        assert len(batch) == 30  # 返回被压批次：40-keep(10)=30 条
        assert batch[0] == {"role": "user", "content": "用户消息0"}  # 最老那批的纯值
        assert llm.calls == 1
        async with f() as db:
            s = await db.get(SessionModel, sid)
            assert s.summary == "融合摘要：用户咨询过订单与退款。"
            assert s.summary_upto_id == 30  # 指针前进到 id=30
        # 压缩后再 load：未压缩数应降到 keep=10
        msgs, summary = await st.load(sid)
        assert summary.startswith("融合摘要")
        assert len(msgs) == 10  # id 31..40

    @pytest.mark.asyncio
    async def test_rolling_fuses_prev_summary(self):
        f, sid = await _seed(20)  # 40 条
        llm = _StubLLM()
        st = ShortTermMemory(f, llm=llm, trigger=30, keep=10)
        assert len(await st.compress_if_needed(sid)) == 30  # 第一次压缩返回批次
        # 再灌 30 条触发第二次
        async with f() as db:
            for i in range(15):
                db.add(MessageModel(session_id=sid, role="user", content=f"新用户{i}"))
                db.add(MessageModel(session_id=sid, role="assistant", content=f"新助手{i}"))
            await db.commit()
        assert len(await st.compress_if_needed(sid)) > 0  # 第二次也压
        assert llm.calls == 2  # 第二次压缩又调一次（融合既有摘要）

    @pytest.mark.asyncio
    async def test_missing_session_returns_empty(self):
        f = await init_db("sqlite+aiosqlite:///:memory:")
        st = ShortTermMemory(f, llm=_StubLLM(), trigger=1)
        assert await st.compress_if_needed("nope") == []

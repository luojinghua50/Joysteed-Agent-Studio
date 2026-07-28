"""短期记忆（对话历史）：滚动摘要 + 最近 N 条原文加载。

解决两个缺口：
1) 每轮加载不再全量 SELECT——只取 id>summary_upto_id 的未压缩原文，带 LIMIT。
2) 超窗口的老消息不再直接丢弃——回合结束后压缩成滚动摘要（session.summary），
   摘要经 memory_context 注入 SystemMessage（永不被各 agent 的历史窗口切掉）。

滚动累积：session.summary 存已压缩老消息的摘要，summary_upto_id 记已覆盖到的最大
message id。加载 = 摘要 + (id>upto 的原文)，无 gap、无重叠。

优雅降级：llm 为空 → 不压缩，仅保留 LIMIT 加载优化；任何 DB/LLM 异常 → 记 warning
不抛，绝不阻断对话。
"""
import structlog
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

logger = structlog.get_logger()


SUMMARY_PROMPT = """你是对话摘要助手。把「已有摘要」和「新增的较早对话」融合成一段简洁的中文摘要，
供后续轮次快速回顾。保留：主要话题、已解决事项、关键实体（订单号/金额/地址等）、未决问题。
不要逐句复述，控制在 200 字内。

## 已有摘要（可能为空）
{prev_summary}

## 新增的较早对话
{new_dialogue}

## 只输出融合后的摘要正文，不要额外说明
"""


class ShortTermMemory:
    """会话对话历史的加载与滚动摘要压缩。"""

    def __init__(self, session_factory, llm=None,
                 load_limit: int = 30, trigger: int = 30, keep: int = 10):
        self.session_factory = session_factory
        self.llm = llm
        self.load_limit = load_limit
        self.trigger = trigger
        self.keep = keep

    async def load(self, session_id: str) -> tuple[list[BaseMessage], str]:
        """返回 (最近原文消息列表, 摘要文本)。

        原文 = id>summary_upto_id 的消息，按 id 倒序取最多 load_limit 条再正序还原
        （带 LIMIT，避免全量 SELECT）。摘要 = session.summary（可能为空）。
        """
        from sqlalchemy import select
        from src.database import SessionModel, MessageModel

        async with self.session_factory() as db:
            session = await db.get(SessionModel, session_id)
            upto = session.summary_upto_id if session else 0
            summary = (session.summary if session else "") or ""

            rows = (await db.execute(
                select(MessageModel)
                .where(MessageModel.session_id == session_id, MessageModel.id > upto)
                .order_by(MessageModel.id.desc())
                .limit(self.load_limit)
            )).scalars().all()

        recent = list(reversed(rows))  # 倒序取最近 N 条 → 正序还原时间顺序
        messages: list[BaseMessage] = []
        for m in recent:
            if m.role == "user":
                messages.append(HumanMessage(content=m.content))
            elif m.role == "assistant":
                messages.append(AIMessage(content=m.content))
        return messages, summary

    async def compress_if_needed(self, session_id: str) -> list[dict]:
        """回合结束后调用：未压缩消息数超过 trigger 时，把最老的一批压进滚动摘要。

        返回**被压缩的那批原文** [{"role","content"}]，供调用方（MemoryManager）对这批
        做增量事实抽取（原文在手、细节全、量小）；无压缩/无 llm → 返回 []。全程 try/except
        不抛。压缩逻辑只管摘要，不碰 facts——保持 ShortTermMemory 与长期记忆解耦。
        """
        if self.llm is None:
            return []
        try:
            from sqlalchemy import select, func
            from src.database import SessionModel, MessageModel

            async with self.session_factory() as db:
                session = await db.get(SessionModel, session_id)
                if session is None:
                    return []
                upto = session.summary_upto_id or 0

                pending = (await db.execute(
                    select(func.count()).select_from(MessageModel)
                    .where(MessageModel.session_id == session_id, MessageModel.id > upto)
                )).scalar() or 0
                if pending < self.trigger:
                    return []

                # 压缩最老的 (pending - keep) 条，保留最近 keep 条原文
                to_compress = pending - self.keep
                rows = (await db.execute(
                    select(MessageModel)
                    .where(MessageModel.session_id == session_id, MessageModel.id > upto)
                    .order_by(MessageModel.id.asc())
                    .limit(to_compress)
                )).scalars().all()
                if not rows:
                    return []

                prev_summary = session.summary or ""
                # 在会话内取出纯值，避免 ORM 对象出会话后失效
                batch = [{"role": m.role, "content": m.content} for m in rows]
                new_upto = rows[-1].id

            new_dialogue = "\n".join(
                f"{'用户' if b['role'] == 'user' else '客服'}: {b['content']}" for b in batch
            )[:6000]

            # LLM 调用放在 DB 会话外，避免长事务占连接
            resp = await self.llm.ainvoke(SUMMARY_PROMPT.format(
                prev_summary=prev_summary or "（无）", new_dialogue=new_dialogue,
            ))
            summary_text = (getattr(resp, "content", str(resp)) or "").strip()
            if not summary_text:
                return []

            async with self.session_factory() as db:
                session = await db.get(SessionModel, session_id)
                if session is None:
                    return []
                session.summary = summary_text
                session.summary_upto_id = new_upto
                await db.commit()

            logger.info("short_term_compressed", session=session_id,
                        compressed=len(batch), summary_upto_id=new_upto)
            return batch
        except Exception as e:
            logger.warning("short_term_compress_failed", session=session_id, error=str(e))
            return []

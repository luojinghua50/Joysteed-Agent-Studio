import json

import structlog

from src.memory.working import WorkingMemory
from src.memory.long_term.profile import ProfileMemory, UserProfile
from src.memory.long_term.episodic import EpisodicMemory
from src.memory.long_term.semantic import SemanticMemory
from src.memory.decay import format_fact_for_prompt

logger = structlog.get_logger()


def _format_dialogue(messages: list) -> str:
    """把消息批次（[{"role","content"}] 或 LangChain 消息对象）拼成对话文本，供 LLM 输入。
    统一走"用户/客服"角色标注；上限 6000 字防超长。"""
    lines = []
    for m in messages or []:
        if isinstance(m, dict):
            role, content = m.get("role", ""), m.get("content", "")
        else:
            role = getattr(m, "type", "")
            content = getattr(m, "content", str(m))
        who = "用户" if role in ("user", "human") else "客服"
        lines.append(f"{who}: {content}")
    return "\n".join(lines)[:6000]


# EPISODE 摘要（检索用，可容忍丢细节）：输入 = 短期滚动摘要 + 最近未压缩原文，
# 不再看全量原文（省 token、不与短期记忆重复摘要）。只出会话级元信息，不抽事实。
EPISODE_DIGEST_PROMPT = """你是客服会话分析助手。根据「历史摘要」和「最近对话」概括这次会话，输出 JSON。

## 输出字段
- summary: 一句话会话摘要（用户问了什么、怎么解决的）
- intent: 主要意图（如 order_query / refund / complaint / tech_support / general）
- resolution: resolved / unresolved / escalated 三选一
- satisfaction: 0~1 的满意度估计，无法判断填 null

## 历史摘要（较早对话的压缩，可能为空）
{summary}

## 最近对话
{recent}

## 只输出 JSON，不要额外文字
"""


# FACT 抽取（要精确，不容丢）：对**一批原文**抽用户明确告知的硬信息。在短期记忆每次
# 压缩那批时增量调用（原文在手、细节全），故绝不从摘要抽——摘要会抹掉低频细节。
FACT_EXTRACT_PROMPT = """你是客服信息抽取助手。仔细阅读这批对话，抽取**用户明确告知的持久信息**，输出 JSON。
务必保留用户亲口说出的硬信息（电话/地址/证件号/邮箱/偏好等），不要遗漏、不要臆测。

## 输出字段
- facts: 对象，用户明确告知的持久事实（如 {{"address": "北京朝阳", "phone": "138..."}}），无则 {{}}
- profile: 对象，可含 communication_style（简洁/详细/情绪化）、sensitive_points（字符串数组，如投诉敏感点），无则 {{}}

## 对话
{conversation}

## 只输出 JSON，不要额外文字
"""


class MemoryManager:
    """Unified memory management entry point."""

    def __init__(
        self,
        working: WorkingMemory | None = None,
        profile: ProfileMemory | None = None,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
        llm=None,
        short_term=None,
    ):
        self.working = working or WorkingMemory()
        self.profile = profile or ProfileMemory()
        self.episodic = episodic or EpisodicMemory()
        self.semantic = semantic or SemanticMemory()
        self.llm = llm  # 会话摘要/抽取用；None 时 on_session_end 走极简降级
        # 短期记忆（对话历史加载 + 滚动摘要）；routes 装配注入，None 时 routes 回退全量加载
        self.short_term = short_term

    async def load_context(
        self, customer_id: str, session_id: str, current_message: str
    ) -> dict:
        """Load all relevant memory context before agent execution."""
        profile = await self.profile.get(customer_id)
        history = await self.episodic.search(current_message, customer_id, top_k=3)
        facts = await self._load_scored_facts(customer_id)
        entities = await self.working.get(session_id)

        return {
            "profile": profile,
            "relevant_history": history,
            "facts": facts,
            "entities": entities,
        }

    async def _load_scored_facts(self, customer_id: str) -> list[dict]:
        """事实带当前置信度（DB 后端衰减；内存后端也支持）。"""
        get_scored = getattr(self.semantic, "get_facts_scored", None)
        if get_scored is not None:
            return await get_scored(customer_id)
        # 极老 fallback：无置信度，按满分处理
        facts = await self.semantic.get_facts(customer_id)
        return [{"key": k, "value": v, "confidence": 1.0} for k, v in facts.items()]

    async def on_turn_end(self, customer_id: str, session_id: str):
        """回合结束入口：触发短期记忆压缩；若有消息被压缩，对**恰好这批**做增量事实抽取。

        增量抽取的意义（做法1核心）：老消息在被摘要"抹掉细节"之前，趁原文还在就把
        用户明确告知的硬信息（电话/地址等）抽进 facts/profile。避免会话结束才从摘要
        抽取导致低频细节丢失。short_term 未装配则空操作。
        """
        if self.short_term is None:
            return
        batch = await self.short_term.compress_if_needed(session_id)
        if batch:
            await self._extract_facts(customer_id, batch)

    async def _extract_facts(self, customer_id: str, messages: list):
        """对一批原文（[{"role","content"}] 或消息对象）用 LLM 抽 facts/profile 并落库。

        set_fact 按 (customer_id,key) upsert、update_from_conversation merge 去重 →
        重复调用幂等，增量累积安全。无 llm 或空批次 → 跳过。全程 try/except 不抛。
        """
        if self.llm is None or not messages:
            return
        convo = _format_dialogue(messages)
        data = await self._llm_json(FACT_EXTRACT_PROMPT.format(conversation=convo), "fact_extract")
        for key, value in (data.get("facts") or {}).items():
            try:
                await self.update_fact(customer_id, key, str(value), source="user_explicit")
            except Exception as e:
                logger.warning("extract_fact_failed", key=key, error=str(e))
        prof = data.get("profile") or {}
        if prof:
            try:
                await self.profile.update_from_conversation(customer_id, prof)
            except Exception as e:
                logger.warning("extract_profile_failed", error=str(e))

    async def on_session_end(
        self, customer_id: str, session_id: str, messages: list
    ):
        """会话结束归档：生成 episode 摘要（复用短期摘要+最近原文）+ 补抽未压缩尾巴的
        事实 → 落长期记忆 → 清工作记忆。

        做法1：episode 摘要用"短期滚动摘要 + 最近未压缩原文"（省 token、不重复摘要）；
        facts/profile 只对未压缩尾巴补抽（老的已在 on_turn_end 压缩时增量抽过），保证
        每条消息事实被抽且仅抽一次。每步独立 try/except，不阻断其余。
        """
        # —— 取 episode 的输入：优先复用短期记忆的 (最近原文, 摘要)；无 short_term 回退全量 ——
        recent, summary_text = messages, ""
        if self.short_term is not None:
            try:
                recent, summary_text = await self.short_term.load(session_id)
            except Exception as e:
                logger.warning("session_end_load_failed", session=session_id, error=str(e))
                recent, summary_text = messages, ""

        digest = await self._episode_digest(recent, summary_text)

        # 1) 历史摘要（结构化真相源 + 向量）
        try:
            await self.episodic.save_episode(
                session_id, customer_id, recent,
                summary=digest.get("summary"),
                intent=digest.get("intent", "general"),
                resolution=digest.get("resolution", "resolved"),
                satisfaction=digest.get("satisfaction"),
            )
        except Exception as e:
            logger.warning("archive_episode_failed", session=session_id, error=str(e))

        # 2) 补抽未压缩尾巴的事实/画像（老消息已在 on_turn_end 增量抽过）
        try:
            await self._extract_facts(customer_id, recent)
        except Exception as e:
            logger.warning("archive_extract_failed", session=session_id, error=str(e))

        # 3) 清工作记忆（会话内实体不跨会话）
        try:
            await self.working.clear(session_id)
        except Exception as e:
            logger.warning("archive_working_clear_failed", session=session_id, error=str(e))

    async def _episode_digest(self, recent: list, summary_text: str) -> dict:
        """用"短期摘要 + 最近原文"生成 episode 元信息 JSON。无 llm/解析失败 → 空 dict。"""
        if self.llm is None or (not recent and not summary_text):
            return {}
        prompt = EPISODE_DIGEST_PROMPT.format(
            summary=summary_text or "（无）", recent=_format_dialogue(recent) or "（无）",
        )
        return await self._llm_json(prompt, "episode_digest")

    async def _llm_json(self, prompt: str, tag: str) -> dict:
        """调 LLM 并解析 JSON（容忍 ```json 包裹）。异常 → 空 dict + warning。"""
        try:
            resp = await self.llm.ainvoke(prompt)
            content = (getattr(resp, "content", str(resp)) or "").strip()
            content = content.removeprefix("```json").removeprefix("```").removesuffix("```")
            return json.loads(content)
        except Exception as e:
            logger.warning("llm_json_failed", tag=tag, error=str(e))
            return {}

    async def update_fact(
        self, customer_id: str, field: str, value: str, source: str = "agent_inferred"
    ):
        """Update a fact in semantic memory (confidence derived from source)."""
        await self.semantic.set_fact(customer_id, field, value, source=source)

    async def purge_customer_memory(self, customer_id: str):
        """GDPR compliance: delete all memory for a customer."""
        await self.profile.delete(customer_id)
        await self.episodic.delete_all(customer_id)
        await self.semantic.delete_all(customer_id)


def format_memory_for_prompt(memory: dict) -> str:
    """Format memory context as a prompt fragment for agent consumption."""
    parts = []

    profile = memory.get("profile")
    if profile:
        parts.append(
            f"## 用户信息\n"
            f"VIP等级: {profile.vip_level}, "
            f"沟通风格: {profile.communication_style or '未知'}"
        )
        if profile.sensitive_points:
            parts.append(f"注意事项: {', '.join(profile.sensitive_points)}")

    facts = memory.get("facts", [])
    if facts:
        parts.append("## 用户事实")
        for f in facts:
            # facts 可能是 [{key,value,confidence}]（新）或 {k:v}（极老 fallback）
            if isinstance(f, dict) and "key" in f:
                parts.append(format_fact_for_prompt(f["key"], f["value"], f.get("confidence", 1.0)))

    history = memory.get("relevant_history", [])
    if history:
        parts.append("## 相关历史")
        for h in history:
            ts = h.timestamp.strftime("%m-%d") if h.timestamp else "未知"
            parts.append(f"- [{ts}] {h.summary} (结果: {h.resolution})")

    entities = memory.get("entities", {})
    if entities:
        parts.append(f"## 当前会话实体\n{json.dumps(entities, ensure_ascii=False)}")

    return "\n".join(parts)


async def archive_idle_sessions(db_factory, memory_manager, idle_minutes: int = 30) -> int:
    """超时归档兜底：扫 updated_at 超过 idle_minutes 无活动的会话，逐个 on_session_end。

    大量会话不会有人显式点"结束"（关页面/断网），故需此兜底补归档（配合 /end 接口的
    显式触发形成双保险）。本函数不自带调度——生产由 k8s CronJob / APScheduler 定时
    打一个内部端点触发，或在独立 worker 里周期调用。返回归档的会话数。
    """
    from datetime import datetime, timedelta
    from sqlalchemy import select
    from src.database import SessionModel, MessageModel

    cutoff = datetime.now() - timedelta(minutes=idle_minutes)
    archived = 0
    async with db_factory() as db:
        rows = (await db.execute(
            select(SessionModel).where(SessionModel.updated_at < cutoff)
        )).scalars().all()
        sessions = [(s.id, s.customer_id) for s in rows]

    for session_id, customer_id in sessions:
        try:
            async with db_factory() as db:
                msgs = (await db.execute(
                    select(MessageModel).where(MessageModel.session_id == session_id)
                    .order_by(MessageModel.timestamp)
                )).scalars().all()
            from langchain_core.messages import HumanMessage, AIMessage
            history = [
                HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
                for m in msgs
            ]
            await memory_manager.on_session_end(customer_id, session_id, history)
            archived += 1
        except Exception as e:
            logger.warning("idle_archive_failed", session=session_id, error=str(e))
    logger.info("idle_sessions_archived", count=archived)
    return archived

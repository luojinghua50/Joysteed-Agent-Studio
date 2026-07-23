"""Order persistence layer — backs the order MCP tools with PostgreSQL.

Orders used to live in two in-process dicts (MOCK_ORDERS / MOCK_SHIPPING), so
every write (refund/modify/urge) was fake and lost on restart. This module makes
orders, their shipping, and refunds durable via SQLAlchemy async, and turns the
refund flow into a real state machine (see apply_refund below).

Mirrors ticket_server.store.TicketStore: module-level singleton, lazy
ensure_init() that creates tables + seeds demo data on first call.
"""
from __future__ import annotations

from datetime import datetime, timedelta, UTC

from sqlalchemy import (
    String, Text, DateTime, Numeric, Boolean, JSON, ForeignKey, func, select,
)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

VALID_ORDER_STATUSES = ["pending", "shipped", "delivered", "refunded", "partial_refunded"]
VALID_REFUND_STATUSES = ["processing", "completed", "rejected"]
# 退款已完成的订单终态：禁止改单/催发货/再全额退
_LOCKED_STATUSES = {"refunded"}


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    """Naive UTC to match TIMESTAMP WITHOUT TIME ZONE columns (asyncpg rejects tz-aware)."""
    return datetime.now(UTC).replace(tzinfo=None)


class OrderModel(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    amount: Mapped[float] = mapped_column(Numeric(10, 2))
    items: Mapped[list] = mapped_column(JSON, default=list)
    payment_method: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class OrderShippingModel(Base):
    __tablename__ = "order_shippings"

    order_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("orders.order_id", ondelete="CASCADE"), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="not_shipped")
    carrier: Mapped[str] = mapped_column(String(32), default="")
    tracking_number: Mapped[str] = mapped_column(String(32), default="")
    location: Mapped[str] = mapped_column(String(64), default="")
    tracks: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, onupdate=_utcnow)


class RefundModel(Base):
    __tablename__ = "refunds"

    refund_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    order_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("orders.order_id", ondelete="CASCADE"), index=True)
    amount: Mapped[float] = mapped_column(Numeric(10, 2))
    reason: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="completed")
    need_return: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    eta: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class OrderStore:
    """Async order store. One instance per process, initialized at startup."""

    def __init__(self, database_url: str):
        # connect_args.timeout 给 asyncpg 建连设上限：DB 不可达时秒级抛错而非
        # 无限挂起（连接黑洞会拖垮工具调用、也会让本机 pytest 卡到超时）。
        self._engine = create_async_engine(
            database_url, echo=False, connect_args={"timeout": 5},
        )
        self._session = async_sessionmaker(self._engine, expire_on_commit=False)
        self._initialized = False

    async def init(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await self._seed()

    async def ensure_init(self) -> None:
        """Idempotent lazy init — FastMCP manages its own lifespan, so init on
        first tool call instead of a startup hook (same as TicketStore)."""
        if self._initialized:
            return
        await self.init()
        self._initialized = True

    async def _seed(self) -> None:
        """Seed the 3 demo orders + 2 shippings once, so tools have data on first run.

        created_at 用相对时间（非写死的 2024 日期）：ORD-002 设为 2 天前，让
        check_refund_eligibility 的「7天无理由」分支可被演示到，不再永久穿帮。
        """
        async with self._session() as db:
            existing = (await db.execute(select(func.count()).select_from(OrderModel))).scalar()
            if existing:
                return
            now = _utcnow()
            db.add_all([
                OrderModel(
                    order_id="ORD-001", customer_id="C001", status="shipped",
                    amount=299.00, payment_method="支付宝",
                    items=[{"name": "无线耳机", "qty": 1, "price": 299.00}],
                    created_at=now - timedelta(days=5),
                ),
                OrderModel(
                    order_id="ORD-002", customer_id="C001", status="delivered",
                    amount=1599.00, payment_method="微信支付",
                    items=[{"name": "智能手表", "qty": 1, "price": 1599.00}],
                    created_at=now - timedelta(days=2),   # 7天无理由期内，可演示
                ),
                OrderModel(
                    order_id="ORD-003", customer_id="C002", status="pending",
                    amount=89.00, payment_method="信用卡",
                    items=[{"name": "手机壳", "qty": 2, "price": 44.50}],
                    created_at=now - timedelta(days=1),
                ),
            ])
            db.add_all([
                OrderShippingModel(
                    order_id="ORD-001", status="in_transit", carrier="顺丰速运",
                    tracking_number="SF1234567890", location="杭州转运中心",
                    tracks=[
                        {"time": "2024-01-16 08:00", "description": "包裹已揽收"},
                        {"time": "2024-01-16 14:00", "description": "到达深圳分拣中心"},
                        {"time": "2024-01-17 06:00", "description": "到达杭州转运中心"},
                        {"time": "2024-01-17 10:00", "description": "派送中"},
                    ],
                ),
                OrderShippingModel(
                    order_id="ORD-002", status="delivered", carrier="中通快递",
                    tracking_number="ZT9876543210", location="已签收",
                    tracks=[
                        {"time": "2024-01-11 09:00", "description": "包裹已揽收"},
                        {"time": "2024-01-12 15:00", "description": "到达目的城市"},
                        {"time": "2024-01-12 18:00", "description": "已签收"},
                    ],
                ),
            ])
            await db.commit()

    @staticmethod
    def _order_to_dict(o: OrderModel) -> dict:
        return {
            "order_id": o.order_id, "status": o.status, "amount": float(o.amount),
            "created_at": o.created_at.strftime("%Y-%m-%d %H:%M:%S") if o.created_at else "",
            "items": o.items or [], "customer_id": o.customer_id,
            "payment_method": o.payment_method,
        }

    @staticmethod
    def _shipping_to_dict(s: OrderShippingModel) -> dict:
        return {
            "status": s.status, "carrier": s.carrier,
            "tracking_number": s.tracking_number, "location": s.location,
            "tracks": s.tracks or [],
        }

    async def get_order(self, order_id: str) -> dict | None:
        await self.ensure_init()
        async with self._session() as db:
            o = await db.get(OrderModel, order_id)
            return self._order_to_dict(o) if o else None

    async def get_shipping(self, order_id: str) -> dict | None:
        await self.ensure_init()
        async with self._session() as db:
            s = await db.get(OrderShippingModel, order_id)
            return self._shipping_to_dict(s) if s else None

    async def refunded_total(self, db: AsyncSession, order_id: str) -> float:
        """该订单已成功退款（completed）的累计金额，用于部分退款的超额校验。"""
        total = (await db.execute(
            select(func.coalesce(func.sum(RefundModel.amount), 0))
            .where(RefundModel.order_id == order_id, RefundModel.status == "completed")
        )).scalar()
        return float(total or 0)

    async def _next_refund_id(self, db: AsyncSession) -> str:
        count = (await db.execute(select(func.count()).select_from(RefundModel))).scalar() or 0
        return f"RF-{count + 1:05d}"

    @staticmethod
    def _eligibility(order: dict) -> dict:
        """退款资格判定（落库版，逻辑沿用原 mock 的 check_refund_eligibility）。

        返回 {eligible, reason, need_return}。delivered 按 created_at 距今天数判
        7 天无理由——seed 用相对时间后此分支可真实命中。
        """
        status = order["status"]
        if status in ("refunded", "partial_refunded"):
            # 已全额退：不可再退；部分退：交由超额校验把关，这里视作可继续
            if status == "refunded":
                return {"eligible": False, "reason": "订单已全额退款", "need_return": False}
            return {"eligible": True, "reason": "部分退款订单，剩余金额可继续退", "need_return": True}
        if status == "pending":
            return {"eligible": True, "reason": "订单未发货，可全额退款", "need_return": False}
        if status == "shipped":
            return {"eligible": True, "reason": "订单已发货，需要退回商品后退款", "need_return": True}
        if status == "delivered":
            created = datetime.strptime(order["created_at"], "%Y-%m-%d %H:%M:%S")
            days_since = (_utcnow() - created).days
            if days_since <= 7:
                return {"eligible": True, "reason": "7天无理由退款期内", "need_return": True}
            return {"eligible": False, "reason": "已超过7天无理由退款期", "need_return": False}
        return {"eligible": False, "reason": "订单状态不支持退款", "need_return": False}

    async def check_eligibility(self, order_id: str) -> dict:
        order = await self.get_order(order_id)
        if not order:
            return {"error": f"订单 {order_id} 不存在", "eligible": False}
        elig = self._eligibility(order)
        return {"eligible": elig["eligible"], "reason": elig["reason"]}

    async def apply_refund(self, order_id: str, amount: float, reason: str) -> dict:
        """退款落库 + 订单状态联动（方案 B：一步置完成，无 returning 中间态）。

        硬规则：①先查资格，不符合（如 delivered 超 7 天）直接拒、不建退款单；
        ②已退总额 + 本次 > 订单金额则拒；③全额→refunded，部分→partial_refunded。
        """
        await self.ensure_init()
        async with self._session() as db:
            o = await db.get(OrderModel, order_id)
            if not o:
                return {"error": f"订单 {order_id} 不存在"}

            order_dict = self._order_to_dict(o)
            elig = self._eligibility(order_dict)
            if not elig["eligible"]:
                return {"error": f"订单不符合退款条件：{elig['reason']}"}

            order_amount = float(o.amount)
            already = await self.refunded_total(db, order_id)
            if amount <= 0:
                return {"error": "退款金额必须大于 0"}
            if already + amount > order_amount:
                return {"error": f"退款金额 {amount} 加上已退 {already} 超过订单金额 {order_amount}"}

            refund_id = await self._next_refund_id(db)
            eta = _utcnow() + timedelta(days=3)
            db.add(RefundModel(
                refund_id=refund_id, order_id=order_id, amount=amount, reason=reason,
                status="completed", need_return=elig["need_return"], eta=eta,
            ))

            # 订单状态联动：累计退款达订单金额→全额 refunded，否则 partial_refunded
            new_total = already + amount
            o.status = "refunded" if new_total >= order_amount else "partial_refunded"
            o.updated_at = _utcnow()
            await db.commit()

            return {
                "refund_id": refund_id, "order_id": order_id, "amount": amount,
                "status": "completed", "order_status": o.status,
                "eta": eta.strftime("%Y-%m-%d"),
                "message": f"退款已到账（{'全额' if o.status == 'refunded' else '部分'}退款），退款单号 {refund_id}",
            }

    async def modify_order(self, order_id: str, field: str, value: str) -> dict:
        await self.ensure_init()
        async with self._session() as db:
            o = await db.get(OrderModel, order_id)
            if not o:
                return {"error": f"订单 {order_id} 不存在"}
            if o.status in _LOCKED_STATUSES:
                return {"error": "订单已退款，无法修改"}
            if o.status != "pending":
                return {"error": "订单已发货，无法修改"}
            allowed_fields = ["shipping_address", "phone", "note"]
            if field not in allowed_fields:
                return {"error": f"不支持修改字段: {field}，可修改: {allowed_fields}"}
            # 这些字段不在 orders 表落列（演示用），仅校验通过即认为成功
            return {"success": True, "message": f"订单 {order_id} 的 {field} 已更新为: {value}"}

    async def urge_shipping(self, order_id: str) -> dict:
        await self.ensure_init()
        async with self._session() as db:
            o = await db.get(OrderModel, order_id)
            if not o:
                return {"error": f"订单 {order_id} 不存在"}
            if o.status in _LOCKED_STATUSES:
                return {"success": False, "message": "订单已退款，无需催促"}
            if o.status == "pending":
                return {"success": True, "message": "已通知仓库加急处理，预计24小时内发货"}
            if o.status == "shipped":
                return {"success": True, "message": "已联系快递公司催促配送，预计今日送达"}
            return {"success": False, "message": "订单已完成，无需催促"}

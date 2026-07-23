# Order 工具落库（mock → 真实 PostgreSQL）技术方案

> 把 order-mcp 从内存 mock dict 改为真实 PostgreSQL 持久化，对齐 ticket 的落库范式。
> 生成日期：2026-06-29

---

## 1. 现状与目标

**现状**：`agent-tools/order_server/server.py` 第 1 行即 `"""Order MCP Server with mock data."""`，订单/物流是写死的内存 `dict`（`MOCK_ORDERS` 3 笔、`MOCK_SHIPPING` 2 笔），所有写操作（退款/改单/催发货）只返回成功消息、**不落库**，重启即恢复初始值。

**目标**：order 像 ticket 一样用 PostgreSQL 持久化，订单状态、退款记录可跨重启保留、可被多方（AI 工具 / 未来的后台 UI）共享。

**直接照搬范式**：项目里 ticket（`ticket_server/store.py` + `server.py`）已经是「SQLAlchemy async + asyncpg + 模块级单例 store + `ensure_init` 懒建表 seed」的成熟范式，order 复刻这套即可，无需新引依赖（`sqlalchemy[asyncio]` / asyncpg 已在用）。

---

## 2. 改动范围总览

| 文件 | 改动 | 量级 |
|---|---|---|
| `agent-tools/order_server/store.py` | **新建**，OrderStore + 三张表模型（orders / order_shippings / refunds）+ 退款状态流转 | 中（~240 行，仿 ticket store） |
| `agent-tools/order_server/server.py` | 6 个工具改为调 store；删 MOCK 常量 | 中（改约 100 行） |
| `db-init/01-create-databases.sql` | 无需改（agent_core 库已存在，建表交给 ensure_init） | 无 |
| `docker-compose.yml` order-mcp | 加 `ORDER_DATABASE_URL` env + `depends_on: postgres(healthy)` | 小（4 行） |
| `agent-core/tests/test_agents/test_order_mcp.py` | 套 skip 守卫 + module loop_scope（仿刚修的 ticket 测试） | 中（改约 40 行） |
| `test_idempotency.py` / `test_registry_grants.py` | **不动**（只用 order 工具名字符串，不碰数据） | 无 |

**整体评估：中等改动，1 个新文件 + 4 处修改，无架构级变更、无新依赖、无 schema 迁移工具需求。** 风险点集中在「写操作语义从假变真」和「7 天无理由的时间穿帮」两处业务逻辑，下面详述。

---

## 3. 数据模型设计（三表）

订单与物流拆开：物流有独立的生命周期（揽收/转运/签收）和多条轨迹，不应塞进 orders 的 JSON 列。退款单独成表，才能支撑「累计退款不超订单金额」「部分退款多次」等内存 mock 做不到的规则。

### orders 表
| 列 | 类型 | 说明 |
|---|---|---|
| order_id | String(16) PK | ORD-001 |
| customer_id | String(32) index | C001 |
| status | String(16) index | pending/shipped/delivered/refunded/partial_refunded |
| amount | Numeric(10,2) | 订单总额 |
| items | JSON | `[{"name","qty","price"}]` |
| payment_method | String(32) | 支付宝/微信/信用卡 |
| created_at | DateTime server_default | seed 用相对时间，见 4.3 |
| updated_at | DateTime onupdate | |

### order_shippings 表
| 列 | 类型 | 说明 |
|---|---|---|
| order_id | String(16) PK, FK→orders | 一对一 |
| status | String(16) | not_shipped/in_transit/delivered |
| carrier | String(32) | 顺丰速运/中通快递 |
| tracking_number | String(32) | SF1234567890 |
| location | String(64) | 当前位置/已签收 |
| tracks | JSON | `[{"time","description"}]` 轨迹列表 |
| updated_at | DateTime onupdate | |

> pending 订单没有 shipping 行（track_shipping 查不到行 → 返回 not_shipped）。

### refunds 表
| 列 | 类型 | 说明 |
|---|---|---|
| refund_id | String(16) PK | RF-00001（顺序号，替代 mock 的 random） |
| order_id | String(16) FK→orders, index | |
| amount | Numeric(10,2) | 本次退款金额 |
| reason | Text | |
| status | String(16) | processing/completed/rejected（B 方案下实际只用 completed/rejected） |
| need_return | Boolean | 是否需退回商品（shipped/delivered=true） |
| created_at | DateTime | |
| eta | DateTime | 预计到账 |

---

## 3b. 退款状态流转（本次落库的核心业务设计）

落库后存在**两个状态机**，通过退款动作联动。mock 现状是 `apply_refund` 只返回一个 `processing`、订单状态完全不变——落库后必须真正联动。

### 订单状态机（orders.status）
```
pending ─┐
shipped ─┼─→ refunded          (全额退款，apply_refund 一步到位)
delivered┘ └→ partial_refunded  (部分退款，金额<订单总额；剩余仍可再退)
```
- pending / shipped / delivered — 现有不变
- refunded — 全额退款完成（终态，状态锁）
- partial_refunded — 部分退款完成（剩余金额仍可再退）
- ~~refunding~~ — **B 方案下不实现**：apply_refund 直接置达终态，无「处理中」锁定态

### 退款单状态机（refunds.status）
```
processing ─→ completed   (apply_refund 一步置完成，见决策)
         └──→ rejected    (不符合资格)
```
- processing — 落库瞬时态（apply_refund 内部短暂经过，随即置 completed）
- completed — 退款到账（apply_refund 直接置达）
- rejected — 不符合条件被拒（超 7 天 / 超额）
- ~~returning~~ — **已决定不实现**（见下方决策）：无真实物流回调，不引入「退货中」中间态

### 联动规则（apply_refund 按资格分支决定路径）
| 订单状态 | 资格判定（沿用 check_refund_eligibility） | 退款单 | 订单转为 |
|---|---|---|---|
| pending | 可全额退 | completed | refunded（全额）/ partial_refunded（部分） |
| shipped | 需退回商品后退 | completed | refunded / partial_refunded |
| delivered ≤7天 | 7天无理由 | completed | refunded / partial_refunded |
| delivered >7天 | 不可退 | **拒绝创建**（rejected 不入库或入库标记） | 不变 |

> need_return 字段仍保留（记录该退款「业务上是否需退货」，shipped/delivered=true），仅用于展示/留痕，不再驱动状态流转。

### 三条必须落库的硬规则
1. **超 7 天直接拒**：mock 的 apply_refund 现在不调 check_refund_eligibility，任何 delivered 都能退。落库后必须先查资格，超 7 天返回 error、不建退款单。**这是 mock 当前的逻辑漏洞，落库时一并修掉。**
2. **累计退款不超订单金额**：部分退款可多次，但 `已退总额(completed) + 本次 > amount` 要拒。需对 refunds 表按 order_id 求和——内存 mock 做不到，正是落库价值点。全额退→refunded，部分退→partial_refunded（剩余仍可再退）。
3. **退款完成后状态锁**：订单进入 refunded 后，modify_order / urge_shipping / 再次 apply_refund（全额）应拒绝；partial_refunded 仍允许对剩余金额继续退。

### 决策已定：returning → completed 推进方式 = **方案 B（直接置完成）**
apply_refund 资格通过后，退款单直接置 completed、订单直接置 refunded/partial_refunded，**不引入 returning 中间态、不新增 confirm_refund 工具**。理由：demo 无真实物流回调，「退货中」中间态无法真实推进。

→ **工具数保持 6 个不变**，refunds.status 实际只用到 completed / rejected。

---

## 4. 关键实现要点（含已知坑）

### 4.1 复刻 ticket 的落库范式
- `OrderStore.__init__` 建 async engine，**务必带 `connect_args={"timeout": 5}`**（刚在 ticket store 加过，避免 DB 不可达时连接挂死拖垮工具调用）。
- `ensure_init()` 懒建表 + seed 三笔 demo 订单（ORD-001/002/003），幂等（已有数据则跳过），与 ticket 一致——FastMCP 没有干净的 startup hook，靠首次工具调用触发。
- 写 DateTime 列**必须用 naive UTC**（`datetime.now(UTC).replace(tzinfo=None)`），否则 asyncpg 对 tz-aware 报错——这是 rag 那条已知坑（[[rag-naive-utc-datetime]]）在 order 的复现，seed 和 created_at 都要注意。

### 4.2 写操作语义从「假成功」变「真落库」
mock 里 `apply_refund`/`modify_order`/`urge_shipping` 都只返回消息、不改数据。落库后要真正改状态：
- `apply_refund` → 写 refunds 表 + 订单状态流转（可选）。
- `modify_order` → 真正 UPDATE orders（mock 现在连改地址都是假的）。
- `urge_shipping` → 可保持「只返回消息」（催发货本就是触发外部动作，无状态变更），或记一个催单计数。

这是行为变化最大的地方：**集成测试和 demo 脚本里凡是依赖「退款后订单依然不变」的隐含假设都会变**。需要逐个工具确认新语义。

### 4.3 ⚠️ 7 天无理由退款的时间穿帮（务必处理）
`check_refund_eligibility` 对 delivered 订单用 `created_at`(写死 2024-01-10) 和 `datetime.now()`(2026) 算天数 → 永远落进「已超过 7 天」。**落库 seed 时若仍用 2024 硬编码日期，这条业务逻辑在 demo 里永久穿帮。** 方案：seed 用相对时间（`now - timedelta(days=2)` 之类），让「7 天内可退」分支能被演示到。

### 4.4 并发与连接池（测试侧）
order_server 同样是模块级单例 `store`，连接池绑事件循环。test_order_mcp.py 改造时要套用刚在 ticket 测试验证过的模板：
- module 级 `order_db` fixture 用短超时探测 DB，连不上 `pytest.skip`；
- 4+ 个 DB 依赖测试标 `@pytest.mark.asyncio(loop_scope="module")` 共享同一事件循环，避免 asyncpg `another operation is in progress`（[[agent-core-test-infra-pitfalls]]）。
- 本机真跑设 `ORDER_DATABASE_URL=...@localhost:5432/agent_core`。

---

## 5. 实施步骤（建议顺序）

1. 写 `order_server/store.py`（OrderStore + 模型 + ensure_init + seed），seed 用相对时间。
2. 改 `order_server/server.py`：6 个工具改调 store，删 MOCK 常量；保留工具签名与返回结构不变（这样 agent-core 侧、registry 授权、幂等缓存全不用动）。
3. 改 docker-compose.yml order-mcp：加 `ORDER_DATABASE_URL` + `depends_on: postgres(healthy)`。
4. 改 test_order_mcp.py：套 skip 守卫 + module loop_scope；调整依赖「假写操作」的断言。
5. 重建 order-mcp 容器，端到端验证（查单 / 退款落库 / 7天资格 / 物流）。
6. 全套件回归。

**保持工具的入参和返回 JSON 结构不变**是控制改动范围的关键——只要返回形状一致，agent-core 的工具调用层、prompt、registry 授权、幂等守卫都零改动，改动被完全限制在 agent-tools + 一个测试文件内。

---

## 6. 工作量与风险结论

- **工作量**：中等偏上。1 个新文件（store.py ~240 行：三表模型 + 退款状态流转）+ 1 个 server 改写（~100 行，工具数保持 6 个不变）+ compose 4 行 + 1 个测试文件改造（~50 行，含退款流转新断言）。约半天到一天可完成并验证。
- **无新依赖、无 schema 迁移工具**（懒建表）、**无 agent-core 侧改动**（工具契约不变）。
- **主要风险**：
  1. 退款状态流转是真正的新业务逻辑（订单/退款两状态机联动 + 三条硬规则），不是纯搬运——需逐工具确认新行为，更新依赖「假写操作」的测试/demo；
  2. 7 天无理由时间穿帮——seed 必须用相对时间，且 apply_refund 落库时要真正接上 check_refund_eligibility（修掉 mock 漏洞）；
  3. naive UTC datetime 坑——复用已知规避；
  4. 测试连接池跨循环——复用刚验证的 module loop_scope 模板。
- **决策已定**：returning→completed 采用方案 B（apply_refund 直接置完成），不新增工具，工具数保持 6 个。
- 除退款流转属新逻辑外，其余风险点都有项目内现成规避先例（ticket store / rag / 刚修的测试基础设施），不存在未知技术障碍。

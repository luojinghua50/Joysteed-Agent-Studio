import { useCallback, useEffect, useState } from 'react';
import { listTickets, createTicket, type Ticket } from '@/services/ticket';
import {
  s,
  statusColor,
  priorityColor,
  STATUS_LABELS,
  PRIORITY_LABELS,
} from '@/styles/theme';
import { TicketDetail } from './TicketDetail';

interface TicketBoardProps {
  seat: string;
  agents: string[];
}

export function TicketBoard({ seat, agents }: TicketBoardProps) {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [priorityFilter, setPriorityFilter] = useState('');
  const [mineOnly, setMineOnly] = useState(false);
  const [openId, setOpenId] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);

  // create form
  const [cTitle, setCTitle] = useState('');
  const [cCustomer, setCCustomer] = useState('C001');
  const [cDesc, setCDesc] = useState('');
  const [cPriority, setCPriority] = useState('medium');

  const refresh = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const { tickets } = await listTickets({
        status: statusFilter || undefined,
        priority: priorityFilter || undefined,
        assigned_to: mineOnly ? seat : undefined,
      });
      setTickets(tickets);
    } catch (e) {
      setError(`加载工单失败:${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [statusFilter, priorityFilter, mineOnly, seat]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleCreate() {
    if (!cTitle.trim() || !cCustomer.trim()) return;
    setError('');
    try {
      await createTicket({
        customer_id: cCustomer.trim(),
        title: cTitle.trim(),
        description: cDesc.trim(),
        priority: cPriority,
      });
      setCTitle('');
      setCDesc('');
      setCPriority('medium');
      setShowCreate(false);
      await refresh();
    } catch (e) {
      setError(`创建失败:${(e as Error).message}`);
    }
  }

  return (
    <div style={s.container}>
      {error && <div style={s.error}>{error}</div>}

      <div style={s.card}>
        <div style={{ ...s.row, justifyContent: 'space-between' }}>
          <div style={s.row}>
            <select style={s.select} value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
              <option value="">全部状态</option>
              {Object.entries(STATUS_LABELS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v}
                </option>
              ))}
            </select>
            <select style={s.select} value={priorityFilter} onChange={(e) => setPriorityFilter(e.target.value)}>
              <option value="">全部优先级</option>
              {Object.entries(PRIORITY_LABELS).map(([k, v]) => (
                <option key={k} value={k}>
                  优先级 {v}
                </option>
              ))}
            </select>
            <label style={{ ...s.muted, display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
              <input type="checkbox" checked={mineOnly} onChange={(e) => setMineOnly(e.target.checked)} />
              只看我的 ({seat})
            </label>
            <button style={s.btnGhost} onClick={refresh}>
              刷新
            </button>
          </div>
          <button style={s.btn} onClick={() => setShowCreate((v) => !v)}>
            {showCreate ? '取消' : '+ 新建工单'}
          </button>
        </div>

        {showCreate && (
          <div style={{ marginTop: 16, paddingTop: 16, borderTop: `1px solid #eee` }}>
            <div style={{ ...s.row, marginBottom: 10 }}>
              <input style={{ ...s.input, flex: '1 1 200px' }} placeholder="标题(必填)" value={cTitle} onChange={(e) => setCTitle(e.target.value)} />
              <input style={{ ...s.input, width: 140 }} placeholder="客户ID" value={cCustomer} onChange={(e) => setCCustomer(e.target.value)} />
              <select style={s.select} value={cPriority} onChange={(e) => setCPriority(e.target.value)}>
                {Object.entries(PRIORITY_LABELS).map(([k, v]) => (
                  <option key={k} value={k}>
                    优先级 {v}
                  </option>
                ))}
              </select>
            </div>
            <div style={s.row}>
              <input style={{ ...s.input, flex: 1 }} placeholder="问题描述" value={cDesc} onChange={(e) => setCDesc(e.target.value)} />
              <button style={{ ...s.btn, opacity: cTitle.trim() ? 1 : 0.6 }} disabled={!cTitle.trim()} onClick={handleCreate}>
                创建
              </button>
            </div>
          </div>
        )}
      </div>

      <div style={s.card}>
        {loading ? (
          <div style={s.empty}>加载中…</div>
        ) : tickets.length === 0 ? (
          <div style={s.empty}>没有符合条件的工单。</div>
        ) : (
          <table style={s.table}>
            <thead>
              <tr>
                <th style={s.th}>工单号</th>
                <th style={s.th}>标题</th>
                <th style={s.th}>客户</th>
                <th style={s.th}>优先级</th>
                <th style={s.th}>状态</th>
                <th style={s.th}>坐席</th>
                <th style={s.th}>更新时间</th>
              </tr>
            </thead>
            <tbody>
              {tickets.map((t) => (
                <tr key={t.ticket_id} style={s.trClickable} onClick={() => setOpenId(t.ticket_id)}>
                  <td style={{ ...s.td, ...s.link }}>{t.ticket_id}</td>
                  <td style={s.td}>{t.title}</td>
                  <td style={s.td}>{t.customer_id}</td>
                  <td style={s.td}>
                    <span style={{ ...s.pill, background: priorityColor(t.priority) }}>
                      {PRIORITY_LABELS[t.priority] || t.priority}
                    </span>
                  </td>
                  <td style={s.td}>
                    <span style={{ ...s.pill, background: statusColor(t.status) }}>
                      {STATUS_LABELS[t.status] || t.status}
                    </span>
                  </td>
                  <td style={s.td}>{t.assigned_to}</td>
                  <td style={{ ...s.td, ...s.muted }}>{t.updated_at}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {openId && (
        <TicketDetail
          ticketId={openId}
          seat={seat}
          agents={agents}
          onClose={() => setOpenId(null)}
          onChanged={refresh}
        />
      )}
    </div>
  );
}

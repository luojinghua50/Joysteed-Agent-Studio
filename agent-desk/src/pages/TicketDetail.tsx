import { useEffect, useState } from 'react';
import {
  getTicket,
  updateStatus,
  reassignTicket,
  addComment,
  type TicketDetail as TicketDetailData,
} from '@/services/ticket';
import {
  s,
  statusColor,
  priorityColor,
  STATUS_LABELS,
  PRIORITY_LABELS,
  tokens,
} from '@/styles/theme';

interface TicketDetailProps {
  ticketId: string;
  seat: string;
  agents: string[];
  onClose: () => void;
  onChanged: () => void;
}

const STATUSES = ['open', 'in_progress', 'resolved', 'closed'];

export function TicketDetail({ ticketId, seat, agents, onClose, onChanged }: TicketDetailProps) {
  const [t, setT] = useState<TicketDetailData | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [comment, setComment] = useState('');

  async function refresh() {
    setError('');
    try {
      setT(await getTicket(ticketId));
    } catch (e) {
      setError((e as Error).message);
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticketId]);

  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    setError('');
    try {
      await fn();
      await refresh();
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function submitComment() {
    if (!comment.trim()) return;
    await act(() => addComment(ticketId, seat, comment.trim()));
    setComment('');
  }

  return (
    <div style={s.drawer} onClick={onClose}>
      <div style={s.drawerPanel} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={s.sectionTitle}>工单详情 {ticketId}</div>
          <button style={s.btnGhost} onClick={onClose}>
            关闭
          </button>
        </div>
        {error && <div style={s.error}>{error}</div>}
        {!t ? (
          <div style={s.empty}>加载中…</div>
        ) : (
          <>
            <div style={{ ...s.card, marginBottom: 16 }}>
              <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8 }}>{t.title}</div>
              <div style={{ ...s.muted, marginBottom: 12, lineHeight: 1.6 }}>{t.description}</div>
              <div style={{ ...s.row, gap: 8 }}>
                <span style={{ ...s.pill, background: statusColor(t.status) }}>
                  {STATUS_LABELS[t.status] || t.status}
                </span>
                <span style={{ ...s.pill, background: priorityColor(t.priority) }}>
                  优先级 {PRIORITY_LABELS[t.priority] || t.priority}
                </span>
                <span style={s.muted}>客户 {t.customer_id}</span>
                <span style={s.muted}>坐席 {t.assigned_to}</span>
              </div>
              <div style={{ ...s.muted, marginTop: 8, fontSize: 12 }}>
                创建 {t.created_at} · 更新 {t.updated_at}
              </div>
            </div>

            <div style={{ ...s.card, marginBottom: 16 }}>
              <div style={s.sectionTitle}>操作</div>
              <div style={{ marginBottom: 12 }}>
                <div style={{ ...s.muted, marginBottom: 6 }}>状态流转</div>
                <div style={s.row}>
                  {STATUSES.map((st) => (
                    <button
                      key={st}
                      style={{
                        ...s.btnGhost,
                        ...(t.status === st
                          ? { borderColor: statusColor(st), color: statusColor(st), fontWeight: 600 }
                          : {}),
                      }}
                      disabled={busy || t.status === st}
                      onClick={() => act(() => updateStatus(ticketId, st))}
                    >
                      {STATUS_LABELS[st]}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <div style={{ ...s.muted, marginBottom: 6 }}>转派坐席</div>
                <select
                  style={s.select}
                  value={t.assigned_to}
                  disabled={busy}
                  onChange={(e) => act(() => reassignTicket(ticketId, e.target.value))}
                >
                  {!agents.includes(t.assigned_to) && (
                    <option value={t.assigned_to}>{t.assigned_to}</option>
                  )}
                  {agents.map((a) => (
                    <option key={a} value={a}>
                      {a}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div style={s.card}>
              <div style={s.sectionTitle}>处理记录 ({t.comments.length})</div>
              {t.comments.length === 0 ? (
                <div style={{ ...s.muted, marginBottom: 12 }}>暂无记录</div>
              ) : (
                t.comments.map((c) => (
                  <div key={c.id} style={s.comment}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <strong style={{ color: c.author === 'system' ? tokens.textMuted : tokens.text }}>
                        {c.author}
                      </strong>
                      <span style={{ ...s.muted, fontSize: 12 }}>{c.created_at}</span>
                    </div>
                    <div style={{ marginTop: 4 }}>{c.comment}</div>
                  </div>
                ))
              )}
              <div style={{ ...s.row, marginTop: 12 }}>
                <input
                  style={{ ...s.input, flex: 1 }}
                  placeholder="添加处理记录…"
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && submitComment()}
                />
                <button
                  style={{ ...s.btn, opacity: busy || !comment.trim() ? 0.6 : 1 }}
                  disabled={busy || !comment.trim()}
                  onClick={submitComment}
                >
                  提交
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

import { useEffect, useState } from 'react';
import { TicketBoard } from '@/pages/TicketBoard';
import { listAgents } from '@/services/ticket';
import { s, tokens } from '@/styles/theme';

type View = 'tickets' | 'handoff' | 'calls';

const NAV: { key: View; label: string; icon: string }[] = [
  { key: 'tickets', label: '工单处理', icon: '🎫' },
  { key: 'handoff', label: '人工会话', icon: '💬' },
  { key: 'calls', label: '智能外呼', icon: '📞' },
];

const VIEW_TITLES: Record<View, string> = {
  tickets: '工单处理',
  handoff: '人工会话接管',
  calls: '智能外呼',
};

function Placeholder({ text }: { text: string }) {
  return (
    <div style={s.container}>
      <div style={s.card}>
        <div style={s.empty}>{text}</div>
      </div>
    </div>
  );
}

function App() {
  const [view, setView] = useState<View>('tickets');
  const [agents, setAgents] = useState<string[]>([]);
  const [seat, setSeat] = useState('agent-001');

  useEffect(() => {
    listAgents()
      .then(({ agents }) => {
        setAgents(agents);
        if (agents.length > 0) setSeat(agents[0]);
      })
      .catch(() => setAgents([]));
  }, []);

  return (
    <div style={s.app}>
      <aside style={s.sidebar}>
        <div style={s.sidebarBrand}>
          <span>🎧</span> 坐席工作台
        </div>
        {NAV.map((n) => (
          <div
            key={n.key}
            style={{ ...s.navItem, ...(view === n.key ? s.navItemActive : {}) }}
            onClick={() => setView(n.key)}
          >
            <span>{n.icon}</span> {n.label}
          </div>
        ))}
        <div style={s.seatBar}>
          <div style={{ marginBottom: 6 }}>当前坐席</div>
          <select
            value={seat}
            onChange={(e) => setSeat(e.target.value)}
            style={{
              width: '100%',
              padding: '6px 8px',
              borderRadius: 6,
              border: '1px solid rgba(255,255,255,0.2)',
              background: 'rgba(255,255,255,0.08)',
              color: '#fff',
              fontSize: 13,
            }}
          >
            {(agents.length ? agents : [seat]).map((a) => (
              <option key={a} value={a} style={{ color: tokens.text }}>
                {a}
              </option>
            ))}
          </select>
        </div>
      </aside>

      <main style={s.main}>
        <div style={s.topbar}>
          <span>{VIEW_TITLES[view]}</span>
          <span style={s.muted}>Agent Desk</span>
        </div>
        {view === 'tickets' && <TicketBoard seat={seat} agents={agents} />}
        {view === 'handoff' && <Placeholder text="人工会话接管 — P1-B 阶段接入(待接管队列 + 实时接管聊天)" />}
        {view === 'calls' && <Placeholder text="智能外呼 — P2 阶段接入(外呼任务队列 + 通话控制 + 小结)" />}
      </main>
    </div>
  );
}

export default App;

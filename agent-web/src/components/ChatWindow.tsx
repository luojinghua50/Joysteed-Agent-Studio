import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '@/services/api';

interface ChatWindowProps {
  messages: ChatMessage[];
  isLoading: boolean;
  currentAgent: string;
  onSend: (content: string) => void;
  onStop: () => void;
  onNewChat: () => void;
  onLogout?: () => void;
}

const quickActions = [
  { label: '📦 查询订单', text: '我想查询订单状态' },
  { label: '🔄 退换货服务', text: '我需要退换货服务' },
  { label: '💡 产品咨询', text: '我想咨询产品信息' },
  { label: '📝 投诉建议', text: '我要提交投诉建议' },
];

// agent 名 → 中文友好标签。后端 supervisor 路由后通过 SSE done 事件回传 agent 名。
const AGENT_LABELS: Record<string, string> = {
  faq: '常见问题',
  order: '订单服务',
  complaint: '投诉处理',
  tech_support: '技术支持',
  human_handoff: '人工客服',
  fallback: '智能客服',
};

function agentLabel(agent: string): string {
  return AGENT_LABELS[agent] || agent;
}

function BotAvatar({ size = 24 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none">
      <rect x="12" y="20" width="40" height="34" rx="10" fill="#fff" stroke="#4facfe" strokeWidth="2.5"/>
      <rect x="22" y="8" width="20" height="16" rx="8" fill="#4facfe"/>
      <circle cx="27" cy="35" r="5" fill="#333"/>
      <circle cx="37" cy="35" r="5" fill="#333"/>
      <circle cx="28.5" cy="33.5" r="1.5" fill="#fff"/>
      <circle cx="38.5" cy="33.5" r="1.5" fill="#fff"/>
      <path d="M26 44 Q32 49 38 44" stroke="#ff6b8a" strokeWidth="2.5" strokeLinecap="round" fill="none"/>
      <rect x="8" y="30" width="6" height="12" rx="3" fill="#4facfe"/>
      <rect x="50" y="30" width="6" height="12" rx="3" fill="#4facfe"/>
      <line x1="32" y1="4" x2="32" y2="9" stroke="#4facfe" strokeWidth="2.5" strokeLinecap="round"/>
      <circle cx="32" cy="3" r="2.5" fill="#ff6b8a"/>
      <rect x="20" y="52" width="8" height="6" rx="3" fill="#667eea"/>
      <rect x="36" y="52" width="8" height="6" rx="3" fill="#667eea"/>
    </svg>
  );
}

export function ChatWindow({ messages, isLoading, currentAgent, onSend, onStop, onNewChat, onLogout }: ChatWindowProps) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSend(input.trim());
    setInput('');
  };

  return (
    <div style={styles.container}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.avatar}><BotAvatar size={22} /></div>
          <div style={{ flex: 1 }}>
            <h3 style={styles.headerTitle}>智能客服</h3>
            {currentAgent && (
              <span style={styles.headerSubtitle}>当前服务: {agentLabel(currentAgent)}</span>
            )}
          </div>
          <button
            type="button"
            style={styles.newChatBtn}
            onClick={onNewChat}
            title="开启一段全新对话"
          >
            ✛ 新对话
          </button>
          {onLogout && (
            <button
              type="button"
              style={styles.newChatBtn}
              onClick={onLogout}
              title="退出登录"
            >
              ⎋ 退出
            </button>
          )}
        </div>
      </div>

      {/* Messages Area */}
      <div style={styles.messagesArea}>
        {messages.length === 0 ? (
          <div style={styles.welcome}>
            <div style={styles.welcomeAvatar}><BotAvatar size={40} /></div>
            <h2 style={styles.welcomeTitle}>您好！我是智能客服助手</h2>
            <p style={styles.welcomeSubtitle}>有什么可以帮您的？</p>
            <div style={styles.quickActions}>
              {quickActions.map((action) => (
                <button
                  key={action.text}
                  style={styles.quickActionBtn}
                  onClick={() => onSend(action.text)}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = '#e8f4ff';
                    e.currentTarget.style.borderColor = '#4facfe';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = '#fff';
                    e.currentTarget.style.borderColor = '#e8e8e8';
                  }}
                >
                  {action.label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((msg, i) => {
              const isLastAssistant = isLoading && msg.role === 'assistant' && i === messages.length - 1;
              if (isLastAssistant && !msg.content) return null;
              // 用户停止生成且没有任何已生成内容：用占位文本，不渲染空气泡
              const stoppedEmpty = msg.role === 'assistant' && msg.stopped && !msg.content;
              return (
                <div key={i} style={{
                  display: 'flex',
                  justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                  marginBottom: '16px',
                  alignItems: 'flex-start',
                }}>
                  {msg.role === 'assistant' && (
                    <div style={styles.msgAvatar}><BotAvatar size={18} /></div>
                  )}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxWidth: '70%' }}>
                    <div style={msg.role === 'user' ? styles.userBubble : styles.assistantBubble}>
                      {stoppedEmpty
                        ? <span style={styles.stoppedPlaceholder}>已停止生成</span>
                        : msg.content}
                    </div>
                    {isLastAssistant && (
                      <div style={styles.streamingHint}>生成中...</div>
                    )}
                    {msg.stopped && msg.content && (
                      <div style={styles.stoppedHint}>已停止生成</div>
                    )}
                  </div>
                </div>
              );
            })}
            {isLoading && (!messages.length || !messages[messages.length - 1].content) && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
                <div style={styles.msgAvatar}><BotAvatar size={18} /></div>
                <div style={styles.typingIndicator}>
                  <span style={styles.dot1} />
                  <span style={styles.dot2} />
                  <span style={styles.dot3} />
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Persistent quick-service bar — lets users switch service mid-conversation.
          Hidden on the welcome screen since the welcome grid already shows them. */}
      {messages.length > 0 && (
        <div style={styles.quickBar}>
          {quickActions.map((action) => (
            <button
              key={action.text}
              type="button"
              style={styles.quickChip}
              disabled={isLoading}
              onClick={() => onSend(action.text)}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#e8f4ff';
                e.currentTarget.style.borderColor = '#4facfe';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = '#fff';
                e.currentTarget.style.borderColor = '#e8e8e8';
              }}
            >
              {action.label}
            </button>
          ))}
        </div>
      )}

      {/* Input Area */}
      <form onSubmit={handleSubmit} style={styles.inputArea}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="输入您的问题..."
          disabled={isLoading}
          style={styles.input}
          onFocus={(e) => { e.currentTarget.style.borderColor = '#4facfe'; e.currentTarget.style.boxShadow = '0 0 0 2px rgba(79,172,254,0.15)'; }}
          onBlur={(e) => { e.currentTarget.style.borderColor = '#e8e8e8'; e.currentTarget.style.boxShadow = 'none'; }}
        />
        {isLoading ? (
          <button type="button" onClick={onStop} style={styles.stopBtn}>
            ⏹ 停止
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim()}
            style={{
              ...styles.sendBtn,
              opacity: input.trim() ? 1 : 0.5,
              cursor: input.trim() ? 'pointer' : 'not-allowed',
            }}
          >
            发送 ➤
          </button>
        )}
      </form>

      <style>{typingAnimation}</style>
    </div>
  );
}

const typingAnimation = `
@keyframes bounce {
  0%, 60%, 100% { transform: translateY(0); }
  30% { transform: translateY(-4px); }
}
`;

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    background: '#f5f7fa',
    borderRadius: '12px',
    overflow: 'hidden',
  },
  header: {
    padding: '16px 20px',
    background: 'linear-gradient(135deg, #4facfe 0%, #667eea 100%)',
    color: '#fff',
    boxShadow: '0 2px 8px rgba(79,172,254,0.3)',
  },
  headerContent: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  avatar: {
    width: '40px',
    height: '40px',
    borderRadius: '50%',
    background: 'rgba(255,255,255,0.2)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '20px',
  },
  headerTitle: {
    margin: 0,
    fontSize: '16px',
    fontWeight: 600,
  },
  headerSubtitle: {
    fontSize: '12px',
    opacity: 0.85,
  },
  newChatBtn: {
    padding: '6px 12px',
    background: 'rgba(255,255,255,0.2)',
    color: '#fff',
    border: '1px solid rgba(255,255,255,0.4)',
    borderRadius: '16px',
    fontSize: '12px',
    fontWeight: 500,
    cursor: 'pointer',
    whiteSpace: 'nowrap' as const,
    flexShrink: 0,
  },
  quickBar: {
    display: 'flex',
    gap: '8px',
    padding: '10px 16px',
    overflowX: 'auto' as const,
    borderTop: '1px solid #eee',
    background: '#fafbfe',
  },
  quickChip: {
    padding: '6px 12px',
    background: '#fff',
    border: '1px solid #e8e8e8',
    borderRadius: '16px',
    fontSize: '13px',
    cursor: 'pointer',
    whiteSpace: 'nowrap' as const,
    transition: 'all 0.2s',
    flexShrink: 0,
  },
  messagesArea: {
    flex: 1,
    overflow: 'auto',
    padding: '20px',
  },
  welcome: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100%',
    textAlign: 'center',
    padding: '40px 20px',
  },
  welcomeAvatar: {
    width: '72px',
    height: '72px',
    borderRadius: '50%',
    background: 'linear-gradient(135deg, #4facfe 0%, #667eea 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '36px',
    marginBottom: '16px',
    boxShadow: '0 4px 12px rgba(79,172,254,0.3)',
  },
  welcomeTitle: {
    margin: '0 0 8px 0',
    fontSize: '20px',
    fontWeight: 600,
    color: '#1a1a1a',
  },
  welcomeSubtitle: {
    margin: '0 0 28px 0',
    fontSize: '14px',
    color: '#888',
  },
  quickActions: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '12px',
    maxWidth: '360px',
    width: '100%',
  },
  quickActionBtn: {
    padding: '12px 16px',
    background: '#fff',
    border: '1px solid #e8e8e8',
    borderRadius: '10px',
    fontSize: '14px',
    cursor: 'pointer',
    transition: 'all 0.2s',
    textAlign: 'left' as const,
    boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
  },
  msgAvatar: {
    width: '32px',
    height: '32px',
    borderRadius: '50%',
    background: 'linear-gradient(135deg, #4facfe 0%, #667eea 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '14px',
    marginRight: '8px',
    flexShrink: 0,
  },
  userBubble: {
    padding: '10px 14px',
    borderRadius: '14px 14px 4px 14px',
    background: 'linear-gradient(135deg, #4facfe 0%, #667eea 100%)',
    color: '#fff',
    fontSize: '14px',
    lineHeight: '1.5',
    whiteSpace: 'pre-wrap' as const,
    boxShadow: '0 2px 6px rgba(79,172,254,0.2)',
  },
  assistantBubble: {
    padding: '10px 14px',
    borderRadius: '14px 14px 14px 4px',
    background: '#fff',
    color: '#333',
    fontSize: '14px',
    lineHeight: '1.5',
    whiteSpace: 'pre-wrap' as const,
    boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
  },
  streamingHint: {
    fontSize: '12px',
    color: '#999',
    paddingLeft: '4px',
  },
  stoppedHint: {
    fontSize: '12px',
    color: '#bbb',
    paddingLeft: '4px',
  },
  stoppedPlaceholder: {
    color: '#bbb',
    fontStyle: 'italic',
  },
  typingIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    padding: '10px 14px',
    background: '#fff',
    borderRadius: '14px 14px 14px 4px',
    boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
  },
  dot1: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    background: '#999',
    animation: 'bounce 1.2s infinite',
    animationDelay: '0s',
  },
  dot2: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    background: '#999',
    animation: 'bounce 1.2s infinite',
    animationDelay: '0.2s',
  },
  dot3: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    background: '#999',
    animation: 'bounce 1.2s infinite',
    animationDelay: '0.4s',
  },
  inputArea: {
    padding: '14px 20px',
    borderTop: '1px solid #eee',
    display: 'flex',
    gap: '10px',
    background: '#fff',
  },
  input: {
    flex: 1,
    padding: '12px 16px',
    border: '1px solid #e8e8e8',
    borderRadius: '24px',
    fontSize: '14px',
    outline: 'none',
    transition: 'border-color 0.2s, box-shadow 0.2s',
    background: '#fafafa',
  },
  sendBtn: {
    padding: '10px 20px',
    background: 'linear-gradient(135deg, #4facfe 0%, #667eea 100%)',
    color: '#fff',
    border: 'none',
    borderRadius: '24px',
    fontSize: '14px',
    fontWeight: 500,
    whiteSpace: 'nowrap' as const,
  },
  stopBtn: {
    padding: '10px 20px',
    background: '#ff4d4f',
    color: '#fff',
    border: 'none',
    borderRadius: '24px',
    fontSize: '14px',
    fontWeight: 500,
    cursor: 'pointer',
    whiteSpace: 'nowrap' as const,
  },
};

import type { CSSProperties } from 'react';

// Shared design tokens — blue/purple gradient palette aligned with agent-web/admin.
export const tokens = {
  brand: '#4facfe',
  brandDeep: '#667eea',
  gradient: 'linear-gradient(135deg, #667eea 0%, #4facfe 100%)',
  bg: '#f5f6fa',
  surface: '#ffffff',
  border: '#e6e8ef',
  text: '#1f2330',
  textMuted: '#6b7280',
  danger: '#ef4444',
  success: '#10b981',
  warn: '#f59e0b',
  info: '#3b82f6',
  radius: '12px',
  shadow: '0 4px 20px rgba(0,0,0,0.06)',
};

// Ticket/call status → color, for status pills.
export function statusColor(status?: string): string {
  switch (status) {
    case 'open':
    case 'queued':
      return tokens.info;
    case 'in_progress':
    case 'dialing':
    case 'ringing':
    case 'connected':
      return tokens.warn;
    case 'resolved':
    case 'closed':
    case 'completed':
      return tokens.success;
    case 'failed':
    case 'no_answer':
    case 'busy':
      return tokens.danger;
    default:
      return tokens.textMuted;
  }
}

export function priorityColor(priority?: string): string {
  switch (priority) {
    case 'high':
      return tokens.danger;
    case 'medium':
      return tokens.warn;
    case 'low':
      return tokens.textMuted;
    default:
      return tokens.textMuted;
  }
}

export const STATUS_LABELS: Record<string, string> = {
  open: '待处理',
  in_progress: '处理中',
  resolved: '已解决',
  closed: '已关闭',
};

export const PRIORITY_LABELS: Record<string, string> = {
  high: '高',
  medium: '中',
  low: '低',
};

// PLACEHOLDER_STYLES
export const s: Record<string, CSSProperties> = {
  app: { display: 'flex', minHeight: '100vh', background: tokens.bg },
  sidebar: {
    width: 200,
    background: '#1f2330',
    color: '#fff',
    display: 'flex',
    flexDirection: 'column',
    flexShrink: 0,
  },
  sidebarBrand: {
    padding: '20px 18px',
    fontSize: 16,
    fontWeight: 600,
    borderBottom: '1px solid rgba(255,255,255,0.1)',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  navItem: {
    padding: '13px 18px',
    fontSize: 14,
    cursor: 'pointer',
    color: 'rgba(255,255,255,0.75)',
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    borderLeft: '3px solid transparent',
  },
  navItemActive: {
    background: 'rgba(255,255,255,0.08)',
    color: '#fff',
    borderLeft: `3px solid ${tokens.brand}`,
  },
  seatBar: {
    marginTop: 'auto',
    padding: '14px 18px',
    borderTop: '1px solid rgba(255,255,255,0.1)',
    fontSize: 13,
    color: 'rgba(255,255,255,0.7)',
  },
  main: { flex: 1, overflow: 'auto' },
  topbar: {
    background: tokens.surface,
    borderBottom: `1px solid ${tokens.border}`,
    padding: '16px 28px',
    fontSize: 17,
    fontWeight: 600,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  container: { padding: '24px 28px' },
  card: {
    background: tokens.surface,
    border: `1px solid ${tokens.border}`,
    borderRadius: tokens.radius,
    boxShadow: tokens.shadow,
    padding: 20,
    marginBottom: 20,
  },
  sectionTitle: { fontSize: 15, fontWeight: 600, marginBottom: 14, color: tokens.text },
  row: { display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' },
  input: {
    padding: '9px 12px',
    border: `1px solid ${tokens.border}`,
    borderRadius: 8,
    fontSize: 14,
    outline: 'none',
    background: '#fff',
    color: tokens.text,
  },
  select: { padding: '9px 12px', border: `1px solid ${tokens.border}`, borderRadius: 8, fontSize: 14, cursor: 'pointer', background: '#fff', color: tokens.text },
  btn: {
    padding: '9px 16px',
    border: 'none',
    borderRadius: 8,
    background: tokens.gradient,
    color: '#fff',
    fontSize: 14,
    fontWeight: 500,
    cursor: 'pointer',
  },
  btnGhost: {
    padding: '7px 13px',
    border: `1px solid ${tokens.border}`,
    borderRadius: 8,
    background: '#fff',
    color: tokens.text,
    fontSize: 13,
    cursor: 'pointer',
  },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 14 },
  th: {
    textAlign: 'left',
    padding: '10px 12px',
    borderBottom: `2px solid ${tokens.border}`,
    color: tokens.textMuted,
    fontWeight: 600,
    fontSize: 13,
  },
  td: { padding: '11px 12px', borderBottom: `1px solid ${tokens.border}`, color: tokens.text },
  trClickable: { cursor: 'pointer' },
  pill: {
    display: 'inline-block',
    padding: '2px 10px',
    borderRadius: 999,
    fontSize: 12,
    fontWeight: 600,
    color: '#fff',
  },
  muted: { color: tokens.textMuted, fontSize: 13 },
  link: { color: tokens.brandDeep, cursor: 'pointer', fontWeight: 500 },
  error: {
    background: '#fef2f2',
    border: `1px solid ${tokens.danger}`,
    color: tokens.danger,
    padding: '10px 14px',
    borderRadius: 8,
    fontSize: 13,
    marginBottom: 14,
  },
  empty: { textAlign: 'center', color: tokens.textMuted, padding: '40px 0', fontSize: 14 },
  drawer: {
    position: 'fixed',
    inset: 0,
    background: 'rgba(0,0,0,0.35)',
    display: 'flex',
    justifyContent: 'flex-end',
    zIndex: 100,
  },
  drawerPanel: {
    width: 560,
    maxWidth: '92vw',
    height: '100%',
    background: tokens.surface,
    padding: 24,
    overflowY: 'auto',
    boxShadow: '-8px 0 32px rgba(0,0,0,0.12)',
  },
  comment: {
    padding: '10px 12px',
    border: `1px solid ${tokens.border}`,
    borderRadius: 8,
    marginBottom: 8,
    background: '#fafbfe',
    fontSize: 13,
  },
};


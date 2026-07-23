import type { CSSProperties } from 'react';

// Shared design tokens — blue/purple gradient palette aligned with agent-web.
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
  radius: '12px',
  shadow: '0 4px 20px rgba(0,0,0,0.06)',
};

// Status → color, for document/version status pills.
export function statusColor(status?: string): string {
  switch (status) {
    case 'active':
    case 'ready':
      return tokens.success;
    case 'processing':
      return tokens.warn;
    case 'failed':
      return tokens.danger;
    default:
      return tokens.textMuted;
  }
}

// 知识形态展示元信息。与 agent-rag _form_defaults 一一对应：运营选形态，
// 后端绑定切片/检索/短路默认；这里只负责展示标签、配色与一句话说明。
export interface KbFormMeta {
  label: string;
  color: string;
  hint: string;
}

export const KB_FORMS: Record<string, KbFormMeta> = {
  standard: {
    label: '标准',
    color: tokens.brandDeep,
    hint: '通用文档库：按标题切分、hybrid 检索，适合政策/手册等长文。',
  },
  faq: {
    label: 'FAQ',
    color: tokens.success,
    hint: '问答库：按问答对切分，高置信命中可短路直答，优先级最高。',
  },
  temporal: {
    label: '时效',
    color: tokens.warn,
    hint: '时效库：检索自动按有效期过滤（需定义 effective_ts / expire_ts 时间字段）。',
  },
  multimodal: {
    label: '多模态',
    color: '#8b5cf6',
    hint: '多模态库：纯向量检索，面向图文等非纯文本召回。',
  },
};

export function kbFormMeta(form?: string): KbFormMeta {
  return KB_FORMS[form ?? 'standard'] ?? KB_FORMS.standard;
}

export const s: Record<string, CSSProperties> = {
  page: {
    minHeight: '100vh',
    background: tokens.bg,
  },
  header: {
    background: tokens.gradient,
    color: '#fff',
    padding: '18px 32px',
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    boxShadow: tokens.shadow,
  },
  headerTitle: { fontSize: 18, fontWeight: 600 },
  headerSub: { fontSize: 13, opacity: 0.85, marginLeft: 4 },
  container: { maxWidth: 1100, margin: '0 auto', padding: '28px 32px' },
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
  btnDanger: {
    padding: '7px 13px',
    border: `1px solid ${tokens.danger}`,
    borderRadius: 8,
    background: '#fff',
    color: tokens.danger,
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
};

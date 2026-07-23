import { useEffect, useState } from 'react';
import {
  listKbs,
  createKb,
  deleteKb,
  type KnowledgeBase,
  type KbForm,
} from '@/services/rag';
import { s, tokens, KB_FORMS, kbFormMeta } from '@/styles/theme';

interface KbListProps {
  onOpen: (kb: KnowledgeBase) => void;
}

const FORM_ORDER: KbForm[] = ['standard', 'faq', 'temporal', 'multimodal'];

export function KbList({ onOpen }: KbListProps) {
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [kbForm, setKbForm] = useState<KbForm>('standard');
  const [creating, setCreating] = useState(false);

  async function refresh() {
    setLoading(true);
    setError('');
    try {
      setKbs(await listKbs());
    } catch (e) {
      setError(`加载知识库失败:${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleCreate() {
    if (!name.trim()) return;
    setCreating(true);
    setError('');
    try {
      await createKb(name.trim(), description.trim(), kbForm);
      setName('');
      setDescription('');
      setKbForm('standard');
      await refresh();
    } catch (e) {
      setError(`创建失败:${(e as Error).message}`);
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(kb: KnowledgeBase) {
    if (!confirm(`确认删除知识库「${kb.name}」?其下所有文档与向量索引都会被清除,且不可恢复。`)) return;
    setError('');
    try {
      await deleteKb(kb.id);
      await refresh();
    } catch (e) {
      setError(`删除失败:${(e as Error).message}`);
    }
  }

  return (
    <div>
      <div style={s.card}>
        <div style={s.sectionTitle}>新建知识库</div>
        <div style={s.row}>
          <input
            style={{ ...s.input, flex: '1 1 200px' }}
            placeholder="名称(必填)"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <input
            style={{ ...s.input, flex: '2 1 280px' }}
            placeholder="描述(可选)"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
          <select
            style={{ ...s.input, cursor: 'pointer' }}
            value={kbForm}
            onChange={(e) => setKbForm(e.target.value as KbForm)}
          >
            {FORM_ORDER.map((f) => (
              <option key={f} value={f}>
                {KB_FORMS[f].label}
              </option>
            ))}
          </select>
          <button
            style={{ ...s.btn, opacity: creating || !name.trim() ? 0.6 : 1 }}
            onClick={handleCreate}
            disabled={creating || !name.trim()}
          >
            {creating ? '创建中…' : '创建'}
          </button>
        </div>
        <div style={{ ...s.muted, marginTop: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{ ...s.pill, background: kbFormMeta(kbForm).color }}
          >
            {kbFormMeta(kbForm).label}
          </span>
          <span>{kbFormMeta(kbForm).hint}</span>
        </div>
      </div>

      <div style={s.card}>
        <div style={s.sectionTitle}>知识库列表</div>
        {error && <div style={s.error}>{error}</div>}
        {loading ? (
          <div style={s.empty}>加载中…</div>
        ) : kbs.length === 0 ? (
          <div style={s.empty}>还没有知识库,先在上方创建一个。</div>
        ) : (
          <table style={s.table}>
            <thead>
              <tr>
                <th style={s.th}>名称</th>
                <th style={s.th}>形态</th>
                <th style={s.th}>描述</th>
                <th style={s.th}>分块策略</th>
                <th style={s.th}>文档数</th>
                <th style={{ ...s.th, textAlign: 'right' }}>操作</th>
              </tr>
            </thead>
            <tbody>
              {kbs.map((kb) => (
                <tr key={kb.id}>
                  <td style={s.td}>
                    <span style={s.link} onClick={() => onOpen(kb)}>
                      {kb.name}
                    </span>
                    <div style={{ ...s.muted, fontSize: 11 }}>{kb.id}</div>
                  </td>
                  <td style={s.td}>
                    <span style={{ ...s.pill, background: kbFormMeta(kb.kb_form).color }}>
                      {kbFormMeta(kb.kb_form).label}
                    </span>
                  </td>
                  <td style={{ ...s.td, color: tokens.textMuted }}>{kb.description || '—'}</td>
                  <td style={s.td}>{kb.chunking_strategy}</td>
                  <td style={s.td}>{kb.document_count ?? 0}</td>
                  <td style={{ ...s.td, textAlign: 'right' }}>
                    <button style={{ ...s.btnGhost, marginRight: 8 }} onClick={() => onOpen(kb)}>
                      管理
                    </button>
                    <button style={s.btnDanger} onClick={() => handleDelete(kb)}>
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

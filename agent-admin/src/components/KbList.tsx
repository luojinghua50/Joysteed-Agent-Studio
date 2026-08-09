import { useEffect, useState } from 'react';
import {
  deleteKb,
  listKbs,
  type KnowledgeBase,
} from '@/services/rag';
import { kbFormMeta, s, tokens } from '@/styles/theme';

// 中文展示名映射
const CHUNKING_LABEL: Record<string, string> = {
  auto: '自动', recursive: '递归', heading: '标题', fixed: '固定大小',
  table: '表格', qa_pair: 'Q&A 对', parent_child: '父子分块', semantic: '语义',
};
const RETRIEVAL_LABEL: Record<string, string> = {
  vector: '向量检索', fulltext: '全文检索', hybrid: '混合检索',
};
const RETRIEVAL_COLOR: Record<string, string> = {
  hybrid: tokens.success, vector: tokens.brandDeep, fulltext: tokens.warn,
};

interface KbListProps {
  onOpen: (kb: KnowledgeBase) => void;
  onCreate: () => void;
  onEdit: (kb: KnowledgeBase) => void;
}

export function KbList({ onOpen, onCreate, onEdit }: KbListProps) {
  const [kbs, setKbs] = useState<KnowledgeBase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  async function refresh() {
    setLoading(true);
    setError('');
    try {
      setKbs((await listKbs()).slice().sort((a, b) => {
        if (!a.created_at || !b.created_at) return 0;
        return b.created_at.localeCompare(a.created_at);
      }));
    } catch (e) {
      setError(`加载知识库失败:${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

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
      <div style={{ ...s.row, justifyContent: 'space-between', marginBottom: 18 }}>
        <div>
          <div style={{ fontSize: 19, fontWeight: 700, color: tokens.text }}>知识库列表</div>
          <div style={{ ...s.muted, marginTop: 5 }}>管理知识库、文档版本和检索配置</div>
        </div>
        <div style={s.row}>
          <button style={s.btnGhost} onClick={refresh}>
            刷新
          </button>
          <button style={s.btn} onClick={onCreate}>
            新建知识库
          </button>
        </div>
      </div>

      <div style={s.card}>
        {error && <div style={s.error}>{error}</div>}
        {loading ? (
          <div style={s.empty}>加载中...</div>
        ) : kbs.length === 0 ? (
          <div style={s.empty}>还没有知识库，点击右上角新建。</div>
        ) : (
          <table style={s.table}>
            <thead>
              <tr>
                <th style={s.th}>名称</th>
                <th style={s.th}>描述</th>
                <th style={s.th}>分块策略</th>
                <th style={s.th}>检索模式</th>
                <th style={s.th}>文档数</th>
                <th style={{ ...s.th, textAlign: 'right' }}>操作</th>
              </tr>
            </thead>
            <tbody>
              {kbs.map((kb) => {
                const meta = kbFormMeta(kb.kb_form);
                const chunkLabel = CHUNKING_LABEL[kb.chunking_strategy] ?? kb.chunking_strategy;
                const retrievalLabel = RETRIEVAL_LABEL[kb.retrieval_mode] ?? kb.retrieval_mode;
                const retrievalColor = RETRIEVAL_COLOR[kb.retrieval_mode] ?? tokens.textMuted;
                return (
                  <tr key={kb.id}>
                    <td style={s.td}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={s.link} onClick={() => onOpen(kb)}>{kb.name}</span>
                        <span style={{ ...s.pill, background: meta.color, fontSize: 11 }}>{meta.label}</span>
                      </div>
                      <div style={{ ...s.muted, fontSize: 11, marginTop: 3 }}>{kb.id}</div>
                    </td>
                    <td style={{ ...s.td, color: tokens.textMuted, maxWidth: 200 }}>
                      <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {kb.description || '—'}
                      </div>
                    </td>
                    <td style={s.td}>
                      {chunkLabel}（{kb.chunking_strategy}）
                    </td>
                    <td style={s.td}>
                      <span style={{ ...s.pill, background: retrievalColor, fontSize: 11 }}>
                        {retrievalLabel}（{kb.retrieval_mode}）
                      </span>
                    </td>
                    <td style={s.td}>{kb.document_count ?? 0}</td>
                    <td style={{ ...s.td, textAlign: 'right', whiteSpace: 'nowrap' }}>
                      <button style={{ ...s.btnGhost, marginRight: 8 }} onClick={() => onOpen(kb)}>
                        管理
                      </button>
                      <button style={{ ...s.btnGhost, marginRight: 8 }} onClick={() => onEdit(kb)}>
                        编辑
                      </button>
                      <button style={s.btnDanger} onClick={() => handleDelete(kb)}>
                        删除
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

import { useCallback, useEffect, useState } from 'react';
import {
  listDocuments,
  deleteDocument,
  type KnowledgeBase,
  type DocumentInfo,
} from '@/services/rag';
import { s, statusColor, kbFormMeta } from '@/styles/theme';
import { UploadPanel } from './UploadPanel';
import { SearchTester } from './SearchTester';
import { VersionDrawer } from './VersionDrawer';
import { MetadataFields } from './MetadataFields';
import { ThresholdSetting } from './ThresholdSetting';

interface KbDetailProps {
  kb: KnowledgeBase;
  onBack: () => void;
}

function fmtSize(bytes?: number): string {
  if (!bytes) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function KbDetail({ kb: initialKb, onBack }: KbDetailProps) {
  // 本地持有 kb，使阈值更新后头部徽章与高级设置即时同步。
  const [kb, setKb] = useState<KnowledgeBase>(initialKb);
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [versionDoc, setVersionDoc] = useState<DocumentInfo | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setDocs(await listDocuments(kb.id));
    } catch (e) {
      setError(`加载文档失败:${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [kb.id]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleDelete(doc: DocumentInfo) {
    if (!confirm(`确认删除文档「${doc.filename}」?其所有版本与向量都会被清除。`)) return;
    setError('');
    try {
      await deleteDocument(doc.id);
      await refresh();
    } catch (e) {
      setError(`删除失败:${(e as Error).message}`);
    }
  }

  return (
    <div>
      <div style={{ ...s.row, marginBottom: 18 }}>
        <button style={s.btnGhost} onClick={onBack}>
          ← 返回列表
        </button>
        <div>
          <span style={{ fontSize: 17, fontWeight: 600 }}>{kb.name}</span>
          <span style={{ ...s.pill, background: kbFormMeta(kb.kb_form).color, marginLeft: 10 }}>
            {kbFormMeta(kb.kb_form).label}
          </span>
          <span style={{ ...s.muted, marginLeft: 10 }}>
            {kb.description || '无描述'} · 策略 {kb.chunking_strategy} · {kb.id}
          </span>
        </div>
      </div>

      <UploadPanel kbId={kb.id} onUploaded={refresh} />

      <MetadataFields kbId={kb.id} />

      {kb.kb_form === 'faq' && <ThresholdSetting kb={kb} onUpdated={setKb} />}

      <div style={s.card}>
        <div style={{ ...s.row, justifyContent: 'space-between' }}>
          <div style={{ ...s.sectionTitle, marginBottom: 0 }}>文档列表</div>
          <button style={s.btnGhost} onClick={refresh}>
            刷新
          </button>
        </div>
        {error && <div style={{ ...s.error, marginTop: 14 }}>{error}</div>}
        {loading ? (
          <div style={s.empty}>加载中…</div>
        ) : docs.length === 0 ? (
          <div style={s.empty}>该知识库还没有文档,先在上方上传。</div>
        ) : (
          <table style={{ ...s.table, marginTop: 14 }}>
            <thead>
              <tr>
                <th style={s.th}>文件名</th>
                <th style={s.th}>类型</th>
                <th style={s.th}>大小</th>
                <th style={s.th}>版本</th>
                <th style={s.th}>状态</th>
                <th style={s.th}>chunk</th>
                <th style={{ ...s.th, textAlign: 'right' }}>操作</th>
              </tr>
            </thead>
            <tbody>
              {docs.map((doc) => (
                <tr key={doc.id}>
                  <td style={s.td}>
                    {doc.filename}
                    <div style={{ ...s.muted, fontSize: 11 }}>{doc.id}</div>
                  </td>
                  <td style={s.td}>{doc.file_type || '—'}</td>
                  <td style={s.td}>{fmtSize(doc.file_size)}</td>
                  <td style={s.td}>{doc.version_no ? `v${doc.version_no}` : '—'}</td>
                  <td style={s.td}>
                    {doc.status ? (
                      <span style={{ ...s.pill, background: statusColor(doc.status) }}>
                        {doc.status}
                      </span>
                    ) : (
                      '—'
                    )}
                  </td>
                  <td style={s.td}>{doc.chunk_count ?? '—'}</td>
                  <td style={{ ...s.td, textAlign: 'right' }}>
                    <button
                      style={{ ...s.btnGhost, marginRight: 8 }}
                      onClick={() => setVersionDoc(doc)}
                    >
                      版本
                    </button>
                    <button style={s.btnDanger} onClick={() => handleDelete(doc)}>
                      删除
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <SearchTester kbId={kb.id} />

      {versionDoc && (
        <VersionDrawer
          docId={versionDoc.id}
          filename={versionDoc.filename}
          onClose={() => setVersionDoc(null)}
          onRolledBack={refresh}
        />
      )}
    </div>
  );
}

import { useEffect, useState } from 'react';
import { listVersions, rollbackDocument, type VersionInfo } from '@/services/rag';
import { s, tokens, statusColor } from '@/styles/theme';

interface VersionDrawerProps {
  docId: string;
  filename: string;
  onClose: () => void;
  onRolledBack: () => void;
}

export function VersionDrawer({ docId, filename, onClose, onRolledBack }: VersionDrawerProps) {
  const [versions, setVersions] = useState<VersionInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  async function refresh() {
    setLoading(true);
    setError('');
    try {
      setVersions(await listVersions(docId));
    } catch (e) {
      setError(`加载版本失败:${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [docId]);

  async function handleRollback(v: VersionInfo) {
    if (!confirm(`回滚到版本 v${v.version_no}?当前激活版本会被切换。`)) return;
    setBusy(true);
    setError('');
    try {
      await rollbackDocument(docId, v.version_no);
      await refresh();
      onRolledBack();
    } catch (e) {
      setError(`回滚失败:${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.35)',
        display: 'flex',
        justifyContent: 'flex-end',
        zIndex: 100,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: 520,
          maxWidth: '90vw',
          height: '100%',
          background: tokens.surface,
          padding: 24,
          overflowY: 'auto',
          boxShadow: '-8px 0 32px rgba(0,0,0,0.12)',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={s.sectionTitle}>版本历史 — {filename}</div>
          <button style={s.btnGhost} onClick={onClose}>
            关闭
          </button>
        </div>
        {error && <div style={s.error}>{error}</div>}
        {loading ? (
          <div style={s.empty}>加载中…</div>
        ) : versions.length === 0 ? (
          <div style={s.empty}>没有版本记录。</div>
        ) : (
          <table style={s.table}>
            <thead>
              <tr>
                <th style={s.th}>版本</th>
                <th style={s.th}>状态</th>
                <th style={s.th}>chunk</th>
                <th style={{ ...s.th, textAlign: 'right' }}>操作</th>
              </tr>
            </thead>
            <tbody>
              {versions.map((v) => (
                <tr key={v.id}>
                  <td style={s.td}>
                    v{v.version_no}
                    {v.is_current && (
                      <span style={{ ...s.pill, background: tokens.success, marginLeft: 8 }}>
                        当前
                      </span>
                    )}
                  </td>
                  <td style={s.td}>
                    <span style={{ ...s.pill, background: statusColor(v.status) }}>{v.status}</span>
                  </td>
                  <td style={s.td}>{v.chunk_count}</td>
                  <td style={{ ...s.td, textAlign: 'right' }}>
                    {v.is_current ? (
                      <span style={s.muted}>—</span>
                    ) : (
                      <button
                        style={{ ...s.btnGhost, opacity: busy ? 0.6 : 1 }}
                        onClick={() => handleRollback(v)}
                        disabled={busy}
                      >
                        回滚到此版本
                      </button>
                    )}
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

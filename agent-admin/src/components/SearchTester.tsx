import { useEffect, useState } from 'react';
import { search, type SearchResultItem } from '@/services/rag';
import { s, tokens } from '@/styles/theme';

interface SearchTesterProps {
  kbId: string;
  defaultTopK?: number;
}

export function SearchTester({ kbId, defaultTopK = 5 }: SearchTesterProps) {
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(defaultTopK);
  const [results, setResults] = useState<SearchResultItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setTopK(defaultTopK);
  }, [defaultTopK]);

  async function run() {
    if (!query.trim()) return;
    setLoading(true);
    setError('');
    try {
      const resp = await search(kbId, query.trim(), topK);
      setResults(resp.results);
    } catch (e) {
      setError(`检索失败:${(e as Error).message}`);
      setResults(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={s.card}>
      <div style={{ ...s.sectionTitle, marginBottom: 12 }}>检索测试</div>
      {error && <div style={{ ...s.error, marginBottom: 10 }}>{error}</div>}
      <div style={s.row}>
        <input
          style={{ ...s.input, flex: '1 1 320px' }}
          placeholder="输入查询，例如：SP-BT500 价格"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && run()}
        />
        <label style={{ ...s.muted, display: 'flex', alignItems: 'center', gap: 6 }}>
          top_k
          <input
            type="number" min={1} max={50}
            style={{ ...s.input, width: 64 }}
            value={topK}
            onChange={(e) => setTopK(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
          />
        </label>
        <button
          style={{ ...s.btn, opacity: loading || !query.trim() ? 0.6 : 1 }}
          onClick={run}
          disabled={loading || !query.trim()}
        >
          {loading ? '检索中…' : '检索'}
        </button>
      </div>

      {results !== null && (
        <div style={{ marginTop: 16 }}>
          {results.length === 0 ? (
            <div style={s.empty}>没有命中任何 chunk。确认该知识库已上传文档且索引完成。</div>
          ) : (
            results.map((r, i) => (
              <div
                key={r.chunk_id || i}
                style={{
                  padding: '12px 14px',
                  border: `1px solid ${tokens.border}`,
                  borderRadius: 10,
                  marginBottom: 10,
                  background: '#fafbfe',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <span style={{ ...s.muted, fontSize: 12 }}>
                    #{i + 1} · doc {r.doc_id || '—'}
                  </span>
                  <span
                    style={{ ...s.pill, background: tokens.brandDeep, fontFamily: 'monospace' }}
                    title="库级检索相似度分"
                  >
                    score {r.score.toFixed(3)}
                  </span>
                </div>
                <div style={{ fontSize: 14, lineHeight: 1.6, color: tokens.text }}>{r.text}</div>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}


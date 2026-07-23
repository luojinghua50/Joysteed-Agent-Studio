import { useState, type CSSProperties } from 'react';
import { search, routeSearch, type SearchResultItem } from '@/services/rag';
import { s, tokens } from '@/styles/theme';

interface SearchTesterProps {
  kbId: string;
}

type Mode = 'single' | 'route';

interface RouteMeta {
  shortcut: boolean;
  routed_kbs: string[];
}

export function SearchTester({ kbId }: SearchTesterProps) {
  const [mode, setMode] = useState<Mode>('single');
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState<SearchResultItem[] | null>(null);
  const [routeMeta, setRouteMeta] = useState<RouteMeta | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function run() {
    if (!query.trim()) return;
    setLoading(true);
    setError('');
    setRouteMeta(null);
    try {
      if (mode === 'route') {
        // scope 限定到当前库，验证其形态行为（faq 库可触发短路）。
        const resp = await routeSearch(query.trim(), [kbId], topK);
        setResults(resp.results);
        setRouteMeta({ shortcut: resp.shortcut, routed_kbs: resp.routed_kbs });
      } else {
        const resp = await search(kbId, query.trim(), topK);
        setResults(resp.results);
      }
    } catch (e) {
      setError(`检索失败:${(e as Error).message}`);
      setResults(null);
    } finally {
      setLoading(false);
    }
  }

  const tab = (m: Mode): CSSProperties => ({
    ...s.btnGhost,
    borderColor: mode === m ? tokens.brandDeep : tokens.border,
    color: mode === m ? tokens.brandDeep : tokens.text,
    fontWeight: mode === m ? 600 : 400,
    background: mode === m ? '#f0f4ff' : '#fff',
  });

  return (
    <div style={s.card}>
      <div style={{ ...s.row, justifyContent: 'space-between' }}>
        <div style={{ ...s.sectionTitle, marginBottom: 0 }}>检索测试</div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button style={tab('single')} onClick={() => setMode('single')}>
            单库检索
          </button>
          <button style={tab('route')} onClick={() => setMode('route')}>
            路由检索
          </button>
        </div>
      </div>
      <div style={{ ...s.muted, margin: '8px 0 14px' }}>
        {mode === 'single'
          ? '直连 /api/search,单库召回,用库级检索模式,score 为相似度分。'
          : '走生产链路 /api/route-search(限定当前库),展示 faq 短路命中与路由溯源。注:结果分为 RRF 融合排名分(非语义相似度);短路置信用的是内部 vector 探针分。'}
      </div>
      {error && <div style={s.error}>{error}</div>}
      <div style={s.row}>
        <input
          style={{ ...s.input, flex: '1 1 320px' }}
          placeholder="输入查询,例如:退款多久能到账"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && run()}
        />
        <label style={{ ...s.muted, display: 'flex', alignItems: 'center', gap: 6 }}>
          top_k
          <input
            type="number"
            min={1}
            max={20}
            style={{ ...s.input, width: 64 }}
            value={topK}
            onChange={(e) => setTopK(Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
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

      {routeMeta && (
        <div
          style={{
            ...s.row,
            marginTop: 14,
            padding: '10px 14px',
            borderRadius: 8,
            background: routeMeta.shortcut ? '#f0fdf4' : '#f5f6fa',
            border: `1px solid ${routeMeta.shortcut ? tokens.success : tokens.border}`,
          }}
        >
          <span
            style={{
              ...s.pill,
              background: routeMeta.shortcut ? tokens.success : tokens.textMuted,
            }}
          >
            {routeMeta.shortcut ? 'FAQ 短路命中' : '未短路 · 走融合'}
          </span>
          <span style={s.muted}>
            参与库:{routeMeta.routed_kbs.length ? routeMeta.routed_kbs.join(', ') : '无'}
          </span>
        </div>
      )}

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
                    {mode === 'route' && r.version_no != null && ` · v${r.version_no}`}
                    {mode === 'route' && r.source && ` · ${r.source}`}
                    {mode === 'route' && routeMeta?.shortcut && i === 0 && (
                      <span style={{ ...s.pill, background: tokens.success, marginLeft: 8 }}>
                        短路直答
                      </span>
                    )}
                  </span>
                  <span
                    title={
                      mode === 'route'
                        ? 'RRF 融合排名分(尺度 ~0.03，仅反映排序，非语义相似度;短路置信用的是内部 vector 探针分，不在此返回)'
                        : '库级检索相似度分'
                    }
                    style={{
                      ...s.pill,
                      background: mode === 'route' ? tokens.textMuted : tokens.brandDeep,
                      fontFamily: 'monospace',
                    }}
                  >
                    {mode === 'route' ? 'RRF' : 'score'} {r.score.toFixed(3)}
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

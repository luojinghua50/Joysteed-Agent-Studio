import { useCallback, useEffect, useState } from 'react';
import {
  listMetadataFields,
  createMetadataField,
  deleteMetadataField,
  type MetadataField,
  type MetaFieldType,
  type KnowledgeBase,
} from '@/services/rag';
import { s, tokens } from '@/styles/theme';

interface MetadataFieldsProps {
  kb: KnowledgeBase;
}

const FIELD_TYPES: { value: MetaFieldType; label: string; desc: string }[] = [
  { value: 'string', label: '字符串', desc: '分类、标签等文本值，如 category = "退款政策"' },
  { value: 'number', label: '数值',   desc: '优先级、评分等数字，如 priority = 1' },
  { value: 'time',   label: '时间',   desc: '时效库必用，effective_ts / expire_ts 有效期过滤' },
];

const TYPE_COLOR: Record<MetaFieldType, string> = {
  string: tokens.brandDeep,
  number: tokens.success,
  time:   tokens.warn,
};

// 时效库必须的两个时间字段
const TEMPORAL_REQUIRED = ['effective_ts', 'expire_ts'];

export function MetadataFields({ kb }: MetadataFieldsProps) {
  const [fields, setFields] = useState<MetadataField[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [name, setName] = useState('');
  const [fieldType, setFieldType] = useState<MetaFieldType>('string');
  const [creating, setCreating] = useState(false);
  const [typeHint, setTypeHint] = useState('');

  const isTemporalKb = kb.kb_form === 'temporal';

  const refresh = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setFields(await listMetadataFields(kb.id));
    } catch (e) {
      setError(`加载字段失败：${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [kb.id]);

  useEffect(() => { refresh(); }, [refresh]);

  function handleTypeChange(t: MetaFieldType) {
    setFieldType(t);
    const found = FIELD_TYPES.find((o) => o.value === t);
    setTypeHint(found?.desc ?? '');
  }

  async function handleCreate() {
    if (!name.trim()) return;
    setCreating(true);
    setError('');
    try {
      await createMetadataField(kb.id, name.trim(), fieldType);
      setName('');
      setFieldType('string');
      setTypeHint('');
      await refresh();
    } catch (e) {
      setError(`新增字段失败：${(e as Error).message}`);
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(field: MetadataField) {
    if (!confirm(`确认删除字段「${field.name}」？已上传文档的该标签值不受影响，但将无法再按此字段过滤。`)) return;
    setError('');
    try {
      await deleteMetadataField(kb.id, field.id);
      await refresh();
    } catch (e) {
      setError(`删除失败：${(e as Error).message}`);
    }
  }

  // 时效库：检查必填字段是否已定义
  const definedNames = fields.map((f) => f.name);
  const missingTemporalFields = isTemporalKb
    ? TEMPORAL_REQUIRED.filter((r) => !definedNames.includes(r))
    : [];

  return (
    <div style={s.card}>
      <div style={{ ...s.row, justifyContent: 'space-between', marginBottom: 6 }}>
        <div style={s.sectionTitle}>元数据字段</div>
        {/* 非时效库折叠提示 */}
        {!isTemporalKb && (
          <span style={{ fontSize: 12, color: tokens.textMuted }}>
            普通知识库可跳过此项
          </span>
        )}
      </div>

      {/* 用途说明 */}
      <div style={{ fontSize: 13, color: tokens.textMuted, marginBottom: 14, lineHeight: 1.6 }}>
        给文档打标签，让检索时可以按条件过滤。例如定义 <code>category</code> 字段后，
        上传文档时标注"退款政策"或"物流说明"，检索时就能只搜某一类文档。
        {isTemporalKb && (
          <span style={{ color: tokens.warn, fontWeight: 500 }}>
            {' '}时效库需要定义 effective_ts 和 expire_ts 两个时间字段，检索时自动过滤过期内容。
          </span>
        )}
      </div>

      {/* 时效库缺少必填字段时的警告 */}
      {missingTemporalFields.length > 0 && (
        <div style={{
          marginBottom: 14,
          padding: '10px 14px',
          background: '#fffbeb',
          border: `1px solid ${tokens.warn}`,
          borderRadius: 8,
          fontSize: 13,
          color: tokens.text,
        }}>
          ⚠ 时效库还缺少必要字段：
          {missingTemporalFields.map((f) => (
            <button
              key={f}
              style={{
                marginLeft: 8,
                padding: '2px 10px',
                border: `1px solid ${tokens.warn}`,
                borderRadius: 999,
                background: '#fff',
                color: tokens.warn,
                fontSize: 12,
                cursor: 'pointer',
                fontWeight: 600,
              }}
              onClick={() => { setName(f); setFieldType('time'); setTypeHint(FIELD_TYPES[2].desc); }}
            >
              + {f}
            </button>
          ))}
          <span style={{ color: tokens.textMuted, marginLeft: 8 }}>点击快速添加</span>
        </div>
      )}

      {error && <div style={{ ...s.error, marginBottom: 12 }}>{error}</div>}

      {/* 新增字段表单 */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <div style={s.row}>
          <input
            style={{ ...s.input, flex: '1 1 200px' }}
            placeholder="字段名，如 category / effective_ts"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
          />
          <select
            style={{ ...s.input, cursor: 'pointer' }}
            value={fieldType}
            onChange={(e) => handleTypeChange(e.target.value as MetaFieldType)}
          >
            {FIELD_TYPES.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
          <button
            style={{ ...s.btn, opacity: creating || !name.trim() ? 0.6 : 1 }}
            onClick={handleCreate}
            disabled={creating || !name.trim()}
          >
            {creating ? '新增中…' : '新增字段'}
          </button>
        </div>
        {/* 类型说明联动 */}
        {typeHint && (
          <div style={{ fontSize: 12, color: tokens.textMuted, paddingLeft: 4 }}>
            💡 {typeHint}
          </div>
        )}
      </div>

      {/* 已定义字段列表 */}
      <div style={{ marginTop: 14 }}>
        {loading ? (
          <div style={s.empty}>加载中…</div>
        ) : fields.length === 0 ? (
          <div style={{ fontSize: 13, color: tokens.textMuted }}>
            {isTemporalKb ? '尚未定义字段，请先添加上方必要字段。' : '尚未定义元数据字段。'}
          </div>
        ) : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {fields.map((f) => (
              <div
                key={f.id}
                style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  padding: '7px 10px 7px 12px',
                  border: `1px solid ${tokens.border}`,
                  borderRadius: 999,
                  background: '#fafbfe',
                }}
              >
                <span style={{ fontSize: 13, fontWeight: 500, color: tokens.text }}>{f.name}</span>
                <span style={{ ...s.pill, background: TYPE_COLOR[f.field_type], fontSize: 11 }}>
                  {FIELD_TYPES.find((o) => o.value === f.field_type)?.label ?? f.field_type}
                </span>
                <span
                  onClick={() => handleDelete(f)}
                  title="删除字段"
                  style={{ cursor: 'pointer', color: tokens.textMuted, fontSize: 16, lineHeight: 1, padding: '0 2px' }}
                >
                  ×
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

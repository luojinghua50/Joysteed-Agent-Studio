import { useCallback, useEffect, useState } from 'react';
import {
  listMetadataFields,
  createMetadataField,
  deleteMetadataField,
  type MetadataField,
  type MetaFieldType,
} from '@/services/rag';
import { s, tokens } from '@/styles/theme';

interface MetadataFieldsProps {
  kbId: string;
}

const FIELD_TYPES: { value: MetaFieldType; label: string }[] = [
  { value: 'string', label: '字符串' },
  { value: 'number', label: '数值' },
  { value: 'time', label: '时间(有效期过滤用)' },
];

const TYPE_COLOR: Record<MetaFieldType, string> = {
  string: tokens.brandDeep,
  number: tokens.success,
  time: tokens.warn,
};

export function MetadataFields({ kbId }: MetadataFieldsProps) {
  const [fields, setFields] = useState<MetadataField[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const [name, setName] = useState('');
  const [fieldType, setFieldType] = useState<MetaFieldType>('string');
  const [creating, setCreating] = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setFields(await listMetadataFields(kbId));
    } catch (e) {
      setError(`加载字段失败:${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [kbId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleCreate() {
    if (!name.trim()) return;
    setCreating(true);
    setError('');
    try {
      await createMetadataField(kbId, name.trim(), fieldType);
      setName('');
      setFieldType('string');
      await refresh();
    } catch (e) {
      setError(`新增字段失败:${(e as Error).message}`);
    } finally {
      setCreating(false);
    }
  }

  async function handleDelete(field: MetadataField) {
    if (!confirm(`确认删除字段「${field.name}」?已上传文档的该标签值不受影响,但将无法再按此字段过滤。`))
      return;
    setError('');
    try {
      await deleteMetadataField(kbId, field.id);
      await refresh();
    } catch (e) {
      setError(`删除失败:${(e as Error).message}`);
    }
  }

  return (
    <div style={s.card}>
      <div style={s.sectionTitle}>元数据字段</div>
      <div style={{ ...s.muted, marginBottom: 14 }}>
        定义库内可过滤字段。上传文档时按这些字段打标签,检索时即可按标签过滤;time
        类型字段供时效库做有效期过滤。
      </div>
      {error && <div style={s.error}>{error}</div>}
      <div style={s.row}>
        <input
          style={{ ...s.input, flex: '1 1 200px' }}
          placeholder="字段名,如 category / effective_ts"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
        />
        <select
          style={{ ...s.input, cursor: 'pointer' }}
          value={fieldType}
          onChange={(e) => setFieldType(e.target.value as MetaFieldType)}
        >
          {FIELD_TYPES.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
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

      <div style={{ marginTop: 14 }}>
        {loading ? (
          <div style={s.empty}>加载中…</div>
        ) : fields.length === 0 ? (
          <div style={s.empty}>还没有定义元数据字段。</div>
        ) : (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {fields.map((f) => (
              <div
                key={f.id}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '7px 10px 7px 12px',
                  border: `1px solid ${tokens.border}`,
                  borderRadius: 999,
                  background: '#fafbfe',
                }}
              >
                <span style={{ fontSize: 13, fontWeight: 500, color: tokens.text }}>{f.name}</span>
                <span style={{ ...s.pill, background: TYPE_COLOR[f.field_type], fontSize: 11 }}>
                  {f.field_type}
                </span>
                <span
                  onClick={() => handleDelete(f)}
                  title="删除字段"
                  style={{
                    cursor: 'pointer',
                    color: tokens.textMuted,
                    fontSize: 16,
                    lineHeight: 1,
                    padding: '0 2px',
                  }}
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

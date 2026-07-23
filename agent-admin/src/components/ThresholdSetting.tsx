import { useState } from 'react';
import { updateKbThreshold, type KnowledgeBase } from '@/services/rag';
import { s, tokens } from '@/styles/theme';

interface ThresholdSettingProps {
  kb: KnowledgeBase;
  onUpdated: (kb: KnowledgeBase) => void;
}

// faq 库高置信短路阈值的编辑卡片。仅在 faq 库渲染（调用方保证）。
// 这是把危险旋钮：过低误短路答错，过高永不触发，所以默认折叠在「高级设置」下。
export function ThresholdSetting({ kb, onUpdated }: ThresholdSettingProps) {
  const [open, setOpen] = useState(false);
  const [value, setValue] = useState(String(kb.shortcut_threshold ?? 0));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [saved, setSaved] = useState(false);

  const parsed = Number(value);
  const valid = value.trim() !== '' && !Number.isNaN(parsed) && parsed >= 0 && parsed <= 1;
  const dirty = parsed !== kb.shortcut_threshold;

  async function handleSave() {
    if (!valid || !dirty) return;
    setSaving(true);
    setError('');
    setSaved(false);
    try {
      const updated = await updateKbThreshold(kb.id, parsed);
      onUpdated(updated);
      setValue(String(updated.shortcut_threshold));
      setSaved(true);
    } catch (e) {
      setError(`保存失败:${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div style={s.card}>
      <div
        style={{ ...s.row, justifyContent: 'space-between', cursor: 'pointer' }}
        onClick={() => setOpen((v) => !v)}
      >
        <div style={{ ...s.sectionTitle, marginBottom: 0 }}>
          高级设置 <span style={s.muted}>· 短路阈值</span>
        </div>
        <span style={s.muted}>{open ? '收起 ▲' : '展开 ▼'}</span>
      </div>

      {open && (
        <div style={{ marginTop: 14 }}>
          <div style={{ ...s.muted, marginBottom: 12, lineHeight: 1.6 }}>
            高置信短路阈值:路由检索时，faq 库用 vector 探针语义相似度(0–1)与此值比较，
            达到即直答、跳过融合。<strong>过低</strong>会误短路答错,<strong>过高</strong>会永不触发。
            当前基准:精确问法约 0.81、同义约 0.65、无关约 0.20,推荐 0.70。换 embedding
            模型后需据真实校准重设。
          </div>
          {error && <div style={s.error}>{error}</div>}
          <div style={s.row}>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              style={{ ...s.input, width: 120 }}
              value={value}
              onChange={(e) => {
                setValue(e.target.value);
                setSaved(false);
              }}
            />
            <button
              style={{ ...s.btn, opacity: !valid || !dirty || saving ? 0.6 : 1 }}
              onClick={handleSave}
              disabled={!valid || !dirty || saving}
            >
              {saving ? '保存中…' : '保存'}
            </button>
            {!valid && <span style={{ ...s.muted, color: tokens.danger }}>需为 0–1 之间的数值</span>}
            {valid && saved && !dirty && (
              <span style={{ ...s.muted, color: tokens.success }}>已保存</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

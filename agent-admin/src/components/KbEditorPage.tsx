import { useEffect, useState, type CSSProperties } from 'react';
import {
  createKb,
  updateKbConfig,
  type ChunkingStrategy,
  type KbForm,
  type KnowledgeBase,
  type RetrievalMode,
} from '@/services/rag';
import { kbFormMeta, s, tokens } from '@/styles/theme';

type EditorMode = 'create' | 'edit';

interface KbEditorPageProps {
  mode: EditorMode;
  kb?: KnowledgeBase;
  onCancel: () => void;
  onSaved: (kb: KnowledgeBase) => void;
}

interface EditorState {
  name: string;
  description: string;
  kbForm: KbForm;
  chunkingStrategy: ChunkingStrategy;
  chunkSize: number;
  chunkOverlap: number;
  retrievalMode: RetrievalMode;
  topK: number;
  vectorWeight: number;
  scoreThreshold: number;
  shortcutThreshold: number;
  embeddingModel: string;
  rerankModel: string;
}

const FORM_ORDER: KbForm[] = ['standard', 'faq', 'temporal', 'multimodal'];
const CHUNKING_OPTIONS: ChunkingStrategy[] = [
  'auto', 'recursive', 'heading', 'fixed', 'table', 'qa_pair', 'parent_child',
  // 'semantic' 暂未实现（降级为 recursive），待 embedding 语义切分上线后启用
];
const RETRIEVAL_OPTIONS: RetrievalMode[] = ['vector', 'fulltext', 'hybrid'];

// 这些策略不需要 chunk_size / overlap
const NO_SIZE_STRATEGIES: ChunkingStrategy[] = ['qa_pair', 'table', 'parent_child'];

// 编辑模式下不可修改的字段（已有数据依赖这些参数，改了会导致版本不一致）
const EDIT_LOCKED: Array<keyof EditorState> = ['embeddingModel', 'chunkingStrategy', 'chunkSize', 'chunkOverlap'];

const CHUNKING_DESCRIPTIONS: Record<ChunkingStrategy, { label: string; desc: string; recommend: string; color: string }> = {
  auto:         { label: '自动',     color: tokens.brandDeep, desc: '根据文件类型自动选择策略：PDF/TXT → 递归切分，Markdown/Word → 标题切分，表格文件 → 表格切分。', recommend: '不确定时首选，适合混合文件类型的知识库。' },
  recursive:    { label: '递归',     color: tokens.brand,     desc: '按分隔符层级递归切分：先段落（\\n\\n）→ 换行 → 句子，直到满足 chunk_size。保留语义完整性。', recommend: '通用文档、长文本、PDF 正文。' },
  heading:      { label: '标题',     color: '#06b6d4',        desc: '按 Markdown / Word 标题（H1~H3）切分，每个标题段落作为一个 chunk，保留文档结构层级。', recommend: '有明确章节结构的文档，如产品手册、操作指南。' },
  fixed:        { label: '固定大小', color: tokens.textMuted, desc: '按固定字符数切分，相邻 chunk 有 overlap 重叠，不考虑语义边界。', recommend: '对切分位置无要求、只需严格控制 token 长度时使用。' },
  semantic:     { label: '语义',     color: tokens.textMuted, desc: '暂未实现，实际等同于递归切分。', recommend: '' },
  table:        { label: '表格',     color: tokens.warn,      desc: '按行切分表格（xlsx/csv），自动将多行合并直到达到 chunk_size，保留表头语境。不需要 size 参数。', recommend: '结构化数据、价格表、规格对照表。' },
  qa_pair:      { label: 'Q&A 对',   color: tokens.success,   desc: '按"Q："/"问："格式切分问答对，每个 Q&A 对作为一个完整 chunk。不需要 size 参数。', recommend: 'FAQ 文档、客服话术库，与 faq 快速模板搭配使用。' },
  parent_child: { label: '父子分块', color: '#8b5cf6',        desc: '子块（细粒度）用于向量检索，命中后返回父块（完整段落）作为上下文，兼顾精准召回与完整语境。不需要 size 参数。', recommend: '长文档精读场景，需要精准定位又要保留上下文时。' },
};

const RETRIEVAL_DESCRIPTIONS: Record<RetrievalMode, { label: string; desc: string; recommend: string; color: string }> = {
  vector:   { label: '向量检索', color: tokens.brandDeep, desc: '将查询和文档转换为 embedding 向量，通过余弦相似度找最近邻。能理解语义和同义词，对精确关键词匹配较弱。', recommend: '开放式问答、语义相似场景，如"有没有类似 XX 的产品"。' },
  fulltext: { label: '全文检索', color: tokens.warn,       desc: '基于 BM25 关键词频率匹配，对精确词汇匹配效果好，不理解语义（"降噪耳机"不等于"主动降噪"）。速度快、无需 embedding 模型。', recommend: '有大量专有名词、编号、型号的场景，如订单号、产品 SKU、规格参数。' },
  hybrid:   { label: '混合检索', color: tokens.success,   desc: '同时跑向量检索和 BM25，各自召回候选后用 RRF 算法融合排序。综合语义理解和关键词精度，效果最好。', recommend: '生产环境首选。兼顾语义理解和精确匹配，覆盖面最广。' },
};

const FORM_DEFAULTS: Record<KbForm, Pick<
  EditorState,
  'chunkingStrategy' | 'retrievalMode' | 'vectorWeight' | 'scoreThreshold' | 'shortcutThreshold'
>> = {
  standard:  { chunkingStrategy: 'heading',  retrievalMode: 'hybrid', vectorWeight: 0.6, scoreThreshold: 0, shortcutThreshold: 0 },
  faq:       { chunkingStrategy: 'qa_pair',  retrievalMode: 'hybrid', vectorWeight: 0.7, scoreThreshold: 0, shortcutThreshold: 0.7 },
  temporal:  { chunkingStrategy: 'auto',     retrievalMode: 'hybrid', vectorWeight: 0.6, scoreThreshold: 0, shortcutThreshold: 0 },
  multimodal:{ chunkingStrategy: 'auto',     retrievalMode: 'vector', vectorWeight: 1,   scoreThreshold: 0, shortcutThreshold: 0 },
};

// ── 样式常量 ──────────────────────────────────────────
const INPUT_H = 38; // 统一所有 input/select 的高度
const field: CSSProperties = { display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0 };
const fieldLabel: CSSProperties = { fontSize: 13, fontWeight: 600, color: tokens.text, lineHeight: '20px' };
const fieldHelper: CSSProperties = { fontSize: 12, color: tokens.textMuted, lineHeight: 1.5, marginTop: 2, minHeight: 18 };
// 统一控件高度，保证同行的 select 和 input 框高度一致
const inputStyle: CSSProperties = { ...s.input, height: INPUT_H, boxSizing: 'border-box' };
// alignItems:'start'，helper 占位高度统一，input 顶部对齐
const row3: CSSProperties = { display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, alignItems: 'start' };
const row2: CSSProperties = { display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 16, alignItems: 'start' };

function infoBox(color: string): CSSProperties {
  return {
    marginTop: 12,
    padding: '10px 14px',
    background: `${color}12`,
    borderRadius: 6,
    borderLeft: `3px solid ${color}`,
  };
}

function defaultState(form: KbForm = 'standard'): EditorState {
  const preset = FORM_DEFAULTS[form];
  return {
    name: '', description: '', kbForm: form,
    chunkingStrategy: preset.chunkingStrategy,
    chunkSize: 512, chunkOverlap: 50,
    retrievalMode: preset.retrievalMode,
    topK: 5,
    vectorWeight: preset.vectorWeight,
    scoreThreshold: preset.scoreThreshold,
    shortcutThreshold: preset.shortcutThreshold,
    embeddingModel: 'BAAI/bge-small-zh-v1.5',
    rerankModel: 'BAAI/bge-reranker-base',
  };
}

function stateFromKb(kb: KnowledgeBase): EditorState {
  return {
    name: kb.name, description: kb.description ?? '',
    kbForm: kb.kb_form,
    chunkingStrategy: kb.chunking_strategy as ChunkingStrategy,
    chunkSize: kb.chunk_size ?? 512, chunkOverlap: kb.chunk_overlap ?? 50,
    retrievalMode: kb.retrieval_mode,
    topK: kb.top_k ?? 5,
    vectorWeight: kb.vector_weight ?? 0.6,
    scoreThreshold: kb.score_threshold ?? 0,
    shortcutThreshold: kb.shortcut_threshold ?? 0,
    embeddingModel: kb.embedding_model ?? '',
    rerankModel: kb.rerank_model ?? '',
  };
}

export function KbEditorPage({ mode, kb, onCancel, onSaved }: KbEditorPageProps) {
  const [values, setValues] = useState<EditorState>(() => (kb ? stateFromKb(kb) : defaultState()));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    setValues(kb ? stateFromKb(kb) : defaultState());
    setError('');
  }, [kb?.id, mode]);

  const keywordWeight = Number((1 - values.vectorWeight).toFixed(2));
  const showSizeFields = !NO_SIZE_STRATEGIES.includes(values.chunkingStrategy);
  const showHybridWeight = values.retrievalMode === 'hybrid';
  const showShortcut = values.kbForm === 'faq';
  // 编辑模式下，已上传文档依赖这些参数，不允许修改
  const isLocked = (key: keyof EditorState) => mode === 'edit' && EDIT_LOCKED.includes(key);

  const valid =
    values.name.trim().length > 0 &&
    values.chunkSize >= 100 && values.chunkSize <= 4000 &&
    values.chunkOverlap >= 0 && values.chunkOverlap < values.chunkSize &&
    values.topK >= 1 && values.topK <= 50 &&
    values.vectorWeight >= 0 && values.vectorWeight <= 1 &&
    values.scoreThreshold >= 0 && values.scoreThreshold <= 1 &&
    values.shortcutThreshold >= 0 && values.shortcutThreshold <= 1;

  function setField<K extends keyof EditorState>(key: K, value: EditorState[K]) {
    if (isLocked(key)) return;
    setValues((prev) => ({ ...prev, [key]: value }));
  }

  function applyTemplate(nextForm: KbForm) {
    const preset = FORM_DEFAULTS[nextForm];
    setValues((prev) => ({
      ...prev, kbForm: nextForm,
      // 分块相关在编辑模式锁定，不覆盖
      ...(mode === 'create' ? {
        chunkingStrategy: preset.chunkingStrategy,
      } : {}),
      retrievalMode: preset.retrievalMode,
      vectorWeight: preset.vectorWeight,
      scoreThreshold: preset.scoreThreshold,
      shortcutThreshold: preset.shortcutThreshold,
    }));
  }

  function setRetrievalMode(next: RetrievalMode) {
    setValues((prev) => ({
      ...prev, retrievalMode: next,
      vectorWeight: next === 'vector' ? 1 : next === 'fulltext' ? 0 : prev.vectorWeight,
    }));
  }

  async function handleSave() {
    if (!valid || saving) return;
    if (mode === 'edit' && !kb) { setError('缺少要编辑的知识库'); return; }
    setSaving(true);
    setError('');
    try {
      const finalVectorWeight = values.retrievalMode === 'hybrid' ? values.vectorWeight : values.retrievalMode === 'vector' ? 1 : 0;
      const finalKeywordWeight = values.retrievalMode === 'hybrid' ? keywordWeight : values.retrievalMode === 'fulltext' ? 1 : 0;
      const payload = {
        name: values.name.trim(), description: values.description.trim(),
        kb_form: values.kbForm,
        chunking_strategy: values.chunkingStrategy,
        chunk_size: values.chunkSize, chunk_overlap: values.chunkOverlap,
        retrieval_mode: values.retrievalMode,
        top_k: values.topK,
        vector_weight: finalVectorWeight, keyword_weight: finalKeywordWeight,
        score_threshold: values.scoreThreshold,
        shortcut_threshold: values.shortcutThreshold,
        embedding_model: values.embeddingModel.trim(),
        rerank_model: values.rerankModel.trim(),
      };
      const saved = mode === 'create' ? await createKb(payload) : await updateKbConfig(kb!.id, payload);
      onSaved(saved);
    } catch (e) {
      setError(`保存失败：${(e as Error).message}`);
    } finally {
      setSaving(false);
    }
  }

  const saveLabel = saving ? '保存中...' : mode === 'create' ? '创建知识库' : '保存修改';

  const chunkDesc = CHUNKING_DESCRIPTIONS[values.chunkingStrategy];
  const retrievalDesc = RETRIEVAL_DESCRIPTIONS[values.retrievalMode];

  // 锁定字段的样式
  const lockedInput: CSSProperties = { ...inputStyle, background: '#f5f6fa', color: tokens.textMuted, cursor: 'not-allowed' };
  const lockedHint: CSSProperties = { fontSize: 11, color: tokens.warn, marginTop: 2 };

  return (
    <div style={{ maxWidth: 760, margin: '0 auto' }}>
      {/* 顶部 */}
      <div style={{ ...s.row, justifyContent: 'space-between', marginBottom: 24 }}>
        <div style={s.row}>
          <button style={s.btnGhost} onClick={onCancel}>← 返回</button>
          <div>
            <div style={{ fontSize: 18, fontWeight: 700, color: tokens.text }}>
              {mode === 'create' ? '新建知识库' : '编辑知识库'}
            </div>
            <div style={{ ...fieldHelper, marginTop: 2 }}>
              {mode === 'create' ? '配置基础信息、分块策略和检索参数' : `${kb?.name ?? ''} · ${kb?.id ?? ''}`}
            </div>
          </div>
        </div>
      </div>

      {error && <div style={{ ...s.error, marginBottom: 16 }}>{error}</div>}

      {/* 基础信息 */}
      <section style={{ ...s.card, marginBottom: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: tokens.text, marginBottom: 14 }}>基础信息</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
          <label style={field}>
            <span style={fieldLabel}>名称 <span style={{ color: tokens.danger }}>*</span></span>
            <input
              style={inputStyle} value={values.name}
              placeholder="例如：售后 FAQ"
              onChange={(e) => setField('name', e.target.value)}
            />
          </label>
          <label style={field}>
            <span style={fieldLabel}>描述</span>
            <textarea
              style={{ ...s.input, minHeight: 80, resize: 'vertical', lineHeight: 1.55 }}
              value={values.description}
              placeholder="记录知识库用途、适用业务线或维护说明"
              onChange={(e) => setField('description', e.target.value)}
            />
          </label>
        </div>
      </section>

      {/* 快速模板 */}
      <section style={{ ...s.card, marginBottom: 16 }}>
        <div style={{ ...s.row, justifyContent: 'space-between', marginBottom: 14 }}>
          <div style={{ fontSize: 14, fontWeight: 700, color: tokens.text }}>快速模板（可选）</div>
          <span style={{ fontSize: 12, color: tokens.textMuted }}>选择后自动填充推荐配置，也可跳过手动配置</span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
          {FORM_ORDER.map((form) => {
            const meta = kbFormMeta(form);
            const active = values.kbForm === form;
            const templateLocked = mode === 'edit';
            return (
              <button
                key={form}
                type="button"
                disabled={templateLocked}
                onClick={() => !templateLocked && (active ? applyTemplate('standard') : applyTemplate(form))}
                style={{
                  padding: '10px 12px', minHeight: 76, textAlign: 'left',
                  border: `1.5px solid ${active ? meta.color : tokens.border}`,
                  borderRadius: 8,
                  background: templateLocked ? '#f5f6fa' : active ? `${meta.color}12` : '#fff',
                  color: templateLocked ? tokens.textMuted : tokens.text,
                  cursor: templateLocked ? 'not-allowed' : 'pointer',
                  transition: 'all 0.15s',
                  opacity: templateLocked ? 0.6 : 1,
                }}
              >
                <span style={{ ...s.pill, background: templateLocked ? tokens.textMuted : meta.color, display: 'inline-block', marginBottom: 6 }}>{meta.label}</span>
                <div style={{ fontSize: 11, color: tokens.textMuted, lineHeight: 1.45 }}>{meta.hint}</div>
              </button>
            );
          })}
        </div>
        <div style={{ ...fieldHelper, marginTop: 10 }}>
          {mode === 'edit'
            ? '⚠ 编辑模式下模板不可切换，知识形态在创建后固定。'
            : '模板只是快捷方式，选中后可以在下方自由调整任意参数。'}
        </div>
      </section>

      {/* 分块设置 */}
      <section style={{ ...s.card, marginBottom: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: tokens.text, marginBottom: 14 }}>分块设置</div>
        <div style={showSizeFields ? row3 : { display: 'grid', gridTemplateColumns: '1fr', gap: 16, alignItems: 'start' }}>
          <label style={field}>
            <span style={fieldLabel}>分块策略</span>
            <select
              style={isLocked('chunkingStrategy') ? lockedInput : inputStyle}
              value={values.chunkingStrategy}
              disabled={isLocked('chunkingStrategy')}
              onChange={(e) => setField('chunkingStrategy', e.target.value as ChunkingStrategy)}
            >
              {CHUNKING_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>{CHUNKING_DESCRIPTIONS[opt].label}（{opt}）</option>
              ))}
            </select>
            {isLocked('chunkingStrategy') && <span style={lockedHint}>⚠ 已有文档，不可修改</span>}
            {!isLocked('chunkingStrategy') && <span style={fieldHelper}>{/* 占位 */}</span>}
          </label>
          {showSizeFields && (
            <label style={field}>
              <span style={fieldLabel}>Chunk Size</span>
              <input
                type="number" min={100} max={4000}
                style={isLocked('chunkSize') ? lockedInput : inputStyle}
                value={values.chunkSize}
                disabled={isLocked('chunkSize')}
                onChange={(e) => setField('chunkSize', Number(e.target.value) || 0)}
              />
              {isLocked('chunkSize') && <span style={lockedHint}>⚠ 已有文档，不可修改</span>}
              {!isLocked('chunkSize') && <span style={fieldHelper}>{/* 占位 */}</span>}
            </label>
          )}
          {showSizeFields && (
            <label style={field}>
              <span style={fieldLabel}>Overlap</span>
              <input
                type="number" min={0}
                style={isLocked('chunkOverlap') ? lockedInput : inputStyle}
                value={values.chunkOverlap}
                disabled={isLocked('chunkOverlap')}
                onChange={(e) => setField('chunkOverlap', Number(e.target.value) || 0)}
              />
              {isLocked('chunkOverlap') && <span style={lockedHint}>⚠ 已有文档，不可修改</span>}
              {!isLocked('chunkOverlap') && <span style={fieldHelper}>{/* 占位 */}</span>}
            </label>
          )}
        </div>
        {/* 分块策略联动说明 */}
        <div style={infoBox(chunkDesc.color)}>
          <div style={{ fontSize: 13, color: tokens.text, marginBottom: 4, fontWeight: 500 }}>
            {chunkDesc.label}
          </div>
          <div style={{ fontSize: 13, color: tokens.text, marginBottom: chunkDesc.recommend ? 6 : 0 }}>
            {chunkDesc.desc}
          </div>
          {chunkDesc.recommend && (
            <div style={{ fontSize: 12, color: tokens.textMuted }}>
              💡 推荐场景：{chunkDesc.recommend}
            </div>
          )}
        </div>
        {showSizeFields && !isLocked('chunkSize') && (
          <div style={{ ...fieldHelper, marginTop: 8 }}>分块参数影响后续上传或重建索引的文档，已有版本不会自动重切。</div>
        )}
      </section>

      {/* 检索设置 */}
      <section style={{ ...s.card, marginBottom: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: tokens.text, marginBottom: 14 }}>检索设置</div>
        <div style={row3}>
          <label style={field}>
            <span style={fieldLabel}>检索模式</span>
            <select
              style={inputStyle} value={values.retrievalMode}
              onChange={(e) => setRetrievalMode(e.target.value as RetrievalMode)}
            >
              {RETRIEVAL_OPTIONS.map((opt) => (
                <option key={opt} value={opt}>{RETRIEVAL_DESCRIPTIONS[opt].label}（{opt}）</option>
              ))}
            </select>
            <span style={fieldHelper}>{/* 占位，保持三列等高 */}</span>
          </label>
          <label style={field}>
            <span style={fieldLabel}>Top K</span>
            <input
              type="number" min={1} max={50} style={inputStyle}
              value={values.topK}
              onChange={(e) => setField('topK', Number(e.target.value) || 0)}
            />
            <span style={fieldHelper}>最多返回的结果条数</span>
          </label>
          <label style={field}>
            <span style={fieldLabel}>Score 阈值</span>
            <input
              type="number" min={0} max={1} step={0.01} style={inputStyle}
              value={values.scoreThreshold}
              onChange={(e) => setField('scoreThreshold', Number(e.target.value) || 0)}
            />
            <span style={fieldHelper}>过滤低分结果，0 = 不过滤</span>
          </label>
        </div>
        {/* 检索模式联动说明 */}
        <div style={infoBox(retrievalDesc.color)}>
          <div style={{ fontSize: 13, color: tokens.text, marginBottom: 4, fontWeight: 500 }}>
            {retrievalDesc.label}
          </div>
          <div style={{ fontSize: 13, color: tokens.text, marginBottom: 6 }}>
            {retrievalDesc.desc}
          </div>
          <div style={{ fontSize: 12, color: tokens.textMuted }}>
            💡 推荐场景：{retrievalDesc.recommend}
          </div>
        </div>

        {showHybridWeight && (
          <div style={{ marginTop: 16 }}>
            <div style={{ ...s.row, justifyContent: 'space-between', marginBottom: 8 }}>
              <span style={fieldLabel}>语义 / 关键词权重</span>
              <span style={{ fontSize: 12, color: tokens.textMuted }}>
                语义 {Math.round(values.vectorWeight * 100)}% / 关键词 {Math.round((1 - values.vectorWeight) * 100)}%
              </span>
            </div>
            <input
              type="range" min={0} max={1} step={0.05} style={{ width: '100%' }}
              value={values.vectorWeight}
              onChange={(e) => setField('vectorWeight', Number(e.target.value))}
            />
            <div style={{ ...fieldHelper, marginTop: 6 }}>
              💡 推荐值：通用场景 语义 60% / 关键词 40%，FAQ / 专有名词多时可调至 语义 40% / 关键词 60%。Score 阈值作用于融合前的原始召回分，过滤后再做 RRF 融合。
            </div>
          </div>
        )}

        {showShortcut && (
          <div style={{ marginTop: 16 }}>
            <label style={field}>
              <span style={fieldLabel}>FAQ 短路阈值</span>
              <input
                type="number" min={0} max={1} step={0.01}
                style={{ ...inputStyle, width: 160 }}
                value={values.shortcutThreshold}
                onChange={(e) => setField('shortcutThreshold', Number(e.target.value) || 0)}
              />
              <span style={fieldHelper}>
                向量语义相似度达到此值时直接返回 FAQ 答案，跳过 LLM 生成，降低延迟和成本。0 = 不启用。推荐值 0.7~0.8。
              </span>
            </label>
          </div>
        )}
      </section>

      {/* 模型设置 */}
      <section style={{ ...s.card, marginBottom: 24 }}>
        <div style={{ fontSize: 14, fontWeight: 700, color: tokens.text, marginBottom: 14 }}>模型设置</div>
        <div style={row2}>
          <label style={field}>
            <span style={fieldLabel}>Embedding 模型</span>
            <input
              style={isLocked('embeddingModel') ? lockedInput : inputStyle}
              value={values.embeddingModel}
              placeholder="BAAI/bge-small-zh-v1.5"
              disabled={isLocked('embeddingModel')}
              onChange={(e) => setField('embeddingModel', e.target.value)}
            />
            {isLocked('embeddingModel') && <span style={lockedHint}>⚠ 已有文档，不可修改（改变向量维度会导致旧索引失效）</span>}
            {!isLocked('embeddingModel') && <span style={fieldHelper}>{/* 占位 */}</span>}
          </label>
          <label style={field}>
            <span style={fieldLabel}>Rerank 模型</span>
            <input
              style={inputStyle} value={values.rerankModel}
              placeholder="BAAI/bge-reranker-base"
              onChange={(e) => setField('rerankModel', e.target.value)}
            />
            <span style={fieldHelper}>{/* 占位 */}</span>
          </label>
        </div>
      </section>

      {/* 底部操作栏 */}
      <div style={{ ...s.row, justifyContent: 'space-between' }}>
        {!valid ? (
          <span style={{ fontSize: 13, color: tokens.danger }}>
            请检查：名称不能为空，Chunk Size 100~4000，Overlap 需小于 Size，Top K 1~50，阈值 0~1。
          </span>
        ) : (
          <span style={{ fontSize: 13, color: tokens.textMuted }}>配置保存后立即用于后续检索和新文档构建。</span>
        )}
        <div style={s.row}>
          <button style={s.btnGhost} onClick={onCancel}>取消</button>
          <button
            style={{ ...s.btn, opacity: !valid || saving ? 0.6 : 1 }}
            onClick={handleSave}
            disabled={!valid || saving}
          >
            {saveLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

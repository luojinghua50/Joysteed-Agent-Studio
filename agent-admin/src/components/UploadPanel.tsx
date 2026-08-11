import { useRef, useState } from 'react';
import { uploadDocument, type DocumentInfo } from '@/services/rag';
import { s, tokens } from '@/styles/theme';

interface UploadPanelProps {
  kbId: string;
  onUploaded: (doc: DocumentInfo) => void;
}

// 支持的格式及说明
const ACCEPT = '.md,.txt,.csv,.json,.markdown,.text,.log,.tsv,.yaml,.yml,.pdf,.docx,.doc,.xlsx';
const FORMATS = [
  { ext: '.pdf', desc: 'PDF 文档，自动提取正文文字' },
  { ext: '.docx / .doc', desc: 'Word 文档，提取段落和表格' },
  { ext: '.xlsx', desc: 'Excel 表格，转为文本行' },
  { ext: '.md / .markdown', desc: '产品文档、知识文章' },
  { ext: '.txt / .text', desc: '纯文本、日志' },
  { ext: '.csv / .tsv', desc: '表格数据、FAQ 导出' },
  { ext: '.json', desc: '结构化数据' },
  { ext: '.yaml / .yml', desc: '配置类文档' },
];

export function UploadPanel({ kbId, onUploaded }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [last, setLast] = useState<DocumentInfo | null>(null);
  const [showFormats, setShowFormats] = useState(false);

  async function doUpload(file: File) {
    setUploading(true);
    setError('');
    setLast(null);
    try {
      const doc = await uploadDocument(kbId, file);
      setLast(doc);
      onUploaded(doc);
    } catch (e) {
      setError(`上传失败：${(e as Error).message}`);
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = '';
    }
  }

  function onPick(files: FileList | null) {
    if (files && files.length > 0) doUpload(files[0]);
  }

  return (
    <div style={s.card}>
      <div style={{ ...s.row, justifyContent: 'space-between', marginBottom: 14 }}>
        <div style={s.sectionTitle}>上传文档</div>
        <button
          style={{ ...s.btnGhost, fontSize: 12 }}
          onClick={() => setShowFormats((v) => !v)}
        >
          {showFormats ? '收起格式说明 ▲' : '支持哪些格式？▼'}
        </button>
      </div>

      {/* 格式说明展开区 */}
      {showFormats && (
        <div style={{
          marginBottom: 14,
          padding: '12px 16px',
          background: '#f0f7ff',
          border: `1px solid ${tokens.brand}`,
          borderRadius: 8,
        }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: tokens.text, marginBottom: 8 }}>
            支持的文件格式
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '6px 24px' }}>
            {FORMATS.map((f) => (
              <div key={f.ext} style={{ fontSize: 13, color: tokens.text }}>
                <span style={{ fontWeight: 500, color: tokens.brandDeep }}>{f.ext}</span>
                <span style={{ color: tokens.textMuted }}> — {f.desc}</span>
              </div>
            ))}
          </div>
          <div style={{ fontSize: 12, color: tokens.textMuted, marginTop: 10 }}>
            PDF / Word / Excel 会自动提取文字后再切分索引，扫描件（纯图片 PDF）无法提取文字。
          </div>
        </div>
      )}

      {error && <div style={{ ...s.error, marginBottom: 12 }}>{error}</div>}

      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); onPick(e.dataTransfer.files); }}
        onClick={() => !uploading && inputRef.current?.click()}
        style={{
          border: `2px dashed ${dragging ? tokens.brand : tokens.border}`,
          borderRadius: tokens.radius,
          padding: '28px 20px',
          textAlign: 'center',
          cursor: uploading ? 'not-allowed' : 'pointer',
          background: dragging ? '#f0f7ff' : '#fafbfe',
          transition: 'all 0.15s',
        }}
      >
        <div style={{ fontSize: 28, marginBottom: 8 }}>📄</div>
        <div style={{ fontWeight: 500, color: tokens.text, marginBottom: 4 }}>
          {uploading ? '上传并索引中…' : '点击选择文件，或拖拽到此处'}
        </div>
        <div style={{ fontSize: 12, color: tokens.textMuted }}>
          支持 .pdf / .docx / .md / .txt / .csv / .json 等格式，自动提取文字后索引
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          style={{ display: 'none' }}
          onChange={(e) => onPick(e.target.files)}
        />
      </div>

      {last && (
        <div style={{
          marginTop: 12,
          padding: '10px 14px',
          background: '#f0fdf4',
          border: `1px solid ${tokens.success}`,
          borderRadius: 8,
          fontSize: 13,
          color: tokens.text,
        }}>
          ✓ 已上传 <strong>{last.filename}</strong> — 版本 v{last.version_no}，切分{' '}
          <strong>{last.chunk_count}</strong> 个 chunk，状态 <strong>{last.status}</strong>
        </div>
      )}
    </div>
  );
}

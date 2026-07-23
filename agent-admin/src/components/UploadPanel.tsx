import { useRef, useState } from 'react';
import { uploadDocument, type DocumentInfo } from '@/services/rag';
import { s, tokens } from '@/styles/theme';

interface UploadPanelProps {
  kbId: string;
  onUploaded: (doc: DocumentInfo) => void;
}

// Backend extracts text via content.decode("utf-8") — text formats only.
const ACCEPT = '.md,.txt,.csv,.json,.markdown,.text,.log,.tsv,.yaml,.yml';
const ACCEPT_HINT = '仅支持纯文本格式(.md / .txt / .csv / .json 等)。PDF、Word 等二进制文件会解析为乱码。';

export function UploadPanel({ kbId, onUploaded }: UploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [last, setLast] = useState<DocumentInfo | null>(null);

  async function doUpload(file: File) {
    setUploading(true);
    setError('');
    setLast(null);
    try {
      const doc = await uploadDocument(kbId, file);
      setLast(doc);
      onUploaded(doc);
    } catch (e) {
      setError(`上传失败:${(e as Error).message}`);
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
      <div style={s.sectionTitle}>上传文档</div>
      {error && <div style={s.error}>{error}</div>}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          onPick(e.dataTransfer.files);
        }}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `2px dashed ${dragging ? tokens.brand : tokens.border}`,
          borderRadius: tokens.radius,
          padding: '32px 20px',
          textAlign: 'center',
          cursor: 'pointer',
          background: dragging ? '#f0f7ff' : '#fafbfe',
          transition: 'all 0.15s',
        }}
      >
        <div style={{ fontSize: 28, marginBottom: 8 }}>📄</div>
        <div style={{ fontWeight: 500, color: tokens.text }}>
          {uploading ? '上传并索引中…' : '点击选择文件,或拖拽到此处'}
        </div>
        <div style={{ ...s.muted, marginTop: 6 }}>{ACCEPT_HINT}</div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          style={{ display: 'none' }}
          onChange={(e) => onPick(e.target.files)}
        />
      </div>

      {last && (
        <div
          style={{
            marginTop: 14,
            padding: '10px 14px',
            background: '#f0fdf4',
            border: `1px solid ${tokens.success}`,
            borderRadius: 8,
            fontSize: 13,
            color: tokens.text,
          }}
        >
          已上传 <strong>{last.filename}</strong> — 版本 v{last.version_no}、状态{' '}
          <strong>{last.status}</strong>、切分 <strong>{last.chunk_count}</strong> 个 chunk。
        </div>
      )}
    </div>
  );
}

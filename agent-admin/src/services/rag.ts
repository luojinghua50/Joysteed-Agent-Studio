// RAG (agent-rag) API client. All requests go through the `/rag` prefix, which
// the Vite dev proxy and the nginx prod config rewrite onto agent-rag:8010.
//
// Tenant is fixed to `default` for now (matches agent-rag's _tenant header
// default). Replace X_TENANT once real admin auth lands.

const BASE = '/rag';
const TENANT = 'default';

function headers(extra?: Record<string, string>): Record<string, string> {
  return { 'x-tenant-id': TENANT, ...extra };
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
    } catch {
      // non-JSON error body, keep the status line
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

// ===== Types =====

export type ChunkingStrategy =
  | 'auto'
  | 'recursive'
  | 'heading'
  | 'fixed'
  | 'semantic'
  | 'table'
  | 'qa_pair'
  | 'parent_child';

export type RetrievalMode = 'vector' | 'fulltext' | 'hybrid';

// 知识形态：运营只选形态，后端按形态绑定切片/检索/短路默认配置。
export type KbForm = 'standard' | 'faq' | 'temporal' | 'multimodal';

export type MetaFieldType = 'string' | 'number' | 'time';

export interface KnowledgeBase {
  id: string;
  tenant_id: string;
  name: string;
  description: string;
  chunking_strategy: string;
  chunk_size: number;
  chunk_overlap: number;
  embedding_model: string;
  rerank_model: string;
  document_count: number;
  kb_form: KbForm;
  retrieval_mode: RetrievalMode;
  top_k: number;
  priority_weight: number;
  vector_weight: number;
  keyword_weight: number;
  score_threshold: number;
  shortcut_threshold: number;
  created_at?: string;
}

export interface KbConfigInput {
  name?: string;
  description?: string;
  kb_form?: KbForm;
  chunking_strategy?: ChunkingStrategy;
  chunk_size?: number;
  chunk_overlap?: number;
  retrieval_mode?: RetrievalMode;
  top_k?: number;
  priority_weight?: number;
  vector_weight?: number;
  keyword_weight?: number;
  score_threshold?: number;
  shortcut_threshold?: number;
  embedding_model?: string;
  rerank_model?: string;
}

export interface MetadataField {
  id: string;
  name: string;
  field_type: MetaFieldType;
}

export interface DocumentInfo {
  id: string;
  tenant_id: string;
  kb_id: string;
  filename: string;
  current_version_id: string | null;
  version_no?: number;
  status?: string;
  file_type?: string;
  file_size?: number;
  chunk_count?: number;
}

export interface VersionInfo {
  id: string;
  version_no: number;
  status: string;
  file_hash: string;
  file_size: number;
  chunk_count: number;
  created_by: string;
  is_current: boolean;
}

export interface ChunkPositionMeta {
  page?: number;
  line_start?: number;
  line_end?: number;
  total_lines?: number;
  [key: string]: unknown;
}

export interface SearchResultItem {
  chunk_id: string;
  doc_id: string;
  text: string;
  score: number;
  kb_id?: string;
  source?: string;
  version_no?: number | null;
  metadata?: ChunkPositionMeta;
}

export interface SearchResponse {
  query: string;
  results: SearchResultItem[];
  total: number;
}

// 聚合检索响应：带路由溯源（命中哪些库、是否走了 faq 短路）。
export interface RouteSearchResponse {
  query: string;
  results: SearchResultItem[];
  total: number;
  shortcut: boolean;
  routed_kbs: string[];
}

// ===== Knowledge bases =====

export function listKbs(): Promise<KnowledgeBase[]> {
  return fetch(`${BASE}/api/knowledge-bases`, { headers: headers() }).then(asJson<KnowledgeBase[]>);
}

function appendConfigParams(qs: URLSearchParams, config: KbConfigInput, mode: 'create' | 'update') {
  if (config.name !== undefined) qs.set('name', config.name);
  if (config.description !== undefined) qs.set('description', config.description);
  if (config.kb_form !== undefined) qs.set('kb_form', config.kb_form);
  if (config.chunking_strategy !== undefined) {
    qs.set(mode === 'create' ? 'strategy' : 'chunking_strategy', config.chunking_strategy);
  }
  if (config.chunk_size !== undefined) qs.set('chunk_size', String(config.chunk_size));
  if (config.chunk_overlap !== undefined) qs.set('chunk_overlap', String(config.chunk_overlap));
  if (config.retrieval_mode !== undefined) qs.set('retrieval_mode', config.retrieval_mode);
  if (config.top_k !== undefined) qs.set('top_k', String(config.top_k));
  if (config.priority_weight !== undefined) qs.set('priority_weight', String(config.priority_weight));
  if (config.vector_weight !== undefined) qs.set('vector_weight', String(config.vector_weight));
  if (config.keyword_weight !== undefined) qs.set('keyword_weight', String(config.keyword_weight));
  if (config.score_threshold !== undefined) qs.set('score_threshold', String(config.score_threshold));
  if (config.shortcut_threshold !== undefined) qs.set('shortcut_threshold', String(config.shortcut_threshold));
  if (config.embedding_model !== undefined) qs.set('embedding_model', config.embedding_model);
  if (config.rerank_model !== undefined) qs.set('rerank_model', config.rerank_model);
}

export function createKb(config: KbConfigInput & { name: string }): Promise<KnowledgeBase> {
  // NOTE: agent-rag takes these as QUERY params, not a JSON body. Chinese must
  // be URL-encoded (URLSearchParams handles that) or the server returns 400.
  const qs = new URLSearchParams({
    name: config.name,
    description: config.description ?? '',
    kb_form: config.kb_form ?? 'standard',
  });
  appendConfigParams(qs, config, 'create');
  return fetch(`${BASE}/api/knowledge-bases?${qs}`, {
    method: 'POST',
    headers: headers(),
  }).then(asJson<KnowledgeBase>);
}

export function deleteKb(kbId: string): Promise<unknown> {
  return fetch(`${BASE}/api/knowledge-bases/${kbId}`, {
    method: 'DELETE',
    headers: headers(),
  }).then(asJson);
}

export function updateKbConfig(
  kbId: string,
  config: KbConfigInput,
): Promise<KnowledgeBase> {
  const qs = new URLSearchParams();
  appendConfigParams(qs, config, 'update');
  return fetch(`${BASE}/api/knowledge-bases/${kbId}?${qs}`, {
    method: 'PATCH',
    headers: headers(),
  }).then(asJson<KnowledgeBase>);
}

export function updateKbThreshold(kbId: string, shortcutThreshold: number): Promise<KnowledgeBase> {
  return updateKbConfig(kbId, { shortcut_threshold: shortcutThreshold });
}

// ===== Documents =====

export function listDocuments(kbId: string): Promise<DocumentInfo[]> {
  return fetch(`${BASE}/api/knowledge-bases/${kbId}/documents`, { headers: headers() }).then(
    asJson<DocumentInfo[]>,
  );
}

export function uploadDocument(kbId: string, file: File): Promise<DocumentInfo> {
  const form = new FormData();
  form.append('file', file);
  return fetch(`${BASE}/api/knowledge-bases/${kbId}/documents`, {
    method: 'POST',
    headers: headers(), // do NOT set Content-Type; browser sets multipart boundary
    body: form,
  }).then(asJson<DocumentInfo>);
}

export function listVersions(docId: string): Promise<VersionInfo[]> {
  return fetch(`${BASE}/api/documents/${docId}/versions`, { headers: headers() }).then(
    asJson<VersionInfo[]>,
  );
}

export function rollbackDocument(docId: string, targetVersionNo: number): Promise<unknown> {
  const qs = new URLSearchParams({ target_version_no: String(targetVersionNo) });
  return fetch(`${BASE}/api/documents/${docId}/rollback?${qs}`, {
    method: 'POST',
    headers: headers(),
  }).then(asJson);
}

export function deleteDocument(docId: string): Promise<unknown> {
  return fetch(`${BASE}/api/documents/${docId}`, {
    method: 'DELETE',
    headers: headers(),
  }).then(asJson);
}

// ===== Search =====

export function search(kbId: string, query: string, topK = 5): Promise<SearchResponse> {
  return fetch(`${BASE}/api/search`, {
    method: 'POST',
    headers: headers({ 'Content-Type': 'application/json' }),
    body: JSON.stringify({ query, kb_id: kbId, top_k: topK }),
  }).then(asJson<SearchResponse>);
}

// 聚合检索（生产链路）：跨库路由 + faq 短路 + 加权 RRF。
// scope 限定参与库（传 kb_id 或 kb_form）；空数组/undefined = 租户下全部库。
export function routeSearch(
  query: string,
  scope?: string[],
  topK = 5,
): Promise<RouteSearchResponse> {
  const body: Record<string, unknown> = { query, top_k: topK };
  if (scope && scope.length > 0) body.scope = scope;
  return fetch(`${BASE}/api/route-search`, {
    method: 'POST',
    headers: headers({ 'Content-Type': 'application/json' }),
    body: JSON.stringify(body),
  }).then(asJson<RouteSearchResponse>);
}

// ===== Metadata fields (库内可过滤字段定义) =====

export function listMetadataFields(kbId: string): Promise<MetadataField[]> {
  return fetch(`${BASE}/api/knowledge-bases/${kbId}/metadata-fields`, {
    headers: headers(),
  }).then(asJson<MetadataField[]>);
}

export function createMetadataField(
  kbId: string,
  name: string,
  fieldType: MetaFieldType = 'string',
): Promise<MetadataField> {
  // 后端取 query 参数；name 含中文需 URL 编码。
  const qs = new URLSearchParams({ name, field_type: fieldType });
  return fetch(`${BASE}/api/knowledge-bases/${kbId}/metadata-fields?${qs}`, {
    method: 'POST',
    headers: headers(),
  }).then(asJson<MetadataField>);
}

export function deleteMetadataField(kbId: string, fieldId: string): Promise<unknown> {
  return fetch(`${BASE}/api/knowledge-bases/${kbId}/metadata-fields/${fieldId}`, {
    method: 'DELETE',
    headers: headers(),
  }).then(asJson);
}

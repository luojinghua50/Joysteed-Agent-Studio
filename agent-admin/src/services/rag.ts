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

export type ChunkingStrategy = 'auto' | 'recursive' | 'heading' | 'fixed' | 'qa_pair';

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
  document_count: number;
  // 形态及其绑定的检索配置（后端 _kb_dict 暴露）。
  kb_form: KbForm;
  retrieval_mode: string;
  priority_weight: number;
  shortcut_threshold: number; // 仅 faq 库有意义：高置信短路阈值，可经 PATCH 调整。
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

export interface SearchResultItem {
  chunk_id: string;
  doc_id: string;
  text: string;
  score: number;
  // route-search 溯源：命中来自哪个库、来源标签、版本号、元数据。
  kb_id?: string;
  source?: string;
  version_no?: number | null;
  metadata?: Record<string, unknown>;
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

export function createKb(
  name: string,
  description = '',
  kbForm: KbForm = 'standard',
  strategy?: ChunkingStrategy,
): Promise<KnowledgeBase> {
  // NOTE: agent-rag takes these as QUERY params, not a JSON body. Chinese must
  // be URL-encoded (URLSearchParams handles that) or the server returns 400.
  // strategy 可选：省略时后端按 kb_form 绑定默认切片策略。
  const qs = new URLSearchParams({ name, description, kb_form: kbForm });
  if (strategy) qs.set('strategy', strategy);
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

// 更新 faq 库的高置信短路阈值。后端取 query 参数、值域 [0,1]、仅 faq 库有效。
export function updateKbThreshold(
  kbId: string,
  shortcutThreshold: number,
): Promise<KnowledgeBase> {
  const qs = new URLSearchParams({ shortcut_threshold: String(shortcutThreshold) });
  return fetch(`${BASE}/api/knowledge-bases/${kbId}?${qs}`, {
    method: 'PATCH',
    headers: headers(),
  }).then(asJson<KnowledgeBase>);
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

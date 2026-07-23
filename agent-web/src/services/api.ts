import { authFetch } from './auth';

const API_BASE = '/v1';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
  agent?: string;
  stopped?: boolean;   // 用户中途停止生成；UI 据此显示「已停止」而非空气泡
}

export async function sendMessage(
  sessionId: string,
  content: string,
  customerId: string,
  onToken: (token: string) => void,
  onDone: (agent: string) => void,
  onError: (error: string) => void,
): Promise<void> {
  const response = await authFetch(`${API_BASE}/chat/${sessionId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, customer_id: customerId }),
  });

  if (!response.ok) {
    onError(`HTTP error: ${response.status}`);
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    onError('No response body');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const data = line.slice(6).trim();
      if (!data) continue;

      try {
        const parsed = JSON.parse(data);
        if (parsed.type === 'token') {
          onToken(parsed.content);
        } else if (parsed.type === 'done') {
          onDone(parsed.agent || '');
        } else if (parsed.type === 'error') {
          onError(parsed.message);
        }
      } catch {
        // skip malformed JSON
      }
    }
  }
}

export async function getChatHistory(
  sessionId: string,
  customerId: string,
): Promise<ChatMessage[]> {
  const params = new URLSearchParams({ customer_id: customerId });
  const response = await authFetch(`${API_BASE}/chat/${sessionId}/history?${params}`);
  if (!response.ok) return [];
  const data = await response.json();
  return data.messages || [];
}

export async function createSession(customerId: string): Promise<string> {
  const response = await authFetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content: 'init', customer_id: customerId }),
  });
  const data = await response.json();
  return data.session_id;
}

export async function submitApproval(
  sessionId: string,
  approved: boolean,
  customerId: string,
  reason: string = '',
): Promise<void> {
  await authFetch(`${API_BASE}/chat/${sessionId}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approved, reason, customer_id: customerId }),
  });
}

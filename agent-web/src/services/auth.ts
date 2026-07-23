/**
 * Client-side auth: token storage + auth API + an auth-aware fetch wrapper.
 *
 * Migration note (P1-C before P1-B): the backend still trusts the self-reported
 * customer_id in the request body. We attach the Bearer token now so that when
 * P1-B flips the chat endpoints to token-derived identity, the frontend already
 * sends what's needed and nothing breaks on flag day.
 */

const ACCESS_KEY = 'agent_access_token';
const REFRESH_KEY = 'agent_refresh_token';
const CUSTOMER_KEY = 'agent_customer_id';

export interface AuthTokens {
  access_token: string;
  refresh_token: string | null;
  customer_id: string;
}

export function getAccessToken(): string | null {
  return localStorage.getItem(ACCESS_KEY);
}

export function getRefreshToken(): string | null {
  return localStorage.getItem(REFRESH_KEY);
}

export function isLoggedIn(): boolean {
  return !!getAccessToken();
}

/** The customer_id in use: the logged-in user's id, or a persistent guest id. */
export function getCustomerId(): string {
  let id = localStorage.getItem(CUSTOMER_KEY);
  if (!id) {
    id = `guest-${crypto.randomUUID()}`;
    localStorage.setItem(CUSTOMER_KEY, id);
  }
  return id;
}

function storeTokens(t: AuthTokens): void {
  localStorage.setItem(ACCESS_KEY, t.access_token);
  if (t.refresh_token) localStorage.setItem(REFRESH_KEY, t.refresh_token);
  localStorage.setItem(CUSTOMER_KEY, t.customer_id);
}

export function clearTokens(): void {
  localStorage.removeItem(ACCESS_KEY);
  localStorage.removeItem(REFRESH_KEY);
  // Drop the logged-in customer_id; a fresh guest id is minted on next use.
  localStorage.removeItem(CUSTOMER_KEY);
}

const API_BASE = '/v1';

export async function register(
  username: string,
  password: string,
  displayName?: string,
): Promise<AuthTokens> {
  const resp = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password, display_name: displayName }),
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `注册失败 (${resp.status})`);
  }
  const tokens: AuthTokens = await resp.json();
  storeTokens(tokens);
  return tokens;
}

export async function login(username: string, password: string): Promise<AuthTokens> {
  const resp = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `登录失败 (${resp.status})`);
  }
  const tokens: AuthTokens = await resp.json();
  storeTokens(tokens);
  return tokens;
}

/** Obtain an anonymous guest token so the visitor flow still carries a trusted,
 *  unforgeable identity (identity is always token-derived after P1-B). */
export async function guestLogin(): Promise<AuthTokens> {
  const resp = await fetch(`${API_BASE}/auth/guest`, { method: 'POST' });
  if (!resp.ok) throw new Error(`访客登录失败 (${resp.status})`);
  const tokens: AuthTokens = await resp.json();
  storeTokens(tokens);
  return tokens;
}

export function logout(): void {
  clearTokens();
}

/** Try to refresh the access token. Returns true on success. */
async function tryRefresh(): Promise<boolean> {
  const refresh_token = getRefreshToken();
  if (!refresh_token) return false;
  const resp = await fetch(`${API_BASE}/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token }),
  });
  if (!resp.ok) return false;
  storeTokens(await resp.json());
  return true;
}

/**
 * fetch wrapper that injects the Bearer token and, on a 401, attempts one
 * refresh-and-retry. Callers handle a still-401 response (e.g. redirect to
 * login). Streaming callers (SSE) use this too — the body is consumed by them.
 */
export async function authFetch(input: string, init: RequestInit = {}): Promise<Response> {
  const withAuth = (token: string | null): RequestInit => {
    const headers = new Headers(init.headers || {});
    if (token) headers.set('Authorization', `Bearer ${token}`);
    return { ...init, headers };
  };

  let resp = await fetch(input, withAuth(getAccessToken()));
  if (resp.status === 401 && (await tryRefresh())) {
    resp = await fetch(input, withAuth(getAccessToken()));
  }
  return resp;
}

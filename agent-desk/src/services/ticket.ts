// Ticket REST client. Routes go through `/ticket`, rewritten by the Vite dev
// proxy and nginx onto ticket-mcp:8003's /api/* REST endpoints.

const BASE = '/ticket/api';

export interface Ticket {
  ticket_id: string;
  customer_id: string;
  title: string;
  description: string;
  status: string;
  priority: string;
  assigned_to: string;
  created_at: string;
  updated_at: string;
}

export interface TicketComment {
  id: number;
  ticket_id: string;
  author: string;
  comment: string;
  created_at: string;
}

export interface TicketDetail extends Ticket {
  comments: TicketComment[];
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.error) detail = body.error;
    } catch {
      // keep status line
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const jsonHeaders = { 'Content-Type': 'application/json' };

export function listTickets(filters: {
  status?: string;
  assigned_to?: string;
  priority?: string;
}): Promise<{ tickets: Ticket[]; total: number }> {
  const qs = new URLSearchParams();
  if (filters.status) qs.set('status', filters.status);
  if (filters.assigned_to) qs.set('assigned_to', filters.assigned_to);
  if (filters.priority) qs.set('priority', filters.priority);
  const q = qs.toString();
  return fetch(`${BASE}/tickets${q ? `?${q}` : ''}`).then(
    asJson<{ tickets: Ticket[]; total: number }>,
  );
}

export function getTicket(ticketId: string): Promise<TicketDetail> {
  return fetch(`${BASE}/tickets/${ticketId}`).then(asJson<TicketDetail>);
}

export function createTicket(body: {
  customer_id: string;
  title: string;
  description?: string;
  priority?: string;
  assigned_to?: string;
}): Promise<Ticket> {
  return fetch(`${BASE}/tickets`, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify(body),
  }).then(asJson<Ticket>);
}

export function updateStatus(ticketId: string, status: string): Promise<Ticket> {
  return fetch(`${BASE}/tickets/${ticketId}/status`, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ status }),
  }).then(asJson<Ticket>);
}

export function reassignTicket(ticketId: string, newAgent: string): Promise<Ticket> {
  return fetch(`${BASE}/tickets/${ticketId}/reassign`, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ new_agent: newAgent }),
  }).then(asJson<Ticket>);
}

export function addComment(
  ticketId: string,
  author: string,
  comment: string,
): Promise<TicketComment> {
  return fetch(`${BASE}/tickets/${ticketId}/comments`, {
    method: 'POST',
    headers: jsonHeaders,
    body: JSON.stringify({ author, comment }),
  }).then(asJson<TicketComment>);
}

export function listAgents(): Promise<{ agents: string[] }> {
  return fetch(`${BASE}/agents`).then(asJson<{ agents: string[] }>);
}

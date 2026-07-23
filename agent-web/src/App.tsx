import { useState, useEffect, useCallback } from 'react';
import { ChatWindow } from '@/components/ChatWindow';
import { LoginForm } from '@/components/LoginForm';
import { useChat } from '@/hooks/useChat';
import { createSession, listSessions, SessionSummary } from '@/services/api';
import { getCustomerId, isLoggedIn, logout, guestLogin } from '@/services/auth';

function App() {
  const [sessionId, setSessionId] = useState<string>('');
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  // null = not yet decided; once set, we have a customer_id (guest or user).
  const [customerId, setCustomerId] = useState<string | null>(
    () => (isLoggedIn() ? getCustomerId() : null),
  );

  const refreshSessions = useCallback(async (): Promise<SessionSummary[]> => {
    const list = await listSessions();
    setSessions(list);
    return list;
  }, []);

  // On login/guest, restore the conversation from the server rather than from
  // localStorage alone: pick the last-used session if it's still ours, else the
  // most recent one, else create a fresh session. This is what lets history
  // survive a logout/login (the session_id in localStorage is not enough).
  useEffect(() => {
    if (!customerId) return;
    let cancelled = false;
    (async () => {
      const list = await refreshSessions();
      if (cancelled) return;
      const stored = localStorage.getItem('agent_session_id');
      const storedIsMine = stored && list.some((s) => s.session_id === stored);
      let id: string;
      if (storedIsMine) {
        id = stored!;
      } else if (list.length > 0) {
        id = list[0].session_id; // newest first, from the backend ordering
      } else {
        id = await createSession(customerId);
        if (cancelled) return;
        await refreshSessions();
      }
      localStorage.setItem('agent_session_id', id);
      setSessionId(id);
    })();
    return () => { cancelled = true; };
  }, [customerId, refreshSessions]);

  const { messages, isLoading, currentAgent, send, stop, reset } = useChat(
    sessionId,
    customerId ?? '',
  );

  const handleNewChat = async () => {
    if (!customerId) return;
    reset();
    localStorage.removeItem('agent_session_id');
    const id = await createSession(customerId);
    localStorage.setItem('agent_session_id', id);
    setSessionId(id);
    await refreshSessions();
  };

  const handleSelectSession = (id: string) => {
    if (id === sessionId) return;
    reset();
    localStorage.setItem('agent_session_id', id);
    setSessionId(id);
  };

  const handleLogout = () => {
    logout();
    localStorage.removeItem('agent_session_id');
    reset();
    setSessions([]);
    setSessionId('');
    setCustomerId(null);
  };

  // Not authenticated and not yet opted into guest mode -> show login.
  if (!customerId) {
    return (
      <LoginForm
        onAuthed={() => setCustomerId(getCustomerId())}
        onGuest={async () => {
          await guestLogin();
          setCustomerId(getCustomerId());
        }}
      />
    );
  }

  if (!sessionId) return null;

  return (
    <div style={{ height: '100vh', maxWidth: '480px', margin: '0 auto', padding: '20px 0', boxSizing: 'border-box' }}>
      <div style={{ height: '100%', borderRadius: '16px', overflow: 'hidden', boxShadow: '0 8px 32px rgba(0,0,0,0.1)' }}>
        <ChatWindow
          messages={messages}
          isLoading={isLoading}
          currentAgent={currentAgent}
          sessions={sessions}
          activeSessionId={sessionId}
          onSelectSession={handleSelectSession}
          onSend={send}
          onStop={stop}
          onNewChat={handleNewChat}
          onLogout={isLoggedIn() ? handleLogout : undefined}
        />
      </div>
    </div>
  );
}

export default App;

import { useState, useEffect } from 'react';
import { ChatWindow } from '@/components/ChatWindow';
import { LoginForm } from '@/components/LoginForm';
import { useChat } from '@/hooks/useChat';
import { createSession } from '@/services/api';
import { getCustomerId, isLoggedIn, logout, guestLogin } from '@/services/auth';

function App() {
  const [sessionId, setSessionId] = useState<string>('');
  // null = not yet decided; once set, we have a customer_id (guest or user).
  const [customerId, setCustomerId] = useState<string | null>(
    () => (isLoggedIn() ? getCustomerId() : null),
  );

  useEffect(() => {
    if (!customerId) return;
    const stored = localStorage.getItem('agent_session_id');
    if (stored) {
      setSessionId(stored);
    } else {
      createSession(customerId).then((id) => {
        localStorage.setItem('agent_session_id', id);
        setSessionId(id);
      });
    }
  }, [customerId]);

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
  };

  const handleLogout = () => {
    logout();
    localStorage.removeItem('agent_session_id');
    reset();
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

import { useState } from 'react';
import { login, register } from '@/services/auth';

interface LoginFormProps {
  onAuthed: () => void;
  onGuest: () => Promise<void>;
}

/** Login / register card. On success stores tokens and calls onAuthed.
 *  "以访客身份继续" obtains a guest token so identity stays token-derived. */
export function LoginForm({ onAuthed, onGuest }: LoginFormProps) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setBusy(true);
    try {
      if (mode === 'login') {
        await login(username, password);
      } else {
        await register(username, password, displayName || undefined);
      }
      onAuthed();
    } catch (err) {
      setError(err instanceof Error ? err.message : '出错了，请重试');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ height: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <form
        onSubmit={submit}
        style={{ width: '320px', padding: '32px', borderRadius: '16px', boxShadow: '0 8px 32px rgba(0,0,0,0.1)', display: 'flex', flexDirection: 'column', gap: '12px' }}
      >
        <h2 style={{ margin: 0, fontSize: '20px' }}>{mode === 'login' ? '登录' : '注册'}</h2>

        <input
          aria-label="用户名"
          placeholder="用户名"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
          required
          style={inputStyle}
        />
        {mode === 'register' && (
          <input
            aria-label="昵称"
            placeholder="昵称（可选）"
            value={displayName}
            onChange={(e) => setDisplayName(e.target.value)}
            style={inputStyle}
          />
        )}
        <input
          aria-label="密码"
          type="password"
          placeholder="密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
          required
          style={inputStyle}
        />

        {error && <div role="alert" style={{ color: '#d33', fontSize: '13px' }}>{error}</div>}

        <button type="submit" disabled={busy} style={primaryBtn}>
          {busy ? '请稍候…' : mode === 'login' ? '登录' : '注册'}
        </button>

        <button
          type="button"
          onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(''); }}
          style={linkBtn}
        >
          {mode === 'login' ? '没有账号？去注册' : '已有账号？去登录'}
        </button>

        <button
          type="button"
          onClick={async () => {
            setError('');
            setBusy(true);
            try {
              await onGuest();
            } catch (err) {
              setError(err instanceof Error ? err.message : '访客登录失败');
            } finally {
              setBusy(false);
            }
          }}
          disabled={busy}
          style={linkBtn}
        >
          以访客身份继续
        </button>
      </form>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  padding: '10px 12px', borderRadius: '8px', border: '1px solid #ddd', fontSize: '14px',
};
const primaryBtn: React.CSSProperties = {
  padding: '10px', borderRadius: '8px', border: 'none', background: '#4f46e5', color: '#fff', fontSize: '14px', cursor: 'pointer',
};
const linkBtn: React.CSSProperties = {
  padding: '4px', border: 'none', background: 'none', color: '#4f46e5', fontSize: '13px', cursor: 'pointer',
};

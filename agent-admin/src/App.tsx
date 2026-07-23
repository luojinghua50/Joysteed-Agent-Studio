import { useState } from 'react';
import { KbList } from '@/components/KbList';
import { KbDetail } from '@/components/KbDetail';
import { type KnowledgeBase } from '@/services/rag';
import { s } from '@/styles/theme';

function App() {
  const [active, setActive] = useState<KnowledgeBase | null>(null);

  return (
    <div style={s.page}>
      <header style={s.header}>
        <span style={{ fontSize: 22 }}>📚</span>
        <span style={s.headerTitle}>知识库管理</span>
        <span style={s.headerSub}>Agent Admin · RAG Console</span>
      </header>
      <div style={s.container}>
        {active ? (
          <KbDetail kb={active} onBack={() => setActive(null)} />
        ) : (
          <KbList onOpen={setActive} />
        )}
      </div>
    </div>
  );
}

export default App;

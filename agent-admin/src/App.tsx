import { useState } from 'react';
import { KbList } from '@/components/KbList';
import { KbDetail } from '@/components/KbDetail';
import { KbEditorPage } from '@/components/KbEditorPage';
import { type KnowledgeBase } from '@/services/rag';
import { s } from '@/styles/theme';

type View =
  | { name: 'list' }
  | { name: 'create' }
  | { name: 'detail'; kb: KnowledgeBase }
  | { name: 'edit'; kb: KnowledgeBase; backTo: 'list' | 'detail' };

function App() {
  const [view, setView] = useState<View>({ name: 'list' });
  const [listVersion, setListVersion] = useState(0);

  function closeEditor() {
    if (view.name === 'edit' && view.backTo === 'detail') {
      setView({ name: 'detail', kb: view.kb });
      return;
    }
    setView({ name: 'list' });
  }

  function handleSaved(kb: KnowledgeBase) {
    if (view.name === 'edit' && view.backTo === 'list') {
      setListVersion((v) => v + 1);
      setView({ name: 'list' });
      return;
    }
    setView({ name: 'detail', kb });
  }

  return (
    <div style={s.page}>
      <header style={s.header}>
        <span style={{ fontSize: 22 }}>📚</span>
        <span style={s.headerTitle}>知识库管理</span>
        <span style={s.headerSub}>Agent Admin · RAG Console</span>
      </header>
      <div style={s.container}>
        {view.name === 'list' && (
          <KbList
            key={listVersion}
            onOpen={(kb) => setView({ name: 'detail', kb })}
            onCreate={() => setView({ name: 'create' })}
            onEdit={(kb) => setView({ name: 'edit', kb, backTo: 'list' })}
          />
        )}
        {view.name === 'detail' && (
          <KbDetail
            kb={view.kb}
            onBack={() => setView({ name: 'list' })}
            onEdit={(kb) => setView({ name: 'edit', kb, backTo: 'detail' })}
          />
        )}
        {view.name === 'create' && (
          <KbEditorPage
            mode="create"
            onCancel={() => setView({ name: 'list' })}
            onSaved={handleSaved}
          />
        )}
        {view.name === 'edit' && (
          <KbEditorPage
            mode="edit"
            kb={view.kb}
            onCancel={closeEditor}
            onSaved={handleSaved}
          />
        )}
      </div>
    </div>
  );
}

export default App;

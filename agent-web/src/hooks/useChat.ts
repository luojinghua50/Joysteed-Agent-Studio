import { useState, useCallback, useRef, useEffect } from 'react';
import { ChatMessage, sendMessage, getChatHistory } from '@/services/api';

export function useChat(sessionId: string, customerId: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentAgent, setCurrentAgent] = useState<string>('');
  const abortRef = useRef(false);

  useEffect(() => {
    if (!sessionId) return;
    getChatHistory(sessionId, customerId).then((history) => {
      if (history.length > 0) {
        setMessages(history);
      }
    });
  }, [sessionId, customerId]);

  const send = useCallback(async (content: string) => {
    if (!content.trim() || isLoading || !sessionId) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    abortRef.current = false;

    let assistantContent = '';
    const assistantMessage: ChatMessage = {
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
    };
    setMessages(prev => [...prev, assistantMessage]);

    await sendMessage(
      sessionId,
      content,
      customerId,
      (token) => {
        if (abortRef.current) return;
        assistantContent += token;
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { ...assistantMessage, content: assistantContent };
          return updated;
        });
      },
      (agent) => {
        setCurrentAgent(agent);
        setIsLoading(false);
      },
      (error) => {
        assistantContent = `抱歉，发生了错误：${error}`;
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1] = { ...assistantMessage, content: assistantContent };
          return updated;
        });
        setIsLoading(false);
      },
    );
  }, [sessionId, customerId, isLoading]);

  const stop = useCallback(() => {
    abortRef.current = true;
    setIsLoading(false);
    // 标记最后一条 assistant 消息为已停止：保留已生成内容，UI 据此显示
    // 「已停止生成」而不是渲染成空气泡。
    setMessages(prev => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      if (last.role !== 'assistant') return prev;
      const updated = [...prev];
      updated[updated.length - 1] = { ...last, stopped: true };
      return updated;
    });
  }, []);

  const reset = useCallback(() => {
    setMessages([]);
    setCurrentAgent('');
    abortRef.current = true;
    setIsLoading(false);
  }, []);

  return { messages, isLoading, currentAgent, send, stop, reset };
}

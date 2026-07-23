import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3005,
    proxy: {
      // 工单 REST → ticket-mcp:8003
      '/ticket': {
        target: 'http://localhost:8003',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/ticket/, ''),
      },
      // 坐席接管 + 外呼 → agent-core:8000
      '/desk': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      // 外呼 REST → call-mcp:8005 (P2)
      '/call': {
        target: 'http://localhost:8005',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/call/, ''),
      },
    },
  },
});

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  base: process.env.VITE_ASSET_BASE ?? '/dashboard-ui/',
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/health': 'http://localhost:8000',
      '/dashboard': 'http://localhost:8000',
      '/auto': 'http://localhost:8000',
      '/alerts': 'http://localhost:8000',
      '/backtest': 'http://localhost:8000',
    },
  },
});

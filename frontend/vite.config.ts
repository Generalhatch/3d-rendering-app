import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
    // HMR file watching: prefer fsevents (free, native), but fall back to
    // polling when Vite is launched from a Cursor sandbox or other env that
    // intercepts FS events. Polling adds ~negligible CPU but is reliable.
    watch: {
      usePolling: true,
      interval: 300,
    },
  },
});

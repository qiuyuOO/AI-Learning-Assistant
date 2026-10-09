import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// ★ /api 代理到本地 FastAPI 后端（server.py，默认 8000）。
//   用代理而非直连，前端就是同源请求，省去 CORS 折腾；生产可用 WEB_API 覆盖。
const BACKEND = process.env.WEB_API || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
    },
  },
});

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../src/chromaw/static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      // Backend must be started on a fixed port for the dev proxy to work,
      // e.g. `uv run chromaw . --port 8000`.
      // Override with VITE_API_TARGET if the backend runs elsewhere.
      "/api": {
        target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});

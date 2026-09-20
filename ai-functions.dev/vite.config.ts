import { cloudflare } from "@cloudflare/vite-plugin";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig(({ command }) => ({
  plugins: [
    react(),
    cloudflare({
      configPath:
        command === "serve" ? "./wrangler.local.jsonc" : "./wrangler.jsonc",
    }),
  ],
}));

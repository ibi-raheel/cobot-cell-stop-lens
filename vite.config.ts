import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// Served at ibiraheel.com/demos/cobot-cell-stop-lens/. Data fetches use import.meta.env.BASE_URL.
export default defineConfig({ base: "/demos/cobot-cell-stop-lens/", plugins: [react()] });

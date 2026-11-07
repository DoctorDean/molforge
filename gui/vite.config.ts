import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build lands inside the Python package, so a wheel ships the editor
// and a user never needs Node. `base: "./"` keeps asset URLs relative,
// which is what lets the same bundle work when FastAPI mounts it at /.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../src/molforge/gui/static",
    emptyOutDir: true,
  },
  server: {
    // `npm run dev` serves the UI; the API still comes from the Python
    // server, so there is one source of truth for the block catalog.
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});

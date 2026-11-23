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
    // 3Dmol is ~590 kB and already split into its own chunk by the
    // dynamic import in StructureViewer, so it is fetched the first time
    // someone opens a result and never on load. A molecular viewer is
    // simply this big; warning about it every build trains us to ignore
    // the warning that matters.
    chunkSizeWarningLimit: 700,
  },
  server: {
    // `npm run dev` serves the UI; the API still comes from the Python
    // server, so there is one source of truth for the block catalog.
    proxy: { "/api": "http://127.0.0.1:8765" },
  },
});

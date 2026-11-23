import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end tests against a real `molforge-gui`.
 *
 * The two worst bugs the editor has had — a blank 3D canvas, and a
 * fitView that opened every graph at 4x — were invisible to the Python
 * suite and to `tsc`. Both were found by looking at the thing. These
 * tests are that look, automated: they drive a real browser against a
 * real server, so a regression in the viewer fails CI instead of
 * shipping.
 *
 * The server is the actual CLI, not a mock. A fake API would have agreed
 * with whatever the frontend believed, which is how a typed client drifts
 * from its backend without either side noticing.
 */
export default defineConfig({
  testDir: "./tests",
  // The structure viewer waits on WebGL and a fetch; the default 5s
  // assertion timeout is tight for software rendering in CI.
  expect: { timeout: 15_000 },
  timeout: 60_000,
  fullyParallel: false, // the server allows one run at a time, by design
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["github"], ["list"]] : [["list"]],
  use: {
    baseURL: process.env.MOLFORGE_GUI_URL ?? "http://127.0.0.1:8777",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: process.env.MOLFORGE_GUI_URL
    ? undefined
    : {
        // `python -m` rather than the console script: it works from a
        // source checkout without the package being installed by name.
        command:
          `${process.env.MOLFORGE_PYTHON ?? "python"} -m molforge.gui.server ` +
          "--no-browser --port 8777",
        url: "http://127.0.0.1:8777/api/blocks",
        reuseExistingServer: !process.env.CI,
        timeout: 60_000,
        cwd: "..",
      },
});

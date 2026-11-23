/**
 * The editor, driven the way a person drives it.
 *
 * These assert on behaviour that only exists once HTML, CSS, WebGL and
 * the API are all in the same place: that a wire is refused, that the
 * canvas actually drew something, that a finished block shows its
 * result. None of it is reachable from the Python suite or from `tsc`,
 * and all of it has broken at least once.
 *
 * Only the blocks that run anywhere are used — `molforge.io.fetch` needs
 * no engine installed. Asserting on Boltz would make the suite a report
 * on what the runner happens to have.
 */

import { expect, test } from "@playwright/test";
import type { Locator, Page } from "@playwright/test";

const FETCH_BLOCK = "molforge.io.fetch";

async function addBlock(page: Page, name: string): Promise<void> {
  await page.getByRole("button", { name }).first().click();
}

/** Add a fetch block and give it a PDB ID, leaving it ready to run. */
async function addFetch(page: Page, pdbId = "1CRN"): Promise<void> {
  await addBlock(page, FETCH_BLOCK);
  await page.getByPlaceholder("required").first().fill(pdbId);
}

/** React Flow's wrapper — what you count, drag from, and click. */
function nodes(page: Page): Locator {
  return page.locator(".react-flow__node");
}

/** The block itself, inside the wrapper. State lives here, not on the
 *  wrapper React Flow owns. */
function block(page: Page, index = 0): Locator {
  return nodes(page).nth(index).locator(".block-node");
}

async function dragHandle(page: Page, from: Locator, to: Locator): Promise<void> {
  const a = await from.boundingBox();
  const b = await to.boundingBox();
  if (!a || !b) throw new Error("a handle was not on screen");
  await page.mouse.move(a.x + a.width / 2, a.y + a.height / 2);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 12 });
  await page.mouse.up();
}

test.beforeEach(async ({ page }) => {
  const failures: string[] = [];
  page.on("pageerror", (e) => failures.push(String(e)));
  await page.goto("/");
  await expect(page.locator(".palette")).toBeVisible();
  // Every test fails on an uncaught exception, not just the ones that
  // happen to assert on the symptom.
  test.info().annotations.push({ type: "pageerrors", description: () => failures.join("\n") });
});

test.describe("the palette", () => {
  test("lists blocks grouped by what they do", async ({ page }) => {
    await expect(page.getByRole("heading", { name: "Sources" })).toBeVisible();
    await expect(page.getByRole("button", { name: FETCH_BLOCK }).first()).toBeVisible();
  });

  test("marks engines that are not installed", async ({ page }) => {
    // Nothing is guaranteed absent on a given machine, but something is:
    // a developer box with every engine installed is not a thing.
    const missing = page.locator('.block-chip[data-unavailable]');
    if ((await missing.count()) > 0) {
      await expect(missing.first().locator(".pill")).toHaveText("missing");
    }
  });

  test("filters", async ({ page }) => {
    await page.getByLabel("Filter blocks").fill("fetch");
    await expect(page.locator(".block-chip")).toHaveCount(1);
  });
});

test.describe("wiring", () => {
  test("refuses a wire between mismatched port types", async ({ page }) => {
    // fetch produces a structure; MMseqs2 takes a sequence.
    await addFetch(page);
    await addBlock(page, "MMseqs2");
    await expect(nodes(page)).toHaveCount(2);

    await dragHandle(
      page,
      nodes(page).nth(0).locator(".react-flow__handle-right"),
      nodes(page).nth(1).locator(".react-flow__handle-left"),
    );
    await expect(page.locator(".react-flow__edge")).toHaveCount(0);
  });

  test("connects matching types and says the port is wired", async ({ page }) => {
    await addFetch(page);
    await addBlock(page, "fpocket");

    await dragHandle(
      page,
      nodes(page).nth(0).locator(".react-flow__handle-right"),
      nodes(page).nth(1).locator(".react-flow__handle-left"),
    );
    await expect(page.locator(".react-flow__edge")).toHaveCount(1);
    await expect(nodes(page).nth(1)).toContainText("wired");
  });
});

test.describe("checking", () => {
  test("blames the node that is wrong", async ({ page }) => {
    await addBlock(page, FETCH_BLOCK); // no pdb_id
    await page.getByRole("button", { name: "Check" }).click();
    await expect(page.locator(".statusbar")).toHaveAttribute("data-kind", "error");
    await expect(block(page)).toHaveAttribute("data-problem", "true");
  });

  test("passes a complete graph", async ({ page }) => {
    await addFetch(page);
    await page.getByRole("button", { name: "Check" }).click();
    await expect(page.locator(".statusbar")).toHaveAttribute("data-kind", "ok");
  });

  test("reports two unconnected chains", async ({ page }) => {
    await addFetch(page);
    await addFetch(page);
    await page.getByRole("button", { name: "Check" }).click();
    await expect(page.locator(".statusbar .message")).toContainText("chains");
  });
});

test.describe("running", () => {
  test("runs a pipeline and reports it on the block", async ({ page }) => {
    await addFetch(page, "1CRN");
    await page.getByRole("button", { name: "Run", exact: true }).click();

    await expect(block(page)).toHaveAttribute("data-state", "succeeded");
    await expect(block(page)).toContainText("residues");
    await expect(page.locator(".statusbar .progress")).toHaveText("1/1 steps");
  });

  test("shows the result in 3D when the block is clicked", async ({ page }) => {
    await addFetch(page, "1CRN");
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(block(page)).toHaveAttribute("data-state", "succeeded");

    await nodes(page).first().click();
    await expect(page.locator(".results")).toBeVisible();
    await expect(page.locator(".facts dd").first()).toHaveText("46"); // 1CRN residues

    // The canvas existing is not the test — it existed throughout the
    // bug where the viewer rendered nothing. What matters is that
    // something was drawn into it.
    await expect(page.locator(".viewer-canvas canvas")).toBeVisible();
    await expect
      .poll(() => canvasDrewSomething(page), { timeout: 25_000 })
      .toBe(true);
  });

  test("serves the structure it drew", async ({ page, request }) => {
    await addFetch(page, "1CRN");
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(block(page)).toHaveAttribute("data-state", "succeeded");

    const runs = await (await request.get("/api/runs")).json();
    const id = runs.runs.at(-1).id;
    const pdb = await request.get(`/api/runs/${id}/steps/1/structure/structure`);
    expect(pdb.ok()).toBeTruthy();
    expect(await pdb.text()).toContain("ATOM");
  });
});

test.describe("saving and opening", () => {
  test("a saved pipeline opens back as the same graph", async ({ page }) => {
    await addFetch(page, "4HHB");

    const download = page.waitForEvent("download");
    await page.getByRole("button", { name: "Save" }).click();
    const saved = await download;
    const path = await saved.path();
    expect(saved.suggestedFilename()).toMatch(/pipeline\.(yaml|json)/);

    await page.getByRole("button", { name: "Clear" }).click();
    await expect(nodes(page)).toHaveCount(0);

    await page.locator('input[type="file"]').setInputFiles(path);
    await expect(nodes(page)).toHaveCount(1);
    await expect(nodes(page).first()).toContainText("4HHB");
  });
});

/**
 * Whether the WebGL canvas actually drew a structure.
 *
 * Reading the buffer back from the page does not work: the context is
 * created without `preserveDrawingBuffer`, and asking for it again
 * returns the existing context with the original attributes, so
 * `readPixels` after a presented frame sees black. Playwright screenshots
 * the *composited* canvas, which is what is really on screen.
 *
 * The signal is compressed size. At this canvas size a solid background
 * PNG is ~1.2 kB; the rendered structure measured 36 kB over ten thousand
 * distinct colours. Anything above a few kB is geometry, and the gap is
 * wide enough that it does not need to be precise.
 */
const BLANK_CANVAS_BYTES = 4_000;

async function canvasDrewSomething(page: Page): Promise<boolean> {
  const shot = await page.locator(".viewer-canvas canvas").screenshot();
  return shot.byteLength > BLANK_CANVAS_BYTES;
}

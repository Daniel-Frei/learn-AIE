import { expect, test } from "@playwright/test";

const route = "/learn/stanford-cme296/diffusiongemma/presentation";

test.describe.configure({ mode: "serial" });

test("presents the DiffusionGemma paper as a navigable 26-slide deck", async ({
  page,
}) => {
  await page.goto("/learn/stanford-cme296", { waitUntil: "domcontentloaded" });
  await expect(
    page.getByText(/1 learning page \/ 0 quiz-linked min/i),
  ).toBeVisible();
  const presentationLink = page.getByRole("link", {
    name: /DiffusionGemma Talk Deck/i,
  });
  await expect(presentationLink).toHaveAttribute("href", route);
  await presentationLink.click();
  await expect(page.getByTestId("diffusiongemma-presentation")).toHaveAttribute(
    "data-ready",
    "true",
    { timeout: 15_000 },
  );

  await expect(
    page.getByRole("heading", { level: 1, name: /DiffusionGemma/i }),
  ).toBeVisible();
  await expect(page.getByText(/It said YES/i)).toBeVisible();
  await expect(page.getByText(/Then it changed its mind/i)).toBeVisible();
  await expect(
    page.getByText(/How can rewriting a draft deliver/i),
  ).toBeVisible();
  await expect(page.getByTestId("diffusiongemma-slide")).toHaveCount(26);
  await expect(
    page.getByText(/Choose a route; learn the local moves/i),
  ).toBeVisible();
  await expect(
    page.getByText(/Parallel inside a block; autoregressive across blocks/i),
  ).toBeVisible();
  await expect(
    page.getByText(/A new speed frontier—with an accuracy bill/i),
  ).toBeVisible();

  await page.keyboard.press("Home");
  await page.keyboard.press("ArrowRight");
  await expect
    .poll(() => page.evaluate(() => window.scrollY))
    .toBeGreaterThan(500);
  await page.keyboard.press("Home");
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBeLessThan(75);
});

test("supports the denoising and entropy-sampling teaching interactions", async ({
  page,
}) => {
  await page.goto(route, { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("diffusiongemma-presentation")).toHaveAttribute(
    "data-ready",
    "true",
    { timeout: 15_000 },
  );

  const canvas = page.getByTestId("denoising-canvas");
  await canvas.scrollIntoViewIfNeeded();
  await expect(canvas.getByText(/Step 0 \/ 4/i)).toBeVisible();
  await canvas.getByRole("button", { name: "Next denoising step" }).click();
  await expect(canvas.getByText(/Step 1 \/ 4/i)).toBeVisible();
  await canvas.getByRole("button", { name: "Show denoising step 4" }).click();
  await expect(canvas.getByText(/mean confidence 97%/i)).toBeVisible();

  const sampler = page.getByTestId("entropy-sampler");
  await sampler.scrollIntoViewIfNeeded();
  await sampler.locator('input[type="range"]').fill("20");
  await expect(sampler.locator('[data-committed="true"]')).toHaveCount(3);
  await expect(sampler.locator('[data-committed="false"]')).toHaveCount(3);
});

test("enlarges paper figures and remains horizontally safe on mobile", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(route, { waitUntil: "domcontentloaded" });
  await expect(page.getByTestId("diffusiongemma-presentation")).toHaveAttribute(
    "data-ready",
    "true",
    { timeout: 15_000 },
  );

  const figure = page.getByRole("button", {
    name: /Enlarge Paper Figure 3/i,
  });
  await figure.scrollIntoViewIfNeeded();
  await figure.click();
  await expect(
    page.getByTestId("diffusiongemma-figure-lightbox"),
  ).toBeVisible();
  await page.getByRole("button", { name: "Close enlarged figure" }).click();

  const overflow = await page.evaluate(
    () =>
      document.documentElement.scrollWidth -
      document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});

import { expect, test } from "@playwright/test";

const route = "/learn/stanford-cme296/diffusiongemma/presentation";

test.describe.configure({ mode: "serial" });

test("presents the DiffusionGemma paper as a navigable 45-slide deck", async ({
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
    page.getByRole("heading", {
      level: 1,
      name: /Nearly the same intelligence\. Seven times the speed/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByText(/They moved the speed–capability frontier/i),
  ).toBeVisible();
  await expect(
    page.getByText(/The usual escape routes only bend the chain/i),
  ).toHaveCount(0);
  await expect(page.getByText(/Today's route/i)).toBeVisible();
  await expect(page.locator("[data-section-divider]")).toHaveCount(6);
  await expect(
    page.getByText(/The shortest mental model: one Transformer, two modes/i),
  ).toBeVisible();
  await expect(
    page.getByText(/Five steps turn Gemma 4 into DiffusionGemma/i),
  ).toBeVisible();
  await expect(
    page.getByTestId("conversion-roadmap").getByRole("listitem"),
  ).toHaveCount(5);
  await expect(
    page.getByText(/Step 1 · Start from a model that already knows language/i),
  ).toBeVisible();
  await expect(
    page.getByText(/Step 2 · Change who each canvas token can see/i),
  ).toBeVisible();
  await expect(
    page.getByText(
      /Steps 3–4 · Learn the whole clean block—and reuse the last guess/i,
    ),
  ).toBeVisible();
  await expect(
    page.getByText(/At inference, the trained model alternates the two modes/i),
  ).toBeVisible();
  await expect(
    page.getByText(/Step 5 · Make a capable denoiser fast/i),
  ).toBeVisible();
  await expect(
    page.getByText(
      /The result: diffusion within blocks, autoregression between blocks/i,
    ),
  ).toBeVisible();
  await expect(page.getByText(/Meet the checkpoint/i)).toHaveCount(0);
  await expect(page.getByTestId("diffusiongemma-slide")).toHaveCount(45);
  await expect(page.getByText(/^\d+ min$/)).toHaveCount(0);
  await expect(
    page.getByText(/Choose a route; learn the local moves/i),
  ).toBeVisible();
  await expect(
    page.getByText(/A new speed frontier—with an accuracy bill/i),
  ).toBeVisible();

  await expect(
    page.getByRole("button", {
      name: /Enlarge Artificial Analysis · Output Speed/i,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: /Enlarge Artificial Analysis Intelligence Index/i,
    }),
  ).toBeVisible();
  await expect(
    page.locator('[data-section-divider="01"]').getByTestId("section-motif"),
  ).toHaveCount(0);
  await expect(
    page.locator('[data-section-divider="02"]').getByTestId("section-motif"),
  ).toHaveCount(0);
  await expect(page.getByTestId("section-motif")).toHaveCount(4);

  const agendaPositions = await page
    .getByTestId("agenda-item")
    .evaluateAll((items) =>
      items.slice(0, 6).map((item) => {
        const box = item.getBoundingClientRect();
        return { x: box.x, y: box.y };
      }),
    );
  expect(agendaPositions).toHaveLength(6);
  expect(new Set(agendaPositions.map(({ x }) => Math.round(x))).size).toBe(1);
  expect(agendaPositions.map(({ y }) => y)).toEqual(
    [...agendaPositions.map(({ y }) => y)].sort((a, b) => a - b),
  );

  const imagesOutsideZoomControls = await page
    .locator('[data-testid="diffusiongemma-slide"] img')
    .evaluateAll((images) =>
      images
        .filter((image) => !image.closest('button[aria-label^="Enlarge "]'))
        .map((image) => image.getAttribute("alt")),
    );
  expect(imagesOutsideZoomControls).toEqual([]);

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

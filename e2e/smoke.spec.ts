import { expect, type Page, test } from "@playwright/test";

async function openFilters(page: Page) {
  const panelHeading = page.getByText(
    /pick mode, question types, series, lectures, topics and question elo range/i,
  );
  const toggle = page.getByRole("button", {
    name: /choose filters|close selection/i,
  });

  await expect(toggle).toBeVisible();
  await expect(async () => {
    if (!(await panelHeading.isVisible())) {
      await toggle.click();
    }
    await expect(panelHeading).toBeVisible({ timeout: 1000 });
  }).toPass({ timeout: 10000 });
}

function trackHydrationErrors(page: Page) {
  const errors: string[] = [];
  const hydrationPattern = /hydration failed/i;

  page.on("console", (message) => {
    if (message.type() === "error" && hydrationPattern.test(message.text())) {
      errors.push(message.text());
    }
  });
  page.on("pageerror", (error) => {
    if (hydrationPattern.test(error.message)) {
      errors.push(error.message);
    }
  });

  return errors;
}

test("renders core quiz controls on the home page", async ({ page }) => {
  const hydrationErrors = trackHydrationErrors(page);

  await page.goto("/");

  await expect(
    page.getByRole("heading", { name: "Learning AI" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /close selection/i }),
  ).toBeVisible();
  await expect(
    page.getByText(
      /pick mode, question types, series, lectures, topics and question elo range/i,
    ),
  ).toBeVisible();
  await expect(
    page.getByRole("checkbox", { name: /multiple-select/i }),
  ).toBeChecked();
  await expect(
    page.getByRole("checkbox", { name: /assertion-reason/i }),
  ).toBeChecked();
  await expect(page.getByRole("spinbutton").nth(0)).toHaveValue("0");
  await expect(page.getByRole("spinbutton").nth(1)).toHaveValue("3000");
  await expect(
    page.getByRole("button", { name: /reset glicko rating/i }),
  ).toBeVisible();
  await expect(
    page.getByText(/no questions for this selection/i),
  ).toBeVisible();
  await expect(
    page.getByText(/adjust the filters above to see available questions/i),
  ).toBeVisible();
  expect(hydrationErrors, "home page should hydrate cleanly").toEqual([]);
});

test("updates filter checkboxes immediately after clicking", async ({
  page,
}) => {
  await page.goto("/");

  await openFilters(page);

  const seriesCheckbox = page.getByRole("checkbox", {
    name: "AIE Foundations Book",
  });
  const seriesLabel = page
    .locator("summary")
    .filter({ hasText: "AIE Foundations Book" })
    .getByText("AIE Foundations Book");
  await expect(seriesCheckbox).not.toBeChecked();
  await seriesLabel.click();
  await expect(seriesCheckbox).not.toBeChecked();
  await seriesCheckbox.click();
  await expect(seriesCheckbox).toBeChecked();
});

test("resets the participant Glicko rating from the filter panel", async ({
  page,
}) => {
  await page.goto("/");

  await openFilters(page);
  await page.getByRole("button", { name: /reset glicko rating/i }).click();

  await expect(page.getByRole("status")).toHaveText(/rating reset/i);
  await expect(
    page.getByText(/glicko rating:\s*1500\s*\+\/-\s*350/i),
  ).toBeVisible();
});

test("reveals question elo after answering and resets the timer for the next question", async ({
  page,
}) => {
  await page.goto("/");

  await openFilters(page);
  await expect(page.getByText(/question elo range:/i)).toBeVisible();
  await page.getByRole("button", { name: /select all topics/i }).click();
  await page.getByRole("button", { name: /apply selection/i }).click();

  const contextIcon = page.locator('[aria-label^="Question context:"]').first();
  await expect(contextIcon).toBeVisible();
  await expect(page.getByText(/^Context:/i)).toHaveCount(0);
  const contextText = (
    (await contextIcon.getAttribute("aria-label")) ?? ""
  ).replace(/^Question context:\s*/, "");
  const contextTooltip = page
    .locator('[role="tooltip"]')
    .filter({ hasText: contextText });
  await expect(contextTooltip).toBeHidden();
  await contextIcon.hover();
  await expect(contextTooltip).toBeVisible();

  await expect(page.getByText(/question elo:/i)).toHaveCount(0);
  await expect(page.getByTestId("question-rating-line")).toHaveCount(0);
  await expect(page.getByText(/difficulty:/i)).toHaveCount(0);
  await expect(page.getByText(/answered:/i)).toHaveCount(0);
  await expect(page.getByText(/correct:/i)).toHaveCount(0);
  await expect(page.getByTestId("quiz-accuracy")).toHaveCSS(
    "text-align",
    "right",
  );
  await expect(page.getByText(/time:\s*0:00\s*\/\s*3:00/i)).toBeVisible();

  await page.waitForTimeout(1200);
  await expect(page.getByText(/time:\s*0:0[1-9]\s*\/\s*3:00/i)).toBeVisible();

  const promptBeforeSubmit = await page
    .getByTestId("question-prompt")
    .boundingBox();
  expect(promptBeforeSubmit).not.toBeNull();

  await page.route("**/api/answers", async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    await route.continue();
  });
  const answerResponse = page.waitForResponse(
    (response) =>
      response.url().includes("/api/answers") &&
      response.request().method() === "POST",
  );

  await page.locator("section").nth(0).getByRole("checkbox").first().click();
  await page.getByRole("button", { name: /submit answer/i }).click();
  await expect(page.getByText(/question elo:/i)).toBeVisible({
    timeout: 500,
  });
  const promptAfterSubmit = await page
    .getByTestId("question-prompt")
    .boundingBox();
  expect(
    Math.abs((promptAfterSubmit?.y ?? 0) - (promptBeforeSubmit?.y ?? 0)),
  ).toBeLessThanOrEqual(1);
  const timeBox = await page.getByTestId("question-timer").boundingBox();
  const questionRatingBox = await page
    .getByTestId("question-rating-line")
    .boundingBox();
  expect(timeBox, "timer should be visible").not.toBeNull();
  expect(questionRatingBox, "question rating should be visible").not.toBeNull();
  expect(questionRatingBox!.y).toBeGreaterThan(timeBox!.y);
  await expect(page.getByTestId("question-rating-line")).toHaveText(
    /question elo:.*[+-]\d+/i,
  );
  await expect(page.getByTestId("user-rating-delta")).toHaveText(/[+-]\d+/);
  await expect(page.getByTestId("question-rating-delta")).toHaveText(/[+-]\d+/);
  await answerResponse;

  await page.getByRole("button", { name: /next question/i }).click();

  await expect(page.getByText(/question elo:/i)).toHaveCount(0);
  await expect(page.getByTestId("question-rating-line")).toHaveCount(0);
  await expect(page.getByTestId("user-rating-delta")).toHaveCount(0);
  await expect(page.getByTestId("question-rating-delta")).toHaveCount(0);
  await expect(page.getByText(/time:\s*0:00\s*\/\s*3:00/i)).toBeVisible();
});

test("lets users select answer text without toggling the answer", async ({
  page,
}) => {
  await page.goto("/");

  await openFilters(page);
  await page.getByRole("button", { name: /select all topics/i }).click();
  await page.getByRole("button", { name: /apply selection/i }).click();

  const firstOption = page
    .locator("section")
    .nth(0)
    .getByRole("checkbox")
    .first();
  const firstOptionText = page.getByTestId("answer-option-text").first();

  await expect(firstOption).toHaveAttribute("aria-checked", "false");
  await expect(firstOptionText).toHaveCSS("cursor", "pointer");

  const box = await firstOptionText.boundingBox();
  expect(box, "first answer option text should be visible").not.toBeNull();
  if (!box) return;

  await page.mouse.move(box.x + 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(
    box.x + Math.min(box.width - 2, 420),
    box.y + box.height / 2,
    { steps: 12 },
  );
  await page.mouse.up();

  const selectedText = await page.evaluate(
    () => window.getSelection()?.toString().trim() ?? "",
  );

  expect(selectedText.length).toBeGreaterThan(0);
  await expect(firstOption).toHaveAttribute("aria-checked", "false");

  await page.evaluate(() => window.getSelection()?.removeAllRanges());
  await firstOption.click();
  await expect(firstOption).toHaveAttribute("aria-checked", "true");
});

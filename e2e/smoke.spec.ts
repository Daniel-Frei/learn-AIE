import { expect, type Page, test } from "@playwright/test";

async function openFilters(page: Page) {
  const panelHeading = page.getByText(
    /pick mode, series, lectures, topics and question elo range/i,
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

test("renders core quiz controls on the home page", async ({ page }) => {
  await page.goto("/");

  await expect(
    page.getByRole("heading", { name: "Learning AI" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /close selection/i }),
  ).toBeVisible();
  await expect(
    page.getByText(
      /pick mode, series, lectures, topics and question elo range/i,
    ),
  ).toBeVisible();
  await expect(
    page.getByText(/no questions for this selection/i),
  ).toBeVisible();
  await expect(
    page.getByText(/adjust the filters above to see available questions/i),
  ).toBeVisible();
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

test("shows question elo and a per-question timer that resets for the next question", async ({
  page,
}) => {
  await page.goto("/");

  await openFilters(page);
  await expect(page.getByText(/question elo range:/i)).toBeVisible();
  await page.getByRole("button", { name: /select all topics/i }).click();
  await page.getByRole("button", { name: /apply selection/i }).click();

  await expect(page.getByText(/question elo:/i)).toBeVisible();
  await expect(page.getByText(/difficulty:/i)).toHaveCount(0);
  await expect(page.getByText(/time:\s*0:00\s*\/\s*3:00/i)).toBeVisible();

  await page.waitForTimeout(1200);
  await expect(page.getByText(/time:\s*0:0[1-9]\s*\/\s*3:00/i)).toBeVisible();

  await page.locator("section").nth(0).getByRole("button").first().click();
  await page.getByRole("button", { name: /submit answer/i }).click();
  await page.getByRole("button", { name: /next question/i }).click();

  await expect(page.getByText(/question elo:/i)).toBeVisible();
  await expect(page.getByText(/time:\s*0:00\s*\/\s*3:00/i)).toBeVisible();
});

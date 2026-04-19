import { expect, test } from "@playwright/test";

test("renders core quiz controls on the home page", async ({ page }) => {
  await page.goto("/");

  await expect(
    page.getByRole("button", { name: /choose filters/i }),
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

  await page.getByRole("button", { name: /choose filters/i }).click();

  const seriesCheckbox = page.getByRole("checkbox", {
    name: "AIE Foundations Book",
  });
  const seriesToggle = page
    .locator("summary button")
    .filter({ hasText: "AIE Foundations Book" });
  await expect(seriesCheckbox).not.toBeChecked();
  await seriesToggle.click();
  await expect(seriesCheckbox).toBeChecked();
});

test("shows question elo and a per-question timer that resets for the next question", async ({
  page,
}) => {
  await page.goto("/");

  await page.getByRole("button", { name: /choose filters/i }).click();
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

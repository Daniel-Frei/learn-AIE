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

  await page
    .getByText(/3 lectures\/chapters/i)
    .first()
    .click();

  const lectureCheckbox = page.getByRole("checkbox", {
    name: "Chapter 1 only",
  });
  const lectureToggle = page.getByRole("button", { name: /chapter 1 only/i });
  await expect(lectureCheckbox).not.toBeChecked();
  await lectureToggle.click();
  await expect(lectureCheckbox).toBeChecked();
});

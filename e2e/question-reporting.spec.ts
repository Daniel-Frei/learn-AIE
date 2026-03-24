import { expect, test } from "@playwright/test";

test("reports a question locally and exports the report file", async ({
  page,
}) => {
  await page.goto("/");

  await page.getByRole("button", { name: /choose filters/i }).click();
  await page.getByRole("button", { name: /select all topics/i }).click();
  await page.getByRole("button", { name: /apply selection/i }).click();

  await expect(page.getByText(/question 1 of/i)).toBeVisible();

  await page.getByRole("button", { name: /report question/i }).click();
  await page.getByRole("button", { name: /submit report/i }).click();

  await expect(
    page.getByRole("alert").filter({ hasText: /enter a comment/i }),
  ).toBeVisible();

  await page
    .getByRole("textbox", { name: /comment/i })
    .fill("Prompt is too vague and needs clarification.");
  await page.getByRole("button", { name: /submit report/i }).click();

  await expect(page.getByRole("status")).toHaveText(/report saved locally/i);
  await expect(page.getByText(/reports for this question:/i)).toHaveCount(0);
  await expect(page.getByText(/reports saved locally:/i)).toHaveCount(0);

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: /export reports/i }).click();
  const download = await downloadPromise;

  expect(download.suggestedFilename()).toBe("quiz-question-reports.json");
});

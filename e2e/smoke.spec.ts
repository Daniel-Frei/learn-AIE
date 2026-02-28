import { expect, test } from "@playwright/test";

test("renders core quiz controls on the home page", async ({ page }) => {
  await page.goto("/");

  await expect(
    page.getByRole("button", { name: /choose sources & range/i }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: /submit answer/i }),
  ).toBeVisible();
});

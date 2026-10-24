import { expect, test } from "@playwright/test";
import { createDefaultRatingState } from "../lib/ratingEngine";
import type { SubmitQuestionReportRequest } from "../lib/quizSync";

for (const source of [
  { id: "cme295-2026-lect2", context: /CME295 2026.*Lecture 2/i },
  { id: "cme295-lect4", context: /LLM training lecture about pretraining/i },
]) {
  test(
    "shared item keeps its rating ID and active source: " + source.id,
    async ({ page }, testInfo) => {
      const oldId = "cme295-lect4-q104";
      const ratingState = createDefaultRatingState();
      ratingState.questions[oldId] = {
        rating: 2200,
        rd: 80,
        sigma: 0.06,
        lastUpdatedAt: Date.now(),
        gamesPlayed: 12,
        legacyCorrect: 7,
        legacyWrong: 5,
      };
      const runtimeErrors: string[] = [];
      const failedRequests: string[] = [];
      page.on("pageerror", (error) => runtimeErrors.push(error.message));
      page.on("console", (message) => {
        if (message.type() === "error") runtimeErrors.push(message.text());
      });
      page.on("requestfailed", (request) => failedRequests.push(request.url()));
      await page.route("**/api/quiz-state?*", async (route) => {
        await route.fulfill({
          json: {
            participantId: new URL(route.request().url()).searchParams.get(
              "participantId",
            ),
            ratingState,
            reportSummary: { totalReportCount: 0, countsByQuestion: {} },
            legacyMigrationCompleted: true,
          },
        });
      });
      let report: SubmitQuestionReportRequest | undefined;
      await page.route("**/api/question-reports", async (route) => {
        report = route.request().postDataJSON() as SubmitQuestionReportRequest;
        await route.fulfill({
          json: { totalReportCount: 1, questionReportCount: 1 },
        });
      });

      const stateResponse = page.waitForResponse("**/api/quiz-state?*");
      await page.goto("/?source=" + source.id);
      await stateResponse;
      // Snapshot before the first interaction records the filter state used below.
      await testInfo.attach("initial-source-state", {
        body: await page.locator("body").ariaSnapshot(),
        contentType: "text/plain",
      });
      const filterToggle = page.getByRole("button", {
        name: /choose filters|close selection/i,
      });
      const min = page.getByRole("spinbutton").nth(0);
      if (!(await min.isVisible())) await filterToggle.click();
      await min.fill("2100");
      await page.getByRole("spinbutton").nth(1).fill("2300");
      await page.getByRole("button", { name: /apply selection/i }).click();
      // Exactly one item proves that selection read the old-ID 2200 rating.
      await expect(page.getByText(/question 1 of 1\b/i)).toBeVisible();
      const context = page.locator('[aria-label^="Question context:"]');
      await expect(context).toHaveAttribute("aria-label", source.context);
      await context.hover();
      await expect(page.getByRole("tooltip")).toBeVisible();
      await testInfo.attach("shared-item-context", {
        body: await page.screenshot(),
        contentType: "image/png",
      });
      await page.getByRole("button", { name: /report question/i }).click();
      await page
        .getByRole("textbox", { name: /comment/i })
        .fill("Check shared question source context.");
      await page.getByRole("button", { name: /submit report/i }).click();
      await expect(page.getByRole("status")).toHaveText(/report submitted/i);
      expect(report?.draft.questionId).toBe(oldId);
      expect(report?.draft.snapshot.sourceId).toBe(source.id);
      expect(report?.draft.snapshot.seriesId).toBe(
        source.id.includes("2026") ? "stanford-cme295-2026" : "stanford-cme295",
      );
      expect(runtimeErrors).toEqual([]);
      expect(failedRequests).toEqual([]);
    },
  );
}

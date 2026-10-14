import { expect, type Page, test } from "@playwright/test";
import { QUESTION_SOURCES } from "../lib/quiz";
import { createDefaultRatingState } from "../lib/ratingEngine";

async function openOnlyQuestion(
  page: Page,
  sourceId: string,
  questionId: string,
) {
  const source = QUESTION_SOURCES.find((item) => item.id === sourceId)!;
  const ratingState = createDefaultRatingState();
  for (const question of source.questions) {
    ratingState.questions[question.id] = {
      ...ratingState.user,
      rating: question.id === questionId ? 1500 : 500,
      legacyCorrect: 0,
      legacyWrong: 0,
    };
  }
  await page.route("**/api/quiz-state?*", async (route) => {
    await route.fulfill({
      json: {
        participantId: "reported-question-rendering-test",
        ratingState,
        reportSummary: { totalReportCount: 0, countsByQuestion: {} },
        legacyMigrationCompleted: true,
      },
    });
  });
  await page.goto(`/?source=${sourceId}`);
  await page.getByRole("button", { name: /choose filters/i }).click();
  await page.getByRole("spinbutton").nth(0).fill("1400");
  await page.getByRole("spinbutton").nth(1).fill("1600");
  await page.getByRole("button", { name: /apply selection/i }).click();
  await expect(
    page.getByText("Question 1 of 1", { exact: true }),
  ).toBeVisible();
}

test("shows every numeric logarithm answer at desktop and mobile widths", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await openOnlyQuestion(
    page,
    "math-logs-exp-roots",
    "math-logs-exp-roots-q67",
  );
  await expect(page.getByTestId("question-prompt")).toContainText("81");
  const answers = page.getByTestId("answer-option-text");
  await expect(answers).toHaveCount(4);
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 844 });
    for (let index = 0; index < 4; index += 1) {
      await expect(answers.nth(index)).toBeVisible();
      await expect(answers.nth(index).locator(".katex-html")).toHaveText(
        /^(3|4|27|81)$/,
      );
      await expect(answers.nth(index).locator("ol, .katex-error")).toHaveCount(
        0,
      );
    }
  }
  await page.screenshot({
    path: test.info().outputPath("numeric-answers-mobile.png"),
  });
  expect(errors).toEqual([]);
});

for (const number of [57, 69]) {
  test(`renders reported policy math in question ${number}`, async ({
    page,
  }) => {
    await openOnlyQuestion(page, "cs224r-lect5", `cs224r-lect5-q${number}`);
    await expect(page.locator(".katex-error")).toHaveCount(0);
    const math =
      number === 69
        ? page.getByTestId("question-prompt")
        : page.getByTestId("answer-option-text").first();
    await expect(math.locator(".katex").first()).toBeVisible();
    await expect(
      page.getByTestId("question-prompt").locator("table"),
    ).toHaveCount(number === 69 ? 1 : 0);
    await page.screenshot({
      path: test.info().outputPath(`policy-math-${number}.png`),
    });
  });
}

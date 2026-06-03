import { describe, expect, it } from "vitest";
import { QUESTION_SOURCES } from "@/lib/quiz";

const MIT15773_SOURCE_IDS = [
  "mit15773-l1",
  "mit15773-l2",
  "mit15773-l3",
  "mit15773-l4",
  "mit15773-l5",
] as const;

const APPROXIMATE_BUCKET_TOLERANCE_RATIO = 0.1;

function getDistribution(sourceId: (typeof MIT15773_SOURCE_IDS)[number]) {
  const source = QUESTION_SOURCES.find((entry) => entry.id === sourceId);
  expect(source).toBeDefined();

  const distribution = { 1: 0, 2: 0, 3: 0, 4: 0 } as Record<number, number>;
  for (const question of source!.questions) {
    const correctCount = question.options.filter(
      (option) => option.isCorrect,
    ).length;
    distribution[correctCount] += 1;
  }

  return { source: source!, distribution };
}

function getDistributionFromSource(source: (typeof QUESTION_SOURCES)[number]) {
  const distribution = { 1: 0, 2: 0, 3: 0, 4: 0 } as Record<number, number>;

  for (const question of source.questions) {
    const correctCount = question.options.filter(
      (option) => option.isCorrect,
    ).length;
    distribution[correctCount] += 1;
  }

  return { distribution, totalQuestions: source.questions.length };
}

describe("MIT 15.773 answer distributions", () => {
  it("keeps each Spring 2024 lecture file balanced across answer patterns", () => {
    for (const sourceId of MIT15773_SOURCE_IDS) {
      const { source, distribution } = getDistribution(sourceId);

      expect(source.questions).toHaveLength(40);
      expect(
        distribution,
        `${source.id} should have 10 questions each with 1, 2, 3, and 4 correct answers`,
      ).toEqual({
        1: 10,
        2: 10,
        3: 10,
        4: 10,
      });
    }
  });

  it("keeps all lib question banks outside lib/llm roughly balanced", () => {
    const failures: string[] = [];

    for (const source of QUESTION_SOURCES) {
      if (source.balance === false) continue;

      const { distribution, totalQuestions } =
        getDistributionFromSource(source);
      const idealBucketSize = totalQuestions / 4;
      const allowedDeviation = Math.max(
        1,
        Math.ceil(totalQuestions * APPROXIMATE_BUCKET_TOLERANCE_RATIO),
      );

      const outOfRangeBuckets = [1, 2, 3, 4].filter(
        (bucketSize) =>
          Math.abs(distribution[bucketSize] - idealBucketSize) >
          allowedDeviation,
      );

      if (outOfRangeBuckets.length > 0) {
        failures.push(
          `${source.id}: total=${totalQuestions}, distribution=${JSON.stringify(distribution)}, ideal=${idealBucketSize.toFixed(2)}, allowedDeviation=${allowedDeviation}`,
        );
      }
    }

    expect(
      failures,
      "Question-bank sources should stay roughly balanced across 1/2/3/4-correct patterns unless balance is false.",
    ).toEqual([]);
  });
});

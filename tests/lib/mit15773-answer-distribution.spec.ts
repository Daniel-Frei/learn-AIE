import { describe, expect, it } from "vitest";
import { QUESTION_SOURCES } from "@/lib/quiz";

const MIT15773_SOURCE_IDS = [
  "mit15773-l1",
  "mit15773-l2",
  "mit15773-l3",
  "mit15773-l4",
  "mit15773-l5",
] as const;

const CORRECT_ANSWER_BUCKETS = [1, 2, 3, 4] as const;
type CorrectAnswerBucket = (typeof CORRECT_ANSWER_BUCKETS)[number];
type CorrectAnswerDistribution = Record<CorrectAnswerBucket, number>;

const STRICT_BALANCE_LEGACY_EXCEPTIONS: Record<
  string,
  {
    totalQuestions: number;
    distribution: CorrectAnswerDistribution;
  }
> = {
  "chapter-2": {
    totalQuestions: 80,
    distribution: { 1: 12, 2: 28, 3: 28, 4: 12 },
  },
  "langchain-deepagents": {
    totalQuestions: 64,
    distribution: { 1: 9, 2: 15, 3: 23, 4: 17 },
  },
};

function getAllowedBucketDeviation(totalQuestions: number) {
  return Math.max(1, Math.min(5, Math.ceil(totalQuestions * 0.1)));
}

function getDistribution(sourceId: (typeof MIT15773_SOURCE_IDS)[number]) {
  const source = QUESTION_SOURCES.find((entry) => entry.id === sourceId);
  expect(source).toBeDefined();

  const distribution: CorrectAnswerDistribution = { 1: 0, 2: 0, 3: 0, 4: 0 };
  for (const question of source!.questions) {
    const correctCount = question.options.filter(
      (option) => option.isCorrect,
    ).length;
    expect(CORRECT_ANSWER_BUCKETS).toContain(correctCount);
    distribution[correctCount as CorrectAnswerBucket] += 1;
  }

  return { source: source!, distribution };
}

function getDistributionFromSource(source: (typeof QUESTION_SOURCES)[number]) {
  const distribution: CorrectAnswerDistribution = { 1: 0, 2: 0, 3: 0, 4: 0 };

  for (const question of source.questions) {
    const correctCount = question.options.filter(
      (option) => option.isCorrect,
    ).length;
    expect(CORRECT_ANSWER_BUCKETS).toContain(correctCount);
    distribution[correctCount as CorrectAnswerBucket] += 1;
  }

  return { distribution, totalQuestions: source.questions.length };
}

function matchesLegacyException(
  sourceId: string,
  totalQuestions: number,
  distribution: CorrectAnswerDistribution,
) {
  const exception = STRICT_BALANCE_LEGACY_EXCEPTIONS[sourceId];
  if (!exception || exception.totalQuestions !== totalQuestions) return false;

  return CORRECT_ANSWER_BUCKETS.every(
    (bucket) => exception.distribution[bucket] === distribution[bucket],
  );
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
      const allowedDeviation = getAllowedBucketDeviation(totalQuestions);

      const outOfRangeBuckets = CORRECT_ANSWER_BUCKETS.filter(
        (bucketSize) =>
          Math.abs(distribution[bucketSize] - idealBucketSize) >
          allowedDeviation,
      );

      if (outOfRangeBuckets.length > 0) {
        if (matchesLegacyException(source.id, totalQuestions, distribution)) {
          continue;
        }

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

  it("uses a balance tolerance whose relative size shrinks for larger sets", () => {
    expect(getAllowedBucketDeviation(11)).toBe(2);
    expect(getAllowedBucketDeviation(40)).toBe(4);
    expect(getAllowedBucketDeviation(85)).toBe(5);
    expect(getAllowedBucketDeviation(100)).toBe(5);
  });

  it("keeps strict-balance legacy exceptions exact and explicit", () => {
    expect(Object.keys(STRICT_BALANCE_LEGACY_EXCEPTIONS).sort()).toEqual([
      "chapter-2",
      "langchain-deepagents",
    ]);

    for (const sourceId of Object.keys(STRICT_BALANCE_LEGACY_EXCEPTIONS)) {
      const source = QUESTION_SOURCES.find((entry) => entry.id === sourceId);
      expect(source, `${sourceId} should remain registered`).toBeDefined();

      const { distribution, totalQuestions } = getDistributionFromSource(
        source!,
      );
      expect({ totalQuestions, distribution }).toEqual(
        STRICT_BALANCE_LEGACY_EXCEPTIONS[sourceId],
      );
    }
  });
});

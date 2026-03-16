import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { QUESTION_SOURCES } from "@/lib/quiz";

const MIT15773_SOURCE_IDS = [
  "mit15773-l1",
  "mit15773-l2",
  "mit15773-l3",
  "mit15773-l4",
  "mit15773-l5",
] as const;

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, "../..");
const libRoot = path.join(repoRoot, "lib");
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

function listQuestionFiles(dir: string): string[] {
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  return entries.flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (fullPath === path.join(libRoot, "llm")) return [];
      return listQuestionFiles(fullPath);
    }

    if (!fullPath.endsWith(".ts")) return [];

    const content = fs.readFileSync(fullPath, "utf8");
    if (!/export\s+const\s+\w+\s*:\s*Question\[\]/.test(content)) return [];

    return [fullPath];
  });
}

function getDistributionFromFile(filePath: string) {
  const content = fs.readFileSync(filePath, "utf8");
  const questionBlocks = [
    ...content.matchAll(/options:\s*\[(.*?)\]\s*,\s*explanation:/gs),
  ];
  const distribution = { 1: 0, 2: 0, 3: 0, 4: 0 } as Record<number, number>;

  for (const block of questionBlocks) {
    const correctCount = (block[1].match(/isCorrect:\s*true/g) || []).length;
    distribution[correctCount] += 1;
  }

  return { distribution, totalQuestions: questionBlocks.length };
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

    for (const filePath of listQuestionFiles(libRoot)) {
      const relativePath = path
        .relative(repoRoot, filePath)
        .replace(/\\/g, "/");
      const { distribution, totalQuestions } =
        getDistributionFromFile(filePath);
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
          `${relativePath}: total=${totalQuestions}, distribution=${JSON.stringify(distribution)}, ideal=${idealBucketSize.toFixed(2)}, allowedDeviation=${allowedDeviation}`,
        );
      }
    }

    expect(
      failures,
      "Question-bank files should stay roughly balanced across 1/2/3/4-correct patterns.",
    ).toEqual([]);
  });
});

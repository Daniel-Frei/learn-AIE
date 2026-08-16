import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  QUESTION_SOURCES,
  getQuestionType,
  stanfordCME296Lecture2ScoreMatchingQuestions,
  stanfordCME296Lecture3FlowMatchingQuestions,
  stanfordCME296Lecture6ModelTrainingQuestions,
  stanfordCME296Lecture8TextDiffusionQuestions,
  type Question,
} from "@/lib/quiz";

const testDir = path.dirname(fileURLToPath(import.meta.url));

const sets = [
  {
    sourceId: "cme296-lect2",
    filename: "lecture2_score_matching.ts",
    questions: stanfordCME296Lecture2ScoreMatchingQuestions,
    count: 12,
    difficulty: { easy: 4, medium: 4, hard: 4 },
    assertionReasonCount: 2,
    multipleSelectCounts: { 1: 2, 2: 3, 3: 3, 4: 2 },
  },
  {
    sourceId: "cme296-lect3",
    filename: "lecture3_flow_matching.ts",
    questions: stanfordCME296Lecture3FlowMatchingQuestions,
    count: 33,
    difficulty: { easy: 11, medium: 11, hard: 11 },
    assertionReasonCount: 5,
    multipleSelectCounts: { 1: 7, 2: 7, 3: 7, 4: 7 },
  },
  {
    sourceId: "cme296-lect6",
    filename: "lecture6_model_training.ts",
    questions: stanfordCME296Lecture6ModelTrainingQuestions,
    count: 16,
    difficulty: { easy: 5, medium: 6, hard: 5 },
    assertionReasonCount: 4,
    multipleSelectCounts: { 1: 2, 2: 4, 3: 3, 4: 3 },
  },
  {
    sourceId: "cme296-lect8",
    filename: "lecture8_text_diffusion.ts",
    questions: stanfordCME296Lecture8TextDiffusionQuestions,
    count: 12,
    difficulty: { easy: 4, medium: 4, hard: 4 },
    assertionReasonCount: 2,
    multipleSelectCounts: { 1: 2, 2: 3, 3: 3, 4: 2 },
  },
] as const;

function countDifficulty(questions: readonly Question[]) {
  const counts = { easy: 0, medium: 0, hard: 0 };
  for (const question of questions) counts[question.difficulty] += 1;
  return counts;
}

function countMultipleSelectAnswers(questions: readonly Question[]) {
  const counts = { 1: 0, 2: 0, 3: 0, 4: 0 };
  for (const question of questions) {
    if (getQuestionType(question) === "assertion-reason") continue;
    const correctCount = question.options.filter(({ isCorrect }) => isCorrect)
      .length as 1 | 2 | 3 | 4;
    counts[correctCount] += 1;
  }
  return counts;
}

describe("DiffusionGemma-focused CME296 question sets", () => {
  it("registers only the study-guide lectures in course order", () => {
    const cme296Sources = QUESTION_SOURCES.filter(
      ({ seriesId }) => seriesId === "stanford-cme296",
    );

    expect(cme296Sources.map(({ id }) => id)).toEqual([
      "cme296-lect1",
      "cme296-lect2",
      "cme296-lect3",
      "cme296-lect6",
      "cme296-lect8",
      "cme296-diffusiongemma",
    ]);
    expect(cme296Sources.map(({ questions }) => questions.length)).toEqual([
      25, 12, 33, 16, 12, 60,
    ]);
  });

  for (const set of sets) {
    it(`keeps ${set.sourceId} proportional, balanced, and structurally valid`, () => {
      const source = QUESTION_SOURCES.find(({ id }) => id === set.sourceId);
      expect(source).toMatchObject({
        seriesId: "stanford-cme296",
        questions: set.questions,
      });
      expect(set.questions).toHaveLength(set.count);
      expect(set.questions.map(({ id }) => id)).toEqual(
        Array.from(
          { length: set.count },
          (_, index) =>
            `${set.sourceId}-q${String(index + 1).padStart(2, "0")}`,
        ),
      );
      expect(countDifficulty(set.questions)).toEqual(set.difficulty);
      expect(countMultipleSelectAnswers(set.questions)).toEqual(
        set.multipleSelectCounts,
      );
      expect(
        set.questions.filter(
          (question) => getQuestionType(question) === "assertion-reason",
        ),
      ).toHaveLength(set.assertionReasonCount);

      for (const question of set.questions) {
        const assertionReason =
          getQuestionType(question) === "assertion-reason";
        expect(question.options).toHaveLength(assertionReason ? 5 : 4);
        expect(question.options.some(({ isCorrect }) => isCorrect)).toBe(true);
        if (assertionReason) {
          expect(
            question.options.filter(({ isCorrect }) => isCorrect),
          ).toHaveLength(1);
        }
        const sentenceCount =
          question.explanation.match(/[.!?](?:\s|$)/g)?.length ?? 0;
        expect(question.explanation.length).toBeGreaterThanOrEqual(200);
        expect(sentenceCount).toBeGreaterThanOrEqual(2);
      }
    });

    it(`hardcodes every ${set.sourceId} ID at its helper call site`, () => {
      const filePath = path.resolve(
        testDir,
        "../../lib/lectures/Stanford CME296 Diffusion & Large Vision Models",
        set.filename,
      );
      const fileContents = fs.readFileSync(filePath, "utf8");
      const questionArraySource = fileContents.slice(
        fileContents.indexOf("export const stanfordCME296Lecture"),
      );
      const helperCalls = [
        ...questionArraySource.matchAll(/make(?:AssertionReason)?Question\(/g),
      ];
      const hardcodedIds = [
        ...questionArraySource.matchAll(
          /make(?:AssertionReason)?Question\(\s*\n\s*"([^"]+)"/g,
        ),
      ].map((match) => match[1]);

      expect(hardcodedIds).toHaveLength(helperCalls.length);
      expect(hardcodedIds).toEqual(set.questions.map(({ id }) => id));
    });
  }
});

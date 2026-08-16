import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  QUESTION_SOURCES,
  diffusionGemmaTechnicalReportQuestions,
  getQuestionType,
} from "@/lib/quiz";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const questionFilePath = path.resolve(
  testDir,
  "../../lib/lectures/Stanford CME296 Diffusion & Large Vision Models/DiffusionGemma Technical Report.ts",
);

describe("DiffusionGemma technical report question set", () => {
  it("registers all 60 paper questions as a selectable CME296 source", () => {
    const source = QUESTION_SOURCES.find(
      ({ id }) => id === "cme296-diffusiongemma",
    );

    expect(source).toMatchObject({
      label: "DiffusionGemma Paper",
      title: "DiffusionGemma Technical Report",
      seriesId: "stanford-cme296",
      seriesLabel: "Stanford CME296 Diffusion & Large Vision Models",
      topic: "NLP",
      questions: diffusionGemmaTechnicalReportQuestions,
    });
    expect(diffusionGemmaTechnicalReportQuestions).toHaveLength(60);
    expect(diffusionGemmaTechnicalReportQuestions.map(({ id }) => id)).toEqual(
      Array.from(
        { length: 60 },
        (_, index) =>
          "cme296-diffusiongemma-q" + String(index + 1).padStart(2, "0"),
      ),
    );
  });

  it("balances difficulty, question types, and correct-answer counts", () => {
    const difficultyCounts = { easy: 0, medium: 0, hard: 0 };
    const correctAnswerCounts = { 1: 0, 2: 0, 3: 0, 4: 0 };
    const assertionReasonCorrectPositions = [0, 0, 0, 0, 0];
    let assertionReasonCount = 0;

    for (const question of diffusionGemmaTechnicalReportQuestions) {
      difficultyCounts[question.difficulty] += 1;
      const correctPositions = question.options.flatMap((option, index) =>
        option.isCorrect ? [index] : [],
      );

      if (getQuestionType(question) === "assertion-reason") {
        assertionReasonCount += 1;
        expect(correctPositions).toHaveLength(1);
        assertionReasonCorrectPositions[correctPositions[0]] += 1;
      }

      const correctCount = correctPositions.length as 1 | 2 | 3 | 4;
      correctAnswerCounts[correctCount] += 1;
    }

    expect(difficultyCounts).toEqual({ easy: 20, medium: 20, hard: 20 });
    expect(correctAnswerCounts).toEqual({ 1: 15, 2: 15, 3: 15, 4: 15 });
    expect(assertionReasonCount).toBe(5);
    expect(assertionReasonCorrectPositions).toEqual([1, 1, 1, 1, 1]);
  });

  it("hardcodes every stable id at its helper call site", () => {
    const fileContents = fs.readFileSync(questionFilePath, "utf8");
    const questionArraySource = fileContents.slice(
      fileContents.indexOf(
        "export const diffusionGemmaTechnicalReportQuestions",
      ),
    );
    const helperCalls = [
      ...questionArraySource.matchAll(/make(?:AssertionReason)?Question\(/g),
    ];
    const hardcodedIds = [
      ...questionArraySource.matchAll(
        /make(?:AssertionReason)?Question\(\s*\n\s*"([^"]+)"/g,
      ),
    ].map((match) => match[1]);

    expect(helperCalls).toHaveLength(60);
    expect(hardcodedIds).toEqual(
      diffusionGemmaTechnicalReportQuestions.map(({ id }) => id),
    );
  });

  it("keeps every item self-contained, structurally valid, and instructional", () => {
    const directSourceReferences =
      /\b(?:in|from|according to) (?:the )?(?:paper|report|source|slides|lecture)\b|the paper's|previous question|next question|as above/i;

    for (const question of diffusionGemmaTechnicalReportQuestions) {
      const assertionReason = getQuestionType(question) === "assertion-reason";
      expect(question.options).toHaveLength(assertionReason ? 5 : 4);
      expect(question.options.some(({ isCorrect }) => isCorrect)).toBe(true);
      expect(question.prompt).not.toMatch(directSourceReferences);
      expect(question.explanation.length).toBeGreaterThanOrEqual(200);
      expect(
        question.explanation.match(/[.!?](?:\s|$)/g)?.length ?? 0,
      ).toBeGreaterThanOrEqual(2);
    }
  });
});

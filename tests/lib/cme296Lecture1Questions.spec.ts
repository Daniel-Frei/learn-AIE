import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  QUESTION_SOURCES,
  getQuestionType,
  stanfordCME296Lecture1DiffusionQuestions,
} from "@/lib/quiz";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const questionFilePath = path.resolve(
  testDir,
  "../../lib/lectures/Stanford CME296 Diffusion & Large Vision Models/lecture1_diffusion.ts",
);

describe("CME296 Lecture 1 question set", () => {
  it("registers one 40-question deep-learning source with contiguous stable IDs", () => {
    const source = QUESTION_SOURCES.find(
      (candidate) => candidate.id === "cme296-lect1",
    );
    const expectedIds = Array.from(
      { length: 40 },
      (_, index) => `cme296-lect1-q${String(index + 1).padStart(2, "0")}`,
    );

    expect(source).toMatchObject({
      seriesId: "stanford-cme296",
      topic: "DL",
      questions: stanfordCME296Lecture1DiffusionQuestions,
    });
    expect(stanfordCME296Lecture1DiffusionQuestions).toHaveLength(40);
    expect(
      stanfordCME296Lecture1DiffusionQuestions.map((question) => question.id),
    ).toEqual(expectedIds);
  });

  it("hardcodes every exported question ID instead of deriving it at runtime", () => {
    const fileContents = fs.readFileSync(questionFilePath, "utf8");
    const questionArraySource = fileContents.slice(
      fileContents.indexOf(
        "export const stanfordCME296Lecture1DiffusionQuestions",
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

    expect(hardcodedIds).toHaveLength(helperCalls.length);
    expect(hardcodedIds).toEqual(
      Array.from(
        { length: 40 },
        (_, index) => `cme296-lect1-q${String(index + 1).padStart(2, "0")}`,
      ),
    );
  });

  it("balances difficulty, question types, and answer counts", () => {
    const difficultyCounts = { easy: 0, medium: 0, hard: 0 };
    const allCorrectAnswerCounts = { 1: 0, 2: 0, 3: 0, 4: 0 };
    const multipleSelectCorrectAnswerCounts = { 1: 0, 2: 0, 3: 0, 4: 0 };
    const assertionReasonCorrectPositions = [0, 0, 0, 0, 0];
    let assertionReasonCount = 0;

    for (const question of stanfordCME296Lecture1DiffusionQuestions) {
      difficultyCounts[question.difficulty] += 1;
      const correctPositions = question.options.flatMap((option, index) =>
        option.isCorrect ? [index] : [],
      );
      const correctCount = correctPositions.length as 1 | 2 | 3 | 4;
      allCorrectAnswerCounts[correctCount] += 1;

      if (getQuestionType(question) === "assertion-reason") {
        assertionReasonCount += 1;
        expect(correctPositions).toHaveLength(1);
        assertionReasonCorrectPositions[correctPositions[0]] += 1;
      } else {
        multipleSelectCorrectAnswerCounts[correctCount] += 1;
      }
    }

    expect(difficultyCounts).toEqual({ easy: 13, medium: 14, hard: 13 });
    expect(assertionReasonCount).toBe(8);
    expect(assertionReasonCorrectPositions).toEqual([2, 2, 1, 2, 1]);
    expect(allCorrectAnswerCounts).toEqual({ 1: 10, 2: 10, 3: 10, 4: 10 });
    expect(multipleSelectCorrectAnswerCounts).toEqual({
      1: 2,
      2: 10,
      3: 10,
      4: 10,
    });
  });

  it("keeps every item structurally valid and every explanation instructional", () => {
    const structurallyInvalid = stanfordCME296Lecture1DiffusionQuestions.filter(
      (question) => {
        const expectedOptionCount =
          getQuestionType(question) === "assertion-reason" ? 5 : 4;
        return (
          question.options.length !== expectedOptionCount ||
          question.options.every((option) => !option.isCorrect)
        );
      },
    );
    const shortExplanations = stanfordCME296Lecture1DiffusionQuestions.filter(
      (question) => {
        const sentenceCount =
          question.explanation.match(/[.!?](?:\s|$)/g)?.length ?? 0;
        return question.explanation.length < 200 || sentenceCount < 2;
      },
    );

    expect(structurallyInvalid).toEqual([]);
    expect(shortExplanations).toEqual([]);
  });
});

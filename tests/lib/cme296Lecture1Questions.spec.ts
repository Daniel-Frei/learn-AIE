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
  const expectedIds = [
    "cme296-lect1-q01",
    "cme296-lect1-q02",
    "cme296-lect1-q06",
    "cme296-lect1-q07",
    "cme296-lect1-q08",
    "cme296-lect1-q09",
    "cme296-lect1-q10",
    ...Array.from({ length: 18 }, (_, index) => `cme296-lect1-q${index + 23}`),
  ];

  it("registers the 25 questions selected by the DiffusionGemma study guide", () => {
    const source = QUESTION_SOURCES.find(
      (candidate) => candidate.id === "cme296-lect1",
    );

    expect(source).toMatchObject({
      seriesId: "stanford-cme296",
      topic: "DL",
      questions: stanfordCME296Lecture1DiffusionQuestions,
    });
    expect(stanfordCME296Lecture1DiffusionQuestions).toHaveLength(25);
    expect(
      stanfordCME296Lecture1DiffusionQuestions.map((question) => question.id),
    ).toEqual(expectedIds);
  });

  it("hardcodes every candidate and explicitly selects the exported stable IDs", () => {
    const fileContents = fs.readFileSync(questionFilePath, "utf8");
    const questionArraySource = fileContents.slice(
      fileContents.indexOf("const lecture1QuestionCandidates"),
      fileContents.indexOf("const diffusionGemmaStudyGuideQuestionIds"),
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
    expect(hardcodedIds).toHaveLength(40);
    expect(fileContents).toContain("diffusionGemmaStudyGuideQuestionIds");
    expect(
      stanfordCME296Lecture1DiffusionQuestions.map(({ id }) => id),
    ).toEqual(expectedIds);
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

    expect(difficultyCounts).toEqual({ easy: 7, medium: 9, hard: 9 });
    expect(assertionReasonCount).toBe(4);
    expect(assertionReasonCorrectPositions).toEqual([0, 2, 1, 1, 0]);
    expect(allCorrectAnswerCounts).toEqual({ 1: 6, 2: 7, 3: 6, 4: 6 });
    expect(multipleSelectCorrectAnswerCounts).toEqual({
      1: 2,
      2: 7,
      3: 6,
      4: 6,
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

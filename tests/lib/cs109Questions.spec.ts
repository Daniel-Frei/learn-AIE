import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  QUESTION_SOURCE_CONTEXT,
  QUESTION_SOURCES,
  stanfordCS109Lecture1WelcomeCountingQuestions,
  stanfordCS109Lecture2CombinatoricsQuestions,
  type Question,
} from "@/lib/quiz";

const testDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(testDir, "../..");
const courseRoot = path.join(
  repoRoot,
  "lib",
  "lectures",
  "Stanford CS109 Probability for Computer Scientists",
);

const sets = [
  {
    sourceId: "cs109-lect1",
    chapter: 1,
    questions: stanfordCS109Lecture1WelcomeCountingQuestions,
    fileName: "lecture1_welcome_counting.ts",
    coverage: [
      "experiment",
      "outcome",
      "product rule",
      "sum rule",
      "inclusion-exclusion",
      "positive result",
    ],
  },
  {
    sourceId: "cs109-lect2",
    chapter: 2,
    questions: stanfordCS109Lecture2CombinatoricsQuestions,
    fileName: "lecture2_combinatorics.ts",
    coverage: [
      "permutation",
      "combination",
      "indistinguishable",
      "bucket",
      "divider",
      "integer solutions",
    ],
  },
] as const;

function getDifficultyDistribution(questions: Question[]) {
  return questions.reduce(
    (counts, question) => {
      counts[question.difficulty] += 1;
      return counts;
    },
    { easy: 0, medium: 0, hard: 0 },
  );
}

function getCorrectAnswerDistribution(questions: Question[]) {
  return questions.reduce(
    (counts, question) => {
      const correctCount = question.options.filter(
        (option) => option.isCorrect,
      ).length;
      counts[correctCount] = (counts[correctCount] ?? 0) + 1;
      return counts;
    },
    {} as Record<number, number>,
  );
}

describe("Stanford CS109 question sets", () => {
  it("keeps both lecture banks at 35 questions with stable hardcoded IDs", () => {
    for (const set of sets) {
      const expectedIds = Array.from(
        { length: 35 },
        (_, index) => set.sourceId + "-q" + String(index + 1).padStart(2, "0"),
      );
      const questionIds = set.questions.map((question) => question.id);
      const fileContent = fs.readFileSync(
        path.join(courseRoot, set.fileName),
        "utf8",
      );
      const authoredIds = [
        ...fileContent.matchAll(
          /makeQuestion\(\s*\n\s*"(cs109-lect[12]-q\d{2})"/g,
        ),
      ].map((match) => match[1]);

      expect(set.questions).toHaveLength(35);
      expect(questionIds).toEqual(expectedIds);
      expect(new Set(questionIds).size).toBe(questionIds.length);
      expect(authoredIds).toEqual(expectedIds);
      expect(
        set.questions.every((question) => question.chapter === set.chapter),
      ).toBe(true);
    }
  });

  it("keeps difficulty and correct-answer counts balanced", () => {
    for (const set of sets) {
      expect(getDifficultyDistribution(set.questions)).toEqual({
        easy: 12,
        medium: 12,
        hard: 11,
      });
      expect(getCorrectAnswerDistribution(set.questions)).toEqual({
        1: 8,
        2: 9,
        3: 9,
        4: 9,
      });
      expect(
        set.questions.every((question) => question.options.length === 4),
      ).toBe(true);
    }
  });

  it("registers a selectable Math course with matching source context", () => {
    const registeredSources = sets.map((set) =>
      QUESTION_SOURCES.find((source) => source.id === set.sourceId),
    );

    expect(
      registeredSources.map((source) => ({
        id: source?.id,
        seriesId: source?.seriesId,
        seriesLabel: source?.seriesLabel,
        topic: source?.topic,
        questionCount: source?.questions.length,
      })),
    ).toEqual([
      {
        id: "cs109-lect1",
        seriesId: "stanford-cs109",
        seriesLabel: "Stanford CS109 Probability for Computer Scientists",
        topic: "Math",
        questionCount: 35,
      },
      {
        id: "cs109-lect2",
        seriesId: "stanford-cs109",
        seriesLabel: "Stanford CS109 Probability for Computer Scientists",
        topic: "Math",
        questionCount: 35,
      },
    ]);
    expect(QUESTION_SOURCE_CONTEXT["cs109-lect1"]).toContain("product");
    expect(QUESTION_SOURCE_CONTEXT["cs109-lect2"]).toContain("divider");
  });

  it("covers the probability and counting source boundary without logistics prompts", () => {
    for (const set of sets) {
      const authoredText = set.questions
        .flatMap((question) => [
          question.prompt,
          question.explanation,
          ...question.options.map((option) => option.text),
        ])
        .join("\n")
        .toLowerCase();

      for (const term of set.coverage) {
        expect(authoredText).toContain(term);
      }
      expect(authoredText).not.toMatch(
        /\b(?:office hours|problem set|grading policy|late policy|honor code|teaching team|midterm date|final exam date)\b/,
      );
    }
  });

  it("stores each lecture's paired transcript and slide deck", () => {
    const sourceRoot = path.join(courseRoot, "transcripts-and-files");
    const expectedFiles = [
      "lecture 1 - slides.pdf",
      "lecture 1 - transcript.md",
      "lecture 2 - slides.pdf",
      "lecture 2 - transcript.md",
    ];

    for (const fileName of expectedFiles) {
      const filePath = path.join(sourceRoot, fileName);
      expect(fs.existsSync(filePath), fileName + " should be stored").toBe(
        true,
      );
      expect(fs.statSync(filePath).size).toBeGreaterThan(1_000);
    }
  });
});

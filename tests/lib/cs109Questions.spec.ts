import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { renderToString } from "katex";
import {
  QUESTION_SOURCE_CONTEXT,
  QUESTION_SOURCES,
  stanfordCS109Lecture1WelcomeCountingQuestions,
  stanfordCS109Lecture2CombinatoricsQuestions,
  stanfordCS109Lecture3IntroProbabilityQuestions,
  stanfordCS109Lecture4ConditioningAndBayesQuestions,
  stanfordCS109Lecture5IndependenceQuestions,
  stanfordCS109Lecture6RandomVariablesQuestions,
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
  {
    sourceId: "cs109-lect3",
    chapter: 3,
    questions: stanfordCS109Lecture3IntroProbabilityQuestions,
    fileName: "lecture3_intro_probability.ts",
    coverage: [
      "sample space",
      "event",
      "equally likely",
      "axiom",
      "complement",
      "mutually exclusive",
    ],
  },
  {
    sourceId: "cs109-lect4",
    chapter: 4,
    questions: stanfordCS109Lecture4ConditioningAndBayesQuestions,
    fileName: "lecture4_conditioning_and_bayes.ts",
    coverage: [
      "conditional probability",
      "chain rule",
      "total probability",
      "bayes",
      "prior",
      "posterior",
    ],
  },
  {
    sourceId: "cs109-lect5",
    chapter: 5,
    questions: stanfordCS109Lecture5IndependenceQuestions,
    fileName: "lecture5_independence.ts",
    coverage: [
      "independent",
      "mutually exclusive",
      "inclusion-exclusion",
      "network",
      "exactly",
      "hash",
    ],
  },
  {
    sourceId: "cs109-lect6",
    chapter: 6,
    questions: stanfordCS109Lecture6RandomVariablesQuestions,
    fileName: "lecture6_random_variables.ts",
    coverage: [
      "conditional independence",
      "random variable",
      "probability mass function",
      "support",
      "expectation",
      "st. petersburg",
    ],
  },
] as const;

// These IDs replace retired items, rather than inheriting their rating history.
const reviewedIdReplacements: Record<string, string> = {
  "cs109-lect1-q18": "cs109-lect1-q37",
  "cs109-lect1-q21": "cs109-lect1-q38",
  "cs109-lect1-q22": "cs109-lect1-q39",
  "cs109-lect1-q31": "cs109-lect1-q40",
  "cs109-lect1-q32": "cs109-lect1-q41",
  "cs109-lect1-q35": "cs109-lect1-q42",
  "cs109-lect2-q10": "cs109-lect2-q36",
  "cs109-lect2-q17": "cs109-lect2-q37",
  "cs109-lect2-q33": "cs109-lect2-q38",
  "cs109-lect2-q35": "cs109-lect2-q39",
  "cs109-lect3-q03": "cs109-lect3-q36",
  "cs109-lect3-q12": "cs109-lect3-q37",
  "cs109-lect3-q24": "cs109-lect3-q38",
  "cs109-lect3-q25": "cs109-lect3-q39",
  "cs109-lect3-q29": "cs109-lect3-q40",
  "cs109-lect3-q31": "cs109-lect3-q41",
  "cs109-lect3-q32": "cs109-lect3-q42",
  "cs109-lect3-q34": "cs109-lect3-q43",
  "cs109-lect4-q23": "cs109-lect4-q36",
  "cs109-lect4-q24": "cs109-lect4-q37",
  "cs109-lect4-q31": "cs109-lect4-q38",
  "cs109-lect4-q34": "cs109-lect4-q39",
  "cs109-lect4-q35": "cs109-lect4-q40",
  "cs109-lect5-q20": "cs109-lect5-q36",
  "cs109-lect5-q26": "cs109-lect5-q37",
  "cs109-lect5-q35": "cs109-lect5-q38",
  "cs109-lect6-q04": "cs109-lect6-q36",
  "cs109-lect6-q22": "cs109-lect6-q37",
  "cs109-lect6-q25": "cs109-lect6-q38",
  "cs109-lect6-q27": "cs109-lect6-q39",
  "cs109-lect6-q28": "cs109-lect6-q40",
  "cs109-lect6-q31": "cs109-lect6-q41",
  "cs109-lect6-q34": "cs109-lect6-q42",
};

const reviewedQuestions = sets.flatMap((set) => set.questions);

function findReviewedQuestion(id: string) {
  const question = reviewedQuestions.find((candidate) => candidate.id === id);
  if (!question) throw new Error("Missing reviewed question: " + id);
  return question;
}

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
  it("keeps all lecture banks at 35 questions with stable hardcoded IDs", () => {
    for (const set of sets) {
      const expectedIds = Array.from({ length: 35 }, (_, index) => {
        const legacyId =
          set.sourceId + "-q" + String(index + 1).padStart(2, "0");
        return reviewedIdReplacements[legacyId] ?? legacyId;
      });
      const questionIds = set.questions.map((question) => question.id);
      const fileContent = fs.readFileSync(
        path.join(courseRoot, set.fileName),
        "utf8",
      );
      const authoredIds = [
        ...fileContent.matchAll(
          /makeQuestion\(\s*\n\s*"(cs109-lect[1-6]-q\d{2})"/g,
        ),
      ].map((match) => match[1]);

      expect(set.questions).toHaveLength(35);
      expect([...questionIds].sort()).toEqual([...expectedIds].sort());
      expect(new Set(questionIds).size).toBe(questionIds.length);
      expect([...authoredIds].sort()).toEqual([...expectedIds].sort());
      expect(
        set.questions.every((question) => question.chapter === set.chapter),
      ).toBe(true);
    }
  });

  it("retires rewritten and rescored IDs without reusing their identities", () => {
    const activeIds = new Set(reviewedQuestions.map((question) => question.id));
    expect(Object.keys(reviewedIdReplacements)).toHaveLength(33);
    expect(new Set(Object.values(reviewedIdReplacements)).size).toBe(33);
    for (const [retired, replacement] of Object.entries(
      reviewedIdReplacements,
    )) {
      expect(activeIds.has(retired), retired).toBe(false);
      expect(activeIds.has(replacement), replacement).toBe(true);
    }
  });

  it("keeps all difficulty levels represented and answer counts balanced", () => {
    for (const set of sets) {
      // Difficulty follows reasoning, not the old forced 12/12/11 quota.
      const difficulties = getDifficultyDistribution(set.questions);
      expect(Object.values(difficulties).reduce((sum, n) => sum + n, 0)).toBe(
        35,
      );
      for (const count of Object.values(difficulties)) {
        expect(count).toBeGreaterThanOrEqual(5);
        expect(count).toBeLessThanOrEqual(18);
      }
      const distribution = getCorrectAnswerDistribution(set.questions);
      for (const count of [1, 2, 3, 4]) {
        expect(Math.abs(distribution[count] - 35 / 4)).toBeLessThanOrEqual(4);
      }
      expect(
        set.questions.every((question) => question.options.length === 4),
      ).toBe(true);
    }
  });

  it("scores the geometric-probability correction consistently with area", () => {
    const question = findReviewedQuestion("cs109-lect3-q41");
    // Equal perimeter does not force equal area: 1x5 and 3x3 rectangles
    // both have perimeter 12 but have areas 5 and 9.
    expect(2 * (1 + 5)).toBe(2 * (3 + 3));
    expect(1 * 5).not.toBe(3 * 3);
    expect(question.options.map((option) => option.isCorrect)).toEqual([
      true,
      true,
      true,
      false,
    ]);
    expect(question.options[2].text).toContain("Replacing area by perimeter");
  });

  it("evaluates the student-weighted class size from enrollment counts", () => {
    const sizes = [5, 10, 150];
    const total = sizes.reduce((sum, size) => sum + size, 0);
    const studentMean =
      sizes.reduce((sum, size) => sum + size * size, 0) / total;
    const question = findReviewedQuestion("cs109-lect6-q37");
    const studentOption = question.options.find((option) =>
      option.text.includes("Choosing one student"),
    );
    expect(studentMean.toFixed(2)).toBe("137.12");
    expect(studentOption?.isCorrect).toBe(true);
    expect(studentOption?.text).toContain(studentMean.toFixed(2));
    expect(question.explanation).toContain("22625/165");
  });

  it("checks constrained counts by enumerating small outcome spaces", () => {
    const validCodes = Array.from({ length: 10_000 }, (_, n) =>
      String(n).padStart(4, "0"),
    ).filter(
      (code) => /^[1-5][0-5]{2}[024]$/.test(code) && new Set(code).size === 4,
    );
    expect(validCodes).toHaveLength(156);
    expect(
      findReviewedQuestion("cs109-lect1-q37").options.find((option) =>
        option.text.includes("156"),
      )?.isCorrect,
    ).toBe(true);

    const allocations = [];
    for (let x1 = 2; x1 <= 8; x1 += 1) {
      for (let x2 = 1; x2 <= 8; x2 += 1) {
        const x3 = 8 - x1 - x2;
        if (x3 >= 0 && x3 <= 2) allocations.push([x1, x2, x3]);
      }
    }
    expect(allocations).toHaveLength(15);
    expect(
      findReviewedQuestion("cs109-lect2-q38").options.find((option) =>
        option.text.includes("21-6=15"),
      )?.isCorrect,
    ).toBe(true);
  });

  it("distinguishes probability of either hash bucket from both buckets", () => {
    const destinations = [
      { bucket: 1, probability: 0.2 },
      { bucket: 2, probability: 0.3 },
      { bucket: 3, probability: 0.5 },
    ];
    let either = 0;
    let both = 0;
    for (const a of destinations) {
      for (const b of destinations) {
        for (const c of destinations) {
          const buckets = new Set([a.bucket, b.bucket, c.bucket]);
          const mass = a.probability * b.probability * c.probability;
          if (buckets.has(1) || buckets.has(2)) either += mass;
          if (buckets.has(1) && buckets.has(2)) both += mass;
        }
      }
    }
    expect(either).toBeCloseTo(0.875);
    expect(both).toBeCloseTo(0.27);
    const question = findReviewedQuestion("cs109-lect5-q38");
    expect(question.options[0].isCorrect).toBe(true);
    expect(question.options[0].text).toContain(either.toFixed(3));
    expect(question.options[1].isCorrect).toBe(false);
    expect(question.options[1].text).toContain(both.toFixed(3));
  });

  it("keeps the fair-frisbee procedure and cutoff game self-contained", () => {
    expect(findReviewedQuestion("cs109-lect6-q14").prompt).toContain("0<p<1");
    expect(findReviewedQuestion("cs109-lect6-q14").prompt).toContain(
      "All flips across rounds are independent",
    );
    const cutoff = findReviewedQuestion("cs109-lect6-q33").prompt;
    expect(cutoff).toContain("first tail");
    expect(cutoff).toContain("preceding heads");
    expect(cutoff).toContain("zero");
    expect(findReviewedQuestion("cs109-lect3-q17").prompt).toContain(
      "four distinct cow toys and three distinct pig toys",
    );
    expect(findReviewedQuestion("cs109-lect4-q21").prompt).toContain(
      "98% sensitivity and a 1% false-positive rate",
    );
  });

  it("renders authored math and rejects accidental string-escape corruption", () => {
    for (const question of reviewedQuestions) {
      const fields = [
        question.prompt,
        question.explanation,
        ...question.options.map((option) => option.text),
      ];
      for (const field of fields) {
        expect(
          [...field].some((char) => {
            const code = char.charCodeAt(0);
            return code < 32 && char !== "\n" && char !== "\t";
          }),
          question.id,
        ).toBe(false);
        for (const match of field.matchAll(/\\\(([\s\S]*?)\\\)/g)) {
          expect(() =>
            renderToString(match[1], { throwOnError: true, strict: "error" }),
          ).not.toThrow();
        }
      }
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
      {
        id: "cs109-lect3",
        seriesId: "stanford-cs109",
        seriesLabel: "Stanford CS109 Probability for Computer Scientists",
        topic: "Math",
        questionCount: 35,
      },
      {
        id: "cs109-lect4",
        seriesId: "stanford-cs109",
        seriesLabel: "Stanford CS109 Probability for Computer Scientists",
        topic: "Math",
        questionCount: 35,
      },
      {
        id: "cs109-lect5",
        seriesId: "stanford-cs109",
        seriesLabel: "Stanford CS109 Probability for Computer Scientists",
        topic: "Math",
        questionCount: 35,
      },
      {
        id: "cs109-lect6",
        seriesId: "stanford-cs109",
        seriesLabel: "Stanford CS109 Probability for Computer Scientists",
        topic: "Math",
        questionCount: 35,
      },
    ]);
    expect(QUESTION_SOURCE_CONTEXT["cs109-lect1"]).toContain("product");
    expect(QUESTION_SOURCE_CONTEXT["cs109-lect2"]).toContain("divider");
    expect(QUESTION_SOURCE_CONTEXT["cs109-lect3"]).toContain("axioms");
    expect(QUESTION_SOURCE_CONTEXT["cs109-lect4"]).toContain("Bayes");
    expect(QUESTION_SOURCE_CONTEXT["cs109-lect5"]).toContain("independence");
    expect(QUESTION_SOURCE_CONTEXT["cs109-lect6"]).toContain("mass functions");
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
      "lecture 3 - slides.pdf",
      "lecture 3 - transcript.md",
      "lecture 4 - slides.pdf",
      "lecture 4 - transcript.md",
      "lecture 5 - slides.pdf",
      "lecture 5 - transcript.md",
      "lecture 6 - slides.pdf",
      "lecture 6 - transcript.md",
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

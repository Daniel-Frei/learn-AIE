import { describe, expect, it } from "vitest";
import { QUESTION_SOURCES, type Question } from "@/lib/quiz";
import { createDefaultRatingState } from "@/lib/ratingEngine";
import {
  buildQuizApiUrl,
  CLIMB_MIN_TARGETED_POOL_SIZE,
  clampDifficultyRange,
  DEFAULT_DIFFICULTY_RANGE,
  evaluateAnswer,
  getEligibleQuestionIds,
  pickClimbQuestionId,
  shuffleItems,
} from "@/lib/quizSession";

describe("quiz session helpers", () => {
  it("shuffles items without mutating the original array", () => {
    const original = ["a", "b", "c"];
    const shuffled = shuffleItems(original, () => 0);

    expect(shuffled).toEqual(["b", "c", "a"]);
    expect(original).toEqual(["a", "b", "c"]);
  });

  it("evaluates multi-select answers and mistake counts", () => {
    const options = [
      { text: "A", isCorrect: true },
      { text: "B", isCorrect: false },
      { text: "C", isCorrect: true },
      { text: "D", isCorrect: false },
    ];

    expect(evaluateAnswer(options, [0, 2])).toMatchObject({
      isCorrect: true,
      mistakeCount: 0,
      incorrectSelectionCount: 0,
      missedCorrectOptionCount: 0,
    });

    expect(evaluateAnswer(options, [0, 1])).toMatchObject({
      isCorrect: false,
      mistakeCount: 2,
      incorrectSelectionCount: 1,
      missedCorrectOptionCount: 1,
    });
  });

  it("clamps difficulty ranges to supported Elo bounds", () => {
    expect(clampDifficultyRange({ min: -100, max: 4000 })).toEqual({
      min: 0,
      max: 3000,
    });

    expect(clampDifficultyRange({ min: 1800, max: 1200 })).toEqual({
      min: 1800,
      max: 1800,
    });
  });

  it("defaults to the full supported question Elo range", () => {
    expect(DEFAULT_DIFFICULTY_RANGE).toEqual({ min: 0, max: 3000 });
  });

  it("returns eligible questions for selected source and Elo range", () => {
    const ratingState = createDefaultRatingState();
    const source = QUESTION_SOURCES[0];

    const eligibleIds = getEligibleQuestionIds({
      sources: [source.id],
      topics: [],
      difficultyRange: { min: 1200, max: 1600 },
      ratingState,
    });
    const sourceIds = new Set(source.questions.map((question) => question.id));

    expect(eligibleIds.length).toBeGreaterThan(0);
    expect(eligibleIds.every((id) => sourceIds.has(id))).toBe(true);
  });

  it("picks a climb question near the participant rating", () => {
    const ratingState = createDefaultRatingState();
    const pool: Question[] = [
      {
        id: "easy-question",
        chapter: 1,
        difficulty: "easy",
        prompt: "Easy",
        options: [],
        explanation: "",
      },
      {
        id: "medium-question",
        chapter: 1,
        difficulty: "medium",
        prompt: "Medium",
        options: [],
        explanation: "",
      },
      {
        id: "hard-question",
        chapter: 1,
        difficulty: "hard",
        prompt: "Hard",
        options: [],
        explanation: "",
      },
    ];

    expect(pickClimbQuestionId(pool, ratingState, [], () => 0)).toBe(
      "medium-question",
    );
  });

  it("returns no climb question when the filtered pool is empty", () => {
    expect(pickClimbQuestionId([], createDefaultRatingState(), [])).toBeNull();
  });

  it("penalizes recently seen climb questions", () => {
    const ratingState = {
      ...createDefaultRatingState(),
      user: { ...createDefaultRatingState().user, rating: 1600 },
    };
    const pool: Question[] = [
      {
        id: "medium-question",
        chapter: 1,
        difficulty: "medium",
        prompt: "Medium",
        options: [],
        explanation: "",
      },
      {
        id: "hard-question",
        chapter: 1,
        difficulty: "hard",
        prompt: "Hard",
        options: [],
        explanation: "",
      },
    ];

    expect(
      pickClimbQuestionId(pool, ratingState, ["medium-question"], () => 0),
    ).toBe("hard-question");
  });

  it("uses random climb selection for a share of questions", () => {
    const ratingState = createDefaultRatingState();
    const pool: Question[] = [
      {
        id: "near-question",
        chapter: 1,
        difficulty: "medium",
        prompt: "Near",
        options: [],
        explanation: "",
      },
      {
        id: "far-question",
        chapter: 1,
        difficulty: "hard",
        prompt: "Far",
        options: [],
        explanation: "",
      },
    ];
    const rolls = [0.95, 0.99];

    expect(
      pickClimbQuestionId(pool, ratingState, [], () => rolls.shift() ?? 0),
    ).toBe("far-question");
  });

  it("keeps at least ten targeted climb candidates when available", () => {
    const base = createDefaultRatingState();
    const pool: Question[] = Array.from(
      { length: CLIMB_MIN_TARGETED_POOL_SIZE + 2 },
      (_, index) => ({
        id: `question-${index}`,
        chapter: 1,
        difficulty: "medium",
        prompt: `Question ${index}`,
        options: [],
        explanation: "",
      }),
    );
    const ratingState = {
      ...base,
      questions: Object.fromEntries(
        pool.map((question, index) => [
          question.id,
          {
            rating: 1500 + index,
            rd: 30,
            sigma: 0.06,
            lastUpdatedAt: 0,
            gamesPlayed: 10,
            legacyCorrect: 0,
            legacyWrong: 0,
            label: "medium" as const,
          },
        ]),
      ),
    };
    const rolls = [0, ...Array.from({ length: pool.length }, () => 0), 0.99];

    expect(
      pickClimbQuestionId(pool, ratingState, [], () => rolls.shift() ?? 0),
    ).toBe(`question-${CLIMB_MIN_TARGETED_POOL_SIZE - 1}`);
  });

  it("builds absolute mobile API URLs", () => {
    expect(
      buildQuizApiUrl("http://localhost:3101", "/api/quiz-state", {
        participantId: "abc 123",
      }),
    ).toBe("http://localhost:3101/api/quiz-state?participantId=abc+123");
    expect(buildQuizApiUrl("http://localhost:3101/", "/api/quiz-state")).toBe(
      "http://localhost:3101/api/quiz-state",
    );
    expect(() => buildQuizApiUrl(" ", "/api/quiz-state")).toThrow(
      "API base URL is required.",
    );
  });
});

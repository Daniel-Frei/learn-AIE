import { describe, expect, it } from "vitest";
import { QUESTION_SOURCES, type Question } from "@/lib/quiz";
import { createDefaultRatingState } from "@/lib/ratingEngine";
import {
  buildQuizApiUrl,
  clampDifficultyRange,
  evaluateAnswer,
  getEligibleQuestionIds,
  pickClimbQuestionId,
} from "@/lib/quizSession";

describe("quiz session helpers", () => {
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

  it("builds absolute mobile API URLs", () => {
    expect(
      buildQuizApiUrl("http://localhost:3101", "/api/quiz-state", {
        participantId: "abc 123",
      }),
    ).toBe("http://localhost:3101/api/quiz-state?participantId=abc+123");
  });
});

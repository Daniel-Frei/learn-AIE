import { describe, expect, it } from "vitest";
import { stanfordCME295Lecture3LLMsQuestions } from "@/lib/quiz";

function getQuestionNumber(id: string) {
  const match = id.match(/^cme295-lect3-q(\d+)$/);
  if (!match) throw new Error("Unexpected CME295 Lecture 3 question ID: " + id);
  return Number(match[1]);
}

describe("CME295 Lecture 3 question set", () => {
  it("keeps the slide-weighted 60-question coverage blueprint", () => {
    const ranges = [
      { label: "LLM overview", first: 181, last: 186, count: 6 },
      { label: "Mixture of Experts", first: 187, last: 195, count: 9 },
      { label: "response generation", first: 196, last: 213, count: 18 },
      { label: "prompting strategies", first: 214, last: 218, count: 5 },
      { label: "inference optimizations", first: 219, last: 240, count: 22 },
    ];

    expect(stanfordCME295Lecture3LLMsQuestions).toHaveLength(60);

    for (const range of ranges) {
      const questionsInRange = stanfordCME295Lecture3LLMsQuestions.filter(
        (question) => {
          const number = getQuestionNumber(question.id);
          return number >= range.first && number <= range.last;
        },
      );

      expect(questionsInRange, range.label).toHaveLength(range.count);
    }
  });

  it("balances difficulty and correct-answer counts", () => {
    const difficultyCounts = { easy: 0, medium: 0, hard: 0 };
    const correctAnswerCounts = { 1: 0, 2: 0, 3: 0, 4: 0 };

    for (const question of stanfordCME295Lecture3LLMsQuestions) {
      difficultyCounts[question.difficulty] += 1;
      const correctCount = question.options.filter((option) => option.isCorrect)
        .length as 1 | 2 | 3 | 4;
      correctAnswerCounts[correctCount] += 1;
    }

    expect(difficultyCounts).toEqual({ easy: 20, medium: 20, hard: 20 });
    expect(correctAnswerCounts).toEqual({ 1: 15, 2: 15, 3: 15, 4: 15 });
  });

  it("keeps new IDs contiguous and preserves substantial applied practice", () => {
    const ids = stanfordCME295Lecture3LLMsQuestions.map(
      (question) => question.id,
    );
    const expectedIds = Array.from(
      { length: 60 },
      (_, index) => "cme295-lect3-q" + (181 + index),
    );
    const quantitativeQuestions = stanfordCME295Lecture3LLMsQuestions.filter(
      (question) =>
        /\\\(|probabilit|logit|parameter|head|GiB|MiB|block|ratio|width-2/i.test(
          question.prompt,
        ),
    );

    expect(ids).toEqual(expectedIds);
    expect(quantitativeQuestions.length).toBeGreaterThanOrEqual(24);
  });

  it("keeps explanations teaching-oriented and avoids definition-stem dominance", () => {
    const shortOrSingleSentenceExplanations =
      stanfordCME295Lecture3LLMsQuestions.filter((question) => {
        const sentenceCount =
          question.explanation.match(/[.!?](?:\s|$)/g)?.length ?? 0;
        return question.explanation.length < 200 || sentenceCount < 2;
      });
    const definitionRecognitionPrompts =
      stanfordCME295Lecture3LLMsQuestions.filter((question) =>
        /which statement best (?:describes|identifies|explains|summarizes)/i.test(
          question.prompt,
        ),
      );

    expect(shortOrSingleSentenceExplanations).toEqual([]);
    expect(definitionRecognitionPrompts.length).toBeLessThanOrEqual(3);
  });
});

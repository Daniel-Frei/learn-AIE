import { describe, expect, it } from "vitest";
import {
  CrashCourseProbabilityL1Questions,
  CrashCourseProbabilityL2Questions,
  CrashCourseProbabilityL3Questions,
  CrashCourseProbabilityL4Questions,
  CrashCourseProbabilityL5Questions,
  QUESTION_SOURCES,
  type Question,
} from "@/lib/quiz";

const sets = [
  {
    sourceId: "crash-probability-l1",
    idPrefix: "crash-probability-l1-q",
    questions: CrashCourseProbabilityL1Questions,
    coverage: [
      /sample space/i,
      /probability axioms/i,
      /complement/i,
      /inclusion-exclusion/i,
      /permutation/i,
      /combination/i,
      /random variable/i,
      /probability mass function/i,
      /probability density function/i,
      /linearity of expectation/i,
      /variance/i,
      /calibrat/i,
    ],
  },
  {
    sourceId: "crash-probability-l2",
    idPrefix: "crash-probability-l2-q",
    questions: CrashCourseProbabilityL2Questions,
    coverage: [
      /conditional probability/i,
      /joint probability/i,
      /marginaliz/i,
      /law of total probability/i,
      /independen/i,
      /common factor/i,
      /Bayes/i,
      /sensitivity/i,
      /specificity/i,
      /prevalence/i,
      /base rate/i,
      /P\(y\\mid x\)/i,
    ],
  },
  {
    sourceId: "crash-probability-l3",
    idPrefix: "crash-probability-l3-q",
    questions: CrashCourseProbabilityL3Questions,
    coverage: [
      /logit/i,
      /softmax/i,
      /likelihood/i,
      /log-likelihood/i,
      /negative log-likelihood/i,
      /cross-entropy/i,
      /entropy/i,
      /temperature/i,
      /calibrat/i,
      /gradient/i,
      /perplexity/i,
    ],
  },
  {
    sourceId: "crash-probability-l4",
    idPrefix: "crash-probability-l4-q",
    questions: CrashCourseProbabilityL4Questions,
    coverage: [
      /transition/i,
      /Markov property/i,
      /stochastic policy/i,
      /exploration/i,
      /discount factor/i,
      /expected return/i,
      /state-value function/i,
      /action value/i,
      /Bellman/i,
      /first-step/i,
      /waiting time/i,
      /two consecutive heads/i,
      /random walk/i,
    ],
  },
  {
    sourceId: "crash-probability-l5",
    idPrefix: "crash-probability-l5-q",
    questions: CrashCourseProbabilityL5Questions,
    coverage: [
      /sampling/i,
      /multinomial/i,
      /temperature/i,
      /top-k/i,
      /top-p/i,
      /entropy/i,
      /latent/i,
      /marginaliz/i,
      /posterior/i,
      /Gaussian/i,
      /forward diffusion/i,
      /reverse Markov/i,
      /classifier-free guidance/i,
      /autoregressive/i,
    ],
  },
] as const;

function countByDifficulty(questions: Question[]) {
  return questions.reduce(
    (counts, question) => {
      counts[question.difficulty] += 1;
      return counts;
    },
    { easy: 0, medium: 0, hard: 0 },
  );
}

function countByCorrectOptions(questions: Question[]) {
  return questions.reduce(
    (counts, question) => {
      const count = question.options.filter(({ isCorrect }) => isCorrect)
        .length as 1 | 2 | 3 | 4;
      counts[count] += 1;
      return counts;
    },
    { 1: 0, 2: 0, 3: 0, 4: 0 },
  );
}

describe("Crash Course Probability Lecture 1-5 question banks", () => {
  for (const set of sets) {
    it(`keeps ${set.sourceId} registered, stable, and structurally valid`, () => {
      const registered = QUESTION_SOURCES.find(({ id }) => id === set.sourceId);
      const expectedIds = Array.from(
        { length: 60 },
        (_, index) => `${set.idPrefix}${index + 61}`,
      );

      expect(registered?.questions).toBe(set.questions);
      expect(set.questions).toHaveLength(60);
      expect(set.questions.map(({ id }) => id)).toEqual(expectedIds);
      expect(new Set(expectedIds).size).toBe(60);
      expect(new Set(set.questions.map(({ prompt }) => prompt)).size).toBe(60);

      expect(
        set.questions
          .filter(({ explanation }) => explanation.length < 200)
          .map(({ id, explanation }) => `${id}:${explanation.length}`),
      ).toEqual([]);

      for (const question of set.questions) {
        expect(question.options, question.id).toHaveLength(4);
        expect(
          new Set(question.options.map(({ text }) => text)).size,
          question.id,
        ).toBe(4);
        expect(
          question.options.filter(({ isCorrect }) => isCorrect).length,
          question.id,
        ).toBeGreaterThanOrEqual(1);
        expect(
          question.explanation.match(/[.!?](?:\s|$)/g)?.length ?? 0,
          question.id,
        ).toBeGreaterThanOrEqual(2);
        expect(question.prompt, question.id).not.toMatch(
          /\b(?:in the lecture|the transcript|the slides|previous question|next question|as above)\b/i,
        );
      }
    });

    it(`balances difficulty and answer patterns for ${set.sourceId}`, () => {
      expect(countByDifficulty(set.questions)).toEqual({
        easy: 20,
        medium: 20,
        hard: 20,
      });
      expect(countByCorrectOptions(set.questions)).toEqual({
        1: 10,
        2: 20,
        3: 20,
        4: 10,
      });
    });

    it(`covers the revised curriculum for ${set.sourceId}`, () => {
      const searchableText = set.questions
        .flatMap(({ prompt, explanation, options }) => [
          prompt,
          explanation,
          ...options.map(({ text }) => text),
        ])
        .join("\n");

      for (const marker of set.coverage) {
        expect(searchableText, `${set.sourceId} is missing ${marker}`).toMatch(
          marker,
        );
      }
    });
  }
});

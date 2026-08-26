import { describe, expect, it } from "vitest";
import {
  bayesPosterior,
  combinations,
  conditionalProbability,
  crossEntropy,
  discountedReturn,
  entropyBits,
  epsilonGreedyDistribution,
  expectedFlipsForConsecutiveHeads,
  expectedValue,
  factorial,
  naturalFrequencyCounts,
  negativeLogLikelihood,
  normalizeWeights,
  oddsToProbability,
  permutations,
  probabilityToOdds,
  softmax,
  topKDistribution,
  topPDistribution,
  totalProbability,
  trajectoryProbability,
  unionProbability,
  variance,
} from "@/lib/probabilityLearning";

describe("probability learning math", () => {
  it("normalizes weights and converts probability to odds and back", () => {
    expect(normalizeWeights([2, 3, 5])).toEqual([0.2, 0.3, 0.5]);
    expect(probabilityToOdds(0.75)).toBeCloseTo(3);
    expect(oddsToProbability(3)).toBeCloseTo(0.75);
  });

  it("counts ordered and unordered selections", () => {
    expect(factorial(4)).toBe(24);
    expect(permutations(10, 3)).toBe(720);
    expect(combinations(10, 3)).toBe(120);
  });

  it("computes event, conditional, total, and Bayesian probabilities", () => {
    expect(unionProbability(4 / 52, 13 / 52, 1 / 52)).toBeCloseTo(16 / 52);
    expect(conditionalProbability(1 / 52, 4 / 52)).toBeCloseTo(0.25);
    expect(
      totalProbability([
        { prior: 0.7, conditional: 0.5 },
        { prior: 0.3, conditional: 7 / 8 },
      ]),
    ).toBeCloseTo(0.6125);
    expect(
      bayesPosterior({
        prior: 0.01,
        likelihood: 0.99,
        falsePositiveRate: 0.01,
      }),
    ).toBeCloseTo(0.5, 1);
  });

  it("builds natural-frequency groups for a diagnostic test", () => {
    expect(
      naturalFrequencyCounts({
        population: 1000,
        prevalence: 0.01,
        sensitivity: 0.99,
        specificity: 0.99,
      }),
    ).toMatchObject({
      condition: 10,
      noCondition: 990,
      truePositive: 10,
      falsePositive: 10,
      positiveTests: 20,
      posterior: 0.5,
    });
  });

  it("computes expectations, variance, softmax loss, and entropy", () => {
    const outcomes = [
      { value: 10, probability: 0.25 },
      { value: 3, probability: 0.5 },
      { value: -4, probability: 0.25 },
    ];
    expect(expectedValue(outcomes)).toBe(3);
    expect(variance(outcomes)).toBeCloseTo(24.5);

    const prediction = softmax([2, 1, 0]);
    expect(prediction.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1);
    expect(prediction[0]).toBeGreaterThan(prediction[1] ?? 0);
    expect(negativeLogLikelihood(0.7)).toBeCloseTo(-Math.log(0.7));
    expect(crossEntropy([1, 0, 0], prediction)).toBeCloseTo(
      -Math.log(prediction[0] ?? 0),
    );
    expect(entropyBits([0.5, 0.5])).toBeCloseTo(1);
  });

  it("renormalizes top-k and top-p distributions", () => {
    expect(topKDistribution([0.5, 0.3, 0.15, 0.05], 2)).toEqual([
      0.625, 0.37499999999999994, 0, 0,
    ]);
    expect(topPDistribution([0.5, 0.3, 0.15, 0.05], 0.8)).toEqual([
      0.625, 0.37499999999999994, 0, 0,
    ]);
  });

  it("computes sequential returns, policies, and recursive waiting times", () => {
    expect(discountedReturn([2, 4, 8], 0.5)).toBe(6);
    expect(trajectoryProbability([0.8, 0.5, 0.25])).toBeCloseTo(0.1);
    expect(epsilonGreedyDistribution(4, 1, 0.2)).toEqual([
      0.05, 0.8500000000000001, 0.05, 0.05,
    ]);
    expect(expectedFlipsForConsecutiveHeads(1)).toBe(2);
    expect(expectedFlipsForConsecutiveHeads(2)).toBe(6);
    expect(expectedFlipsForConsecutiveHeads(3)).toBe(14);
  });
});

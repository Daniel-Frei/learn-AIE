import { describe, expect, it } from "vitest";
import {
  getBudgetEstimate,
  getCombination,
  getConsistencyWinner,
  getGroupRelativeAdvantages,
  getIndependentPassAtK,
  getLengthPenaltyRatio,
  getSampledPassAtK,
} from "@/components/learning/pages/cme295-lecture6/reasoningMath";

describe("CME295 Lecture 6 reasoning math helpers", () => {
  it("computes independent Pass@k from a single-attempt pass rate", () => {
    expect(getIndependentPassAtK(0.4, 1)).toBeCloseTo(0.4, 5);
    expect(getIndependentPassAtK(0.4, 3)).toBeCloseTo(0.784, 3);
    expect(getIndependentPassAtK(0.4, 0)).toBe(0);
    expect(getIndependentPassAtK(2, 2)).toBe(1);
  });

  it("computes combinations and sampled Pass@k from n generated samples", () => {
    expect(getCombination(10, 4)).toBe(210);
    expect(getCombination(6, 4)).toBe(15);
    expect(getCombination(3, 4)).toBe(0);
    expect(
      getSampledPassAtK({
        totalSamples: 10,
        correctSamples: 4,
        attempts: 4,
      }),
    ).toBeCloseTo(0.92857, 5);
    expect(
      getSampledPassAtK({
        totalSamples: 10,
        correctSamples: 4,
        attempts: 8,
      }),
    ).toBe(1);
    expect(
      getSampledPassAtK({
        totalSamples: 10,
        correctSamples: 0,
        attempts: 4,
      }),
    ).toBe(0);
  });

  it("finds the consensus winner and reports ties", () => {
    expect(getConsistencyWinner(["12", "14", "14", "14", "12"])).toEqual({
      answer: "14",
      votes: 3,
      tied: false,
    });
    expect(getConsistencyWinner(["B", "A"])).toEqual({
      answer: "A",
      votes: 1,
      tied: true,
    });
    expect(getConsistencyWinner([])).toBeNull();
  });

  it("estimates thinking-budget quality with useful gain and overthinking penalty", () => {
    const profile = {
      baseQuality: 0.25,
      maxGain: 0.55,
      saturationTokens: 800,
      usefulTokenLimit: 1600,
      overthinkPenaltyPerToken: 0.00005,
    };

    const mediumBudget = getBudgetEstimate(profile, 800);
    const excessiveBudget = getBudgetEstimate(profile, 2600);

    expect(mediumBudget.quality).toBeGreaterThan(0.25);
    expect(mediumBudget.overthinkPenalty).toBe(0);
    expect(excessiveBudget.overthinkPenalty).toBeCloseTo(0.05, 5);
    expect(excessiveBudget.latencyMultiplier).toBeCloseTo(3.6, 5);
  });

  it("computes group-relative advantages from rewards", () => {
    const stats = getGroupRelativeAdvantages([1, 0, 0.5, -0.5]);

    expect(stats.averageReward).toBeCloseTo(0.25, 5);
    expect(stats.advantages).toEqual([0.75, -0.25, 0.25, -0.75]);
    expect(getGroupRelativeAdvantages([])).toEqual({
      averageReward: 0,
      advantages: [],
    });
  });

  it("compares token contribution under per-output length normalization", () => {
    expect(getLengthPenaltyRatio(40, 200)).toBeCloseTo(5, 5);
    expect(getLengthPenaltyRatio(0, 200)).toBe(0);
  });
});

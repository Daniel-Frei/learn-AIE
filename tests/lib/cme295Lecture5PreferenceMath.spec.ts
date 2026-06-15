import { describe, expect, it } from "vitest";
import {
  getBestOfNStats,
  getBradleyTerryStats,
  getDpoLogitStats,
  getPpoClipStats,
  sigmoid,
} from "@/components/learning/pages/cme295-lecture5/preferenceMath";

describe("CME295 Lecture 5 preference tuning math", () => {
  it("computes sigmoid and Bradley-Terry reward-gap statistics", () => {
    expect(sigmoid(0)).toBeCloseTo(0.5);

    const tied = getBradleyTerryStats(0, 0);
    expect(tied.gap).toBe(0);
    expect(tied.probability).toBeCloseTo(0.5);
    expect(tied.loss).toBeCloseTo(Math.log(2));

    const clear = getBradleyTerryStats(1.8, -0.4);
    expect(clear.gap).toBeCloseTo(2.2);
    expect(clear.probability).toBeCloseTo(1 / (1 + Math.exp(-2.2)));
  });

  it("computes PPO clipping for positive and negative advantages", () => {
    const positive = getPpoClipStats({
      currentProbability: 0.3,
      oldProbability: 0.2,
      advantage: 2,
      epsilon: 0.2,
    });

    expect(positive.ratio).toBeCloseTo(1.5);
    expect(positive.clippedRatio).toBeCloseTo(1.2);
    expect(positive.unclippedTerm).toBeCloseTo(3);
    expect(positive.clippedTerm).toBeCloseTo(2.4);
    expect(positive.objectiveTerm).toBeCloseTo(2.4);
    expect(positive.isClipped).toBe(true);

    const negative = getPpoClipStats({
      currentProbability: 0.1,
      oldProbability: 0.2,
      advantage: -2,
      epsilon: 0.2,
    });

    expect(negative.ratio).toBeCloseTo(0.5);
    expect(negative.clippedRatio).toBeCloseTo(0.8);
    expect(negative.unclippedTerm).toBeCloseTo(-1);
    expect(negative.clippedTerm).toBeCloseTo(-1.6);
    expect(negative.objectiveTerm).toBeCloseTo(-1.6);
    expect(negative.isClipped).toBe(true);
  });

  it("computes Best-of-N success probability and generation multiplier", () => {
    const stats = getBestOfNStats({
      samples: 5,
      acceptableProbability: 0.3,
    });

    expect(stats.allFailProbability).toBeCloseTo(0.7 ** 5);
    expect(stats.atLeastOneAcceptable).toBeCloseTo(1 - 0.7 ** 5);
    expect(stats.generationMultiplier).toBe(5);
  });

  it("computes DPO logit contrasts against a frozen reference", () => {
    const stats = getDpoLogitStats({
      policyChosenLogProb: -5,
      policyRejectedLogProb: -7,
      referenceChosenLogProb: -6,
      referenceRejectedLogProb: -6.5,
      beta: 0.1,
    });

    expect(stats.policyGap).toBeCloseTo(2);
    expect(stats.referenceGap).toBeCloseTo(0.5);
    expect(stats.contrast).toBeCloseTo(1.5);
    expect(stats.logit).toBeCloseTo(0.15);
    expect(stats.probability).toBeCloseTo(1 / (1 + Math.exp(-0.15)));
  });
});

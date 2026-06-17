import { describe, expect, it } from "vitest";
import {
  getChanceAgreement,
  getCohensKappa,
  getPassAtK,
  getPassHatK,
  getParetoFrontier,
  getUnigramOverlapStats,
  getWeightedFactualityScore,
  isDominated,
  type FactualityClaim,
  type ParetoModel,
} from "@/components/learning/pages/cme295-lecture8/evaluationMath";

describe("CME295 Lecture 8 evaluation helpers", () => {
  it("computes chance agreement and Cohen kappa from rater marginals", () => {
    const inputs = {
      observedAgreement: 0.78,
      raterAPositiveRate: 0.8,
      raterBPositiveRate: 0.7,
    };

    expect(getChanceAgreement(inputs)).toBeCloseTo(0.62, 5);
    expect(getCohensKappa(inputs)).toBeCloseTo((0.78 - 0.62) / (1 - 0.62), 5);
  });

  it("counts bounded unigram overlap for precision and recall", () => {
    const stats = getUnigramOverlapStats({
      reference: "a plush teddy bear can comfort a child during bedtime",
      candidate: "soft stuffed bears help kids feel safe as they fall asleep",
    });

    expect(stats.referenceUnigrams).toBe(10);
    expect(stats.candidateUnigrams).toBe(11);
    expect(stats.matchedUnigrams).toBe(0);
    expect(stats.precision).toBe(0);
    expect(stats.recall).toBe(0);
  });

  it("aggregates weighted factuality claims with an explicit unverifiable policy", () => {
    const claims: FactualityClaim[] = [
      { weight: 0.35, status: "supported" },
      { weight: 0.25, status: "contradicted" },
      { weight: 0.2, status: "unverifiable" },
      { weight: 0.2, status: "supported" },
    ];

    expect(getWeightedFactualityScore(claims)).toBeCloseTo(0.55, 5);
    expect(getWeightedFactualityScore(claims, 0.5)).toBeCloseTo(0.65, 5);
  });

  it("distinguishes at-least-one success from repeated reliability", () => {
    expect(getPassAtK(0.8, 3)).toBeCloseTo(0.992, 5);
    expect(getPassHatK(0.8, 3)).toBeCloseTo(0.512, 5);
    expect(getPassAtK(0.8, 0)).toBe(0);
    expect(getPassHatK(0.8, 0)).toBe(0);
  });

  it("finds dominated models for a multi-objective Pareto frontier", () => {
    const models: ParetoModel[] = [
      { id: "A", quality: 90, cost: 20, latency: 800, safety: 96 },
      { id: "B", quality: 88, cost: 12, latency: 500, safety: 94 },
      { id: "C", quality: 86, cost: 12, latency: 600, safety: 91 },
      { id: "D", quality: 82, cost: 6, latency: 350, safety: 93 },
      { id: "E", quality: 88, cost: 14, latency: 450, safety: 97 },
    ];

    expect(isDominated(models[2], models[1])).toBe(true);
    expect(isDominated(models[4], models[1])).toBe(false);
    expect(
      getParetoFrontier(models)
        .map((model) => model.id)
        .sort(),
    ).toEqual(["A", "B", "D", "E"]);
  });
});

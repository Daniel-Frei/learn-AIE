import { describe, expect, it } from "vitest";
import {
  getDiscountedCumulativeGain,
  getHybridRetrievalScore,
  getNdcgAtK,
  getReciprocalRankAtK,
  getResidualRiskScore,
  rankRetrievalCandidates,
  type RetrievalCandidate,
} from "@/components/learning/pages/cme295-lecture7/systemsMath";

const candidates: RetrievalCandidate[] = [
  {
    id: "semantic-neighbor",
    title: "nearest semantic neighbor",
    semanticScore: 0.91,
    keywordScore: 0.2,
    relevance: 1,
  },
  {
    id: "exact-id",
    title: "exact identifier match",
    semanticScore: 0.58,
    keywordScore: 0.94,
    relevance: 3,
  },
  {
    id: "hybrid",
    title: "hybrid best match",
    semanticScore: 0.84,
    keywordScore: 0.81,
    relevance: 3,
  },
];

describe("CME295 Lecture 7 systems helpers", () => {
  it("combines semantic and keyword retrieval scores with a bounded weight", () => {
    expect(
      getHybridRetrievalScore({
        semanticScore: 0.8,
        keywordScore: 0.4,
        semanticWeight: 0.75,
      }),
    ).toBeCloseTo(0.7, 5);
    expect(
      getHybridRetrievalScore({
        semanticScore: 0.8,
        keywordScore: 0.4,
        semanticWeight: 2,
      }),
    ).toBeCloseTo(0.8, 5);
  });

  it("ranks candidates according to the selected retrieval method", () => {
    expect(rankRetrievalCandidates(candidates, "semantic")[0]?.id).toBe(
      "semantic-neighbor",
    );
    expect(rankRetrievalCandidates(candidates, "keyword")[0]?.id).toBe(
      "exact-id",
    );
    expect(rankRetrievalCandidates(candidates, "hybrid")[0]?.id).toBe("hybrid");
  });

  it("computes NDCG@k and reciprocal rank from graded relevance", () => {
    const ranking = [0, 3, 1, 2];

    expect(getDiscountedCumulativeGain(ranking, 2)).toBeCloseTo(
      0 + 7 / Math.log2(3),
      5,
    );
    expect(getNdcgAtK(ranking, 4)).toBeGreaterThan(0.6);
    expect(getNdcgAtK([0, 0], 2)).toBe(0);
    expect(getReciprocalRankAtK(ranking, 4)).toBeCloseTo(0.5, 5);
    expect(getReciprocalRankAtK([0, 0, 2], 2)).toBe(0);
  });

  it("reduces residual safety risk with selected safeguards", () => {
    expect(getResidualRiskScore(9, [])).toBe(9);
    expect(
      getResidualRiskScore(9, ["scopedTools", "humanApproval", "auditLog"]),
    ).toBe(3);
    expect(
      getResidualRiskScore(4, ["scopedTools", "humanApproval", "egressFilter"]),
    ).toBe(0);
  });
});

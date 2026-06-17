export type RetrievalMethod = "semantic" | "keyword" | "hybrid";

export type RetrievalCandidate = {
  id: string;
  title: string;
  semanticScore: number;
  keywordScore: number;
  relevance: number;
};

export type RankedRetrievalCandidate = RetrievalCandidate & {
  score: number;
};

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function getHybridRetrievalScore({
  semanticScore,
  keywordScore,
  semanticWeight = 0.55,
}: {
  semanticScore: number;
  keywordScore: number;
  semanticWeight?: number;
}): number {
  const weight = clamp(semanticWeight, 0, 1);
  return semanticScore * weight + keywordScore * (1 - weight);
}

export function rankRetrievalCandidates(
  candidates: readonly RetrievalCandidate[],
  method: RetrievalMethod,
): RankedRetrievalCandidate[] {
  return candidates
    .map((candidate) => {
      const score =
        method === "semantic"
          ? candidate.semanticScore
          : method === "keyword"
            ? candidate.keywordScore
            : getHybridRetrievalScore(candidate);

      return { ...candidate, score };
    })
    .sort((first, second) => {
      const scoreDiff = second.score - first.score;
      return scoreDiff === 0 ? second.relevance - first.relevance : scoreDiff;
    });
}

export function getDiscountedCumulativeGain(
  relevances: readonly number[],
  k: number,
): number {
  return relevances
    .slice(0, Math.max(0, k))
    .reduce((total, relevance, index) => {
      const gain = 2 ** Math.max(0, relevance) - 1;
      return total + gain / Math.log2(index + 2);
    }, 0);
}

export function getNdcgAtK(relevances: readonly number[], k: number): number {
  const dcg = getDiscountedCumulativeGain(relevances, k);
  const ideal = getDiscountedCumulativeGain(
    [...relevances].sort((first, second) => second - first),
    k,
  );

  if (ideal === 0) return 0;
  return dcg / ideal;
}

export function getReciprocalRankAtK(
  relevances: readonly number[],
  k: number,
): number {
  const firstRelevantIndex = relevances
    .slice(0, Math.max(0, k))
    .findIndex((relevance) => relevance > 0);

  return firstRelevantIndex === -1 ? 0 : 1 / (firstRelevantIndex + 1);
}

export type SafetyControl =
  | "scopedTools"
  | "humanApproval"
  | "egressFilter"
  | "budgetLimit"
  | "auditLog";

const CONTROL_RISK_REDUCTION: Record<SafetyControl, number> = {
  scopedTools: 2,
  humanApproval: 3,
  egressFilter: 3,
  budgetLimit: 1,
  auditLog: 1,
};

export function getResidualRiskScore(
  baseRisk: number,
  selectedControls: readonly SafetyControl[],
): number {
  const reduction = selectedControls.reduce(
    (total, control) => total + CONTROL_RISK_REDUCTION[control],
    0,
  );

  return clamp(baseRisk - reduction, 0, 10);
}

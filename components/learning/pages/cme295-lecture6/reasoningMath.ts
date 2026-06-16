export type BudgetProfile = {
  baseQuality: number;
  maxGain: number;
  saturationTokens: number;
  usefulTokenLimit: number;
  overthinkPenaltyPerToken: number;
};

export type BudgetEstimate = {
  quality: number;
  usefulGain: number;
  overthinkPenalty: number;
  latencyMultiplier: number;
};

export function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function getIndependentPassAtK(
  singleAttemptPassRate: number,
  attempts: number,
): number {
  if (attempts <= 0) return 0;

  const clampedPassRate = clamp01(singleAttemptPassRate);
  return clamp01(1 - (1 - clampedPassRate) ** attempts);
}

export function getCombination(total: number, choose: number): number {
  if (choose < 0 || total < 0 || choose > total) return 0;
  if (choose === 0 || choose === total) return 1;

  const reducedChoose = Math.min(choose, total - choose);
  let result = 1;

  for (let index = 1; index <= reducedChoose; index += 1) {
    result = (result * (total - reducedChoose + index)) / index;
  }

  return result;
}

export function getSampledPassAtK({
  totalSamples,
  correctSamples,
  attempts,
}: {
  totalSamples: number;
  correctSamples: number;
  attempts: number;
}): number {
  if (attempts <= 0 || totalSamples <= 0 || correctSamples <= 0) return 0;

  const clampedCorrect = Math.min(correctSamples, totalSamples);
  const clampedAttempts = Math.min(attempts, totalSamples);
  const failedSamples = totalSamples - clampedCorrect;
  const allFailedCombinations = getCombination(failedSamples, clampedAttempts);
  const allCombinations = getCombination(totalSamples, clampedAttempts);

  if (allCombinations === 0) return 0;

  return clamp01(1 - allFailedCombinations / allCombinations);
}

export function getConsistencyWinner(
  answers: readonly string[],
): { answer: string; votes: number; tied: boolean } | null {
  if (answers.length === 0) return null;

  const counts = new Map<string, number>();
  for (const answer of answers) {
    counts.set(answer, (counts.get(answer) ?? 0) + 1);
  }

  const sorted = [...counts.entries()].sort((first, second) => {
    const voteDiff = second[1] - first[1];
    return voteDiff === 0 ? first[0].localeCompare(second[0]) : voteDiff;
  });
  const [answer, votes] = sorted[0];
  const tied = sorted.length > 1 && sorted[1][1] === votes;

  return { answer, votes, tied };
}

export function getBudgetEstimate(
  profile: BudgetProfile,
  thinkingTokens: number,
): BudgetEstimate {
  const tokens = Math.max(0, thinkingTokens);
  const usefulGain =
    profile.maxGain * (1 - Math.exp(-tokens / profile.saturationTokens));
  const excessTokens = Math.max(0, tokens - profile.usefulTokenLimit);
  const overthinkPenalty = excessTokens * profile.overthinkPenaltyPerToken;
  const quality = clamp01(profile.baseQuality + usefulGain - overthinkPenalty);

  return {
    quality,
    usefulGain,
    overthinkPenalty,
    latencyMultiplier: 1 + tokens / 1000,
  };
}

export function getGroupRelativeAdvantages(rewards: readonly number[]): {
  averageReward: number;
  advantages: number[];
} {
  if (rewards.length === 0) {
    return { averageReward: 0, advantages: [] };
  }

  const averageReward =
    rewards.reduce((total, reward) => total + reward, 0) / rewards.length;

  return {
    averageReward,
    advantages: rewards.map((reward) => reward - averageReward),
  };
}

export function getLengthPenaltyRatio(
  shortOutputTokens: number,
  longOutputTokens: number,
): number {
  if (shortOutputTokens <= 0 || longOutputTokens <= 0) return 0;

  return 1 / shortOutputTokens / (1 / longOutputTokens);
}

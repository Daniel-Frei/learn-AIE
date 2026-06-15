export function sigmoid(value: number): number {
  return 1 / (1 + Math.exp(-value));
}

export function getBradleyTerryStats(
  winnerReward: number,
  loserReward: number,
) {
  const gap = winnerReward - loserReward;
  const probability = sigmoid(gap);

  return {
    gap,
    probability,
    loss: -Math.log(probability),
  };
}

export function getPpoClipStats({
  currentProbability,
  oldProbability,
  advantage,
  epsilon,
}: {
  currentProbability: number;
  oldProbability: number;
  advantage: number;
  epsilon: number;
}) {
  if (oldProbability <= 0) {
    throw new Error("oldProbability must be greater than zero.");
  }

  const ratio = currentProbability / oldProbability;
  const lower = 1 - epsilon;
  const upper = 1 + epsilon;
  const clippedRatio = Math.min(Math.max(ratio, lower), upper);
  const unclippedTerm = ratio * advantage;
  const clippedTerm = clippedRatio * advantage;
  const objectiveTerm = Math.min(unclippedTerm, clippedTerm);

  return {
    ratio,
    lower,
    upper,
    clippedRatio,
    unclippedTerm,
    clippedTerm,
    objectiveTerm,
    isClipped: objectiveTerm !== unclippedTerm,
  };
}

export function getBestOfNStats({
  samples,
  acceptableProbability,
}: {
  samples: number;
  acceptableProbability: number;
}) {
  const allFailProbability = (1 - acceptableProbability) ** samples;
  const atLeastOneAcceptable = 1 - allFailProbability;

  return {
    allFailProbability,
    atLeastOneAcceptable,
    generationMultiplier: samples,
  };
}

export function getDpoLogitStats({
  policyChosenLogProb,
  policyRejectedLogProb,
  referenceChosenLogProb,
  referenceRejectedLogProb,
  beta,
}: {
  policyChosenLogProb: number;
  policyRejectedLogProb: number;
  referenceChosenLogProb: number;
  referenceRejectedLogProb: number;
  beta: number;
}) {
  const policyGap = policyChosenLogProb - policyRejectedLogProb;
  const referenceGap = referenceChosenLogProb - referenceRejectedLogProb;
  const contrast = policyGap - referenceGap;
  const logit = beta * contrast;
  const probability = sigmoid(logit);

  return {
    policyGap,
    referenceGap,
    contrast,
    logit,
    probability,
    loss: -Math.log(probability),
  };
}

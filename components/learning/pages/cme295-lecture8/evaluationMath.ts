export type AgreementInputs = {
  observedAgreement: number;
  raterAPositiveRate: number;
  raterBPositiveRate: number;
};

export type UnigramOverlapStats = {
  matchedUnigrams: number;
  candidateUnigrams: number;
  referenceUnigrams: number;
  precision: number;
  recall: number;
};

export type ClaimStatus = "supported" | "contradicted" | "unverifiable";

export type FactualityClaim = {
  weight: number;
  status: ClaimStatus;
};

export type ParetoModel = {
  id: string;
  quality: number;
  cost: number;
  latency: number;
  safety: number;
};

export function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function getChanceAgreement({
  raterAPositiveRate,
  raterBPositiveRate,
}: Pick<AgreementInputs, "raterAPositiveRate" | "raterBPositiveRate">): number {
  const aPositive = clamp01(raterAPositiveRate);
  const bPositive = clamp01(raterBPositiveRate);

  return aPositive * bPositive + (1 - aPositive) * (1 - bPositive);
}

export function getCohensKappa(inputs: AgreementInputs): number {
  const observedAgreement = clamp01(inputs.observedAgreement);
  const chanceAgreement = getChanceAgreement(inputs);

  if (chanceAgreement >= 1) return observedAgreement === 1 ? 1 : 0;

  return (observedAgreement - chanceAgreement) / (1 - chanceAgreement);
}

function tokenizeUnigrams(text: string): string[] {
  return text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
}

export function getUnigramOverlapStats({
  candidate,
  reference,
}: {
  candidate: string;
  reference: string;
}): UnigramOverlapStats {
  const candidateTokens = tokenizeUnigrams(candidate);
  const referenceTokens = tokenizeUnigrams(reference);
  const referenceCounts = new Map<string, number>();

  for (const token of referenceTokens) {
    referenceCounts.set(token, (referenceCounts.get(token) ?? 0) + 1);
  }

  let matchedUnigrams = 0;
  for (const token of candidateTokens) {
    const count = referenceCounts.get(token) ?? 0;
    if (count <= 0) continue;
    matchedUnigrams += 1;
    referenceCounts.set(token, count - 1);
  }

  return {
    matchedUnigrams,
    candidateUnigrams: candidateTokens.length,
    referenceUnigrams: referenceTokens.length,
    precision:
      candidateTokens.length === 0
        ? 0
        : matchedUnigrams / candidateTokens.length,
    recall:
      referenceTokens.length === 0
        ? 0
        : matchedUnigrams / referenceTokens.length,
  };
}

export function getWeightedFactualityScore(
  claims: readonly FactualityClaim[],
  unverifiableCredit = 0,
): number {
  const totalWeight = claims.reduce((total, claim) => total + claim.weight, 0);
  if (totalWeight <= 0) return 0;

  const supportedWeight = claims.reduce((total, claim) => {
    if (claim.status === "supported") return total + claim.weight;
    if (claim.status === "unverifiable") {
      return total + claim.weight * clamp01(unverifiableCredit);
    }
    return total;
  }, 0);

  return supportedWeight / totalWeight;
}

export function getPassAtK(
  successProbability: number,
  attempts: number,
): number {
  if (attempts <= 0) return 0;
  const probability = clamp01(successProbability);
  return clamp01(1 - (1 - probability) ** attempts);
}

export function getPassHatK(
  successProbability: number,
  attempts: number,
): number {
  if (attempts <= 0) return 0;
  return clamp01(successProbability) ** attempts;
}

export function isDominated(
  model: ParetoModel,
  candidate: ParetoModel,
): boolean {
  const atLeastAsGood =
    candidate.quality >= model.quality &&
    candidate.cost <= model.cost &&
    candidate.latency <= model.latency &&
    candidate.safety >= model.safety;
  const strictlyBetter =
    candidate.quality > model.quality ||
    candidate.cost < model.cost ||
    candidate.latency < model.latency ||
    candidate.safety > model.safety;

  return atLeastAsGood && strictlyBetter;
}

export function getParetoFrontier(
  models: readonly ParetoModel[],
): ParetoModel[] {
  return models.filter(
    (model) =>
      !models.some(
        (candidate) =>
          candidate.id !== model.id && isDominated(model, candidate),
      ),
  );
}

export type WeightedOutcome = {
  value: number;
  probability: number;
};

export type TotalProbabilityPath = {
  prior: number;
  conditional: number;
};

export type NaturalFrequencyCounts = {
  population: number;
  condition: number;
  noCondition: number;
  truePositive: number;
  falseNegative: number;
  trueNegative: number;
  falsePositive: number;
  positiveTests: number;
  posterior: number;
};

function requireFinite(value: number, label: string) {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${label} must be finite.`);
  }
}

function requireProbability(value: number, label: string) {
  requireFinite(value, label);
  if (value < 0 || value > 1) {
    throw new RangeError(`${label} must be between 0 and 1.`);
  }
}

export function clampProbability(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function normalizeWeights(weights: readonly number[]): number[] {
  if (weights.length === 0) return [];

  weights.forEach((weight) => {
    requireFinite(weight, "Weight");
    if (weight < 0) throw new RangeError("Weights cannot be negative.");
  });

  const total = weights.reduce((sum, weight) => sum + weight, 0);
  if (total <= 0) {
    throw new RangeError("At least one weight must be positive.");
  }

  return weights.map((weight) => weight / total);
}

export function probabilityToOdds(probability: number): number {
  requireProbability(probability, "Probability");
  if (probability === 1) return Number.POSITIVE_INFINITY;
  return probability / (1 - probability);
}

export function oddsToProbability(odds: number): number {
  requireFinite(odds, "Odds");
  if (odds < 0) throw new RangeError("Odds cannot be negative.");
  return odds / (1 + odds);
}

export function factorial(value: number): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError("Factorial requires a non-negative integer.");
  }

  let result = 1;
  for (let current = 2; current <= value; current += 1) {
    result *= current;
  }
  return result;
}

export function permutations(population: number, selections: number): number {
  if (
    !Number.isInteger(population) ||
    !Number.isInteger(selections) ||
    population < 0 ||
    selections < 0 ||
    selections > population
  ) {
    throw new RangeError("Permutation inputs must satisfy 0 ≤ k ≤ n.");
  }

  let result = 1;
  for (let offset = 0; offset < selections; offset += 1) {
    result *= population - offset;
  }
  return result;
}

export function combinations(population: number, selections: number): number {
  if (
    !Number.isInteger(population) ||
    !Number.isInteger(selections) ||
    population < 0 ||
    selections < 0 ||
    selections > population
  ) {
    throw new RangeError("Combination inputs must satisfy 0 ≤ k ≤ n.");
  }

  const smallerSelection = Math.min(selections, population - selections);
  let result = 1;
  for (let index = 1; index <= smallerSelection; index += 1) {
    result = (result * (population - smallerSelection + index)) / index;
  }
  return Math.round(result);
}

export function expectedValue(outcomes: readonly WeightedOutcome[]): number {
  return outcomes.reduce(
    (sum, outcome) => sum + outcome.value * outcome.probability,
    0,
  );
}

export function variance(outcomes: readonly WeightedOutcome[]): number {
  const mean = expectedValue(outcomes);
  return outcomes.reduce(
    (sum, outcome) => sum + outcome.probability * (outcome.value - mean) ** 2,
    0,
  );
}

export function unionProbability(
  probabilityA: number,
  probabilityB: number,
  intersection: number,
): number {
  requireProbability(probabilityA, "P(A)");
  requireProbability(probabilityB, "P(B)");
  requireProbability(intersection, "P(A ∩ B)");
  return probabilityA + probabilityB - intersection;
}

export function conditionalProbability(
  intersection: number,
  condition: number,
): number {
  requireProbability(intersection, "Intersection probability");
  requireProbability(condition, "Condition probability");
  if (condition === 0) {
    throw new RangeError("Conditional probability requires P(B) > 0.");
  }
  return intersection / condition;
}

export function totalProbability(
  paths: readonly TotalProbabilityPath[],
): number {
  return paths.reduce((sum, path) => {
    requireProbability(path.prior, "Path prior");
    requireProbability(path.conditional, "Path conditional probability");
    return sum + path.prior * path.conditional;
  }, 0);
}

export function bayesPosterior({
  prior,
  likelihood,
  falsePositiveRate,
}: {
  prior: number;
  likelihood: number;
  falsePositiveRate: number;
}): number {
  requireProbability(prior, "Prior");
  requireProbability(likelihood, "Likelihood");
  requireProbability(falsePositiveRate, "False-positive rate");

  const evidence = likelihood * prior + falsePositiveRate * (1 - prior);
  return evidence === 0 ? 0 : (likelihood * prior) / evidence;
}

export function naturalFrequencyCounts({
  population,
  prevalence,
  sensitivity,
  specificity,
}: {
  population: number;
  prevalence: number;
  sensitivity: number;
  specificity: number;
}): NaturalFrequencyCounts {
  if (!Number.isInteger(population) || population <= 0) {
    throw new RangeError("Population must be a positive integer.");
  }
  requireProbability(prevalence, "Prevalence");
  requireProbability(sensitivity, "Sensitivity");
  requireProbability(specificity, "Specificity");

  const condition = Math.round(population * prevalence);
  const noCondition = population - condition;
  const truePositive = Math.round(condition * sensitivity);
  const falseNegative = condition - truePositive;
  const trueNegative = Math.round(noCondition * specificity);
  const falsePositive = noCondition - trueNegative;
  const positiveTests = truePositive + falsePositive;

  return {
    population,
    condition,
    noCondition,
    truePositive,
    falseNegative,
    trueNegative,
    falsePositive,
    positiveTests,
    posterior: positiveTests === 0 ? 0 : truePositive / positiveTests,
  };
}

export function softmax(logits: readonly number[], temperature = 1): number[] {
  if (logits.length === 0) return [];
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError("Temperature must be positive and finite.");
  }

  const scaled = logits.map((logit) => logit / temperature);
  const maximum = Math.max(...scaled);
  const exponentials = scaled.map((value) => Math.exp(value - maximum));
  return normalizeWeights(exponentials);
}

export function entropyBits(probabilities: readonly number[]): number {
  return probabilities.reduce((sum, probability) => {
    requireProbability(probability, "Probability");
    return probability === 0 ? sum : sum - probability * Math.log2(probability);
  }, 0);
}

export function negativeLogLikelihood(probability: number): number {
  requireProbability(probability, "Observed probability");
  return probability === 0 ? Number.POSITIVE_INFINITY : -Math.log(probability);
}

export function crossEntropy(
  target: readonly number[],
  prediction: readonly number[],
): number {
  if (target.length !== prediction.length || target.length === 0) {
    throw new RangeError("Target and prediction must have the same length.");
  }
  return target.reduce((sum, targetProbability, index) => {
    requireProbability(targetProbability, "Target probability");
    const predictedProbability = prediction[index] ?? 0;
    requireProbability(predictedProbability, "Predicted probability");
    if (targetProbability === 0) return sum;
    if (predictedProbability === 0) return Number.POSITIVE_INFINITY;
    return sum - targetProbability * Math.log(predictedProbability);
  }, 0);
}

export function topKDistribution(
  probabilities: readonly number[],
  count: number,
): number[] {
  if (!Number.isInteger(count) || count < 1 || count > probabilities.length) {
    throw new RangeError("Top-k count must be within the distribution.");
  }
  const retained = new Set(
    probabilities
      .map((probability, index) => ({ probability, index }))
      .sort((first, second) => second.probability - first.probability)
      .slice(0, count)
      .map((item) => item.index),
  );
  return normalizeWeights(
    probabilities.map((probability, index) =>
      retained.has(index) ? probability : 0,
    ),
  );
}

export function topPDistribution(
  probabilities: readonly number[],
  threshold: number,
): number[] {
  if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) {
    throw new RangeError("Top-p threshold must be in (0, 1].");
  }

  const ranked = probabilities
    .map((probability, index) => ({ probability, index }))
    .sort((first, second) => second.probability - first.probability);
  const retained = new Set<number>();
  let cumulative = 0;
  for (const item of ranked) {
    retained.add(item.index);
    cumulative += item.probability;
    if (cumulative >= threshold) break;
  }

  return normalizeWeights(
    probabilities.map((probability, index) =>
      retained.has(index) ? probability : 0,
    ),
  );
}

export function discountedReturn(
  rewards: readonly number[],
  discount: number,
): number {
  requireProbability(discount, "Discount factor");
  return rewards.reduce(
    (sum, reward, index) => sum + reward * discount ** index,
    0,
  );
}

export function trajectoryProbability(
  stepProbabilities: readonly number[],
): number {
  return stepProbabilities.reduce((product, probability) => {
    requireProbability(probability, "Step probability");
    return product * probability;
  }, 1);
}

export function epsilonGreedyDistribution(
  actionCount: number,
  greedyIndex: number,
  epsilon: number,
): number[] {
  if (
    !Number.isInteger(actionCount) ||
    actionCount < 1 ||
    !Number.isInteger(greedyIndex) ||
    greedyIndex < 0 ||
    greedyIndex >= actionCount
  ) {
    throw new RangeError("Epsilon-greedy action inputs are invalid.");
  }
  requireProbability(epsilon, "Epsilon");

  const explorationShare = epsilon / actionCount;
  return Array.from({ length: actionCount }, (_, index) =>
    index === greedyIndex ? 1 - epsilon + explorationShare : explorationShare,
  );
}

export function expectedFlipsForConsecutiveHeads(runLength: number): number {
  if (!Number.isInteger(runLength) || runLength < 1) {
    throw new RangeError("Run length must be a positive integer.");
  }
  return 2 ** (runLength + 1) - 2;
}

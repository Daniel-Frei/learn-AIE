// lib/difficultyStore.ts
"use client";

import type { Difficulty } from "./quiz";

export type LegacyQuestionStats = {
  correct: number;
  wrong: number;
};

export type LegacyDifficultyMap = Record<string, LegacyQuestionStats>;

export type RatingEntity = {
  rating: number;
  rd: number;
  sigma: number;
  lastUpdatedAt: number;
  gamesPlayed: number;
};

export type QuestionRating = RatingEntity & {
  legacyCorrect: number;
  legacyWrong: number;
  label?: Difficulty;
};

export type RatingConfig = {
  defaultRating: number;
  defaultRdUser: number;
  defaultRdQuestion: number;
  defaultSigma: number;
  tau: number;
  epsilon: number;
  periodDays: number;
  minRd: number;
  maxRd: number;
  difficultyAnchorRating: number;
  difficultyScale: number;
};

export type RatingStateV2 = {
  version: 2;
  algorithm: "glicko-2";
  config: RatingConfig;
  user: RatingEntity;
  questions: Record<string, QuestionRating>;
};

export type QuestionMetadataMap = Record<string, { label?: Difficulty }>;

const LEGACY_STORAGE_KEY = "aie-quiz-question-stats-v1";
const STORAGE_KEY = "aie-quiz-ratings-v2";

const GLICKO2_SCALE = 173.7178;
const MAX_VOLATILITY_ITERATIONS = 100;
const MIN_PROBABILITY = 1e-12;

const LABEL_PRIOR_RATING: Record<Difficulty, number> = {
  easy: 1300,
  medium: 1500,
  hard: 1700,
};

const DEFAULT_CONFIG: RatingConfig = {
  defaultRating: 1500,
  defaultRdUser: 350,
  defaultRdQuestion: 300,
  defaultSigma: 0.06,
  tau: 0.5,
  epsilon: 1e-6,
  periodDays: 1,
  minRd: 30,
  maxRd: 350,
  difficultyAnchorRating: 1500,
  difficultyScale: 400,
};

type Glicko2Core = {
  mu: number;
  phi: number;
};

type ImportedV2Envelope = RatingStateV2 & {
  exportedAt?: string;
  legacyCounts?: LegacyDifficultyMap;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function isDifficulty(value: unknown): value is Difficulty {
  return value === "easy" || value === "medium" || value === "hard";
}

function nowMs(): number {
  return Date.now();
}

function sanitizeFiniteNumber(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  return fallback;
}

function sanitizeCount(value: unknown): number {
  const numeric = sanitizeFiniteNumber(value, 0);
  if (numeric <= 0) return 0;
  return Math.floor(numeric);
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function clampRd(rd: number, config: RatingConfig): number {
  return clamp(rd, config.minRd, config.maxRd);
}

function getLabelPriorRating(label: Difficulty | undefined): number {
  if (label && isDifficulty(label)) {
    return LABEL_PRIOR_RATING[label];
  }
  return LABEL_PRIOR_RATING.medium;
}

function createDefaultUser(
  timestamp: number,
  config: RatingConfig,
): RatingEntity {
  return {
    rating: config.defaultRating,
    rd: config.defaultRdUser,
    sigma: config.defaultSigma,
    lastUpdatedAt: timestamp,
    gamesPlayed: 0,
  };
}

function createDefaultQuestion(
  label: Difficulty | undefined,
  timestamp: number,
  config: RatingConfig,
): QuestionRating {
  return {
    rating: getLabelPriorRating(label),
    rd: config.defaultRdQuestion,
    sigma: config.defaultSigma,
    lastUpdatedAt: timestamp,
    gamesPlayed: 0,
    legacyCorrect: 0,
    legacyWrong: 0,
    label,
  };
}

function createInitialState(timestamp = nowMs()): RatingStateV2 {
  return {
    version: 2,
    algorithm: "glicko-2",
    config: { ...DEFAULT_CONFIG },
    user: createDefaultUser(timestamp, DEFAULT_CONFIG),
    questions: {},
  };
}

function sanitizeConfig(value: unknown): RatingConfig {
  if (!isRecord(value)) return { ...DEFAULT_CONFIG };

  const base = { ...DEFAULT_CONFIG };
  const minRd = sanitizeFiniteNumber(value.minRd, base.minRd);
  const maxRd = sanitizeFiniteNumber(value.maxRd, base.maxRd);
  const normalizedMinRd = Math.max(1, Math.min(minRd, maxRd));
  const normalizedMaxRd = Math.max(normalizedMinRd, maxRd);

  return {
    defaultRating: sanitizeFiniteNumber(
      value.defaultRating,
      base.defaultRating,
    ),
    defaultRdUser: sanitizeFiniteNumber(
      value.defaultRdUser,
      base.defaultRdUser,
    ),
    defaultRdQuestion: sanitizeFiniteNumber(
      value.defaultRdQuestion,
      base.defaultRdQuestion,
    ),
    defaultSigma: Math.max(
      1e-6,
      sanitizeFiniteNumber(value.defaultSigma, base.defaultSigma),
    ),
    tau: Math.max(1e-6, sanitizeFiniteNumber(value.tau, base.tau)),
    epsilon: Math.max(1e-12, sanitizeFiniteNumber(value.epsilon, base.epsilon)),
    periodDays: Math.max(
      1e-6,
      sanitizeFiniteNumber(value.periodDays, base.periodDays),
    ),
    minRd: normalizedMinRd,
    maxRd: normalizedMaxRd,
    difficultyAnchorRating: sanitizeFiniteNumber(
      value.difficultyAnchorRating,
      base.difficultyAnchorRating,
    ),
    difficultyScale: Math.max(
      1e-6,
      sanitizeFiniteNumber(value.difficultyScale, base.difficultyScale),
    ),
  };
}

function sanitizeEntity(
  value: unknown,
  defaults: RatingEntity,
  config: RatingConfig,
): RatingEntity {
  const src = isRecord(value) ? value : {};
  return {
    rating: sanitizeFiniteNumber(src.rating, defaults.rating),
    rd: clampRd(sanitizeFiniteNumber(src.rd, defaults.rd), config),
    sigma: Math.max(1e-6, sanitizeFiniteNumber(src.sigma, defaults.sigma)),
    lastUpdatedAt: Math.max(
      0,
      Math.floor(
        sanitizeFiniteNumber(src.lastUpdatedAt, defaults.lastUpdatedAt),
      ),
    ),
    gamesPlayed: sanitizeCount(src.gamesPlayed),
  };
}

function sanitizeQuestion(
  value: unknown,
  labelHint: Difficulty | undefined,
  timestamp: number,
  config: RatingConfig,
): QuestionRating {
  const defaults = createDefaultQuestion(labelHint, timestamp, config);
  const entity = sanitizeEntity(value, defaults, config);
  const src = isRecord(value) ? value : {};
  const label = isDifficulty(src.label) ? src.label : labelHint;
  return {
    ...entity,
    legacyCorrect: sanitizeCount(src.legacyCorrect),
    legacyWrong: sanitizeCount(src.legacyWrong),
    label,
  };
}

function sanitizeStateV2(
  value: unknown,
  questionMetadata: QuestionMetadataMap = {},
): RatingStateV2 | null {
  if (!isRecord(value)) return null;
  if (value.version !== 2 || value.algorithm !== "glicko-2") return null;

  const timestamp = nowMs();
  const config = sanitizeConfig(value.config);
  const user = sanitizeEntity(
    value.user,
    createDefaultUser(timestamp, config),
    config,
  );

  const questions: Record<string, QuestionRating> = {};
  if (isRecord(value.questions)) {
    for (const [questionId, rawQuestion] of Object.entries(value.questions)) {
      const hint = questionMetadata[questionId]?.label;
      questions[questionId] = sanitizeQuestion(
        rawQuestion,
        hint,
        timestamp,
        config,
      );
    }
  }

  return {
    version: 2,
    algorithm: "glicko-2",
    config,
    user,
    questions,
  };
}

function isLegacyStats(value: unknown): value is LegacyQuestionStats {
  if (!isRecord(value)) return false;
  return "correct" in value && "wrong" in value;
}

function isLegacyMap(value: unknown): value is LegacyDifficultyMap {
  if (!isRecord(value)) return false;
  const entries = Object.values(value);
  if (entries.length === 0) return true;
  return entries.every((entry) => isLegacyStats(entry));
}

function sanitizeLegacyMap(value: unknown): LegacyDifficultyMap | null {
  if (!isLegacyMap(value)) return null;
  const result: LegacyDifficultyMap = {};
  for (const [questionId, stats] of Object.entries(value)) {
    result[questionId] = {
      correct: sanitizeCount(stats.correct),
      wrong: sanitizeCount(stats.wrong),
    };
  }
  return result;
}

function toGlicko2Core(
  entity: RatingEntity,
  config: RatingConfig,
): Glicko2Core {
  return {
    mu: (entity.rating - config.defaultRating) / GLICKO2_SCALE,
    phi: clampRd(entity.rd, config) / GLICKO2_SCALE,
  };
}

function fromGlicko2Core(
  core: Glicko2Core,
  source: RatingEntity,
  sigma: number,
  config: RatingConfig,
): RatingEntity {
  return {
    rating: core.mu * GLICKO2_SCALE + config.defaultRating,
    rd: clampRd(core.phi * GLICKO2_SCALE, config),
    sigma: Math.max(1e-6, sigma),
    lastUpdatedAt: source.lastUpdatedAt,
    gamesPlayed: source.gamesPlayed,
  };
}

function g(phi: number): number {
  return 1 / Math.sqrt(1 + (3 * phi * phi) / (Math.PI * Math.PI));
}

function expectedScore(
  mu: number,
  muOpponent: number,
  phiOpponent: number,
): number {
  const impact = g(phiOpponent);
  return 1 / (1 + Math.exp(-impact * (mu - muOpponent)));
}

function computeSigmaPrime(
  phi: number,
  sigma: number,
  delta: number,
  v: number,
  config: RatingConfig,
): number {
  const tau = config.tau;
  const epsilon = config.epsilon;
  const a = Math.log(sigma * sigma);
  const deltaSq = delta * delta;
  const phiSq = phi * phi;

  const f = (x: number): number => {
    const expX = Math.exp(x);
    const numerator = expX * (deltaSq - phiSq - v - expX);
    const denominator = 2 * Math.pow(phiSq + v + expX, 2);
    return numerator / denominator - (x - a) / (tau * tau);
  };

  let A = a;
  let B: number;
  if (deltaSq > phiSq + v) {
    B = Math.log(deltaSq - phiSq - v);
  } else {
    let k = 1;
    while (k < MAX_VOLATILITY_ITERATIONS && f(a - k * tau) < 0) {
      k += 1;
    }
    B = a - k * tau;
  }

  let fA = f(A);
  let fB = f(B);
  let iterations = 0;
  while (Math.abs(B - A) > epsilon && iterations < MAX_VOLATILITY_ITERATIONS) {
    if (Math.abs(fB - fA) < MIN_PROBABILITY) break;

    const C = A + ((A - B) * fA) / (fB - fA);
    const fC = f(C);

    if (fC * fB < 0) {
      A = B;
      fA = fB;
    } else {
      fA /= 2;
    }
    B = C;
    fB = fC;
    iterations += 1;
  }

  return Math.exp(A / 2);
}

function inflateForInactivity<T extends RatingEntity>(
  entity: T,
  nowTimestamp: number,
  config: RatingConfig,
): T {
  if (nowTimestamp <= entity.lastUpdatedAt) return entity;

  const msPerPeriod = config.periodDays * 24 * 60 * 60 * 1000;
  if (!Number.isFinite(msPerPeriod) || msPerPeriod <= 0) return entity;

  const elapsedPeriods = (nowTimestamp - entity.lastUpdatedAt) / msPerPeriod;
  if (elapsedPeriods <= 0) return entity;

  const core = toGlicko2Core(entity, config);
  const inflatedPhi = Math.sqrt(
    core.phi * core.phi + elapsedPeriods * entity.sigma * entity.sigma,
  );

  return {
    ...entity,
    rd: clampRd(inflatedPhi * GLICKO2_SCALE, config),
    lastUpdatedAt: nowTimestamp,
  } as T;
}

function rateOneMatch(
  entity: RatingEntity,
  opponent: RatingEntity,
  score: number,
  config: RatingConfig,
): RatingEntity {
  const s = clamp(score, 0, 1);
  const player = toGlicko2Core(entity, config);
  const rival = toGlicko2Core(opponent, config);

  const opponentImpact = g(rival.phi);
  const expected = expectedScore(player.mu, rival.mu, rival.phi);
  const variance =
    1 /
    Math.max(
      MIN_PROBABILITY,
      opponentImpact * opponentImpact * expected * (1 - expected),
    );
  const delta = variance * opponentImpact * (s - expected);
  const sigmaPrime = computeSigmaPrime(
    player.phi,
    entity.sigma,
    delta,
    variance,
    config,
  );

  const phiStar = Math.sqrt(player.phi * player.phi + sigmaPrime * sigmaPrime);
  const phiPrime = 1 / Math.sqrt(1 / (phiStar * phiStar) + 1 / variance);
  const muPrime =
    player.mu + phiPrime * phiPrime * opponentImpact * (s - expected);

  const updated = fromGlicko2Core(
    { mu: muPrime, phi: phiPrime },
    entity,
    sigmaPrime,
    config,
  );

  return {
    ...updated,
    gamesPlayed: entity.gamesPlayed + 1,
  };
}

function getQuestionRating(
  state: RatingStateV2,
  questionId: string,
  label: Difficulty | undefined,
  timestamp: number,
): QuestionRating {
  const existing = state.questions[questionId];
  if (!existing) {
    return createDefaultQuestion(label, timestamp, state.config);
  }
  const sanitized = sanitizeQuestion(existing, label, timestamp, state.config);
  if (!sanitized.label && label) {
    return { ...sanitized, label };
  }
  return sanitized;
}

function buildLegacyCounts(state: RatingStateV2): LegacyDifficultyMap {
  const legacyCounts: LegacyDifficultyMap = {};
  for (const [questionId, question] of Object.entries(state.questions)) {
    legacyCounts[questionId] = {
      correct: sanitizeCount(question.legacyCorrect),
      wrong: sanitizeCount(question.legacyWrong),
    };
  }
  return legacyCounts;
}

function buildSyntheticOutcomes(correct: number, wrong: number): boolean[] {
  const total = correct + wrong;
  if (total <= 0) return [];

  const targetRatio = correct / total;
  const outcomes: boolean[] = [];
  let placedCorrect = 0;
  let placedWrong = 0;

  for (let i = 0; i < total; i++) {
    const remainingCorrect = correct - placedCorrect;
    const remainingWrong = wrong - placedWrong;

    if (remainingCorrect <= 0) {
      outcomes.push(false);
      placedWrong += 1;
      continue;
    }
    if (remainingWrong <= 0) {
      outcomes.push(true);
      placedCorrect += 1;
      continue;
    }

    const nextIndex = i + 1;
    const errorIfCorrect = Math.abs(
      (placedCorrect + 1) / nextIndex - targetRatio,
    );
    const errorIfWrong = Math.abs(placedCorrect / nextIndex - targetRatio);

    if (errorIfCorrect <= errorIfWrong) {
      outcomes.push(true);
      placedCorrect += 1;
    } else {
      outcomes.push(false);
      placedWrong += 1;
    }
  }

  return outcomes;
}

function migrateLegacyMapToV2(
  legacyMap: LegacyDifficultyMap,
  questionMetadata: QuestionMetadataMap = {},
): RatingStateV2 {
  let state = createInitialState();
  const questionIds = Object.keys(legacyMap).sort();

  const queues: Record<string, boolean[]> = {};
  let maxQueueLength = 0;

  for (const questionId of questionIds) {
    const stats = legacyMap[questionId];
    const outcomes = buildSyntheticOutcomes(
      sanitizeCount(stats.correct),
      sanitizeCount(stats.wrong),
    );
    queues[questionId] = outcomes;
    maxQueueLength = Math.max(maxQueueLength, outcomes.length);
  }

  const replayStart = nowMs();
  let tick = 0;
  for (let round = 0; round < maxQueueLength; round++) {
    for (const questionId of questionIds) {
      const outcome = queues[questionId][round];
      if (typeof outcome !== "boolean") continue;
      state = recordAnswer(
        state,
        questionId,
        questionMetadata[questionId]?.label,
        outcome,
        replayStart + tick,
      );
      tick += 1;
    }
  }

  return state;
}

// ---- persistence helpers ----

export function loadRatingState(
  questionMetadata: QuestionMetadataMap = {},
): RatingStateV2 {
  if (typeof window === "undefined") return createInitialState();

  try {
    const rawV2 = window.localStorage.getItem(STORAGE_KEY);
    if (rawV2) {
      const parsed = JSON.parse(rawV2);
      const normalized = sanitizeStateV2(parsed, questionMetadata);
      if (normalized) {
        return normalized;
      }
    }

    const rawLegacy = window.localStorage.getItem(LEGACY_STORAGE_KEY);
    if (rawLegacy) {
      const parsedLegacy = JSON.parse(rawLegacy);
      const legacyMap = sanitizeLegacyMap(parsedLegacy);
      if (legacyMap) {
        const migrated = migrateLegacyMapToV2(legacyMap, questionMetadata);
        saveRatingState(migrated);
        return migrated;
      }
    }
  } catch (err) {
    console.error("Failed to load rating state:", err);
  }

  return createInitialState();
}

export function saveRatingState(state: RatingStateV2): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (err) {
    console.error("Failed to save rating state:", err);
  }
}

// ---- rating updates ----

export function recordAnswer(
  state: RatingStateV2,
  questionId: string,
  label: Difficulty | undefined,
  isCorrect: boolean,
  nowTimestamp: number = nowMs(),
): RatingStateV2 {
  const config = sanitizeConfig(state.config);
  const normalizedState: RatingStateV2 = {
    ...state,
    version: 2,
    algorithm: "glicko-2",
    config,
    user: sanitizeEntity(
      state.user,
      createDefaultUser(nowTimestamp, config),
      config,
    ),
  };

  const baseQuestion = getQuestionRating(
    normalizedState,
    questionId,
    label,
    nowTimestamp,
  );

  const userBefore = inflateForInactivity(
    normalizedState.user,
    nowTimestamp,
    normalizedState.config,
  );
  const questionBefore = inflateForInactivity(
    baseQuestion,
    nowTimestamp,
    normalizedState.config,
  );

  const userScore = isCorrect ? 1 : 0;
  const questionScore = 1 - userScore;

  const userAfter = rateOneMatch(
    userBefore,
    questionBefore,
    userScore,
    normalizedState.config,
  );
  const questionAfterCore = rateOneMatch(
    questionBefore,
    userBefore,
    questionScore,
    normalizedState.config,
  );

  const questionAfter: QuestionRating = {
    ...questionAfterCore,
    legacyCorrect: questionBefore.legacyCorrect + (isCorrect ? 1 : 0),
    legacyWrong: questionBefore.legacyWrong + (isCorrect ? 0 : 1),
    label: questionBefore.label ?? label,
  };

  return {
    ...normalizedState,
    user: userAfter,
    questions: {
      ...normalizedState.questions,
      [questionId]: questionAfter,
    },
  };
}

// ---- question difficulty score (0 = easiest, 1 = hardest) ----

export function computeQuestionDifficultyScore(
  questionId: string,
  label: Difficulty | undefined,
  state: RatingStateV2,
): number {
  const config = sanitizeConfig(state.config);
  const timestamp = nowMs();
  const question = getQuestionRating(
    { ...state, config },
    questionId,
    label,
    timestamp,
  );
  const anchor = config.difficultyAnchorRating;
  const exponent = (anchor - question.rating) / config.difficultyScale;
  const score = 1 / (1 + Math.pow(10, exponent));
  return clamp(score, 0, 1);
}

// ---- import / export ----

export function exportRatingsJson(state: RatingStateV2): string {
  const normalized = sanitizeStateV2(state) ?? createInitialState();
  const envelope: ImportedV2Envelope = {
    ...normalized,
    exportedAt: new Date().toISOString(),
    legacyCounts: buildLegacyCounts(normalized),
  };
  return JSON.stringify(envelope, null, 2);
}

export function importRatingsJson(
  json: string,
  questionMetadata: QuestionMetadataMap = {},
): RatingStateV2 | null {
  try {
    const parsed = JSON.parse(json) as unknown;

    const asV2 = sanitizeStateV2(parsed, questionMetadata);
    if (asV2) return asV2;

    const asLegacy = sanitizeLegacyMap(parsed);
    if (asLegacy) return migrateLegacyMapToV2(asLegacy, questionMetadata);

    if (isRecord(parsed) && isLegacyMap(parsed.legacyCounts)) {
      const nestedLegacy = sanitizeLegacyMap(parsed.legacyCounts);
      if (nestedLegacy) {
        return migrateLegacyMapToV2(nestedLegacy, questionMetadata);
      }
    }
  } catch (err) {
    console.error("Failed to import ratings JSON:", err);
  }

  return null;
}

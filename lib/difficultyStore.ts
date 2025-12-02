// lib/difficultyStore.ts
"use client";

import type { Difficulty } from "./quiz";

export type QuestionStats = {
  correct: number;
  wrong: number;
};

export type DifficultyMap = Record<string, QuestionStats>;

const STORAGE_KEY = "aie-quiz-question-stats-v1";

// ---- persistence helpers ----

export function loadDifficultyMap(): DifficultyMap {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      return parsed as DifficultyMap;
    }
  } catch (err) {
    console.error("Failed to load difficulty map:", err);
  }
  return {};
}

export function saveDifficultyMap(map: DifficultyMap): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
  } catch (err) {
    console.error("Failed to save difficulty map:", err);
  }
}

// ---- updating stats ----

export function updateStats(
  map: DifficultyMap,
  questionId: string,
  isCorrect: boolean
): DifficultyMap {
  const prev = map[questionId] ?? { correct: 0, wrong: 0 };
  const next: QuestionStats = {
    correct: prev.correct + (isCorrect ? 1 : 0),
    wrong: prev.wrong + (isCorrect ? 0 : 1),
  };
  return { ...map, [questionId]: next };
}

// ---- difficulty score (0 = easiest, 1 = hardest) ----

// Beta prior parameters (alpha = "wrong", beta = "correct")
// We use different priors by author-assigned difficulty label
type BetaPrior = { alpha: number; beta: number };

function getPrior(label: Difficulty | undefined): BetaPrior {
  switch (label) {
    case "easy":
      // mean ~0.2 with low pseudo-count
      return { alpha: 0.8, beta: 3.2 };
    case "hard":
      // mean ~0.6 with low pseudo-count
      return { alpha: 2.4, beta: 1.6 };
    case "medium":
    default:
      // mean 0.5 with low pseudo-count
      return { alpha: 2, beta: 2 };
  }
}

/**
 * Compute empirical difficulty score in [0,1].
 *  - We treat difficulty as "probability that a random user answers incorrectly".
 *  - We use a Beta prior (based on author label) + observed correct/wrong counts.
 *  - More answers => more stable; still adapts over time.
 */
export function computeDifficultyScore(
  questionId: string,
  label: Difficulty | undefined,
  map: DifficultyMap
): number {
  const stats = map[questionId];
  const prior = getPrior(label);

  const wrong = stats?.wrong ?? 0;
  const correct = stats?.correct ?? 0;

  const alpha = prior.alpha + wrong; // "wrong" pseudo-counts
  const beta = prior.beta + correct; // "correct" pseudo-counts

  return alpha / (alpha + beta); // expected error rate
}

import {
  getQuestionsForFilters,
  type Question,
  type SourceId,
  type Topic,
} from "./quiz";
import { getQuestionRatingEstimate, type RatingStateV2 } from "./ratingEngine";

export type DifficultyRange = {
  min: number;
  max: number;
};

export type QuestionSelectionMode = "standard" | "climb";

export type QuizOption = {
  text: string;
  isCorrect: boolean;
};

export type AnswerEvaluation = {
  correctIndexes: number[];
  incorrectSelectionCount: number;
  missedCorrectOptionCount: number;
  mistakeCount: number;
  isCorrect: boolean;
};

export const QUESTION_TIMER_TICK_MS = 1000;
export const QUESTION_ELO_FILTER_MIN = 0;
export const QUESTION_ELO_FILTER_MAX = 3000;

export const DEFAULT_DIFFICULTY_RANGE: DifficultyRange = {
  min: QUESTION_ELO_FILTER_MIN,
  max: QUESTION_ELO_FILTER_MAX,
};

export const DEFAULT_QUESTION_SELECTION_MODE: QuestionSelectionMode =
  "standard";

export function shuffleItems<T>(
  items: readonly T[],
  random: () => number = Math.random,
): T[] {
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

export function clampDifficultyRange(range: DifficultyRange): DifficultyRange {
  const min = Math.max(
    QUESTION_ELO_FILTER_MIN,
    Math.min(QUESTION_ELO_FILTER_MAX, range.min),
  );
  const max = Math.max(min, Math.min(QUESTION_ELO_FILTER_MAX, range.max));
  return { min, max };
}

export function getEligibleQuestionIds(payload: {
  sources: SourceId[];
  topics: Topic[];
  difficultyRange: DifficultyRange;
  ratingState: RatingStateV2;
}): string[] {
  const clampedRange = clampDifficultyRange(payload.difficultyRange);
  const uniqueSources = Array.from(new Set(payload.sources));
  const uniqueTopics = Array.from(new Set(payload.topics));

  return getQuestionsForFilters(uniqueSources, uniqueTopics)
    .filter((question) => {
      const rating = getQuestionRatingEstimate(
        question.id,
        question.difficulty,
        payload.ratingState,
      ).rating;
      return rating >= clampedRange.min && rating <= clampedRange.max;
    })
    .map((question) => question.id);
}

export function pickClimbQuestionId(
  pool: Question[],
  ratingState: RatingStateV2,
  recentQuestionIds: string[],
  random: () => number = Math.random,
): string | null {
  if (pool.length === 0) return null;

  const targetRating = ratingState.user.rating;
  const recent = new Set(recentQuestionIds.slice(-6));

  const scored = pool
    .map((question) => {
      const estimate = getQuestionRatingEstimate(
        question.id,
        question.difficulty,
        ratingState,
      );
      const distance = Math.abs(estimate.rating - targetRating);
      const uncertaintyBonus = Math.min(estimate.rd, 250) * 0.25;
      const randomJitter = random() * 90;
      const repeatPenalty = recent.has(question.id) ? 180 : 0;
      return {
        id: question.id,
        score: distance - uncertaintyBonus + randomJitter + repeatPenalty,
      };
    })
    .sort((a, b) => a.score - b.score);

  const shortlistSize = Math.min(8, scored.length);
  const shortlist = scored.slice(0, shortlistSize);
  const choice = shortlist[Math.floor(random() * shortlist.length)];
  return choice?.id ?? null;
}

export function evaluateAnswer(
  options: QuizOption[],
  selectedIndexes: number[],
): AnswerEvaluation {
  const correctIndexes = options
    .map((option, index) => (option.isCorrect ? index : -1))
    .filter((index) => index !== -1);
  const selectedSet = new Set(selectedIndexes);
  const incorrectSelectionCount = options.filter(
    (option, index) => selectedSet.has(index) && !option.isCorrect,
  ).length;
  const missedCorrectOptionCount = options.filter(
    (option, index) => option.isCorrect && !selectedSet.has(index),
  ).length;
  const mistakeCount = incorrectSelectionCount + missedCorrectOptionCount;
  const isCorrect =
    correctIndexes.length === selectedIndexes.length &&
    correctIndexes.every((index) => selectedIndexes.includes(index));

  return {
    correctIndexes,
    incorrectSelectionCount,
    missedCorrectOptionCount,
    mistakeCount,
    isCorrect,
  };
}

export function buildQuizApiUrl(
  baseUrl: string,
  path: string,
  query?: Record<string, string>,
): string {
  const normalizedBase = baseUrl.trim();
  if (!normalizedBase) {
    throw new Error("API base URL is required.");
  }

  const url = new URL(
    path,
    normalizedBase.endsWith("/") ? normalizedBase : `${normalizedBase}/`,
  );
  if (query) {
    for (const [key, value] of Object.entries(query)) {
      url.searchParams.set(key, value);
    }
  }
  return url.toString();
}

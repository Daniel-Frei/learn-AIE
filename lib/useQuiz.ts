// lib/useQuiz.ts
"use client";

import { useMemo, useState } from "react";
import {
  ALL_SOURCE_IDS,
  allQuestions,
  getQuestionsForSources,
  type SourceId,
} from "./quiz";
import {
  loadRatingState,
  saveRatingState,
  recordAnswer,
  computeQuestionDifficultyScore,
  exportRatingsJson,
  importRatingsJson,
  type QuestionMetadataMap,
  type RatingStateV2,
} from "./difficultyStore";

// Difficulty filter is now a numeric range [0,100]
export type DifficultyRange = {
  min: number; // inclusive, 0 = easiest
  max: number; // inclusive, 100 = hardest
};

function shuffle<T>(items: T[]): T[] {
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

const QUESTION_METADATA: QuestionMetadataMap = Object.fromEntries(
  allQuestions.map((q) => [q.id, { label: q.difficulty }]),
);

export function useQuiz() {
  const initialSources = ALL_SOURCE_IDS.length ? [ALL_SOURCE_IDS[0]] : [];
  const initialRange: DifficultyRange = { min: 0, max: 100 };

  // persistent rating state (single user + per-question ratings)
  const [ratingState, setRatingState] = useState<RatingStateV2>(() =>
    loadRatingState(QUESTION_METADATA),
  );

  const [selectedSources, setSelectedSources] =
    useState<SourceId[]>(initialSources);

  // numeric difficulty filter (applied on selection)
  const [difficultyRange, setDifficultyRange] =
    useState<DifficultyRange>(initialRange);

  const [appliedQuestionIds, setAppliedQuestionIds] = useState<string[]>(() => {
    const pool = getQuestionsForSources(initialSources);
    return pool
      .filter((q) => {
        const score = computeQuestionDifficultyScore(
          q.id,
          q.difficulty,
          ratingState,
        );
        const scorePercent = Math.round(score * 100);
        return (
          scorePercent >= initialRange.min && scorePercent <= initialRange.max
        );
      })
      .map((q) => q.id);
  });

  const sourcePool = useMemo(
    () => getQuestionsForSources(selectedSources),
    [selectedSources],
  );

  const questionById = useMemo(() => {
    return new Map(sourcePool.map((q) => [q.id, q]));
  }, [sourcePool]);

  const availableQuestions = useMemo(() => {
    return appliedQuestionIds
      .map((id) => questionById.get(id))
      .filter((q): q is NonNullable<typeof q> => Boolean(q));
  }, [appliedQuestionIds, questionById]);

  const [questionOrder, setQuestionOrder] = useState<number[]>(() =>
    shuffle(Array.from({ length: appliedQuestionIds.length }, (_, i) => i)),
  );
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedIndexes, setSelectedIndexes] = useState<number[]>([]);
  const [showResult, setShowResult] = useState<null | { isCorrect: boolean }>(
    null,
  );
  const [answeredCount, setAnsweredCount] = useState(0);
  const [correctCount, setCorrectCount] = useState(0);

  const currentQuestion = useMemo(() => {
    if (!availableQuestions.length || !questionOrder.length) return null;
    const idxInArray = questionOrder[currentIndex] ?? questionOrder[0];
    return availableQuestions[idxInArray];
  }, [availableQuestions, questionOrder, currentIndex]);

  const shuffledOptions = useMemo(() => {
    if (!currentQuestion) return [];
    return shuffle(currentQuestion.options);
  }, [currentQuestion]);

  // current question difficulty score (0–1)
  const currentDifficultyScore = useMemo(() => {
    if (!currentQuestion) return null;
    return computeQuestionDifficultyScore(
      currentQuestion.id,
      currentQuestion.difficulty,
      ratingState,
    );
  }, [currentQuestion, ratingState]);

  const toggleOption = (idx: number) => {
    if (showResult) return;
    setSelectedIndexes((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx],
    );
  };

  const submitAnswer = () => {
    if (!currentQuestion || showResult) return;

    const correctIndexes = shuffledOptions
      .map((opt, idx) => (opt.isCorrect ? idx : -1))
      .filter((idx) => idx !== -1);

    const isCorrect =
      correctIndexes.length === selectedIndexes.length &&
      correctIndexes.every((idx) => selectedIndexes.includes(idx));

    // Show result and update visible quiz counters
    setShowResult({ isCorrect });
    setAnsweredCount((n) => n + 1);
    setCorrectCount((n) => n + (isCorrect ? 1 : 0));

    // Persist rating update immediately on submit
    setRatingState((prev) => {
      const updated = recordAnswer(
        prev,
        currentQuestion.id,
        currentQuestion.difficulty,
        isCorrect,
      );
      saveRatingState(updated);
      return updated;
    });
  };

  const nextQuestion = () => {
    if (!availableQuestions.length) return;

    // Clear result & selections for the NEXT question
    setShowResult(null);
    setSelectedIndexes([]);

    const next = currentIndex + 1;
    if (next >= availableQuestions.length) {
      const order = shuffle(
        Array.from({ length: availableQuestions.length }, (_, i) => i),
      );
      setQuestionOrder(order);
      setCurrentIndex(0);
    } else {
      setCurrentIndex(next);
    }
  };

  const accuracy =
    answeredCount === 0 ? 0 : Math.round((100 * correctCount) / answeredCount);

  const clampRange = (newRange: DifficultyRange) => {
    const min = Math.max(0, Math.min(100, newRange.min));
    const max = Math.max(min, Math.min(100, newRange.max));
    return { min, max };
  };

  const applySelection = (payload: {
    sources: SourceId[];
    difficultyRange: DifficultyRange;
  }) => {
    const clampedRange = clampRange(payload.difficultyRange);
    const uniqueSources =
      payload.sources.length > 0
        ? Array.from(new Set(payload.sources))
        : [...ALL_SOURCE_IDS];

    const pool = getQuestionsForSources(uniqueSources);
    const eligibleIds = pool
      .filter((q) => {
        const score = computeQuestionDifficultyScore(
          q.id,
          q.difficulty,
          ratingState,
        );
        const scorePercent = Math.round(score * 100);
        return (
          scorePercent >= clampedRange.min && scorePercent <= clampedRange.max
        );
      })
      .map((q) => q.id);

    setSelectedSources(uniqueSources);
    setDifficultyRange(clampedRange);
    setAppliedQuestionIds(eligibleIds);
    setAnsweredCount(0);
    setCorrectCount(0);
    setQuestionOrder(
      shuffle(Array.from({ length: eligibleIds.length }, (_, i) => i)),
    );
    setCurrentIndex(0);
    setSelectedIndexes([]);
    setShowResult(null);
  };

  // -------- EXPORT / IMPORT HELPERS --------

  const exportDifficultyJson = () => {
    return exportRatingsJson(ratingState);
  };

  const importDifficultyFromJson = (json: string) => {
    const parsed = importRatingsJson(json, QUESTION_METADATA);
    if (!parsed) return;
    setRatingState(parsed);
    saveRatingState(parsed);
  };

  return {
    // configuration
    selectedSources,
    difficultyRange,
    applySelection,

    // question set info
    availableCount: availableQuestions.length,
    currentIndex,
    currentQuestion,
    currentDifficultyScore,
    shuffledOptions,

    // answering state
    selectedIndexes,
    showResult,
    toggleOption,
    submitAnswer,
    nextQuestion,

    // stats
    answeredCount,
    correctCount,
    accuracy,

    // export/import
    exportDifficultyJson,
    importDifficultyFromJson,
  };
}

// lib/useQuiz.ts
"use client";

import { useEffect, useMemo, useState } from "react";
import { ALL_SOURCE_IDS, getQuestionsForSources, type SourceId } from "./quiz";
import {
  loadDifficultyMap,
  saveDifficultyMap,
  updateStats,
  computeDifficultyScore,
  type DifficultyMap,
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

export function useQuiz() {
  const [selectedSources, setSelectedSources] = useState<SourceId[]>(() =>
    ALL_SOURCE_IDS.length ? [ALL_SOURCE_IDS[0]] : []
  );

  // numeric difficulty filter
  const [difficultyRange, setDifficultyRange] = useState<DifficultyRange>({
    min: 0,
    max: 100,
  });

  // persistent difficulty stats (per question)
  const [difficultyMap, setDifficultyMap] = useState<DifficultyMap>(() =>
    loadDifficultyMap()
  );

  const baseQuestions = useMemo(
    () => getQuestionsForSources(selectedSources),
    [selectedSources]
  );

  // filter questions by empirical difficulty score
  const availableQuestions = useMemo(() => {
    return baseQuestions.filter((q) => {
      const score = computeDifficultyScore(q.id, q.difficulty, difficultyMap);
      const scorePercent = Math.round(score * 100); // 0–100 for UI/filter
      return (
        scorePercent >= difficultyRange.min &&
        scorePercent <= difficultyRange.max
      );
    });
  }, [baseQuestions, difficultyMap, difficultyRange.min, difficultyRange.max]);

  const [questionOrder, setQuestionOrder] = useState<number[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedIndexes, setSelectedIndexes] = useState<number[]>([]);
  const [showResult, setShowResult] = useState<null | { isCorrect: boolean }>(
    null
  );
  const [answeredCount, setAnsweredCount] = useState(0);
  const [correctCount, setCorrectCount] = useState(0);

  // --- Reset navigation when the *set* of available questions changes ---
  // (e.g. selection / difficulty range / imported difficulty file)
  // We DO NOT touch answeredCount / correctCount here.
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => {
    if (availableQuestions.length === 0) {
      setQuestionOrder([]);
      setCurrentIndex(0);
      setSelectedIndexes([]);
      setShowResult(null);
      return;
    }

    const order = shuffle(
      Array.from({ length: availableQuestions.length }, (_, i) => i)
    );
    setQuestionOrder(order);
    setCurrentIndex(0);
    setSelectedIndexes([]);
    setShowResult(null);
  }, [availableQuestions.length]);

  // --- Reset stats only when filters change (selection or difficulty range) ---
  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => {
    setAnsweredCount(0);
    setCorrectCount(0);
  }, [selectedSources, difficultyRange.min, difficultyRange.max]);

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
    return computeDifficultyScore(
      currentQuestion.id,
      currentQuestion.difficulty,
      difficultyMap
    );
  }, [currentQuestion, difficultyMap]);

  const toggleOption = (idx: number) => {
    if (showResult) return;
    setSelectedIndexes((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx]
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

    // Show result & update stats – but do NOT touch difficultyMap here
    setShowResult({ isCorrect });
    setAnsweredCount((n) => n + 1);
    setCorrectCount((n) => n + (isCorrect ? 1 : 0));
  };

  const nextQuestion = () => {
    if (!availableQuestions.length) return;

    // Apply difficulty update for the question we just finished viewing
    setDifficultyMap((prev) => {
      if (!currentQuestion || !showResult) return prev;

      const updated = updateStats(prev, currentQuestion.id, showResult.isCorrect);
      saveDifficultyMap(updated);
      return updated;
    });

    // Clear result & selections for the NEXT question
    setShowResult(null);
    setSelectedIndexes([]);

    const next = currentIndex + 1;
    if (next >= availableQuestions.length) {
      const order = shuffle(
        Array.from({ length: availableQuestions.length }, (_, i) => i)
      );
      setQuestionOrder(order);
      setCurrentIndex(0);
    } else {
      setCurrentIndex(next);
    }
  };

  const accuracy =
    answeredCount === 0
      ? 0
      : Math.round((100 * correctCount) / answeredCount);

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

    setSelectedSources(uniqueSources);
    setDifficultyRange(clampedRange);
  };

  // -------- EXPORT / IMPORT HELPERS --------

  const exportDifficultyJson = () => {
    return JSON.stringify(difficultyMap, null, 2);
  };

  const importDifficultyFromJson = (json: string) => {
    try {
      const parsed = JSON.parse(json) as DifficultyMap;
      if (!parsed || typeof parsed !== "object") return;
      setDifficultyMap(parsed);
      saveDifficultyMap(parsed);
    } catch (err) {
      console.error("Failed to import difficulty JSON:", err);
    }
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

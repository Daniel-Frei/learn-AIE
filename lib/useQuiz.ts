// lib/useQuiz.ts
"use client";

import { useEffect, useMemo, useState } from "react";
import { chapter1Questions } from "./chapter1";
import { chapter2Questions } from "./chapter2";
import { chapter3Questions } from "./chapter3";
import { aieChapter2Questions } from "./books/AIE_building_apps/chapter2";
import { allQuestions, Question } from "./quiz";
import {
  loadDifficultyMap,
  saveDifficultyMap,
  updateStats,
  computeDifficultyScore,
  type DifficultyMap,
} from "./difficultyStore";

export type Mode =
  | "chapter-1"
  | "chapter-2"
  | "chapter-3"
  | "aie-build-app-ch2"
  | "all";

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

function getQuestionsForMode(mode: Mode): Question[] {
  if (mode === "chapter-1") return chapter1Questions;
  if (mode === "chapter-2") return chapter2Questions;
  if (mode === "chapter-3") return chapter3Questions;
  if (mode === "aie-build-app-ch2") return aieChapter2Questions;
  return allQuestions;
}

export function useQuiz() {
  const [mode, setMode] = useState<Mode>("chapter-1");

  // numeric difficulty filter
  const [difficultyRange, setDifficultyRange] = useState<DifficultyRange>({
    min: 0,
    max: 100,
  });

  // persistent difficulty stats (per question)
  const [difficultyMap, setDifficultyMap] = useState<DifficultyMap>({});

  // load difficulty stats once on mount
  useEffect(() => {
    const loaded = loadDifficultyMap();
    setDifficultyMap(loaded);
  }, []);

  const baseQuestions = useMemo(
    () => getQuestionsForMode(mode),
    [mode]
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
  // (e.g. mode / difficulty range / imported difficulty file)
  // We DO NOT touch answeredCount / correctCount here.
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

  // --- Reset stats only when filters change (mode or difficulty range) ---
  useEffect(() => {
    setAnsweredCount(0);
    setCorrectCount(0);
  }, [mode, difficultyRange.min, difficultyRange.max]);

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

  const changeMode = (newMode: Mode) => {
    setMode(newMode);
  };

  const changeDifficultyRange = (newRange: DifficultyRange) => {
    // clamp + sort to be safe
    const min = Math.max(0, Math.min(100, newRange.min));
    const max = Math.max(min, Math.min(100, newRange.max));
    setDifficultyRange({ min, max });
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
    mode,
    difficultyRange,
    changeMode,
    changeDifficultyRange,

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
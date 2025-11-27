// lib/useQuiz.ts
"use client";

import { useEffect, useMemo, useState } from "react";
import { chapter1Questions } from "./chapter1";
import { chapter2Questions } from "./chapter2";
import { allQuestions, Difficulty, Question } from "./quiz";

export type Mode = "chapter-1" | "chapter-2" | "all";
export type DifficultyFilter = "all" | Difficulty;

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
  return allQuestions;
}

function filterByDifficulty(
  questions: Question[],
  difficulty: DifficultyFilter
): Question[] {
  if (difficulty === "all") return questions;
  return questions.filter((q) => q.difficulty === difficulty);
}

export function useQuiz() {
  const [mode, setMode] = useState<Mode>("chapter-1");
  const [difficultyFilter, setDifficultyFilter] =
    useState<DifficultyFilter>("all");

  const baseQuestions = useMemo(
    () => getQuestionsForMode(mode),
    [mode]
  );

  const availableQuestions = useMemo(
    () => filterByDifficulty(baseQuestions, difficultyFilter),
    [baseQuestions, difficultyFilter]
  );

  const [questionOrder, setQuestionOrder] = useState<number[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedIndexes, setSelectedIndexes] = useState<number[]>([]);
  const [showResult, setShowResult] = useState<null | { isCorrect: boolean }>(
    null
  );
  const [answeredCount, setAnsweredCount] = useState(0);
  const [correctCount, setCorrectCount] = useState(0);

  // Reset order & stats whenever the available question set changes
  useEffect(() => {
    if (availableQuestions.length === 0) {
      setQuestionOrder([]);
      setCurrentIndex(0);
      setSelectedIndexes([]);
      setShowResult(null);
      setAnsweredCount(0);
      setCorrectCount(0);
      return;
    }

    const order = shuffle(
      Array.from({ length: availableQuestions.length }, (_, i) => i)
    );
    setQuestionOrder(order);
    setCurrentIndex(0);
    setSelectedIndexes([]);
    setShowResult(null);
    setAnsweredCount(0);
    setCorrectCount(0);
  }, [availableQuestions.length]);

  const currentQuestion = useMemo(() => {
    if (!availableQuestions.length || !questionOrder.length) return null;
    const idxInArray = questionOrder[currentIndex] ?? questionOrder[0];
    return availableQuestions[idxInArray];
  }, [availableQuestions, questionOrder, currentIndex]);

  const shuffledOptions = useMemo(() => {
    if (!currentQuestion) return [];
    // shuffle only when the question changes
    return shuffle(currentQuestion.options);
  }, [currentQuestion?.id]);

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

    setShowResult({ isCorrect });
    setAnsweredCount((n) => n + 1);
    setCorrectCount((n) => n + (isCorrect ? 1 : 0));
  };

  const nextQuestion = () => {
    if (!availableQuestions.length) return;

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

  const changeDifficulty = (newDifficulty: DifficultyFilter) => {
    setDifficultyFilter(newDifficulty);
  };

  return {
    // configuration
    mode,
    difficultyFilter,
    changeMode,
    changeDifficulty,

    // question set info
    availableCount: availableQuestions.length,
    currentIndex,
    currentQuestion,
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
  };
}

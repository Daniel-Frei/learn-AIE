// components/QuizFooter.tsx
"use client";

import QuestionExplanationChat from "./QuestionExplanationChat";
import type { Question } from "../lib/quiz";

type Option = {
  text: string;
  isCorrect: boolean;
};

type Props = {
  hasQuestion: boolean;
  currentQuestion: Question | null;
  shuffledOptions: Option[];
  selectedIndexes: number[];
  showResult: { isCorrect: boolean } | null;
  submitAnswer: () => void;
  nextQuestion: () => void;
};

export default function QuizFooter({
  hasQuestion,
  currentQuestion,
  shuffledOptions,
  selectedIndexes,
  showResult,
  submitAnswer,
  nextQuestion,
}: Props) {
  if (!hasQuestion) {
    return (
      <footer className="space-y-3 text-xs text-slate-400">
        <p>Adjust the filters above to see available questions.</p>
      </footer>
    );
  }

  return (
    <footer className="space-y-3">
      <div className="flex gap-3">
        {!showResult ? (
          <button
            onClick={submitAnswer}
            disabled={selectedIndexes.length === 0}
            className="px-4 py-2 rounded-lg bg-sky-500 text-slate-950 font-semibold text-sm disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Submit answer
          </button>
        ) : (
          <button
            onClick={nextQuestion}
            className="px-4 py-2 rounded-lg bg-emerald-500 text-slate-950 font-semibold text-sm"
          >
            Next question
          </button>
        )}
      </div>

      {showResult && currentQuestion && (
        <>
          <div
            className={`mt-2 rounded-lg border px-4 py-3 text-sm ${
              showResult.isCorrect
                ? "border-emerald-500 bg-emerald-900/30 text-emerald-100"
                : "border-rose-500 bg-rose-900/30 text-rose-100"
            }`}
          >
            <p className="font-semibold mb-1">
              {showResult.isCorrect
                ? "Correct 🎉"
                : "Not quite – review the explanation:"}
            </p>
            <p>{currentQuestion.explanation}</p>
          </div>

          <QuestionExplanationChat
            question={currentQuestion}
            options={shuffledOptions.map((opt, idx) => ({
              text: opt.text,
              isCorrect: opt.isCorrect,
              selected: selectedIndexes.includes(idx),
            }))}
            isOverallCorrect={showResult.isCorrect}
          />
        </>
      )}
    </footer>
  );
}

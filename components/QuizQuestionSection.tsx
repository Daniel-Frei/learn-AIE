// components/QuizQuestionSection.tsx
"use client";

import type { Question } from "../lib/quiz";

type Option = {
  text: string;
  isCorrect: boolean;
};

type Props = {
  availableCount: number;
  currentIndex: number;
  difficultyPercent: number | null;
  currentQuestion: Question | null;
  shuffledOptions: Option[];
  selectedIndexes: number[];
  showResult: { isCorrect: boolean } | null;
  toggleOption: (idx: number) => void;
};

export default function QuizQuestionSection({
  availableCount,
  currentIndex,
  difficultyPercent,
  currentQuestion,
  shuffledOptions,
  selectedIndexes,
  showResult,
  toggleOption,
}: Props) {
  const hasQuestion = availableCount > 0 && !!currentQuestion;

  return (
    <section className="space-y-4">
      <div className="text-xs uppercase tracking-wide text-slate-400 flex justify-between">
        <span>
          Question{" "}
          {hasQuestion ? currentIndex + 1 : 0} of {availableCount}
        </span>
        {difficultyPercent !== null && hasQuestion && (
          <span>
            Difficulty:{" "}
            <span className="font-semibold">{difficultyPercent}</span>/100
          </span>
        )}
      </div>

      {!hasQuestion ? (
        <div className="rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-6 text-sm text-slate-200">
          <h2 className="text-lg font-semibold mb-2">
            No questions for this selection
          </h2>
          <p className="text-slate-300">
            Try changing the chapter or adjusting the difficulty range.
          </p>
        </div>
      ) : (
        <>
          <h2 className="text-lg md:text-xl font-semibold">
            {currentQuestion!.prompt}
          </h2>

          <div className="space-y-3">
            {shuffledOptions.map((opt, idx) => {
              const selected = selectedIndexes.includes(idx);
              const correct = opt.isCorrect;

              let borderClass = "border-slate-700";
              let bgClass = "bg-slate-900";
              if (showResult) {
                if (selected && correct) {
                  borderClass = "border-emerald-400";
                  bgClass = "bg-emerald-900/30";
                } else if (selected && !correct) {
                  borderClass = "border-rose-400";
                  bgClass = "bg-rose-900/30";
                } else if (!selected && correct) {
                  borderClass = "border-emerald-500/70";
                }
              } else if (selected) {
                borderClass = "border-sky-400";
                bgClass = "bg-slate-800";
              }

              return (
                <button
                  key={idx}
                  type="button"
                  onClick={() => toggleOption(idx)}
                  className={`w-full text-left border ${borderClass} ${bgClass} rounded-xl px-4 py-3 flex gap-3 items-start transition-colors`}
                >
                  <div
                    className={`mt-1 h-4 w-4 flex items-center justify-center rounded border ${
                      selected
                        ? "bg-sky-500 border-sky-400"
                        : "bg-slate-900 border-slate-600"
                    }`}
                  >
                    {selected && (
                      <span className="text-[10px] font-bold text-slate-50">
                        ✓
                      </span>
                    )}
                  </div>
                  <span className="text-sm md:text-base">{opt.text}</span>
                </button>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}

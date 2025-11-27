// app/page.tsx
"use client";

import { useQuiz, Mode, DifficultyFilter } from "../lib/useQuiz";

export default function QuizPage() {
  const {
    mode,
    difficultyFilter,
    changeMode,
    changeDifficulty,
    availableCount,
    currentIndex,
    currentQuestion,
    shuffledOptions,
    selectedIndexes,
    showResult,
    toggleOption,
    submitAnswer,
    nextQuestion,
    answeredCount,
    correctCount,
    accuracy,
  } = useQuiz();

  if (!currentQuestion || availableCount === 0) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-50 flex items-center justify-center px-4 py-10">
        <div className="w-full max-w-xl rounded-2xl bg-slate-900 shadow-xl border border-slate-800 p-6 md:p-8 space-y-4">
          <h1 className="text-2xl font-semibold">
            No questions for this selection
          </h1>
          <p className="text-sm text-slate-300">
            Try changing the chapter or difficulty filter.
          </p>
        </div>
      </div>
    );
  }

  const difficultyLabel =
    difficultyFilter === "all"
      ? "mixed difficulty"
      : `${difficultyFilter} only`;

  const title =
    mode === "chapter-1"
      ? "Chapter 1 Quiz – Analyzing Text Data with Deep Learning"
      : mode === "chapter-2"
      ? "Chapter 2 Quiz – The Transformer and Modern NLP"
      : "All Chapters Quiz – Text & Transformers";

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50 flex items-center justify-center px-4 py-10">
      <div className="w-full max-w-3xl rounded-2xl bg-slate-900 shadow-xl border border-slate-800 p-6 md:p-8 space-y-6">
        {/* HEADER */}
        <header className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-3xl font-semibold">{title}</h1>
            <p className="text-sm text-slate-400 mt-1">
              Multi-select questions –{" "}
              <span className="font-semibold">select all TRUE statements</span>{" "}
              and then submit.
            </p>
          </div>

          <div className="flex flex-col items-start md:items-end gap-2">
            <label className="text-sm text-slate-300 flex items-center gap-2">
              <span>Question source:</span>
              <select
                className="bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-sm"
                value={mode}
                onChange={(e) => changeMode(e.target.value as Mode)}
              >
                <option value="chapter-1">Chapter 1 only</option>
                <option value="chapter-2">Chapter 2 only</option>
                <option value="all">All chapters</option>
              </select>
            </label>

            <label className="text-sm text-slate-300 flex items-center gap-2">
              <span>Difficulty:</span>
              <select
                className="bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-sm"
                value={difficultyFilter}
                onChange={(e) =>
                  changeDifficulty(e.target.value as DifficultyFilter)
                }
              >
                <option value="all">All</option>
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </label>

            <div className="text-xs text-slate-400">
              Answered:{" "}
              <span className="font-semibold text-slate-200">
                {answeredCount}
              </span>{" "}
              · Correct:{" "}
              <span className="font-semibold text-emerald-300">
                {correctCount}
              </span>{" "}
              · Accuracy:{" "}
              <span className="font-semibold">{accuracy}%</span>
            </div>
          </div>
        </header>

        {/* QUESTION */}
        <section className="space-y-4">
          <div className="text-xs uppercase tracking-wide text-slate-400">
            Question {currentIndex + 1} of {availableCount} ·{" "}
            <span className="font-semibold">{difficultyLabel}</span>
          </div>

          <h2 className="text-lg md:text-xl font-semibold">
            {currentQuestion.prompt}
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
        </section>

        {/* FOOTER */}
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

          {showResult && (
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
          )}
        </footer>
      </div>
    </div>
  );
}

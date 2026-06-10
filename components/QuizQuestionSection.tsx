// components/QuizQuestionSection.tsx
"use client";

import { useId } from "react";
import { getQuestionType, type Question } from "../lib/quiz";
import MathText from "./MathText";
import RatingDeltaIndicator from "./RatingDeltaIndicator";

type Option = {
  text: string;
  isCorrect: boolean;
};

type Props = {
  availableCount: number;
  currentIndex: number;
  questionRating: number | null;
  questionRatingDelta: number | null;
  questionElapsedMs: number;
  questionContext: string | null;
  currentQuestion: Question | null;
  shuffledOptions: Option[];
  selectedIndexes: number[];
  showResult: { isCorrect: boolean } | null;
  toggleOption: (idx: number) => void;
};

function formatElapsedMs(elapsedMs: number): string {
  const totalSeconds = Math.max(0, Math.floor(elapsedMs / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

function formatQuestionRating(rating: number): string {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(rating);
}

function QuestionContextTooltip({ context }: { context: string }) {
  return (
    <span
      tabIndex={0}
      aria-label={`Question context: ${context}`}
      className="group ml-2 inline-flex h-5 w-5 align-middle items-center justify-center rounded-full text-slate-400 transition-colors hover:text-sky-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-900"
    >
      <svg
        aria-hidden="true"
        viewBox="0 0 20 20"
        className="h-5 w-5"
        fill="none"
      >
        <circle cx="10" cy="10" r="8.25" stroke="currentColor" />
        <path d="M10 9.25v4.25" stroke="currentColor" strokeLinecap="round" />
        <circle cx="10" cy="6.5" r="0.75" fill="currentColor" />
      </svg>
      <span
        role="tooltip"
        className="pointer-events-none invisible absolute left-0 right-0 top-full z-20 mt-2 rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-left text-xs font-normal leading-5 text-slate-200 opacity-0 shadow-xl transition-opacity group-hover:visible group-hover:opacity-100 group-focus:visible group-focus:opacity-100"
      >
        {context}
      </span>
    </span>
  );
}

function hasTextSelectionInside(element: HTMLElement): boolean {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed || !selection.toString().trim()) {
    return false;
  }

  const anchorNode = selection.anchorNode;
  const focusNode = selection.focusNode;
  return (
    Boolean(anchorNode && element.contains(anchorNode)) ||
    Boolean(focusNode && element.contains(focusNode))
  );
}

function parseAssertionReasonPrompt(prompt: string) {
  const match = prompt.match(
    /^Assertion:\s*([\s\S]*?)\n\s*\nReason:\s*([\s\S]*)$/,
  );
  if (!match) return null;

  return {
    assertion: match[1].trim(),
    reason: match[2].trim(),
  };
}

function QuestionPrompt({
  question,
  questionContext,
}: {
  question: Question;
  questionContext: string | null;
}) {
  const isAssertionReason = getQuestionType(question) === "assertion-reason";
  const assertionReasonPrompt = isAssertionReason
    ? parseAssertionReasonPrompt(question.prompt)
    : null;

  if (assertionReasonPrompt) {
    return (
      <div
        data-testid="question-prompt"
        className="relative space-y-3 text-lg md:text-xl font-semibold leading-relaxed"
      >
        <p>
          <span>Assertion:</span>{" "}
          <MathText text={assertionReasonPrompt.assertion} inline />
        </p>
        <p>
          <span>Reason:</span>{" "}
          <MathText text={assertionReasonPrompt.reason} inline />
          {questionContext && (
            <QuestionContextTooltip context={questionContext} />
          )}
        </p>
      </div>
    );
  }

  return (
    <div data-testid="question-prompt" className="relative">
      <MathText
        text={question.prompt}
        inline
        className="text-lg md:text-xl font-semibold leading-relaxed"
      />
      {questionContext && <QuestionContextTooltip context={questionContext} />}
    </div>
  );
}

export default function QuizQuestionSection({
  availableCount,
  currentIndex,
  questionRating,
  questionRatingDelta,
  questionElapsedMs,
  questionContext,
  currentQuestion,
  shuffledOptions,
  selectedIndexes,
  showResult,
  toggleOption,
}: Props) {
  const hasQuestion = availableCount > 0 && !!currentQuestion;
  const optionIdPrefix = useId();

  return (
    <section className="space-y-4">
      <div className="flex min-h-10 justify-between text-xs uppercase tracking-wide text-slate-400">
        <span>
          Question {hasQuestion ? currentIndex + 1 : 0} of {availableCount}
        </span>
        {hasQuestion && (
          <span className="text-right flex min-w-52 flex-col items-end gap-1">
            <span data-testid="question-timer" className="block leading-4">
              Time:{" "}
              <span className="font-semibold text-slate-100">
                {formatElapsedMs(questionElapsedMs)}
              </span>{" "}
              / 3:00
            </span>
            {showResult && questionRating !== null && (
              <span
                data-testid="question-rating-line"
                className="block min-h-4 whitespace-nowrap leading-4"
              >
                Question Elo:{" "}
                <span className="font-semibold text-slate-100">
                  {formatQuestionRating(questionRating)}
                </span>
                {questionRatingDelta !== null && (
                  <>
                    {" "}
                    <RatingDeltaIndicator
                      delta={questionRatingDelta}
                      label="Question Elo"
                      testId="question-rating-delta"
                      className="align-middle"
                    />
                  </>
                )}
              </span>
            )}
            {!showResult && (
              <span aria-hidden="true" className="block min-h-4 leading-4" />
            )}
          </span>
        )}
      </div>

      {!hasQuestion ? (
        <div className="rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-6 text-sm text-slate-200">
          <h2 className="text-lg font-semibold mb-2">
            No questions for this selection
          </h2>
          <p className="text-slate-300">
            Try selecting a series, lecture/chapter, or topic, then adjust the
            question Elo range if needed.
          </p>
        </div>
      ) : (
        <>
          <QuestionPrompt
            question={currentQuestion}
            questionContext={questionContext}
          />

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
                <div
                  key={idx}
                  role="checkbox"
                  tabIndex={0}
                  aria-checked={selected}
                  aria-labelledby={`${optionIdPrefix}-option-${idx}`}
                  onClick={(event) => {
                    if (hasTextSelectionInside(event.currentTarget)) return;
                    toggleOption(idx);
                  }}
                  onKeyDown={(event) => {
                    if (event.key !== " " && event.key !== "Enter") return;
                    event.preventDefault();
                    toggleOption(idx);
                  }}
                  className={`w-full text-left border ${borderClass} ${bgClass} rounded-xl px-4 py-3 flex gap-3 items-start transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-slate-950`}
                >
                  <div
                    aria-hidden="true"
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
                  <span
                    id={`${optionIdPrefix}-option-${idx}`}
                    data-testid="answer-option-text"
                    className="cursor-pointer select-text"
                  >
                    <MathText
                      text={opt.text}
                      inline
                      className="text-sm md:text-base"
                    />
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}

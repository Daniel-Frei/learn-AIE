// components/QuizFooter.tsx
"use client";

import { useState } from "react";
import QuestionExplanationChat from "./QuestionExplanationChat";
import MathText from "./MathText";
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
  submitQuestionReport: (comment: string) => boolean;
};

type ReportControlsProps = {
  submitQuestionReport: (comment: string) => boolean;
};

function QuestionReportControls({ submitQuestionReport }: ReportControlsProps) {
  const [isReportOpen, setIsReportOpen] = useState(false);
  const [reportComment, setReportComment] = useState("");
  const [reportError, setReportError] = useState("");
  const [reportSuccess, setReportSuccess] = useState("");

  return (
    <>
      <button
        type="button"
        aria-label={isReportOpen ? "Close report" : "Report question"}
        title={isReportOpen ? "Close report" : "Report question"}
        onClick={() => {
          setIsReportOpen((open) => !open);
          setReportError("");
          setReportSuccess("");
        }}
        className={`ml-auto h-10 w-10 rounded-lg text-lg transition-colors ${
          isReportOpen
            ? "bg-slate-800 text-slate-100"
            : "bg-slate-900/60 text-slate-500 hover:bg-slate-800 hover:text-slate-200"
        }`}
      >
        ⚑
      </button>

      {reportSuccess && !isReportOpen && (
        <p className="basis-full text-xs text-emerald-300" role="status">
          {reportSuccess}
        </p>
      )}

      {isReportOpen && (
        <div className="basis-full rounded-lg border border-amber-500/40 bg-amber-950/20 px-4 py-3 space-y-3">
          <div>
            <p className="text-sm font-semibold text-amber-100">
              Report this question
            </p>
            <p className="text-xs text-amber-50/80">
              Add a short note about what seems wrong, vague, or misleading.
            </p>
          </div>

          <label className="block space-y-2">
            <span className="text-xs text-slate-300">Comment</span>
            <textarea
              value={reportComment}
              onChange={(e) => setReportComment(e.target.value)}
              rows={4}
              className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
              placeholder="Example: option B conflicts with the explanation."
            />
          </label>

          {reportError && (
            <p className="text-xs text-rose-300" role="alert">
              {reportError}
            </p>
          )}

          {reportSuccess && (
            <p className="text-xs text-emerald-300" role="status">
              {reportSuccess}
            </p>
          )}

          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => {
                const didSubmit = submitQuestionReport(reportComment);
                if (!didSubmit) {
                  setReportError(
                    "Enter a comment before submitting the report.",
                  );
                  setReportSuccess("");
                  return;
                }

                setReportComment("");
                setReportError("");
                setReportSuccess("Report saved locally.");
                setIsReportOpen(false);
              }}
              className="px-3 py-2 rounded-lg bg-amber-400 text-slate-950 font-semibold text-sm"
            >
              Submit report
            </button>
            <button
              type="button"
              onClick={() => {
                setIsReportOpen(false);
                setReportComment("");
                setReportError("");
                setReportSuccess("");
              }}
              className="px-3 py-2 rounded-lg border border-slate-700 text-sm text-slate-200"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </>
  );
}

export default function QuizFooter({
  hasQuestion,
  currentQuestion,
  shuffledOptions,
  selectedIndexes,
  showResult,
  submitAnswer,
  nextQuestion,
  submitQuestionReport,
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
      <div className="flex flex-wrap gap-3">
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

        <QuestionReportControls
          key={currentQuestion?.id ?? "no-question"}
          submitQuestionReport={submitQuestionReport}
        />
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
            <MathText text={currentQuestion.explanation} />
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

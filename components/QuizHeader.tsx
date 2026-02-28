"use client";

import { useEffect, useRef, useState } from "react";
import type { DifficultyRange } from "../lib/useQuiz";
import { QUESTION_SOURCES, type SourceId } from "../lib/quiz";

type Props = {
  title: string;
  selectedSources: SourceId[];
  difficultyRange: DifficultyRange;
  applySelection: (payload: {
    sources: SourceId[];
    difficultyRange: DifficultyRange;
  }) => void;
  answeredCount: number;
  correctCount: number;
  accuracy: number;
  exportDifficultyJson: () => string;
  importDifficultyFromJson: (json: string) => void;
};

export default function QuizHeader({
  title,
  selectedSources,
  difficultyRange,
  applySelection,
  answeredCount,
  correctCount,
  accuracy,
  exportDifficultyJson,
  importDifficultyFromJson,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isSelectorOpen, setIsSelectorOpen] = useState(false);
  const [pendingSources, setPendingSources] =
    useState<SourceId[]>(selectedSources);
  const [pendingRange, setPendingRange] =
    useState<DifficultyRange>(difficultyRange);

  useEffect(() => {
    setPendingSources(selectedSources);
  }, [selectedSources]);

  useEffect(() => {
    setPendingRange(difficultyRange);
  }, [difficultyRange]);

  const clampRange = (range: DifficultyRange) => {
    const min = Math.max(0, Math.min(100, range.min));
    const max = Math.max(min, Math.min(100, range.max));
    return { min, max };
  };

  const togglePendingSource = (id: SourceId) => {
    setPendingSources((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id],
    );
  };

  const handleSelectAllSources = () => {
    setPendingSources(QUESTION_SOURCES.map((src) => src.id));
  };

  const handleApplySelection = () => {
    const safeSources =
      pendingSources.length === 0
        ? QUESTION_SOURCES.map((src) => src.id)
        : Array.from(new Set(pendingSources));
    const clampedRange = clampRange(pendingRange);

    applySelection({
      sources: safeSources,
      difficultyRange: clampedRange,
    });
    setPendingSources(safeSources);
    setPendingRange(clampedRange);
    setIsSelectorOpen(false);
  };

  const handleCancelSelection = () => {
    setPendingSources(selectedSources);
    setPendingRange(difficultyRange);
    setIsSelectorOpen(false);
  };

  const handleExportClick = () => {
    const json = exportDifficultyJson();
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "quiz-ratings.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        importDifficultyFromJson(reader.result);
      }
      e.target.value = "";
    };
    reader.readAsText(file);
  };

  const selectedLabels = QUESTION_SOURCES.filter((src) =>
    selectedSources.includes(src.id),
  ).map((src) => src.label);
  const allSelected =
    selectedSources.length === QUESTION_SOURCES.length &&
    QUESTION_SOURCES.every((src) => selectedSources.includes(src.id));
  const selectionSummary =
    allSelected && selectedLabels.length
      ? "All sources"
      : selectedLabels.length
        ? selectedLabels.slice(0, 3).join(", ") +
          (selectedLabels.length > 3
            ? ` +${selectedLabels.length - 3} more`
            : "")
        : "No sources selected";

  return (
    <header className="space-y-3">
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-semibold">{title}</h1>
          <p className="text-sm text-slate-400 mt-1">
            Multi-select questions —{" "}
            <span className="font-semibold">select all TRUE statements</span>{" "}
            and then submit.
          </p>
          <p className="text-xs text-slate-400 mt-2">
            Current selection:{" "}
            <span className="font-semibold text-slate-100">
              {selectionSummary}
            </span>{" "}
            Difficulty:{" "}
            <span className="font-semibold text-slate-100">
              {difficultyRange.min}–{difficultyRange.max}
            </span>
          </p>
        </div>

        <div className="flex flex-col items-start md:items-end gap-2 w-full md:w-auto">
          <button
            type="button"
            onClick={() => setIsSelectorOpen((open) => !open)}
            className="px-3 py-2 rounded-md bg-slate-100 text-slate-900 text-sm font-semibold w-full md:w-auto"
          >
            {isSelectorOpen ? "Close selection" : "Choose sources & range"}
          </button>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleExportClick}
              className="px-3 py-1 rounded-md bg-slate-800 border border-slate-700 text-xs"
            >
              Export ratings
            </button>
            <button
              type="button"
              onClick={handleImportClick}
              className="px-3 py-1 rounded-md bg-slate-800 border border-slate-700 text-xs"
            >
              Import ratings
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="application/json"
              className="hidden"
              onChange={handleFileChange}
            />
          </div>

          <div className="text-xs text-slate-400">
            Answered:{" "}
            <span className="font-semibold text-slate-200">
              {answeredCount}
            </span>{" "}
            Correct:{" "}
            <span className="font-semibold text-emerald-300">
              {correctCount}
            </span>{" "}
            Accuracy: <span className="font-semibold">{accuracy}%</span>
          </div>
        </div>
      </div>

      {isSelectorOpen && (
        <div className="w-full rounded-xl border border-slate-800 bg-slate-900/70 p-4 space-y-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <h3 className="text-sm font-semibold text-slate-100">
                Pick topics and difficulty
              </h3>
              <p className="text-xs text-slate-400">
                Tick any mix of chapters or lectures, set your range, then apply
                to reload the quiz.
              </p>
            </div>
            <button
              type="button"
              onClick={handleCancelSelection}
              className="text-xs text-slate-400 hover:text-slate-200"
            >
              Cancel
            </button>
          </div>

          <div className="grid sm:grid-cols-2 gap-3 max-h-72 overflow-y-auto pr-1">
            {QUESTION_SOURCES.map((src) => {
              const checked = pendingSources.includes(src.id);
              return (
                <label
                  key={src.id}
                  className={`flex gap-3 items-start rounded-lg border px-3 py-2 cursor-pointer transition-colors ${
                    checked
                      ? "border-sky-400 bg-slate-800"
                      : "border-slate-700 bg-slate-900/60"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => togglePendingSource(src.id)}
                    className="mt-1 accent-sky-400 h-4 w-4"
                  />
                  <div className="space-y-1">
                    <div className="text-sm font-semibold text-slate-100">
                      {src.label}
                    </div>
                    <div className="text-xs text-slate-400 leading-snug">
                      {src.title}
                    </div>
                  </div>
                </label>
              );
            })}
          </div>

          <div className="flex flex-wrap items-center gap-2 text-sm text-slate-200">
            <span className="text-slate-300">Difficulty range:</span>
            <input
              type="number"
              min={0}
              max={100}
              value={pendingRange.min}
              onChange={(e) =>
                setPendingRange((prev) => ({
                  ...prev,
                  min: Number(e.target.value),
                }))
              }
              className="w-20 bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-sm"
            />
            <span className="text-slate-400">to</span>
            <input
              type="number"
              min={0}
              max={100}
              value={pendingRange.max}
              onChange={(e) =>
                setPendingRange((prev) => ({
                  ...prev,
                  max: Number(e.target.value),
                }))
              }
              className="w-20 bg-slate-800 border border-slate-700 rounded-md px-2 py-1 text-sm"
            />
          </div>

          <div className="flex items-center justify-between gap-3">
            <button
              type="button"
              onClick={handleSelectAllSources}
              className="text-xs px-3 py-2 rounded-md border border-slate-700 text-slate-200 hover:border-slate-500"
            >
              Select all topics
            </button>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={handleCancelSelection}
                className="px-3 py-2 rounded-md border border-slate-700 text-xs text-slate-200"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleApplySelection}
                className="px-3 py-2 rounded-md bg-sky-500 text-slate-950 text-xs font-semibold"
              >
                Apply selection
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import type { DifficultyRange, QuestionSelectionMode } from "../lib/useQuiz";
import {
  ALL_TOPICS,
  QUESTION_SOURCES,
  SOURCE_SERIES,
  type SourceId,
  type SourceSeriesId,
  type Topic,
} from "../lib/quiz";

type Props = {
  title: string;
  selectedSources: SourceId[];
  selectedSeries: SourceSeriesId[];
  selectedTopics: Topic[];
  selectionMode: QuestionSelectionMode;
  difficultyRange: DifficultyRange;
  applySelection: (payload: {
    sources: SourceId[];
    series: SourceSeriesId[];
    topics: Topic[];
    mode: QuestionSelectionMode;
    difficultyRange: DifficultyRange;
  }) => void;
  answeredCount: number;
  correctCount: number;
  accuracy: number;
  userRating: number;
  userRatingRd: number;
  exportDifficultyJson: () => string;
  importDifficultyFromJson: (json: string) => Promise<void>;
  exportReportsJson: () => Promise<string>;
};

export default function QuizHeader({
  title,
  selectedSources,
  selectedSeries,
  selectedTopics,
  selectionMode,
  difficultyRange,
  applySelection,
  answeredCount,
  correctCount,
  accuracy,
  userRating,
  userRatingRd,
  exportDifficultyJson,
  importDifficultyFromJson,
  exportReportsJson,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isSelectorOpen, setIsSelectorOpen] = useState(false);
  const [pendingSources, setPendingSources] =
    useState<SourceId[]>(selectedSources);
  const [pendingSeries, setPendingSeries] =
    useState<SourceSeriesId[]>(selectedSeries);
  const [pendingTopics, setPendingTopics] = useState<Topic[]>(selectedTopics);
  const [pendingMode, setPendingMode] =
    useState<QuestionSelectionMode>(selectionMode);
  const [pendingRange, setPendingRange] =
    useState<DifficultyRange>(difficultyRange);

  useEffect(() => {
    setPendingSources(selectedSources);
  }, [selectedSources]);

  useEffect(() => {
    setPendingSeries(selectedSeries);
  }, [selectedSeries]);

  useEffect(() => {
    setPendingTopics(selectedTopics);
  }, [selectedTopics]);

  useEffect(() => {
    setPendingMode(selectionMode);
  }, [selectionMode]);

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

  const togglePendingSeries = (id: SourceSeriesId) => {
    setPendingSeries((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id],
    );
  };

  const togglePendingTopic = (topic: Topic) => {
    setPendingTopics((prev) =>
      prev.includes(topic) ? prev.filter((t) => t !== topic) : [...prev, topic],
    );
  };

  const handleSelectAllSeries = () => {
    setPendingSeries(SOURCE_SERIES.map((series) => series.id));
  };

  const handleSelectAllTopics = () => {
    setPendingTopics([...ALL_TOPICS]);
  };

  const handleClearAll = () => {
    setPendingSources([]);
    setPendingSeries([]);
    setPendingTopics([]);
  };

  const handleApplySelection = () => {
    const safeSources = Array.from(new Set(pendingSources));
    const safeSeries = Array.from(new Set(pendingSeries));
    const safeTopics = Array.from(new Set(pendingTopics));
    const clampedRange = clampRange(pendingRange);

    applySelection({
      sources: safeSources,
      series: safeSeries,
      topics: safeTopics,
      mode: pendingMode,
      difficultyRange: clampedRange,
    });
    setPendingSources(safeSources);
    setPendingSeries(safeSeries);
    setPendingTopics(safeTopics);
    setPendingRange(clampedRange);
    setIsSelectorOpen(false);
  };

  const handleCancelSelection = () => {
    setPendingSources(selectedSources);
    setPendingSeries(selectedSeries);
    setPendingTopics(selectedTopics);
    setPendingMode(selectionMode);
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

  const handleExportReportsClick = async () => {
    const json = await exportReportsJson();
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = "quiz-question-reports.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const handleFileChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        void importDifficultyFromJson(reader.result);
      }
      e.target.value = "";
    };
    reader.readAsText(file);
  };

  return (
    <header className="space-y-3">
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
        <div>
          <h1 className="text-2xl md:text-3xl font-semibold">{title}</h1>
          <p className="text-sm text-slate-400 mt-1">
            Multi-select questions -{" "}
            <span className="font-semibold">select all TRUE statements</span>{" "}
            and then submit.
          </p>
          <p className="text-xs text-slate-400 mt-2">
            Filters use OR logic (series/lectures/topics). Current mode:{" "}
            <span className="font-semibold text-slate-100">
              {selectionMode}
            </span>
          </p>
          <p className="text-xs text-slate-400 mt-1">
            Glicko rating:{" "}
            <span className="font-semibold text-slate-100">
              {Math.round(userRating)}
            </span>{" "}
            �{" "}
            <span className="font-semibold text-slate-100">
              {Math.round(userRatingRd)}
            </span>
          </p>
        </div>

        <div className="flex flex-col items-start md:items-end gap-2 w-full md:w-auto">
          <button
            type="button"
            onClick={() => setIsSelectorOpen((open) => !open)}
            className="px-3 py-2 rounded-md bg-slate-100 text-slate-900 text-sm font-semibold w-full md:w-auto"
          >
            {isSelectorOpen ? "Close selection" : "Choose filters"}
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
            <button
              type="button"
              onClick={() => {
                void handleExportReportsClick();
              }}
              className="px-3 py-1 rounded-md bg-slate-800 border border-slate-700 text-xs"
            >
              Export reports
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
                Pick mode, series, lectures, topics and difficulty
              </h3>
              <p className="text-xs text-slate-400">
                Keep any combination selected. Questions are included if they
                match at least one selected filter.
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

          <div className="space-y-2">
            <div className="text-xs font-semibold text-slate-300">Mode</div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setPendingMode("standard")}
                className={`px-3 py-2 rounded-md text-xs border ${
                  pendingMode === "standard"
                    ? "border-sky-400 bg-slate-800 text-slate-100"
                    : "border-slate-700 text-slate-300"
                }`}
              >
                Standard
              </button>
              <button
                type="button"
                onClick={() => setPendingMode("climb")}
                className={`px-3 py-2 rounded-md text-xs border ${
                  pendingMode === "climb"
                    ? "border-sky-400 bg-slate-800 text-slate-100"
                    : "border-slate-700 text-slate-300"
                }`}
              >
                Climb
              </button>
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-xs font-semibold text-slate-300">Topics</div>
            <div className="flex flex-wrap gap-2">
              {ALL_TOPICS.map((topic) => {
                const checked = pendingTopics.includes(topic);
                return (
                  <label
                    key={topic}
                    className={`flex items-center gap-2 rounded-md border px-3 py-2 text-xs cursor-pointer ${
                      checked
                        ? "border-sky-400 bg-slate-800 text-slate-100"
                        : "border-slate-700 bg-slate-900/60 text-slate-300"
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => togglePendingTopic(topic)}
                      className="accent-sky-400 h-4 w-4"
                    />
                    {topic}
                  </label>
                );
              })}
            </div>
          </div>

          <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
            <div className="text-xs font-semibold text-slate-300">
              Series and lectures
            </div>
            {SOURCE_SERIES.map((series) => {
              const seriesChecked = pendingSeries.includes(series.id);
              const seriesSources = QUESTION_SOURCES.filter(
                (source) => source.seriesId === series.id,
              );

              return (
                <details
                  key={series.id}
                  className="rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2"
                >
                  <summary className="flex items-center justify-between gap-3 cursor-pointer list-none">
                    <button
                      type="button"
                      className="flex items-center gap-2 text-sm font-semibold text-slate-100"
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        togglePendingSeries(series.id);
                      }}
                    >
                      <input
                        type="checkbox"
                        aria-label={series.label}
                        checked={seriesChecked}
                        readOnly
                        className="pointer-events-none accent-sky-400 h-4 w-4"
                      />
                      {series.label}
                    </button>
                    <span className="text-xs text-slate-400">
                      {seriesSources.length} lectures/chapters
                    </span>
                  </summary>

                  <div className="mt-3 space-y-2">
                    {seriesSources.map((source) => {
                      const sourceChecked = pendingSources.includes(source.id);
                      return (
                        <button
                          type="button"
                          key={source.id}
                          onClick={() => togglePendingSource(source.id)}
                          className="w-full text-left flex gap-3 items-start rounded-md border border-slate-700 bg-slate-900 px-3 py-2 cursor-pointer"
                        >
                          <input
                            type="checkbox"
                            aria-label={source.label}
                            checked={sourceChecked}
                            readOnly
                            className="pointer-events-none mt-1 accent-sky-400 h-4 w-4"
                          />
                          <div className="space-y-1">
                            <div className="text-sm font-semibold text-slate-100">
                              {source.label}
                            </div>
                            <div className="text-xs text-slate-400 leading-snug">
                              {source.title}
                            </div>
                            <div className="text-[11px] text-slate-500">
                              Topic: {source.topic}
                            </div>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                </details>
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
            <div className="flex gap-2">
              <button
                type="button"
                onClick={handleSelectAllSeries}
                className="text-xs px-3 py-2 rounded-md border border-slate-700 text-slate-200 hover:border-slate-500"
              >
                Select all series
              </button>
              <button
                type="button"
                onClick={handleSelectAllTopics}
                className="text-xs px-3 py-2 rounded-md border border-slate-700 text-slate-200 hover:border-slate-500"
              >
                Select all topics
              </button>
              <button
                type="button"
                onClick={handleClearAll}
                className="text-xs px-3 py-2 rounded-md border border-slate-700 text-slate-200 hover:border-slate-500"
              >
                Clear all
              </button>
            </div>
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

"use client";

import { useEffect, useState } from "react";
import type { DifficultyRange, QuestionSelectionMode } from "../lib/useQuiz";
import {
  ALL_TOPICS,
  ALL_SOURCE_IDS,
  QUESTION_SOURCES,
  SOURCE_SERIES,
  getSeriesIdsForSources,
  type SourceId,
  type SourceSeriesId,
  type Topic,
} from "../lib/quiz";

type Props = {
  title: string;
  selectedSources: SourceId[];
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
  resetParticipantRating: () => Promise<boolean>;
};

const QUESTION_ELO_FILTER_MIN = 0;
const QUESTION_ELO_FILTER_MAX = 3000;

export default function QuizHeader({
  title,
  selectedSources,
  selectedTopics,
  selectionMode,
  difficultyRange,
  applySelection,
  answeredCount,
  correctCount,
  accuracy,
  userRating,
  userRatingRd,
  resetParticipantRating,
}: Props) {
  const [isSelectorOpen, setIsSelectorOpen] = useState(true);
  const [isResettingRating, setIsResettingRating] = useState(false);
  const [resetRatingStatus, setResetRatingStatus] = useState<string | null>(
    null,
  );
  const [pendingSources, setPendingSources] =
    useState<SourceId[]>(selectedSources);
  const [pendingTopics, setPendingTopics] = useState<Topic[]>(selectedTopics);
  const [pendingMode, setPendingMode] =
    useState<QuestionSelectionMode>(selectionMode);
  const [pendingRange, setPendingRange] =
    useState<DifficultyRange>(difficultyRange);
  const pendingSeries = getSeriesIdsForSources(pendingSources);

  useEffect(() => {
    setPendingSources(selectedSources);
  }, [selectedSources]);

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
    const min = Math.max(
      QUESTION_ELO_FILTER_MIN,
      Math.min(QUESTION_ELO_FILTER_MAX, range.min),
    );
    const max = Math.max(min, Math.min(QUESTION_ELO_FILTER_MAX, range.max));
    return { min, max };
  };

  const togglePendingSource = (id: SourceId) => {
    setPendingSources((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id],
    );
  };

  const togglePendingSeries = (id: SourceSeriesId) => {
    const series = SOURCE_SERIES.find((entry) => entry.id === id);
    if (!series) return;

    setPendingSources((prev) => {
      const selected = new Set(prev);
      const anySelected = series.sourceIds.some((sourceId) =>
        selected.has(sourceId),
      );

      if (anySelected) {
        return prev.filter((sourceId) => !series.sourceIds.includes(sourceId));
      }

      return Array.from(new Set([...prev, ...series.sourceIds]));
    });
  };

  const togglePendingTopic = (topic: Topic) => {
    setPendingTopics((prev) =>
      prev.includes(topic) ? prev.filter((t) => t !== topic) : [...prev, topic],
    );
  };

  const handleSelectAllSeries = () => {
    setPendingSources(ALL_SOURCE_IDS);
  };

  const handleSelectAllTopics = () => {
    setPendingTopics([...ALL_TOPICS]);
  };

  const handleClearAll = () => {
    setPendingSources([]);
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
    setPendingTopics(safeTopics);
    setPendingRange(clampedRange);
    setIsSelectorOpen(false);
  };

  const handleCancelSelection = () => {
    setPendingSources(selectedSources);
    setPendingTopics(selectedTopics);
    setPendingMode(selectionMode);
    setPendingRange(difficultyRange);
    setIsSelectorOpen(false);
  };

  const handleResetRating = async () => {
    setIsResettingRating(true);
    setResetRatingStatus(null);
    const didReset = await resetParticipantRating();
    setResetRatingStatus(didReset ? "Rating reset" : "Reset failed");
    setIsResettingRating(false);
  };

  return (
    <header className="space-y-3">
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
        <div>
          <h1 className="text-base md:text-lg font-medium text-slate-300">
            {title}
          </h1>
          <p className="text-xs text-slate-400 mt-2">
            Glicko rating:{" "}
            <span className="font-semibold text-slate-100">
              {Math.round(userRating)}
            </span>{" "}
            +/-{" "}
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
                Pick mode, series, lectures, topics and question Elo range
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
            <div className="text-xs font-semibold text-slate-300">
              Glicko rating
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={handleResetRating}
                disabled={isResettingRating}
                className="text-xs px-3 py-2 rounded-md border border-slate-700 text-slate-200 hover:border-slate-500 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isResettingRating ? "Resetting..." : "Reset Glicko rating"}
              </button>
              {resetRatingStatus && (
                <span
                  role="status"
                  className="text-xs text-slate-400"
                  aria-live="polite"
                >
                  {resetRatingStatus}
                </span>
              )}
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
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-100">
                      <input
                        type="checkbox"
                        aria-label={series.label}
                        checked={seriesChecked}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => {
                          e.stopPropagation();
                          togglePendingSeries(series.id);
                        }}
                        className="accent-sky-400 h-4 w-4"
                      />
                      {series.label}
                    </div>
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
            <span className="text-slate-300">Question Elo range:</span>
            <input
              type="number"
              min={QUESTION_ELO_FILTER_MIN}
              max={QUESTION_ELO_FILTER_MAX}
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
              min={QUESTION_ELO_FILTER_MIN}
              max={QUESTION_ELO_FILTER_MAX}
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

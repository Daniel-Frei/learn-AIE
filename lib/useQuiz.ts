// lib/useQuiz.ts
"use client";

import { useEffect, useMemo, useState } from "react";
import {
  getOrCreateParticipantId,
  hasCompletedLegacyMigration,
  markLegacyMigrationCompleted,
} from "./client/participantStorage";
import {
  SOURCE_SERIES,
  allQuestions,
  getQuestionSourceMetadata,
  getQuestionsForFilters,
  type SourceId,
  type SourceSeriesId,
  type Topic,
} from "./quiz";
import { loadRatingState } from "./difficultyStore";
import { loadQuestionReports } from "./questionReportsStore";
import {
  computeQuestionDifficultyScore,
  createDefaultRatingState,
  exportRatingsJson,
  getQuestionRatingEstimate,
  importRatingsJson,
  type QuestionMetadataMap,
  type RatingStateV2,
} from "./ratingEngine";
import type {
  LocalMigrationResponse,
  QuizStateResponse,
  RecordAnswerResponse,
  ReportsExportResponse,
  SubmitQuestionReportResponse,
} from "./quizSync";
import type { Question } from "./quiz";

// Difficulty filter is now a numeric range [0,100]
export type DifficultyRange = {
  min: number; // inclusive, 0 = easiest
  max: number; // inclusive, 100 = hardest
};

export type QuestionSelectionMode = "standard" | "climb";

function shuffle<T>(items: T[]): T[] {
  const arr = [...items];
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

const SERIES_TO_SOURCE_IDS = new Map(
  SOURCE_SERIES.map((series) => [series.id, series.sourceIds]),
);

const QUESTION_METADATA: QuestionMetadataMap = Object.fromEntries(
  allQuestions.map((q) => [q.id, { label: q.difficulty }]),
);

function resolveSources(
  sources: SourceId[],
  seriesIds: SourceSeriesId[],
): SourceId[] {
  const resolved = new Set<SourceId>(sources);
  for (const seriesId of seriesIds) {
    const sourceIds = SERIES_TO_SOURCE_IDS.get(seriesId) ?? [];
    for (const sourceId of sourceIds) {
      resolved.add(sourceId as SourceId);
    }
  }
  return Array.from(resolved);
}

function pickClimbQuestionId(
  pool: Question[],
  ratingState: RatingStateV2,
  recentQuestionIds: string[],
): string | null {
  if (pool.length === 0) return null;

  const targetRating = ratingState.user.rating;
  const recent = new Set(recentQuestionIds.slice(-6));

  const scored = pool
    .map((question) => {
      const estimate = getQuestionRatingEstimate(
        question.id,
        question.difficulty,
        ratingState,
      );
      const distance = Math.abs(estimate.rating - targetRating);
      const uncertaintyBonus = Math.min(estimate.rd, 250) * 0.25;
      const randomJitter = Math.random() * 90;
      const repeatPenalty = recent.has(question.id) ? 180 : 0;
      return {
        id: question.id,
        score: distance - uncertaintyBonus + randomJitter + repeatPenalty,
      };
    })
    .sort((a, b) => a.score - b.score);

  const shortlistSize = Math.min(8, scored.length);
  const shortlist = scored.slice(0, shortlistSize);
  const choice = shortlist[Math.floor(Math.random() * shortlist.length)];
  return choice?.id ?? null;
}

export function useQuiz() {
  const initialSources: SourceId[] = [];
  const initialSeries: SourceSeriesId[] = [];
  const initialTopics: Topic[] = [];
  const initialRange: DifficultyRange = { min: 0, max: 100 };
  const initialMode: QuestionSelectionMode = "standard";

  // Keep initial render deterministic across SSR + hydration.
  const [ratingState, setRatingState] = useState<RatingStateV2>(() =>
    createDefaultRatingState(),
  );
  const [participantId, setParticipantId] = useState<string | null>(null);
  const [reportCountsByQuestion, setReportCountsByQuestion] = useState<
    Record<string, number>
  >({});
  const [totalReportCount, setTotalReportCount] = useState(0);

  const [selectedSources, setSelectedSources] =
    useState<SourceId[]>(initialSources);
  const [selectedSeries, setSelectedSeries] =
    useState<SourceSeriesId[]>(initialSeries);
  const [selectedTopics, setSelectedTopics] = useState<Topic[]>(initialTopics);
  const [selectionMode, setSelectionMode] =
    useState<QuestionSelectionMode>(initialMode);

  // numeric difficulty filter (applied on selection)
  const [difficultyRange, setDifficultyRange] =
    useState<DifficultyRange>(initialRange);

  const [appliedQuestionIds, setAppliedQuestionIds] = useState<string[]>([]);

  const sourcePool = useMemo(() => {
    const resolvedSources = resolveSources(selectedSources, selectedSeries);
    return getQuestionsForFilters(resolvedSources, selectedTopics);
  }, [selectedSources, selectedSeries, selectedTopics]);

  const questionById = useMemo(() => {
    return new Map(sourcePool.map((q) => [q.id, q]));
  }, [sourcePool]);

  const availableQuestions = useMemo(() => {
    return appliedQuestionIds
      .map((id) => questionById.get(id))
      .filter((q): q is NonNullable<typeof q> => Boolean(q));
  }, [appliedQuestionIds, questionById]);

  const [questionOrder, setQuestionOrder] = useState<number[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedIndexes, setSelectedIndexes] = useState<number[]>([]);
  const [showResult, setShowResult] = useState<null | { isCorrect: boolean }>(
    null,
  );
  const [answeredCount, setAnsweredCount] = useState(0);
  const [correctCount, setCorrectCount] = useState(0);
  const [climbQuestionId, setClimbQuestionId] = useState<string | null>(null);
  const [climbRecentIds, setClimbRecentIds] = useState<string[]>([]);

  const applyRemoteState = (response: QuizStateResponse) => {
    setRatingState(response.ratingState);
    setReportCountsByQuestion(response.reportSummary.countsByQuestion);
    setTotalReportCount(response.reportSummary.totalReportCount);
  };

  const fetchJson = async <T>(
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<T> => {
    const res = await fetch(input, {
      headers: {
        "content-type": "application/json",
        ...(init?.headers ?? {}),
      },
      ...init,
    });

    const body = (await res.json()) as T & { error?: string };
    if (!res.ok) {
      throw new Error(body.error ?? `Request failed with status ${res.status}`);
    }
    return body;
  };

  useEffect(() => {
    let cancelled = false;

    const bootstrap = async () => {
      const nextParticipantId = getOrCreateParticipantId();
      if (!nextParticipantId) return;

      setParticipantId(nextParticipantId);

      try {
        const initial = await fetchJson<QuizStateResponse>(
          `/api/quiz-state?participantId=${encodeURIComponent(nextParticipantId)}`,
          { method: "GET" },
        );
        if (cancelled) return;

        applyRemoteState(initial);
        if (initial.legacyMigrationCompleted) {
          markLegacyMigrationCompleted(nextParticipantId);
          return;
        }

        if (hasCompletedLegacyMigration(nextParticipantId)) {
          return;
        }

        const localRatingState = loadRatingState(QUESTION_METADATA);
        const localReportState = loadQuestionReports();
        const hasLocalRatings =
          localRatingState.user.gamesPlayed > 0 ||
          Object.keys(localRatingState.questions).length > 0;
        const hasLocalReports = localReportState.reports.length > 0;

        if (!hasLocalRatings && !hasLocalReports) {
          return;
        }

        const migrated = await fetchJson<LocalMigrationResponse>(
          "/api/local-migration",
          {
            method: "POST",
            body: JSON.stringify({
              participantId: nextParticipantId,
              localRatingState: hasLocalRatings ? localRatingState : undefined,
              localReportState: hasLocalReports ? localReportState : undefined,
            }),
          },
        );
        if (cancelled) return;

        applyRemoteState(migrated);
        markLegacyMigrationCompleted(nextParticipantId);
      } catch (err) {
        console.error("Failed to bootstrap remote quiz state:", err);
      }
    };

    void bootstrap();

    return () => {
      cancelled = true;
    };
  }, []);

  const currentQuestion = useMemo(() => {
    if (!availableQuestions.length) return null;

    if (selectionMode === "climb") {
      if (!climbQuestionId) {
        return availableQuestions[0];
      }
      return questionById.get(climbQuestionId) ?? availableQuestions[0];
    }

    if (!questionOrder.length) return null;
    const idxInArray = questionOrder[currentIndex] ?? questionOrder[0];
    return availableQuestions[idxInArray];
  }, [
    availableQuestions,
    selectionMode,
    climbQuestionId,
    questionById,
    questionOrder,
    currentIndex,
  ]);

  const shuffledOptions = useMemo(() => {
    if (!currentQuestion) return [];
    return shuffle(currentQuestion.options);
  }, [currentQuestion]);

  // current question difficulty score (0–1)
  const currentDifficultyScore = useMemo(() => {
    if (!currentQuestion) return null;
    return computeQuestionDifficultyScore(
      currentQuestion.id,
      currentQuestion.difficulty,
      ratingState,
    );
  }, [currentQuestion, ratingState]);

  const toggleOption = (idx: number) => {
    if (showResult) return;
    setSelectedIndexes((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx],
    );
  };

  const nextQuestion = () => {
    if (!availableQuestions.length) return;

    // Clear result & selections for the NEXT question
    setShowResult(null);
    setSelectedIndexes([]);

    if (selectionMode === "climb") {
      const previousId = currentQuestion?.id;
      const nextRecent = previousId
        ? [...climbRecentIds, previousId].slice(-8)
        : climbRecentIds;
      const nextId = pickClimbQuestionId(
        availableQuestions,
        ratingState,
        nextRecent,
      );

      setClimbRecentIds(nextRecent);
      setClimbQuestionId(nextId);
      setCurrentIndex((prev) =>
        availableQuestions.length > 0
          ? (prev + 1) % availableQuestions.length
          : 0,
      );
      return;
    }

    const next = currentIndex + 1;
    if (next >= availableQuestions.length) {
      const order = shuffle(
        Array.from({ length: availableQuestions.length }, (_, i) => i),
      );
      setQuestionOrder(order);
      setCurrentIndex(0);
    } else {
      setCurrentIndex(next);
    }
  };

  const accuracy =
    answeredCount === 0 ? 0 : Math.round((100 * correctCount) / answeredCount);

  const currentQuestionReportCount = useMemo(() => {
    if (!currentQuestion) return 0;
    return reportCountsByQuestion[currentQuestion.id] ?? 0;
  }, [currentQuestion, reportCountsByQuestion]);

  const submitQuestionReport = async (comment: string): Promise<boolean> => {
    if (!currentQuestion || !participantId) return false;

    const trimmedComment = comment.trim();
    if (!trimmedComment) return false;

    const sourceMetadata = getQuestionSourceMetadata(currentQuestion.id);
    if (!sourceMetadata) return false;

    try {
      const response = await fetchJson<SubmitQuestionReportResponse>(
        "/api/question-reports",
        {
          method: "POST",
          body: JSON.stringify({
            participantId,
            draft: {
              questionId: currentQuestion.id,
              comment: trimmedComment,
              snapshot: {
                sourceId: sourceMetadata.sourceId,
                sourceLabel: sourceMetadata.sourceLabel,
                seriesId: sourceMetadata.seriesId,
                seriesLabel: sourceMetadata.seriesLabel,
                topic: sourceMetadata.topic,
                prompt: currentQuestion.prompt,
              },
            },
          }),
        },
      );

      setTotalReportCount(response.totalReportCount);
      setReportCountsByQuestion((prev) => ({
        ...prev,
        [currentQuestion.id]: response.questionReportCount,
      }));
      return true;
    } catch (err) {
      console.error("Failed to submit question report:", err);
      return false;
    }
  };

  const clampRange = (newRange: DifficultyRange) => {
    const min = Math.max(0, Math.min(100, newRange.min));
    const max = Math.max(min, Math.min(100, newRange.max));
    return { min, max };
  };

  const applySelection = (payload: {
    sources: SourceId[];
    series: SourceSeriesId[];
    topics: Topic[];
    mode: QuestionSelectionMode;
    difficultyRange: DifficultyRange;
  }) => {
    const clampedRange = clampRange(payload.difficultyRange);
    const uniqueSources = Array.from(new Set(payload.sources));
    const uniqueSeries = Array.from(new Set(payload.series));
    const uniqueTopics = Array.from(new Set(payload.topics));
    const resolvedSources = resolveSources(uniqueSources, uniqueSeries);

    const pool = getQuestionsForFilters(resolvedSources, uniqueTopics);
    const eligibleIds = pool
      .filter((q) => {
        const score = computeQuestionDifficultyScore(
          q.id,
          q.difficulty,
          ratingState,
        );
        const scorePercent = Math.round(score * 100);
        return (
          scorePercent >= clampedRange.min && scorePercent <= clampedRange.max
        );
      })
      .map((q) => q.id);

    setSelectedSources(uniqueSources);
    setSelectedSeries(uniqueSeries);
    setSelectedTopics(uniqueTopics);
    setSelectionMode(payload.mode);
    setDifficultyRange(clampedRange);
    setAppliedQuestionIds(eligibleIds);
    setAnsweredCount(0);
    setCorrectCount(0);
    setCurrentIndex(0);
    setSelectedIndexes([]);
    setShowResult(null);

    if (payload.mode === "climb") {
      const eligibleSet = new Set(eligibleIds);
      const eligibleQuestions = pool.filter((q) => eligibleSet.has(q.id));
      const firstQuestionId = pickClimbQuestionId(
        eligibleQuestions,
        ratingState,
        [],
      );
      setClimbQuestionId(firstQuestionId);
      setClimbRecentIds(firstQuestionId ? [firstQuestionId] : []);
      setQuestionOrder([]);
      return;
    }

    setQuestionOrder(
      shuffle(Array.from({ length: eligibleIds.length }, (_, i) => i)),
    );
    setClimbQuestionId(null);
    setClimbRecentIds([]);
  };

  // -------- EXPORT / IMPORT HELPERS --------

  const exportDifficultyJson = () => {
    return exportRatingsJson(ratingState);
  };

  const importDifficultyFromJson = async (json: string) => {
    if (!participantId) return;
    const parsed = importRatingsJson(json, QUESTION_METADATA);
    if (!parsed) return;

    try {
      const migrated = await fetchJson<LocalMigrationResponse>(
        "/api/local-migration",
        {
          method: "POST",
          body: JSON.stringify({
            participantId,
            localRatingState: parsed,
          }),
        },
      );
      applyRemoteState(migrated);
      markLegacyMigrationCompleted(participantId);
    } catch (err) {
      console.error("Failed to import rating JSON into remote store:", err);
    }
  };

  const exportReportsJson = async () => {
    const exported = await fetchJson<ReportsExportResponse>(
      "/api/question-reports/export",
      { method: "GET" },
    );
    return JSON.stringify(exported, null, 2);
  };

  const submitAnswer = async () => {
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

    if (!participantId) return;

    try {
      const response = await fetchJson<RecordAnswerResponse>("/api/answers", {
        method: "POST",
        body: JSON.stringify({
          participantId,
          questionId: currentQuestion.id,
          label: currentQuestion.difficulty,
          isCorrect,
        }),
      });

      setRatingState((prev) => ({
        ...prev,
        user: response.user,
        questions: {
          ...prev.questions,
          [response.questionId]: response.question,
        },
      }));
    } catch (err) {
      console.error("Failed to sync answer to remote store:", err);
    }
  };

  return {
    // configuration
    selectedSources,
    selectedSeries,
    selectedTopics,
    selectionMode,
    difficultyRange,
    applySelection,

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
    userRating: ratingState.user.rating,
    userRatingRd: ratingState.user.rd,
    totalReportCount,
    currentQuestionReportCount,

    // export/import
    exportDifficultyJson,
    importDifficultyFromJson,
    exportReportsJson,
    submitQuestionReport,
  };
}

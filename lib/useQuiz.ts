// lib/useQuiz.ts
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getOrCreateParticipantId,
  hasCompletedLegacyMigration,
  markLegacyMigrationCompleted,
} from "./client/participantStorage";
import {
  allQuestions,
  DEFAULT_QUESTION_TYPES,
  getQuestionSourceContext,
  getQuestionSourceMetadata,
  getQuestionsForFilters,
  type QuestionType,
  type SourceId,
  type Topic,
} from "./quiz";
import { loadRatingState, saveRatingState } from "./difficultyStore";
import {
  createDefaultRatingState,
  getQuestionRatingEstimate,
  QUESTION_TIME_LIMIT_MS,
  recordAnswer,
  type QuestionMetadataMap,
  type RatingStateV2,
} from "./ratingEngine";
import {
  DEFAULT_DIFFICULTY_RANGE,
  DEFAULT_QUESTION_SELECTION_MODE,
  clampDifficultyRange,
  evaluateAnswer,
  getDisplayOptions,
  getEligibleQuestionIds,
  pickClimbQuestionId,
  QUESTION_TIMER_TICK_MS,
  shuffleItems,
  type DifficultyRange,
  type QuestionSelectionMode,
} from "./quizSession";
import type {
  LocalMigrationResponse,
  QuizStateResponse,
  RecordAnswerResponse,
  ResetParticipantRatingResponse,
  SubmitQuestionReportResponse,
} from "./quizSync";
export type { DifficultyRange, QuestionSelectionMode } from "./quizSession";

const QUESTION_METADATA: QuestionMetadataMap = Object.fromEntries(
  allQuestions.map((q) => [q.id, { label: q.difficulty }]),
);

type AnswerRatingSnapshot = {
  questionId: string;
  userRating: number;
  userRatingRd: number;
  questionRating: number;
  userDelta: number;
  questionDelta: number;
};

export type InitialQuizSelection = {
  sources?: readonly SourceId[];
  topics?: readonly Topic[];
  questionTypes?: readonly QuestionType[];
  mode?: QuestionSelectionMode;
  difficultyRange?: DifficultyRange;
};

const EMPTY_INITIAL_SELECTION: InitialQuizSelection = {};

export function useQuiz(
  initialSelection: InitialQuizSelection = EMPTY_INITIAL_SELECTION,
) {
  const initialSources: SourceId[] = useMemo(
    () => Array.from(new Set(initialSelection.sources ?? [])),
    [initialSelection.sources],
  );
  const initialTopics: Topic[] = useMemo(
    () => Array.from(new Set(initialSelection.topics ?? [])),
    [initialSelection.topics],
  );
  const initialQuestionTypes: QuestionType[] = useMemo(
    () =>
      Array.from(
        new Set(initialSelection.questionTypes ?? DEFAULT_QUESTION_TYPES),
      ),
    [initialSelection.questionTypes],
  );
  const initialRange: DifficultyRange = useMemo(
    () =>
      clampDifficultyRange(
        initialSelection.difficultyRange ?? DEFAULT_DIFFICULTY_RANGE,
      ),
    [initialSelection.difficultyRange],
  );
  const initialMode: QuestionSelectionMode =
    initialSelection.mode ?? DEFAULT_QUESTION_SELECTION_MODE;

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
  const [selectedTopics, setSelectedTopics] = useState<Topic[]>(initialTopics);
  const [selectedQuestionTypes, setSelectedQuestionTypes] =
    useState<QuestionType[]>(initialQuestionTypes);
  const [selectionMode, setSelectionMode] =
    useState<QuestionSelectionMode>(initialMode);

  // raw question Elo filter (applied on selection)
  const [difficultyRange, setDifficultyRange] =
    useState<DifficultyRange>(initialRange);

  const [appliedQuestionIds, setAppliedQuestionIds] = useState<string[]>([]);

  const sourcePool = useMemo(() => {
    return getQuestionsForFilters(
      selectedSources,
      selectedTopics,
      selectedQuestionTypes,
    );
  }, [selectedQuestionTypes, selectedSources, selectedTopics]);

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
  const [questionSessionId, setQuestionSessionId] = useState(0);
  const [questionTimerMs, setQuestionTimerMs] = useState(0);
  const [frozenQuestionTimerMs, setFrozenQuestionTimerMs] = useState<
    number | null
  >(null);
  const [answerRatingSnapshot, setAnswerRatingSnapshot] =
    useState<AnswerRatingSnapshot | null>(null);
  const questionStartedAtRef = useRef<number | null>(null);

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
        const hasLocalRatings =
          localRatingState.user.gamesPlayed > 0 ||
          Object.keys(localRatingState.questions).length > 0;

        if (!hasLocalRatings) {
          return;
        }

        const migrated = await fetchJson<LocalMigrationResponse>(
          "/api/local-migration",
          {
            method: "POST",
            body: JSON.stringify({
              participantId: nextParticipantId,
              localRatingState,
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

  const currentQuestionId = currentQuestion?.id ?? null;

  const visibleAnswerRatingSnapshot =
    showResult &&
    currentQuestionId &&
    answerRatingSnapshot?.questionId === currentQuestionId
      ? answerRatingSnapshot
      : null;

  const currentQuestionContext = useMemo(() => {
    if (!currentQuestion) return null;
    return getQuestionSourceContext(currentQuestion.id);
  }, [currentQuestion]);

  useEffect(() => {
    if (!currentQuestionId) {
      questionStartedAtRef.current = null;
      return;
    }

    const startedAt = Date.now();
    questionStartedAtRef.current = startedAt;

    const tick = () => {
      setQuestionTimerMs(
        Math.min(Date.now() - startedAt, QUESTION_TIME_LIMIT_MS),
      );
    };

    const timerId = window.setInterval(tick, QUESTION_TIMER_TICK_MS);
    return () => window.clearInterval(timerId);
  }, [currentQuestionId, questionSessionId]);

  const shuffledOptions = useMemo(() => {
    if (!currentQuestion) return [];
    return getDisplayOptions(currentQuestion);
  }, [currentQuestion]);

  const ratingStateQuestionRating = useMemo(() => {
    if (!currentQuestion) return null;
    return getQuestionRatingEstimate(
      currentQuestion.id,
      currentQuestion.difficulty,
      ratingState,
    ).rating;
  }, [currentQuestion, ratingState]);
  const currentQuestionRating =
    visibleAnswerRatingSnapshot?.questionRating ?? ratingStateQuestionRating;

  const displayedQuestionTimerMs =
    showResult && frozenQuestionTimerMs !== null
      ? frozenQuestionTimerMs
      : questionTimerMs;

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
    setQuestionTimerMs(0);
    setFrozenQuestionTimerMs(null);
    setAnswerRatingSnapshot(null);
    questionStartedAtRef.current = null;
    setQuestionSessionId((id) => id + 1);

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
      const order = shuffleItems(
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

  const applySelection = useCallback(
    (payload: {
      sources: SourceId[];
      series: string[];
      topics: Topic[];
      questionTypes?: QuestionType[];
      mode: QuestionSelectionMode;
      difficultyRange: DifficultyRange;
    }) => {
      const clampedRange = clampDifficultyRange(payload.difficultyRange);
      const uniqueSources = Array.from(new Set(payload.sources));
      const uniqueTopics = Array.from(new Set(payload.topics));
      const uniqueQuestionTypes = Array.from(
        new Set(payload.questionTypes ?? DEFAULT_QUESTION_TYPES),
      );

      const pool = getQuestionsForFilters(
        uniqueSources,
        uniqueTopics,
        uniqueQuestionTypes,
      );
      const eligibleIds = getEligibleQuestionIds({
        sources: uniqueSources,
        topics: uniqueTopics,
        questionTypes: uniqueQuestionTypes,
        difficultyRange: clampedRange,
        ratingState,
      });

      setSelectedSources(uniqueSources);
      setSelectedTopics(uniqueTopics);
      setSelectedQuestionTypes(uniqueQuestionTypes);
      setSelectionMode(payload.mode);
      setDifficultyRange(clampedRange);
      setAppliedQuestionIds(eligibleIds);
      setAnsweredCount(0);
      setCorrectCount(0);
      setCurrentIndex(0);
      setSelectedIndexes([]);
      setShowResult(null);
      setQuestionTimerMs(0);
      setFrozenQuestionTimerMs(null);
      setAnswerRatingSnapshot(null);
      questionStartedAtRef.current = null;
      setQuestionSessionId((id) => id + 1);

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
        shuffleItems(Array.from({ length: eligibleIds.length }, (_, i) => i)),
      );
      setClimbQuestionId(null);
      setClimbRecentIds([]);
    },
    [ratingState],
  );

  const didApplyInitialSelectionRef = useRef(false);

  useEffect(() => {
    if (didApplyInitialSelectionRef.current) return;
    if (initialSources.length === 0 && initialTopics.length === 0) return;

    const frameId = window.requestAnimationFrame(() => {
      if (didApplyInitialSelectionRef.current) return;
      didApplyInitialSelectionRef.current = true;
      applySelection({
        sources: initialSources,
        series: [],
        topics: initialTopics,
        questionTypes: initialQuestionTypes,
        mode: initialMode,
        difficultyRange: initialRange,
      });
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [
    applySelection,
    initialMode,
    initialQuestionTypes,
    initialRange,
    initialSources,
    initialTopics,
  ]);

  const resetParticipantRating = async (): Promise<boolean> => {
    if (!participantId) return false;

    try {
      const response = await fetchJson<ResetParticipantRatingResponse>(
        "/api/participant-rating-reset",
        {
          method: "POST",
          body: JSON.stringify({ participantId }),
        },
      );

      applyRemoteState(response);
      markLegacyMigrationCompleted(participantId);
      saveRatingState(createDefaultRatingState());
      setAnswerRatingSnapshot(null);

      return true;
    } catch (err) {
      console.error("Failed to reset participant rating:", err);
      return false;
    }
  };

  const submitAnswer = async () => {
    if (!currentQuestion || showResult) return;

    const evaluation = evaluateAnswer(shuffledOptions, selectedIndexes);
    const startedAt = questionStartedAtRef.current ?? Date.now();
    const elapsedMs = Math.min(
      Math.max(0, Date.now() - startedAt),
      QUESTION_TIME_LIMIT_MS,
    );

    const { isCorrect, mistakeCount } = evaluation;

    const userRatingBefore = ratingState.user.rating;
    const questionRatingBefore = getQuestionRatingEstimate(
      currentQuestion.id,
      currentQuestion.difficulty,
      ratingState,
    ).rating;
    const optimisticRatingState = recordAnswer(
      ratingState,
      currentQuestion.id,
      currentQuestion.difficulty,
      isCorrect,
      Date.now(),
      {
        elapsedMs,
        mistakeCount,
      },
    );
    const optimisticQuestionRating =
      optimisticRatingState.questions[currentQuestion.id];

    setFrozenQuestionTimerMs(elapsedMs);
    setRatingState(optimisticRatingState);
    setAnswerRatingSnapshot(
      optimisticQuestionRating
        ? {
            questionId: currentQuestion.id,
            userRating: optimisticRatingState.user.rating,
            userRatingRd: optimisticRatingState.user.rd,
            questionRating: optimisticQuestionRating.rating,
            userDelta: optimisticRatingState.user.rating - userRatingBefore,
            questionDelta:
              optimisticQuestionRating.rating - questionRatingBefore,
          }
        : null,
    );
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
          elapsedMs,
          mistakeCount,
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
    selectedTopics,
    selectedQuestionTypes,
    selectionMode,
    difficultyRange,
    applySelection,

    // question set info
    availableCount: availableQuestions.length,
    currentIndex,
    currentQuestion,
    currentQuestionContext,
    currentQuestionRating,
    questionRatingDelta: visibleAnswerRatingSnapshot?.questionDelta ?? null,
    questionElapsedMs: displayedQuestionTimerMs,
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
    userRating:
      visibleAnswerRatingSnapshot?.userRating ?? ratingState.user.rating,
    userRatingRd:
      visibleAnswerRatingSnapshot?.userRatingRd ?? ratingState.user.rd,
    userRatingDelta: visibleAnswerRatingSnapshot?.userDelta ?? null,
    totalReportCount,
    currentQuestionReportCount,

    submitQuestionReport,
    resetParticipantRating,
  };
}

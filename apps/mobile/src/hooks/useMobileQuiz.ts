import { useEffect, useMemo, useRef, useState } from "react";
import {
  DEFAULT_QUESTION_TYPES,
  getQuestionSourceContext,
  getQuestionSourceMetadata,
  getQuestionsForFilters,
  type Question,
  type QuestionType,
  type SourceId,
  type SourceSeriesId,
  type Topic,
} from "../../../../lib/quiz";
import {
  createDefaultRatingState,
  getQuestionRatingEstimate,
  QUESTION_TIME_LIMIT_MS,
  recordAnswer,
  type RatingStateV2,
} from "../../../../lib/ratingEngine";
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
} from "../../../../lib/quizSession";
import {
  clearGuestMobileQuizState,
  createDefaultMobileQuizState,
  GUEST_PROFILE_ID,
  loadMobileQuizState,
  saveMobileQuizState,
  type MobilePersistedQuizState,
  type QueuedAnswerAttempt,
  type QueuedQuestionReport,
} from "../lib/mobileLocalStore";
import {
  rewriteQueuedParticipantIds,
  shouldMigrateGuestState,
} from "../lib/mobileProfileState";
import {
  getConfiguredSupabaseEnv,
  getMobileSupabaseClient,
  toMobileProfile,
  upsertMobileProfile,
  type MobileProfile,
} from "../lib/mobileSupabase";
import { syncMobileQuizState } from "../lib/mobileSync";

export type MobileSyncStatus =
  | "local"
  | "loading"
  | "syncing"
  | "synced"
  | "sync-error";

function makeId(prefix: string): string {
  if (
    typeof globalThis.crypto !== "undefined" &&
    typeof globalThis.crypto.randomUUID === "function"
  ) {
    return globalThis.crypto.randomUUID();
  }

  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function useMobileQuiz() {
  const [persistedState, setPersistedState] =
    useState<MobilePersistedQuizState>(() =>
      createDefaultMobileQuizState(GUEST_PROFILE_ID),
    );
  const [ratingState, setRatingState] = useState<RatingStateV2>(() =>
    createDefaultRatingState(),
  );
  const [profile, setProfile] = useState<MobileProfile | null>(null);
  const [syncStatus, setSyncStatus] = useState<MobileSyncStatus>("loading");
  const [syncMessage, setSyncMessage] = useState("");
  const [reportCountsByQuestion, setReportCountsByQuestion] = useState<
    Record<string, number>
  >({});
  const [totalReportCount, setTotalReportCount] = useState(0);

  const [selectedSources, setSelectedSources] = useState<SourceId[]>([]);
  const [selectedTopics, setSelectedTopics] = useState<Topic[]>([]);
  const [selectedQuestionTypes, setSelectedQuestionTypes] = useState<
    QuestionType[]
  >([...DEFAULT_QUESTION_TYPES]);
  const [selectionMode, setSelectionMode] = useState<QuestionSelectionMode>(
    DEFAULT_QUESTION_SELECTION_MODE,
  );
  const [difficultyRange, setDifficultyRange] = useState<DifficultyRange>(
    DEFAULT_DIFFICULTY_RANGE,
  );
  const [appliedQuestionIds, setAppliedQuestionIds] = useState<string[]>([]);

  const sourcePool = useMemo(
    () =>
      getQuestionsForFilters(
        selectedSources,
        selectedTopics,
        selectedQuestionTypes,
      ),
    [selectedQuestionTypes, selectedSources, selectedTopics],
  );
  const questionById = useMemo(
    () => new Map(sourcePool.map((question) => [question.id, question])),
    [sourcePool],
  );
  const availableQuestions = useMemo(
    () =>
      appliedQuestionIds
        .map((id) => questionById.get(id))
        .filter((question): question is Question => Boolean(question)),
    [appliedQuestionIds, questionById],
  );

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
  const questionStartedAtRef = useRef<number | null>(null);

  const applyPersistedState = async (
    nextState: MobilePersistedQuizState,
  ): Promise<void> => {
    setPersistedState(nextState);
    setRatingState(nextState.ratingState);
    setReportCountsByQuestion(nextState.reportCountsByQuestion);
    setTotalReportCount(nextState.totalReportCount);
    await saveMobileQuizState(nextState);
  };

  const syncNow = async (
    stateOverride?: MobilePersistedQuizState,
  ): Promise<boolean> => {
    const activeState = stateOverride ?? persistedState;
    const activeProfileId =
      profile?.id ??
      (activeState.profileId !== GUEST_PROFILE_ID
        ? activeState.profileId
        : null);

    if (!activeProfileId) {
      setSyncStatus("local");
      setSyncMessage("Sign in to sync ratings, reports, and progress.");
      return false;
    }

    if (!getConfiguredSupabaseEnv()) {
      setSyncStatus("sync-error");
      setSyncMessage("Set EXPO_PUBLIC_SUPABASE_URL and publishable key.");
      return false;
    }

    setSyncStatus("syncing");
    try {
      const syncedState = await syncMobileQuizState({
        ...activeState,
        profileId: activeProfileId,
      });
      await applyPersistedState(syncedState);
      setSyncStatus("synced");
      setSyncMessage("");
      return true;
    } catch (error) {
      console.error("Failed to sync mobile quiz state:", error);
      setSyncStatus("sync-error");
      setSyncMessage(
        "Saved locally. Sync will work when Supabase is reachable.",
      );
      return false;
    }
  };

  const loadProfileState = async (nextProfile: MobileProfile | null) => {
    if (!nextProfile) {
      const guestState = await loadMobileQuizState(GUEST_PROFILE_ID);
      await applyPersistedState(guestState);
      setSyncStatus("local");
      setSyncMessage("Sign in to sync ratings, reports, and progress.");
      return;
    }

    await upsertMobileProfile(nextProfile);

    const [profileState, guestState] = await Promise.all([
      loadMobileQuizState(nextProfile.id),
      loadMobileQuizState(GUEST_PROFILE_ID),
    ]);
    const nextState = shouldMigrateGuestState(profileState, guestState)
      ? rewriteQueuedParticipantIds(guestState, nextProfile.id)
      : profileState;

    await applyPersistedState(nextState);
    if (nextState.profileId === nextProfile.id) {
      await clearGuestMobileQuizState();
    }
    await syncNow(nextState);
  };

  useEffect(() => {
    let cancelled = false;

    const bootstrap = async () => {
      const supabase = getMobileSupabaseClient();
      if (!supabase) {
        const guestState = await loadMobileQuizState(GUEST_PROFILE_ID);
        if (cancelled) return;
        await applyPersistedState(guestState);
        setSyncStatus("local");
        setSyncMessage("Configure Supabase env vars to enable profile sync.");
        return;
      }

      const { data, error } = await supabase.auth.getSession();
      if (cancelled) return;
      if (error) {
        console.error("Failed to load Supabase session:", error);
      }

      const user = data.session?.user ?? null;
      const nextProfile = user ? toMobileProfile(user) : null;
      setProfile(nextProfile);
      await loadProfileState(nextProfile);
    };

    void bootstrap();

    const supabase = getMobileSupabaseClient();
    const subscription = supabase?.auth.onAuthStateChange((_event, session) => {
      const nextProfile = session?.user ? toMobileProfile(session.user) : null;
      setProfile(nextProfile);
      void loadProfileState(nextProfile);
    });

    return () => {
      cancelled = true;
      subscription?.data.subscription.unsubscribe();
    };
    // Initial session hydration should run once; auth changes are handled by the subscription above.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const signIn = async (email: string, password: string): Promise<boolean> => {
    const supabase = getMobileSupabaseClient();
    if (!supabase) {
      setSyncStatus("sync-error");
      setSyncMessage("Configure Supabase env vars to sign in.");
      return false;
    }

    const { data, error } = await supabase.auth.signInWithPassword({
      email: email.trim(),
      password,
    });
    if (error || !data.user) {
      setSyncStatus("sync-error");
      setSyncMessage(error?.message ?? "Sign in failed.");
      return false;
    }

    const nextProfile = toMobileProfile(data.user);
    setProfile(nextProfile);
    await loadProfileState(nextProfile);
    return true;
  };

  const signUp = async (email: string, password: string): Promise<boolean> => {
    const supabase = getMobileSupabaseClient();
    if (!supabase) {
      setSyncStatus("sync-error");
      setSyncMessage("Configure Supabase env vars to sign up.");
      return false;
    }

    const { data, error } = await supabase.auth.signUp({
      email: email.trim(),
      password,
    });
    if (error) {
      setSyncStatus("sync-error");
      setSyncMessage(error.message);
      return false;
    }

    if (data.user) {
      const nextProfile = toMobileProfile(data.user);
      setProfile(nextProfile);
      await loadProfileState(nextProfile);
    } else {
      setSyncStatus("local");
      setSyncMessage("Check your email to confirm the new account.");
    }
    return true;
  };

  const signOut = async (): Promise<void> => {
    const supabase = getMobileSupabaseClient();
    await supabase?.auth.signOut();
    setProfile(null);
    await loadProfileState(null);
  };

  const currentQuestion = useMemo(() => {
    if (!availableQuestions.length) return null;

    if (selectionMode === "climb") {
      if (!climbQuestionId) return availableQuestions[0];
      return questionById.get(climbQuestionId) ?? availableQuestions[0];
    }

    if (!questionOrder.length) return null;
    const idxInArray = questionOrder[currentIndex] ?? questionOrder[0];
    return availableQuestions[idxInArray];
  }, [
    availableQuestions,
    climbQuestionId,
    currentIndex,
    questionById,
    questionOrder,
    selectionMode,
  ]);

  const currentQuestionId = currentQuestion?.id ?? null;

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

    const timerId = setInterval(() => {
      setQuestionTimerMs(
        Math.min(Date.now() - startedAt, QUESTION_TIME_LIMIT_MS),
      );
    }, QUESTION_TIMER_TICK_MS);

    return () => clearInterval(timerId);
  }, [currentQuestionId, questionSessionId]);

  const shuffledOptions = useMemo(() => {
    if (!currentQuestion) return [];
    return getDisplayOptions(currentQuestion);
  }, [currentQuestion]);

  const currentQuestionRating = useMemo(() => {
    if (!currentQuestion) return null;
    return getQuestionRatingEstimate(
      currentQuestion.id,
      currentQuestion.difficulty,
      ratingState,
    ).rating;
  }, [currentQuestion, ratingState]);

  const currentQuestionReportCount = useMemo(() => {
    if (!currentQuestion) return 0;
    return reportCountsByQuestion[currentQuestion.id] ?? 0;
  }, [currentQuestion, reportCountsByQuestion]);

  const displayedQuestionTimerMs =
    showResult && frozenQuestionTimerMs !== null
      ? frozenQuestionTimerMs
      : questionTimerMs;

  const accuracy =
    answeredCount === 0 ? 0 : Math.round((100 * correctCount) / answeredCount);

  const toggleOption = (idx: number) => {
    if (showResult) return;
    setSelectedIndexes((prev) =>
      prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx],
    );
  };

  const resetQuestionState = () => {
    setShowResult(null);
    setSelectedIndexes([]);
    setQuestionTimerMs(0);
    setFrozenQuestionTimerMs(null);
    questionStartedAtRef.current = null;
    setQuestionSessionId((id) => id + 1);
  };

  const nextQuestion = () => {
    if (!availableQuestions.length) return;
    resetQuestionState();

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
      setCurrentIndex((prev) => (prev + 1) % availableQuestions.length);
      return;
    }

    const next = currentIndex + 1;
    if (next >= availableQuestions.length) {
      setQuestionOrder(
        shuffleItems(
          Array.from({ length: availableQuestions.length }, (_, i) => i),
        ),
      );
      setCurrentIndex(0);
    } else {
      setCurrentIndex(next);
    }
  };

  const applySelection = (payload: {
    sources: SourceId[];
    series: SourceSeriesId[];
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
    resetQuestionState();

    if (payload.mode === "climb") {
      const eligibleSet = new Set(eligibleIds);
      const eligibleQuestions = pool.filter((question) =>
        eligibleSet.has(question.id),
      );
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
  };

  const submitAnswer = async () => {
    if (!currentQuestion || showResult) return;

    const evaluation = evaluateAnswer(shuffledOptions, selectedIndexes);
    const startedAt = questionStartedAtRef.current ?? Date.now();
    const elapsedMs = Math.min(
      Math.max(0, Date.now() - startedAt),
      QUESTION_TIME_LIMIT_MS,
    );
    const answeredAt = new Date().toISOString();
    const participantId = profile?.id ?? GUEST_PROFILE_ID;
    const updatedRatingState = recordAnswer(
      ratingState,
      currentQuestion.id,
      currentQuestion.difficulty,
      evaluation.isCorrect,
      Date.parse(answeredAt),
      {
        elapsedMs,
        mistakeCount: evaluation.mistakeCount,
      },
    );
    const queuedAnswer: QueuedAnswerAttempt = {
      attemptId: makeId("answer"),
      participantId,
      questionId: currentQuestion.id,
      label: currentQuestion.difficulty,
      isCorrect: evaluation.isCorrect,
      elapsedMs,
      mistakeCount: evaluation.mistakeCount,
      answeredAt,
      source: "live",
    };
    const nextState: MobilePersistedQuizState = {
      ...persistedState,
      profileId: participantId,
      ratingState: updatedRatingState,
      queuedAnswers: [...persistedState.queuedAnswers, queuedAnswer],
    };

    setFrozenQuestionTimerMs(elapsedMs);
    setShowResult({ isCorrect: evaluation.isCorrect });
    setAnsweredCount((count) => count + 1);
    setCorrectCount((count) => count + (evaluation.isCorrect ? 1 : 0));
    await applyPersistedState(nextState);

    if (profile) {
      void syncNow(nextState);
    } else {
      setSyncStatus("local");
      setSyncMessage("Answer saved locally. Sign in to sync it.");
    }
  };

  const submitQuestionReport = async (comment: string): Promise<boolean> => {
    if (!currentQuestion || !profile) {
      setSyncStatus("local");
      setSyncMessage("Sign in to submit question reports.");
      return false;
    }

    const trimmedComment = comment.trim();
    if (!trimmedComment) return false;

    const sourceMetadata = getQuestionSourceMetadata(currentQuestion.id);
    if (!sourceMetadata) return false;

    const participantId = profile.id;
    const reportId = makeId("report");
    const reportedAt = new Date().toISOString();
    const queuedReport: QueuedQuestionReport = {
      reportId,
      participantId,
      reportedAt,
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
    };
    const nextQuestionReportCount =
      (persistedState.reportCountsByQuestion[currentQuestion.id] ?? 0) + 1;
    const nextState: MobilePersistedQuizState = {
      ...persistedState,
      profileId: participantId,
      queuedReports: [...persistedState.queuedReports, queuedReport],
    };

    setSyncStatus("syncing");
    try {
      const syncedState = await syncMobileQuizState(nextState);
      await applyPersistedState({
        ...syncedState,
        reportCountsByQuestion: {
          ...syncedState.reportCountsByQuestion,
          [currentQuestion.id]:
            syncedState.reportCountsByQuestion[currentQuestion.id] ??
            nextQuestionReportCount,
        },
        totalReportCount: Math.max(
          syncedState.totalReportCount,
          persistedState.totalReportCount + 1,
        ),
      });
      setSyncStatus("synced");
      setSyncMessage("");
      return true;
    } catch (error) {
      console.error("Failed to submit mobile question report:", error);
      setSyncStatus("sync-error");
      setSyncMessage("Report was not submitted. Check sync and try again.");
      return false;
    }
  };

  return {
    selectedSources,
    selectedTopics,
    selectedQuestionTypes,
    selectionMode,
    difficultyRange,
    applySelection,
    availableCount: availableQuestions.length,
    currentIndex,
    currentQuestion,
    currentQuestionContext,
    currentQuestionRating,
    currentQuestionReportCount,
    questionElapsedMs: displayedQuestionTimerMs,
    shuffledOptions,
    selectedIndexes,
    showResult,
    toggleOption,
    submitAnswer,
    nextQuestion,
    answeredCount,
    correctCount,
    accuracy,
    userRating: ratingState.user.rating,
    userRatingRd: ratingState.user.rd,
    totalReportCount,
    queuedAnswerCount: persistedState.queuedAnswers.length,
    submitQuestionReport,
    participantId: profile?.id ?? GUEST_PROFILE_ID,
    profile,
    signIn,
    signUp,
    signOut,
    syncNow: () => syncNow(),
    syncStatus,
    syncMessage,
  };
}

import { afterEach, describe, expect, it, vi } from "vitest";
import type { QuizDataStore } from "@/lib/server/quizDataStore";
import {
  InMemoryQuizDataStore,
  ResilientQuizDataStore,
} from "@/lib/server/quizDataStore";
import {
  getQuizState,
  recordQuizAnswerForParticipant,
} from "@/lib/server/quizDataService";

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function makeFailingStore(message = "fetch failed"): QuizDataStore {
  const fail = async (): Promise<never> => {
    throw new Error(message);
  };

  return {
    getParticipant: fail,
    upsertParticipant: fail,
    listQuestionRatings: fail,
    getQuestionRating: fail,
    upsertQuestionRating: fail,
    hasAnswerAttempt: fail,
    appendAnswerAttempt: fail,
    listQuestionReports: fail,
    hasQuestionReport: fail,
    appendQuestionReport: fail,
  };
}

function makeStringThrowingStore(): QuizDataStore {
  const fail = async (): Promise<never> => {
    throw "fetch failed";
  };

  return {
    getParticipant: fail,
    upsertParticipant: fail,
    listQuestionRatings: fail,
    getQuestionRating: fail,
    upsertQuestionRating: fail,
    hasAnswerAttempt: fail,
    appendAnswerAttempt: fail,
    listQuestionReports: fail,
    hasQuestionReport: fail,
    appendQuestionReport: fail,
  };
}

describe("resilient quiz data store", () => {
  it("falls back to in-memory reads when Supabase is unreachable", async () => {
    const store = new ResilientQuizDataStore(
      makeFailingStore(),
      new InMemoryQuizDataStore(),
    );

    const state = await getQuizState("participant-a", store);

    expect(state.participantId).toBe("participant-a");
    expect(state.ratingState.user.gamesPlayed).toBe(0);
    expect(state.reportSummary.totalReportCount).toBe(0);
    expect(state.legacyMigrationCompleted).toBe(false);
    await expect(store.getParticipant("participant-a")).resolves.toBeNull();
  });

  it("keeps answer writes working after a transient Supabase failure", async () => {
    const store = new ResilientQuizDataStore(
      makeFailingStore(),
      new InMemoryQuizDataStore(),
    );

    const response = await recordQuizAnswerForParticipant(
      {
        participantId: "participant-a",
        questionId: "mit15773-l4-q1",
        label: "medium",
        isCorrect: true,
      },
      store,
    );

    expect(response.participantId).toBe("participant-a");
    expect(response.user.gamesPlayed).toBe(1);
    expect(response.questionId).toBe("mit15773-l4-q1");

    const nextState = await getQuizState("participant-a", store);
    expect(nextState.ratingState.user.gamesPlayed).toBe(1);
    expect(
      nextState.ratingState.questions["mit15773-l4-q1"]?.legacyCorrect,
    ).toBe(1);
  });

  it("reports fallback duration and retries the primary store", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-08-28T12:00:00.000Z"));
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const info = vi.spyOn(console, "info").mockImplementation(() => {});
    let primaryAvailable = false;
    let primaryCalls = 0;
    const participant = {
      id: "participant-a",
      rating: 1600,
      rd: 200,
      sigma: 0.06,
      lastUpdatedAt: Date.now(),
      gamesPlayed: 5,
      legacyMigratedAt: null,
    };
    const primary = makeFailingStore();
    primary.getParticipant = async () => {
      primaryCalls += 1;
      if (!primaryAvailable) throw new Error("fetch failed");
      return participant;
    };
    const store = new ResilientQuizDataStore(
      primary,
      new InMemoryQuizDataStore(),
      { primaryRetryIntervalMs: 1_000 },
    );

    await expect(store.getParticipant("participant-a")).resolves.toBeNull();
    expect(store.getPersistenceHealth()).toEqual({
      mode: "memory",
      unavailableSince: "2026-08-28T12:00:00.000Z",
    });
    expect(warn).toHaveBeenCalledTimes(1);

    primaryAvailable = true;
    await expect(store.getParticipant("participant-a")).resolves.toBeNull();
    expect(primaryCalls).toBe(1);

    vi.advanceTimersByTime(1_000);
    await expect(store.getParticipant("participant-a")).resolves.toEqual(
      participant,
    );
    expect(primaryCalls).toBe(2);
    expect(store.getPersistenceHealth()).toEqual({
      mode: "supabase",
      unavailableSince: null,
    });
    expect(info).toHaveBeenCalledWith(
      expect.stringContaining("Supabase connection recovered"),
    );
  });

  it("rethrows non-transient primary store failures", async () => {
    const store = new ResilientQuizDataStore(
      makeFailingStore("permission denied"),
      new InMemoryQuizDataStore(),
    );

    await expect(store.getParticipant("participant-a")).rejects.toThrow(
      "permission denied",
    );
  });

  it("treats non-Error throws as non-transient failures", async () => {
    const store = new ResilientQuizDataStore(
      makeStringThrowingStore(),
      new InMemoryQuizDataStore(),
    );

    await expect(store.getParticipant("participant-a")).rejects.toBe(
      "fetch failed",
    );
  });
});

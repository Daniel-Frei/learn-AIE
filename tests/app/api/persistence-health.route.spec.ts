import { afterEach, describe, expect, it, vi } from "vitest";
import {
  InMemoryQuizDataStore,
  ResilientQuizDataStore,
  type QuizDataStore,
  setQuizDataStoreForTests,
} from "@/lib/server/quizDataStore";

function makeFailingStore(message: string): QuizDataStore {
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

afterEach(() => {
  setQuizDataStoreForTests(null);
  vi.restoreAllMocks();
});

describe("GET /api/persistence-health", () => {
  it("reports a healthy durable store", async () => {
    setQuizDataStoreForTests(new InMemoryQuizDataStore());
    const { GET } = await import("@/app/api/persistence-health/route");

    const response = await GET();

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      mode: "supabase",
      unavailableSince: null,
    });
  });

  it("reports when the resilient store is using memory", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => {});
    setQuizDataStoreForTests(
      new ResilientQuizDataStore(
        makeFailingStore("fetch failed"),
        new InMemoryQuizDataStore(),
      ),
    );
    const { GET } = await import("@/app/api/persistence-health/route");

    const response = await GET();
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.mode).toBe("memory");
    expect(Date.parse(body.unavailableSince)).not.toBeNaN();
  });

  it("returns 503 for a non-transient persistence failure", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    setQuizDataStoreForTests(
      new ResilientQuizDataStore(
        makeFailingStore("permission denied"),
        new InMemoryQuizDataStore(),
      ),
    );
    const { GET } = await import("@/app/api/persistence-health/route");

    const response = await GET();

    expect(response.status).toBe(503);
    await expect(response.json()).resolves.toEqual({
      error: "Failed to check quiz persistence",
    });
  });
});

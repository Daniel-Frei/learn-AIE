import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { NextRequest } from "next/server";
import { createDefaultRatingState, recordAnswer } from "@/lib/ratingEngine";
import {
  InMemoryQuizDataStore,
  type QuizDataStore,
  type StoredAnswerAttempt,
  setQuizDataStoreForTests,
} from "@/lib/server/quizDataStore";

function buildRequest(
  url: string,
  method: "GET" | "POST",
  body?: unknown,
  headers?: HeadersInit,
): NextRequest {
  const req = new Request(url, {
    method,
    headers: {
      ...(body ? { "content-type": "application/json" } : {}),
      ...(headers ?? {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  return req as NextRequest;
}

function buildMalformedJsonRequest(url: string): NextRequest {
  const req = new Request(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{",
  });
  return req as NextRequest;
}

class CapturingQuizDataStore extends InMemoryQuizDataStore {
  public capturedAnswerAttempts: StoredAnswerAttempt[] = [];

  override async appendAnswerAttempt(attempt: StoredAnswerAttempt) {
    this.capturedAnswerAttempts.push(attempt);
    await super.appendAnswerAttempt(attempt);
  }
}

function makeThrowingStore(): QuizDataStore {
  const fail = async (): Promise<never> => {
    throw new Error("store down");
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

describe("shared quiz data routes", () => {
  let store: CapturingQuizDataStore;

  beforeEach(() => {
    store = new CapturingQuizDataStore();
    setQuizDataStoreForTests(store);
  });

  afterEach(() => {
    setQuizDataStoreForTests(null);
    vi.restoreAllMocks();
  });

  it("bootstraps default quiz state for a participant", async () => {
    const { GET } = await import("@/app/api/quiz-state/route");

    const res = await GET(
      buildRequest(
        "http://localhost/api/quiz-state?participantId=participant-a",
        "GET",
      ),
    );
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.participantId).toBe("participant-a");
    expect(body.ratingState.user.gamesPlayed).toBe(0);
    expect(body.ratingState.questions).toEqual({});
    expect(body.reportSummary.totalReportCount).toBe(0);
    expect(body.legacyMigrationCompleted).toBe(false);
  });

  it("records answers and persists shared question ratings", async () => {
    const { POST } = await import("@/app/api/answers/route");
    const { GET } = await import("@/app/api/quiz-state/route");

    const answerRes = await POST(
      buildRequest("http://localhost/api/answers", "POST", {
        participantId: "participant-a",
        questionId: "mit15773-l4-q1",
        label: "medium",
        isCorrect: true,
        elapsedMs: 15000,
        mistakeCount: 0,
      }),
    );
    const answerBody = await answerRes.json();

    expect(answerRes.status).toBe(200);
    expect(answerBody.user.gamesPlayed).toBe(1);
    expect(answerBody.questionId).toBe("mit15773-l4-q1");
    expect(answerBody.question.legacyCorrect).toBe(1);
    expect(answerBody.question.legacyWrong).toBe(0);
    expect(store.capturedAnswerAttempts).toHaveLength(1);
    expect(store.capturedAnswerAttempts[0]).toMatchObject({
      participantId: "participant-a",
      questionId: "mit15773-l4-q1",
      elapsedMs: 15000,
      mistakeCount: 0,
      source: "live",
    });

    const stateRes = await GET(
      buildRequest(
        "http://localhost/api/quiz-state?participantId=participant-a",
        "GET",
      ),
    );
    const stateBody = await stateRes.json();

    expect(stateBody.ratingState.user.gamesPlayed).toBe(1);
    expect(
      stateBody.ratingState.questions["mit15773-l4-q1"].legacyCorrect,
    ).toBe(1);
  });

  it("resets participant rating without clearing shared question ratings", async () => {
    const { POST: postAnswer } = await import("@/app/api/answers/route");
    const { POST: resetRating } =
      await import("@/app/api/participant-rating-reset/route");

    await postAnswer(
      buildRequest("http://localhost/api/answers", "POST", {
        participantId: "participant-a",
        questionId: "mit15773-l4-q1",
        label: "medium",
        isCorrect: true,
        elapsedMs: 15000,
        mistakeCount: 0,
      }),
    );

    const res = await resetRating(
      buildRequest("http://localhost/api/participant-rating-reset", "POST", {
        participantId: "participant-a",
      }),
    );
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.ratingState.user).toMatchObject({
      rating: 1500,
      rd: 350,
      sigma: 0.06,
      gamesPlayed: 0,
    });
    expect(body.ratingState.questions["mit15773-l4-q1"].legacyCorrect).toBe(1);
    expect(body.legacyMigrationCompleted).toBe(true);
  });

  it("stores append-only reports and exposes counts through quiz state", async () => {
    const { POST } = await import("@/app/api/question-reports/route");
    const { GET } = await import("@/app/api/quiz-state/route");

    const draft = {
      questionId: "mit15773-l4-q1",
      comment: "Prompt is ambiguous.",
      snapshot: {
        sourceId: "mit15773-l4",
        sourceLabel: "MIT 15.773 L4",
        seriesId: "mit-15773-2024",
        seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
        topic: "DL",
        prompt: "What does transfer learning reuse?",
      },
    };

    const first = await POST(
      buildRequest("http://localhost/api/question-reports", "POST", {
        participantId: "participant-a",
        draft,
      }),
    );
    const second = await POST(
      buildRequest("http://localhost/api/question-reports", "POST", {
        participantId: "participant-a",
        draft: { ...draft, comment: "Explanation contradicts option B." },
      }),
    );

    const firstBody = await first.json();
    const secondBody = await second.json();

    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    expect(firstBody.totalReportCount).toBe(1);
    expect(secondBody.totalReportCount).toBe(2);
    expect(secondBody.questionReportCount).toBe(2);

    const stateRes = await GET(
      buildRequest(
        "http://localhost/api/quiz-state?participantId=participant-a",
        "GET",
      ),
    );
    const stateBody = await stateRes.json();

    expect(stateRes.status).toBe(200);
    expect(stateBody.reportSummary).toEqual({
      totalReportCount: 2,
      countsByQuestion: {
        "mit15773-l4-q1": 2,
      },
    });
  });

  it("migrates local rating state once and ignores local reports", async () => {
    const { POST } = await import("@/app/api/local-migration/route");

    const localRatingState = recordAnswer(
      createDefaultRatingState(),
      "mit15773-l4-q1",
      "medium",
      true,
      1000,
    );
    const first = await POST(
      buildRequest("http://localhost/api/local-migration", "POST", {
        participantId: "participant-a",
        localRatingState,
        localReportState: {
          version: 1,
          reports: [
            {
              id: "legacy-report-1",
              questionId: "mit15773-l4-q1",
              comment: "Legacy report",
              reportedAt: "2026-03-24T10:00:00.000Z",
              snapshot: {
                sourceId: "mit15773-l4",
                sourceLabel: "MIT 15.773 L4",
                seriesId: "mit-15773-2024",
                seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
                topic: "DL",
                prompt: "Legacy prompt",
              },
            },
          ],
        },
      }),
    );
    const second = await POST(
      buildRequest("http://localhost/api/local-migration", "POST", {
        participantId: "participant-a",
        localRatingState,
      }),
    );

    const firstBody = await first.json();
    const secondBody = await second.json();

    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    expect(firstBody.legacyMigrationCompleted).toBe(true);
    expect(secondBody.legacyMigrationCompleted).toBe(true);
    expect(firstBody.ratingState.user.gamesPlayed).toBe(1);
    expect(secondBody.ratingState.user.gamesPlayed).toBe(1);
    expect(
      secondBody.ratingState.questions["mit15773-l4-q1"].legacyCorrect,
    ).toBe(1);
    expect(secondBody.reportSummary.totalReportCount).toBe(0);
  });

  it("returns 400 for invalid answer payloads", async () => {
    const { POST } = await import("@/app/api/answers/route");
    const res = await POST(
      buildRequest("http://localhost/api/answers", "POST", {
        participantId: "",
      }),
    );
    const body = await res.json();

    expect(res.status).toBe(400);
    expect(body).toEqual({
      error: "Invalid request payload for answer submission.",
    });
  });

  it("returns 400 for malformed answer JSON", async () => {
    const { POST } = await import("@/app/api/answers/route");
    const res = await POST(
      buildMalformedJsonRequest("http://localhost/api/answers"),
    );
    const body = await res.json();

    expect(res.status).toBe(400);
    expect(body).toEqual({
      error: "Invalid request payload for answer submission.",
    });
  });

  it("returns 400 when quiz state is requested without a participant id", async () => {
    const { GET } = await import("@/app/api/quiz-state/route");

    const res = await GET(
      buildRequest("http://localhost/api/quiz-state", "GET"),
    );
    const body = await res.json();

    expect(res.status).toBe(400);
    expect(body).toEqual({ error: "participantId is required." });
  });

  it("returns 400 when local migration is requested without a participant id", async () => {
    const { POST } = await import("@/app/api/local-migration/route");

    const res = await POST(
      buildRequest("http://localhost/api/local-migration", "POST", {
        participantId: " ",
      }),
    );
    const body = await res.json();

    expect(res.status).toBe(400);
    expect(body).toEqual({
      error: "participantId is required for local migration.",
    });
  });

  it("returns 400 when rating reset is requested without a participant id", async () => {
    const { POST } = await import("@/app/api/participant-rating-reset/route");

    const res = await POST(
      buildRequest("http://localhost/api/participant-rating-reset", "POST", {
        participantId: " ",
      }),
    );
    const body = await res.json();

    expect(res.status).toBe(400);
    expect(body).toEqual({
      error: "participantId is required for rating reset.",
    });
  });

  it("returns 400 for oversized question report comments", async () => {
    const { POST } = await import("@/app/api/question-reports/route");

    const res = await POST(
      buildRequest("http://localhost/api/question-reports", "POST", {
        participantId: "participant-a",
        draft: {
          questionId: "mit15773-l4-q1",
          comment: "x".repeat(2001),
          snapshot: {
            sourceId: "mit15773-l4",
            sourceLabel: "MIT 15.773 L4",
            seriesId: "mit-15773-2024",
            seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
            topic: "DL",
            prompt: "What does transfer learning reuse?",
          },
        },
      }),
    );
    const body = await res.json();

    expect(res.status).toBe(400);
    expect(body).toEqual({
      error: "Invalid request payload for question report.",
    });
  });

  it("accepts Life Science question report payloads", async () => {
    const { POST } = await import("@/app/api/question-reports/route");

    const res = await POST(
      buildRequest("http://localhost/api/question-reports", "POST", {
        participantId: "participant-a",
        draft: {
          questionId: "bio-chem-life-l0-q01",
          comment: "This distractor is too easy to guess.",
          snapshot: {
            sourceId: "bio-chem-life-l0",
            sourceLabel: "Biology & Chemistry L0",
            seriesId: "biology-chemistry-life-science",
            seriesLabel: "Biology & Chemistry for Life Science",
            topic: "Life Science",
            prompt:
              "Which statements describe useful starting assumptions for studying living systems?",
          },
        },
      }),
    );
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body).toEqual({
      totalReportCount: 1,
      questionReportCount: 1,
    });
  });

  it("returns 400 for non-object question report payloads", async () => {
    const { POST } = await import("@/app/api/question-reports/route");

    const res = await POST(
      buildRequest("http://localhost/api/question-reports", "POST", []),
    );
    const body = await res.json();

    expect(res.status).toBe(400);
    expect(body).toEqual({
      error: "Invalid request payload for question report.",
    });
  });

  it("returns zero question count when an existing report id is already stored elsewhere", async () => {
    const { POST } = await import("@/app/api/question-reports/route");
    const existingReportStore = Object.assign(new InMemoryQuizDataStore(), {
      async getParticipant() {
        return {
          id: "participant-a",
          rating: 1500,
          rd: 350,
          sigma: 0.06,
          lastUpdatedAt: 0,
          gamesPlayed: 0,
          legacyMigratedAt: null,
        };
      },
      async hasQuestionReport() {
        return true;
      },
      async listQuestionReports() {
        return [];
      },
    });
    setQuizDataStoreForTests(existingReportStore);

    const res = await POST(
      buildRequest("http://localhost/api/question-reports", "POST", {
        participantId: "participant-a",
        draft: {
          questionId: "mit15773-l4-q1",
          comment: "Prompt is ambiguous.",
          snapshot: {
            sourceId: "mit15773-l4",
            sourceLabel: "MIT 15.773 L4",
            seriesId: "mit-15773-2024",
            seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
            topic: "DL",
            prompt: "What does transfer learning reuse?",
          },
        },
      }),
    );
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body).toEqual({
      totalReportCount: 0,
      questionReportCount: 0,
    });
  });

  it("returns route-specific 500 responses when shared storage fails", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    setQuizDataStoreForTests(makeThrowingStore());

    const { GET: getQuizState } = await import("@/app/api/quiz-state/route");
    const { POST: postAnswer } = await import("@/app/api/answers/route");
    const { POST: postReport } =
      await import("@/app/api/question-reports/route");
    const { POST: migrateLocal } =
      await import("@/app/api/local-migration/route");
    const { POST: resetRating } =
      await import("@/app/api/participant-rating-reset/route");

    const validReportDraft = {
      questionId: "mit15773-l4-q1",
      comment: "Prompt is ambiguous.",
      snapshot: {
        sourceId: "mit15773-l4",
        sourceLabel: "MIT 15.773 L4",
        seriesId: "mit-15773-2024",
        seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
        topic: "DL",
        prompt: "What does transfer learning reuse?",
      },
    };

    const responses = await Promise.all([
      getQuizState(
        buildRequest(
          "http://localhost/api/quiz-state?participantId=participant-a",
          "GET",
        ),
      ),
      postAnswer(
        buildRequest("http://localhost/api/answers", "POST", {
          participantId: "participant-a",
          questionId: "mit15773-l4-q1",
          isCorrect: true,
        }),
      ),
      postReport(
        buildRequest("http://localhost/api/question-reports", "POST", {
          participantId: "participant-a",
          draft: validReportDraft,
        }),
      ),
      migrateLocal(
        buildRequest("http://localhost/api/local-migration", "POST", {
          participantId: "participant-a",
        }),
      ),
      resetRating(
        buildRequest("http://localhost/api/participant-rating-reset", "POST", {
          participantId: "participant-a",
        }),
      ),
    ]);
    const bodies = await Promise.all(responses.map((res) => res.json()));

    expect(responses.map((res) => res.status)).toEqual([
      500, 500, 500, 500, 500,
    ]);
    expect(bodies).toEqual([
      { error: "Failed to load quiz state" },
      { error: "Failed to record answer" },
      { error: "Failed to submit question report" },
      { error: "Failed to migrate local state" },
      { error: "Failed to reset participant rating" },
    ]);
  });
});

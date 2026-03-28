import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { NextRequest } from "next/server";
import {
  appendQuestionReport,
  createDefaultQuestionReportsState,
} from "@/lib/questionReportsStore";
import { createDefaultRatingState, recordAnswer } from "@/lib/ratingEngine";
import {
  InMemoryQuizDataStore,
  setQuizDataStoreForTests,
} from "@/lib/server/quizDataStore";

function buildRequest(
  url: string,
  method: "GET" | "POST",
  body?: unknown,
): NextRequest {
  const req = new Request(url, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  return req as NextRequest;
}

describe("shared quiz data routes", () => {
  beforeEach(() => {
    setQuizDataStoreForTests(new InMemoryQuizDataStore());
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
      }),
    );
    const answerBody = await answerRes.json();

    expect(answerRes.status).toBe(200);
    expect(answerBody.user.gamesPlayed).toBe(1);
    expect(answerBody.questionId).toBe("mit15773-l4-q1");
    expect(answerBody.question.legacyCorrect).toBe(1);
    expect(answerBody.question.legacyWrong).toBe(0);

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

  it("stores append-only reports and exports them from the server", async () => {
    const { POST } = await import("@/app/api/question-reports/route");
    const { GET } = await import("@/app/api/question-reports/export/route");

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

    const exportRes = await GET();
    const exportBody = await exportRes.json();

    expect(exportRes.status).toBe(200);
    expect(exportBody.version).toBe(1);
    expect(exportBody.reports).toHaveLength(2);
  });

  it("migrates local rating/report state once without double-counting", async () => {
    const { POST } = await import("@/app/api/local-migration/route");

    const localRatingState = recordAnswer(
      createDefaultRatingState(),
      "mit15773-l4-q1",
      "medium",
      true,
      1000,
    );
    const localReportState = appendQuestionReport(
      createDefaultQuestionReportsState(),
      {
        questionId: "mit15773-l4-q1",
        comment: "Legacy report",
        snapshot: {
          sourceId: "mit15773-l4",
          sourceLabel: "MIT 15.773 L4",
          seriesId: "mit-15773-2024",
          seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
          topic: "DL",
          prompt: "Legacy prompt",
        },
      },
      {
        reportId: "legacy-report-1",
        reportedAt: "2026-03-24T10:00:00.000Z",
      },
    );

    const first = await POST(
      buildRequest("http://localhost/api/local-migration", "POST", {
        participantId: "participant-a",
        localRatingState,
        localReportState,
      }),
    );
    const second = await POST(
      buildRequest("http://localhost/api/local-migration", "POST", {
        participantId: "participant-a",
        localRatingState,
        localReportState,
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
    expect(secondBody.reportSummary.totalReportCount).toBe(1);
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
});

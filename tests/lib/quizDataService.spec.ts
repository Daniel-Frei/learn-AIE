import { describe, expect, it } from "vitest";
import type { QuestionReportDraft } from "@/lib/questionReportsStore";
import { createDefaultRatingState, recordAnswer } from "@/lib/ratingEngine";
import {
  exportCurrentRatingState,
  migrateLocalStateForParticipant,
  recordQuizAnswerForParticipant,
  resetParticipantRatingForParticipant,
  submitQuestionReportForParticipant,
} from "@/lib/server/quizDataService";
import {
  InMemoryQuizDataStore,
  type StoredAnswerAttempt,
  type StoredParticipant,
  type StoredQuestionRating,
} from "@/lib/server/quizDataStore";

class CapturingStore extends InMemoryQuizDataStore {
  public attempts: StoredAnswerAttempt[] = [];

  override async appendAnswerAttempt(attempt: StoredAnswerAttempt) {
    this.attempts.push(attempt);
    await super.appendAnswerAttempt(attempt);
  }
}

const participant: StoredParticipant = {
  id: "participant-a",
  rating: 1510,
  rd: 220,
  sigma: 0.06,
  lastUpdatedAt: 1000,
  gamesPlayed: 3,
  legacyMigratedAt: "2026-05-01T00:00:00.000Z",
};

const question: StoredQuestionRating = {
  questionId: "q-a",
  rating: 1490,
  rd: 210,
  sigma: 0.06,
  lastUpdatedAt: 1000,
  gamesPlayed: 2,
  legacyCorrect: 1,
  legacyWrong: 1,
  label: "medium",
};

const reportDraft: QuestionReportDraft = {
  questionId: "q-a",
  comment: "Needs review",
  snapshot: {
    sourceId: "mit15773-l4",
    sourceLabel: "MIT 15.773 L4",
    seriesId: "mit-15773-2024",
    seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
    topic: "DL",
    prompt: "Prompt",
  },
};

describe("quiz data service", () => {
  it("returns persisted state for duplicate answer attempts", async () => {
    const store = new InMemoryQuizDataStore();
    await store.upsertParticipant(participant);
    await store.upsertQuestionRating(question);
    await store.appendAnswerAttempt({
      attemptId: "attempt-duplicate",
      participantId: participant.id,
      questionId: question.questionId,
      isCorrect: true,
      elapsedMs: 1000,
      mistakeCount: 0,
      answeredAt: "2026-05-02T00:00:00.000Z",
      source: "live",
    });

    const response = await recordQuizAnswerForParticipant(
      {
        participantId: participant.id,
        questionId: question.questionId,
        attemptId: "attempt-duplicate",
        isCorrect: false,
      },
      store,
    );

    expect(response.user.gamesPlayed).toBe(participant.gamesPlayed);
    expect(response.question.legacyCorrect).toBe(question.legacyCorrect);
    expect(response.question.legacyWrong).toBe(question.legacyWrong);
  });

  it("rejects duplicate attempts when their persisted rows are incomplete", async () => {
    const store = new InMemoryQuizDataStore();
    await store.appendAnswerAttempt({
      attemptId: "attempt-duplicate",
      participantId: participant.id,
      questionId: question.questionId,
      isCorrect: true,
      elapsedMs: 1000,
      mistakeCount: 0,
      answeredAt: "2026-05-02T00:00:00.000Z",
      source: "live",
    });

    await expect(
      recordQuizAnswerForParticipant(
        {
          participantId: participant.id,
          questionId: question.questionId,
          attemptId: "attempt-duplicate",
          isCorrect: true,
        },
        store,
      ),
    ).rejects.toThrow("Duplicate answer attempt was missing persisted state.");
  });

  it("updates existing participant and question rows while defaulting omitted attempt metadata", async () => {
    const store = new CapturingStore();
    await store.upsertParticipant(participant);
    await store.upsertQuestionRating(question);

    const response = await recordQuizAnswerForParticipant(
      {
        participantId: participant.id,
        questionId: question.questionId,
        label: "hard",
        isCorrect: true,
        answeredAt: "2026-05-02T00:00:00.000Z",
      },
      store,
    );

    expect(response.user.gamesPlayed).toBe(participant.gamesPlayed + 1);
    expect(response.question.legacyCorrect).toBe(question.legacyCorrect + 1);
    expect(store.attempts[0]).toMatchObject({
      participantId: participant.id,
      questionId: question.questionId,
      elapsedMs: 0,
      mistakeCount: 0,
      answeredAt: "2026-05-02T00:00:00.000Z",
      source: "live",
    });
    await expect(store.getParticipant(participant.id)).resolves.toMatchObject({
      legacyMigratedAt: participant.legacyMigratedAt,
    });
  });

  it("does not duplicate an existing question report id", async () => {
    const store = new InMemoryQuizDataStore();
    await store.upsertParticipant(participant);

    const first = await submitQuestionReportForParticipant(
      {
        participantId: participant.id,
        draft: reportDraft,
        reportId: "report-a",
        reportedAt: "2026-05-02T00:00:00.000Z",
      },
      store,
    );
    const second = await submitQuestionReportForParticipant(
      {
        participantId: participant.id,
        draft: reportDraft,
        reportId: "report-a",
        reportedAt: "2026-05-02T00:00:00.000Z",
      },
      store,
    );

    expect(first.totalReportCount).toBe(1);
    expect(second.totalReportCount).toBe(1);
    expect(second.countsByQuestion["q-a"]).toBe(1);
  });

  it("rejects invalid report drafts before writing", async () => {
    const store = new InMemoryQuizDataStore();

    await expect(
      submitQuestionReportForParticipant(
        {
          participantId: participant.id,
          draft: {
            ...reportDraft,
            questionId: " ",
          },
        },
        store,
      ),
    ).rejects.toThrow("Invalid report payload.");
  });

  it("resets only the participant rating while preserving shared question data", async () => {
    const store = new InMemoryQuizDataStore();
    await store.upsertParticipant(participant);
    await store.upsertQuestionRating(question);
    await store.appendQuestionReport({
      id: "report-a",
      participantId: participant.id,
      questionId: question.questionId,
      comment: "Needs review",
      reportedAt: "2026-05-02T00:00:00.000Z",
      snapshot: reportDraft.snapshot,
    });

    const response = await resetParticipantRatingForParticipant(
      participant.id,
      store,
    );

    expect(response.ratingState.user).toMatchObject({
      rating: 1500,
      rd: 350,
      sigma: 0.06,
      gamesPlayed: 0,
    });
    expect(response.ratingState.questions[question.questionId]).toMatchObject({
      rating: question.rating,
      rd: question.rd,
      sigma: question.sigma,
      gamesPlayed: question.gamesPlayed,
      legacyCorrect: question.legacyCorrect,
      legacyWrong: question.legacyWrong,
      label: question.label,
    });
    expect(response.reportSummary.countsByQuestion[question.questionId]).toBe(
      1,
    );
    await expect(store.getParticipant(participant.id)).resolves.toMatchObject({
      legacyMigratedAt: participant.legacyMigratedAt,
    });
  });

  it("marks a participant migrated even when there is no local state", async () => {
    const store = new InMemoryQuizDataStore();

    const migrated = await migrateLocalStateForParticipant(
      {
        participantId: "participant-empty",
        questionMetadata: {},
      },
      store,
    );

    expect(migrated.legacyMigrationCompleted).toBe(true);
    expect(migrated.ratingState.user.gamesPlayed).toBe(0);
    await expect(
      store.getParticipant("participant-empty"),
    ).resolves.toMatchObject({
      legacyMigratedAt: expect.any(String),
    });
  });

  it("replays mixed legacy answer counts in a stable migration order", async () => {
    const store = new CapturingStore();

    const migrated = await migrateLocalStateForParticipant(
      {
        participantId: "participant-legacy",
        localRatingState: {
          "q-balanced": { correct: 1, wrong: 2 },
          "q-correct-only": { correct: 2, wrong: 0 },
          "q-empty": { correct: 0, wrong: 0 },
          "q-wrong-only": { correct: 0, wrong: 2 },
        },
        questionMetadata: {
          "q-balanced": { label: "medium" },
          "q-correct-only": { label: "easy" },
          "q-wrong-only": { label: "hard" },
        },
      },
      store,
    );

    expect(migrated.legacyMigrationCompleted).toBe(true);
    expect(migrated.ratingState.questions["q-balanced"]).toMatchObject({
      legacyCorrect: 1,
      legacyWrong: 2,
      label: "medium",
    });
    expect(migrated.ratingState.questions["q-correct-only"]).toMatchObject({
      legacyCorrect: 2,
      legacyWrong: 0,
      label: "easy",
    });
    expect(migrated.ratingState.questions["q-wrong-only"]).toMatchObject({
      legacyCorrect: 0,
      legacyWrong: 2,
      label: "hard",
    });
    expect(store.attempts.map((attempt) => attempt.source)).toEqual(
      Array.from({ length: 7 }, () => "migration"),
    );
  });

  it("exports the current rating state as versioned JSON", () => {
    const state = recordAnswer(
      createDefaultRatingState(),
      "q-export",
      "easy",
      true,
      1000,
    );

    expect(JSON.parse(exportCurrentRatingState(state))).toMatchObject({
      version: 2,
      legacyCounts: {
        "q-export": { correct: 1, wrong: 0 },
      },
    });
  });
});

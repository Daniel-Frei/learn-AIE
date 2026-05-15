import { afterEach, describe, expect, it, vi } from "vitest";
import type {
  StoredAnswerAttempt,
  StoredParticipant,
  StoredQuestionRating,
  StoredQuestionReport,
} from "@/lib/server/quizDataStore";

type TableName =
  | "participants"
  | "question_ratings"
  | "answer_attempts"
  | "question_reports";

type Row = Record<string, unknown>;

type FakeDatabase = {
  rows: Record<TableName, Map<string, Row>>;
  insertErrors: Partial<Record<TableName, string[]>>;
};

const TABLE_KEYS: Record<TableName, string> = {
  participants: "participant_id",
  question_ratings: "question_id",
  answer_attempts: "attempt_id",
  question_reports: "id",
};

function cloneRow(row: Row): Row {
  return { ...row };
}

function createFakeDatabase(
  seed: Partial<Record<TableName, Row[]>> = {},
): FakeDatabase {
  const rows = {
    participants: new Map<string, Row>(),
    question_ratings: new Map<string, Row>(),
    answer_attempts: new Map<string, Row>(),
    question_reports: new Map<string, Row>(),
  };

  for (const [table, tableRows] of Object.entries(seed) as Array<
    [TableName, Row[]]
  >) {
    for (const row of tableRows) {
      rows[table].set(String(row[TABLE_KEYS[table]]), cloneRow(row));
    }
  }

  return { rows, insertErrors: {} };
}

function createFakeSupabaseClient(db: FakeDatabase) {
  return {
    from(tableName: string) {
      const table = tableName as TableName;

      return {
        select() {
          return {
            eq(column: string, value: unknown) {
              return {
                maybeSingle() {
                  const row =
                    Array.from(db.rows[table].values()).find(
                      (entry) => entry[column] === value,
                    ) ?? null;
                  return Promise.resolve({
                    data: row ? cloneRow(row) : null,
                    error: null,
                  });
                },
              };
            },
            order(column: string, options: { ascending: boolean }) {
              return {
                range(from: number, to: number) {
                  const sorted = Array.from(db.rows[table].values()).sort(
                    (a, b) => {
                      const left = String(a[column]);
                      const right = String(b[column]);
                      return options.ascending
                        ? left.localeCompare(right)
                        : right.localeCompare(left);
                    },
                  );
                  return Promise.resolve({
                    data: sorted.slice(from, to + 1).map(cloneRow),
                    error: null,
                  });
                },
              };
            },
          };
        },
        upsert(payload: Row) {
          db.rows[table].set(
            String(payload[TABLE_KEYS[table]]),
            cloneRow(payload),
          );
          return Promise.resolve({ data: cloneRow(payload), error: null });
        },
        insert(payload: Row) {
          const queuedErrors = db.insertErrors[table];
          const message = queuedErrors?.shift();
          if (message) {
            return Promise.resolve({ data: null, error: { message } });
          }

          db.rows[table].set(
            String(payload[TABLE_KEYS[table]]),
            cloneRow(payload),
          );
          return Promise.resolve({ data: cloneRow(payload), error: null });
        },
      };
    },
  };
}

describe("Supabase quiz data store adapter", () => {
  const originalUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const originalServiceRoleKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  afterEach(() => {
    if (originalUrl === undefined) {
      delete process.env.NEXT_PUBLIC_SUPABASE_URL;
    } else {
      process.env.NEXT_PUBLIC_SUPABASE_URL = originalUrl;
    }
    if (originalServiceRoleKey === undefined) {
      delete process.env.SUPABASE_SERVICE_ROLE_KEY;
    } else {
      process.env.SUPABASE_SERVICE_ROLE_KEY = originalServiceRoleKey;
    }
    vi.resetModules();
    vi.doUnmock("@supabase/supabase-js");
    vi.restoreAllMocks();
  });

  it("maps Supabase rows and payloads for all quiz data operations", async () => {
    process.env.NEXT_PUBLIC_SUPABASE_URL = "https://example.supabase.co";
    process.env.SUPABASE_SERVICE_ROLE_KEY = "service-role-key";
    const db = createFakeDatabase({
      participants: [
        {
          participant_id: "participant-a",
          rating: 1510,
          rd: 220,
          sigma: 0.06,
          last_updated_at: 1000,
          games_played: 3,
          legacy_migrated_at: "2026-05-01T00:00:00.000Z",
        },
      ],
      question_ratings: [
        {
          question_id: "q-a",
          rating: 1400,
          rd: 210,
          sigma: 0.06,
          last_updated_at: 1001,
          games_played: 2,
          legacy_correct: 1,
          legacy_wrong: 1,
          label: "medium",
        },
        {
          question_id: "q-b",
          rating: 1600,
          rd: 230,
          sigma: 0.06,
          last_updated_at: 1002,
          games_played: 1,
          legacy_correct: 1,
          legacy_wrong: 0,
          label: null,
        },
      ],
      answer_attempts: [
        {
          attempt_id: "attempt-existing",
          participant_id: "participant-a",
          question_id: "q-a",
        },
      ],
      question_reports: [
        {
          id: "report-existing",
          participant_id: "participant-a",
          question_id: "q-a",
          comment: "Needs review",
          reported_at: "2026-05-02T00:00:00.000Z",
          source_id: "mit15773-l4",
          source_label: "MIT 15.773 L4",
          series_id: "mit-15773-2024",
          series_label: "MIT 15.773 Hands-On Deep Learning 2024",
          topic: "DL",
          prompt: "Prompt",
        },
      ],
    });
    const createClient = vi.fn().mockReturnValue(createFakeSupabaseClient(db));
    vi.doMock("@supabase/supabase-js", () => ({ createClient }));

    const { getQuizDataStore } = await import("@/lib/server/quizDataStore");
    const store = getQuizDataStore();

    await expect(store.getParticipant("participant-a")).resolves.toMatchObject({
      id: "participant-a",
      gamesPlayed: 3,
      legacyMigratedAt: "2026-05-01T00:00:00.000Z",
    });
    await expect(store.getParticipant("missing")).resolves.toBeNull();

    const participant: StoredParticipant = {
      id: "participant-b",
      rating: 1500,
      rd: 350,
      sigma: 0.06,
      lastUpdatedAt: 2000,
      gamesPlayed: 0,
      legacyMigratedAt: null,
    };
    await store.upsertParticipant(participant);
    expect(db.rows.participants.get("participant-b")).toMatchObject({
      participant_id: "participant-b",
      legacy_migrated_at: null,
    });

    await expect(store.listQuestionRatings()).resolves.toEqual([
      expect.objectContaining({ questionId: "q-a", label: "medium" }),
      expect.objectContaining({ questionId: "q-b", label: undefined }),
    ]);
    await expect(store.getQuestionRating("q-a")).resolves.toMatchObject({
      questionId: "q-a",
      legacyCorrect: 1,
      legacyWrong: 1,
    });

    const question: StoredQuestionRating = {
      questionId: "q-c",
      rating: 1700,
      rd: 200,
      sigma: 0.06,
      lastUpdatedAt: 2001,
      gamesPlayed: 1,
      legacyCorrect: 0,
      legacyWrong: 1,
    };
    await store.upsertQuestionRating(question);
    expect(db.rows.question_ratings.get("q-c")).toMatchObject({
      question_id: "q-c",
      label: null,
    });
    await store.upsertQuestionRating({
      ...question,
      questionId: "q-d",
      label: "easy",
    });
    expect(db.rows.question_ratings.get("q-d")).toMatchObject({
      question_id: "q-d",
      label: "easy",
    });

    await expect(store.hasAnswerAttempt("attempt-existing")).resolves.toBe(
      true,
    );
    await expect(store.hasAnswerAttempt("attempt-missing")).resolves.toBe(
      false,
    );

    const attempt: StoredAnswerAttempt = {
      attemptId: "attempt-new",
      participantId: "participant-a",
      questionId: "q-a",
      label: "hard",
      isCorrect: false,
      elapsedMs: 30000,
      mistakeCount: 2,
      answeredAt: "2026-05-03T00:00:00.000Z",
      source: "live",
    };
    await store.appendAnswerAttempt(attempt);
    expect(db.rows.answer_attempts.get("attempt-new")).toMatchObject({
      attempt_id: "attempt-new",
      elapsed_ms: 30000,
      mistake_count: 2,
    });

    await expect(store.listQuestionReports()).resolves.toEqual([
      expect.objectContaining({
        id: "report-existing",
        participantId: "participant-a",
        snapshot: expect.objectContaining({ topic: "DL" }),
      }),
    ]);
    await expect(store.hasQuestionReport("report-existing")).resolves.toBe(
      true,
    );
    await expect(store.hasQuestionReport("report-missing")).resolves.toBe(
      false,
    );

    const report: StoredQuestionReport = {
      id: "report-new",
      participantId: "participant-b",
      questionId: "q-b",
      comment: "Bad wording",
      reportedAt: "2026-05-04T00:00:00.000Z",
      snapshot: {
        sourceId: "mit15773-l4",
        sourceLabel: "MIT 15.773 L4",
        seriesId: "mit-15773-2024",
        seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
        topic: "DL",
        prompt: "Snapshot prompt",
      },
    };
    await store.appendQuestionReport(report);
    expect(db.rows.question_reports.get("report-new")).toMatchObject({
      id: "report-new",
      participant_id: "participant-b",
      source_id: "mit15773-l4",
      prompt: "Snapshot prompt",
    });
  });

  it("falls back to legacy answer-attempt inserts for older schemas", async () => {
    process.env.NEXT_PUBLIC_SUPABASE_URL = "https://example.supabase.co";
    process.env.SUPABASE_SERVICE_ROLE_KEY = "service-role-key";
    const db = createFakeDatabase();
    db.insertErrors.answer_attempts = [
      "schema cache could not find elapsed_ms",
    ];
    vi.doMock("@supabase/supabase-js", () => ({
      createClient: vi.fn().mockReturnValue(createFakeSupabaseClient(db)),
    }));

    const { getQuizDataStore } = await import("@/lib/server/quizDataStore");
    const store = getQuizDataStore();

    await store.appendAnswerAttempt({
      attemptId: "legacy-attempt",
      participantId: "participant-a",
      questionId: "q-a",
      isCorrect: true,
      elapsedMs: 1234,
      mistakeCount: 0,
      answeredAt: "2026-05-03T00:00:00.000Z",
      source: "live",
    });

    expect(db.rows.answer_attempts.get("legacy-attempt")).toEqual({
      attempt_id: "legacy-attempt",
      participant_id: "participant-a",
      question_id: "q-a",
      label: null,
      is_correct: true,
      answered_at: "2026-05-03T00:00:00.000Z",
      source: "live",
    });
  });

  it("rethrows non-schema insert failures from answer attempts", async () => {
    process.env.NEXT_PUBLIC_SUPABASE_URL = "https://example.supabase.co";
    process.env.SUPABASE_SERVICE_ROLE_KEY = "service-role-key";
    const db = createFakeDatabase();
    db.insertErrors.answer_attempts = ["permission denied"];
    vi.doMock("@supabase/supabase-js", () => ({
      createClient: vi.fn().mockReturnValue(createFakeSupabaseClient(db)),
    }));

    const { getQuizDataStore } = await import("@/lib/server/quizDataStore");
    const store = getQuizDataStore();

    await expect(
      store.appendAnswerAttempt({
        attemptId: "failed-attempt",
        participantId: "participant-a",
        questionId: "q-a",
        isCorrect: true,
        elapsedMs: 1234,
        mistakeCount: 0,
        answeredAt: "2026-05-03T00:00:00.000Z",
        source: "live",
      }),
    ).rejects.toThrow("permission denied");
  });

  it("rethrows non-transient Supabase client creation failures", async () => {
    process.env.NEXT_PUBLIC_SUPABASE_URL = "https://example.supabase.co";
    process.env.SUPABASE_SERVICE_ROLE_KEY = "service-role-key";
    vi.doMock("@supabase/supabase-js", () => ({
      createClient: vi.fn(() => {
        throw new Error("bad client options");
      }),
    }));

    const { getQuizDataStore } = await import("@/lib/server/quizDataStore");

    expect(() => getQuizDataStore()).toThrow("bad client options");
  });

  it("uses an in-memory store when the Supabase client cannot be created", async () => {
    delete process.env.NEXT_PUBLIC_SUPABASE_URL;
    delete process.env.SUPABASE_SERVICE_ROLE_KEY;
    vi.doMock("@supabase/supabase-js", () => ({ createClient: vi.fn() }));
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const { getQuizDataStore } = await import("@/lib/server/quizDataStore");
    const store = getQuizDataStore();

    await expect(store.getParticipant("participant-a")).resolves.toBeNull();
    expect(getQuizDataStore()).toBe(store);
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining("Supabase client could not be created"),
    );
  });
});

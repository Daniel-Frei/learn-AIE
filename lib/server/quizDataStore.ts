import type { Difficulty } from "../quiz";
import type {
  QuestionReport,
  QuestionReportSnapshot,
} from "../questionReportsStore";
import type { QuestionRating, RatingEntity } from "../ratingEngine";
import { getSupabaseAdminClient } from "./supabaseAdmin";

export type StoredParticipant = RatingEntity & {
  id: string;
  legacyMigratedAt: string | null;
};

export type StoredQuestionRating = QuestionRating & {
  questionId: string;
};

export type StoredAnswerAttempt = {
  attemptId: string;
  participantId: string;
  questionId: string;
  label?: Difficulty;
  isCorrect: boolean;
  elapsedMs: number;
  mistakeCount: number;
  answeredAt: string;
  source: "live" | "migration";
};

export type StoredQuestionReport = QuestionReport & {
  participantId: string;
};

export interface QuizDataStore {
  getParticipant(participantId: string): Promise<StoredParticipant | null>;
  upsertParticipant(participant: StoredParticipant): Promise<void>;
  listQuestionRatings(): Promise<StoredQuestionRating[]>;
  getQuestionRating(questionId: string): Promise<StoredQuestionRating | null>;
  upsertQuestionRating(question: StoredQuestionRating): Promise<void>;
  hasAnswerAttempt(attemptId: string): Promise<boolean>;
  appendAnswerAttempt(attempt: StoredAnswerAttempt): Promise<void>;
  listQuestionReports(): Promise<StoredQuestionReport[]>;
  hasQuestionReport(reportId: string): Promise<boolean>;
  appendQuestionReport(report: StoredQuestionReport): Promise<void>;
}

type ParticipantRow = {
  participant_id: string;
  rating: number;
  rd: number;
  sigma: number;
  last_updated_at: number;
  games_played: number;
  legacy_migrated_at: string | null;
};

type QuestionRatingRow = {
  question_id: string;
  rating: number;
  rd: number;
  sigma: number;
  last_updated_at: number;
  games_played: number;
  legacy_correct: number;
  legacy_wrong: number;
  label: Difficulty | null;
};

type QuestionReportRow = {
  id: string;
  participant_id: string;
  question_id: string;
  comment: string;
  reported_at: string;
  source_id: string;
  source_label: string;
  series_id: string;
  series_label: string;
  topic: string;
  prompt: string;
};

type PagedSelectQuery = {
  select(columns: string): {
    order(
      column: string,
      options: { ascending: boolean },
    ): {
      range(
        from: number,
        to: number,
      ): PromiseLike<{
        data: unknown[] | null;
        error: { message: string } | null;
      }>;
    };
  };
};

function mapParticipantRow(row: ParticipantRow): StoredParticipant {
  return {
    id: row.participant_id,
    rating: row.rating,
    rd: row.rd,
    sigma: row.sigma,
    lastUpdatedAt: row.last_updated_at,
    gamesPlayed: row.games_played,
    legacyMigratedAt: row.legacy_migrated_at,
  };
}

function mapQuestionRatingRow(row: QuestionRatingRow): StoredQuestionRating {
  return {
    questionId: row.question_id,
    rating: row.rating,
    rd: row.rd,
    sigma: row.sigma,
    lastUpdatedAt: row.last_updated_at,
    gamesPlayed: row.games_played,
    legacyCorrect: row.legacy_correct,
    legacyWrong: row.legacy_wrong,
    label: row.label ?? undefined,
  };
}

function mapQuestionReportRow(row: QuestionReportRow): StoredQuestionReport {
  const snapshot: QuestionReportSnapshot = {
    sourceId: row.source_id as QuestionReportSnapshot["sourceId"],
    sourceLabel: row.source_label,
    seriesId: row.series_id as QuestionReportSnapshot["seriesId"],
    seriesLabel: row.series_label,
    topic: row.topic as QuestionReportSnapshot["topic"],
    prompt: row.prompt,
  };

  return {
    id: row.id,
    participantId: row.participant_id,
    questionId: row.question_id,
    comment: row.comment,
    reportedAt: row.reported_at,
    snapshot,
  };
}

async function throwOnError<T>(
  promise: PromiseLike<{ data: T; error: { message: string } | null }>,
): Promise<T> {
  const { data, error } = await promise;
  if (error) {
    throw new Error(error.message);
  }
  return data;
}

export async function fetchAllRows<T>(
  query: PagedSelectQuery,
  columns: string,
  orderColumn: string,
): Promise<T[]> {
  const pageSize = 1000;
  const rows: T[] = [];

  for (let offset = 0; ; offset += pageSize) {
    const page = (await throwOnError(
      query
        .select(columns)
        .order(orderColumn, { ascending: true })
        .range(offset, offset + pageSize - 1),
    )) as T[] | null;
    const pageRows = page ?? [];
    rows.push(...pageRows);

    if (pageRows.length < pageSize) {
      break;
    }
  }

  return rows;
}

function isTransientSupabaseError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;

  const message = `${err.name}: ${err.message}`.toLowerCase();
  return (
    message.includes("fetch failed") ||
    message.includes("missing required supabase environment variable") ||
    message.includes("econnrefused") ||
    message.includes("enotfound") ||
    message.includes("etimedout")
  );
}

function isMissingAnswerAttemptMetadataError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;

  const message = err.message.toLowerCase();
  return (
    message.includes("elapsed_ms") ||
    message.includes("mistake_count") ||
    message.includes("schema cache")
  );
}

export class ResilientQuizDataStore implements QuizDataStore {
  private primaryFailed = false;

  constructor(
    private readonly primary: QuizDataStore,
    private readonly fallback: QuizDataStore,
  ) {}

  private async run<T>(
    operation: string,
    work: (store: QuizDataStore) => Promise<T>,
  ): Promise<T> {
    if (!this.primaryFailed) {
      try {
        return await work(this.primary);
      } catch (err) {
        if (!isTransientSupabaseError(err)) {
          throw err;
        }

        this.primaryFailed = true;
        console.warn(
          `[quiz-data] Supabase unavailable during ${operation}; using in-memory fallback for this session.`,
        );
      }
    }

    return work(this.fallback);
  }

  async getParticipant(
    participantId: string,
  ): Promise<StoredParticipant | null> {
    return this.run("getParticipant", (store) =>
      store.getParticipant(participantId),
    );
  }

  async upsertParticipant(participant: StoredParticipant): Promise<void> {
    return this.run("upsertParticipant", (store) =>
      store.upsertParticipant(participant),
    );
  }

  async listQuestionRatings(): Promise<StoredQuestionRating[]> {
    return this.run("listQuestionRatings", (store) =>
      store.listQuestionRatings(),
    );
  }

  async getQuestionRating(
    questionId: string,
  ): Promise<StoredQuestionRating | null> {
    return this.run("getQuestionRating", (store) =>
      store.getQuestionRating(questionId),
    );
  }

  async upsertQuestionRating(question: StoredQuestionRating): Promise<void> {
    return this.run("upsertQuestionRating", (store) =>
      store.upsertQuestionRating(question),
    );
  }

  async hasAnswerAttempt(attemptId: string): Promise<boolean> {
    return this.run("hasAnswerAttempt", (store) =>
      store.hasAnswerAttempt(attemptId),
    );
  }

  async appendAnswerAttempt(attempt: StoredAnswerAttempt): Promise<void> {
    return this.run("appendAnswerAttempt", (store) =>
      store.appendAnswerAttempt(attempt),
    );
  }

  async listQuestionReports(): Promise<StoredQuestionReport[]> {
    return this.run("listQuestionReports", (store) =>
      store.listQuestionReports(),
    );
  }

  async hasQuestionReport(reportId: string): Promise<boolean> {
    return this.run("hasQuestionReport", (store) =>
      store.hasQuestionReport(reportId),
    );
  }

  async appendQuestionReport(report: StoredQuestionReport): Promise<void> {
    return this.run("appendQuestionReport", (store) =>
      store.appendQuestionReport(report),
    );
  }
}

class SupabaseQuizDataStore implements QuizDataStore {
  private client = getSupabaseAdminClient();

  async getParticipant(
    participantId: string,
  ): Promise<StoredParticipant | null> {
    const data = await throwOnError(
      this.client
        .from("participants")
        .select(
          "participant_id, rating, rd, sigma, last_updated_at, games_played, legacy_migrated_at",
        )
        .eq("participant_id", participantId)
        .maybeSingle(),
    );

    return data ? mapParticipantRow(data as ParticipantRow) : null;
  }

  async upsertParticipant(participant: StoredParticipant): Promise<void> {
    await throwOnError(
      this.client.from("participants").upsert(
        {
          participant_id: participant.id,
          rating: participant.rating,
          rd: participant.rd,
          sigma: participant.sigma,
          last_updated_at: participant.lastUpdatedAt,
          games_played: participant.gamesPlayed,
          legacy_migrated_at: participant.legacyMigratedAt,
        },
        { onConflict: "participant_id" },
      ),
    );
  }

  async listQuestionRatings(): Promise<StoredQuestionRating[]> {
    const data = await fetchAllRows<QuestionRatingRow>(
      this.client.from("question_ratings"),
      "question_id, rating, rd, sigma, last_updated_at, games_played, legacy_correct, legacy_wrong, label",
      "question_id",
    );

    return data.map(mapQuestionRatingRow);
  }

  async getQuestionRating(
    questionId: string,
  ): Promise<StoredQuestionRating | null> {
    const data = await throwOnError(
      this.client
        .from("question_ratings")
        .select(
          "question_id, rating, rd, sigma, last_updated_at, games_played, legacy_correct, legacy_wrong, label",
        )
        .eq("question_id", questionId)
        .maybeSingle(),
    );

    return data ? mapQuestionRatingRow(data as QuestionRatingRow) : null;
  }

  async upsertQuestionRating(question: StoredQuestionRating): Promise<void> {
    await throwOnError(
      this.client.from("question_ratings").upsert(
        {
          question_id: question.questionId,
          rating: question.rating,
          rd: question.rd,
          sigma: question.sigma,
          last_updated_at: question.lastUpdatedAt,
          games_played: question.gamesPlayed,
          legacy_correct: question.legacyCorrect,
          legacy_wrong: question.legacyWrong,
          label: question.label ?? null,
        },
        { onConflict: "question_id" },
      ),
    );
  }

  async hasAnswerAttempt(attemptId: string): Promise<boolean> {
    const data = await throwOnError(
      this.client
        .from("answer_attempts")
        .select("attempt_id")
        .eq("attempt_id", attemptId)
        .maybeSingle(),
    );

    return Boolean(data);
  }

  async appendAnswerAttempt(attempt: StoredAnswerAttempt): Promise<void> {
    const modernPayload = {
      attempt_id: attempt.attemptId,
      participant_id: attempt.participantId,
      question_id: attempt.questionId,
      label: attempt.label ?? null,
      is_correct: attempt.isCorrect,
      elapsed_ms: attempt.elapsedMs,
      mistake_count: attempt.mistakeCount,
      answered_at: attempt.answeredAt,
      source: attempt.source,
    };

    try {
      await throwOnError(
        this.client.from("answer_attempts").insert(modernPayload),
      );
      return;
    } catch (err) {
      if (!isMissingAnswerAttemptMetadataError(err)) {
        throw err;
      }

      await throwOnError(
        this.client.from("answer_attempts").insert({
          attempt_id: attempt.attemptId,
          participant_id: attempt.participantId,
          question_id: attempt.questionId,
          label: attempt.label ?? null,
          is_correct: attempt.isCorrect,
          answered_at: attempt.answeredAt,
          source: attempt.source,
        }),
      );
    }
  }

  async listQuestionReports(): Promise<StoredQuestionReport[]> {
    const data = await fetchAllRows<QuestionReportRow>(
      this.client.from("question_reports"),
      "id, participant_id, question_id, comment, reported_at, source_id, source_label, series_id, series_label, topic, prompt",
      "id",
    );

    return data.map(mapQuestionReportRow);
  }

  async hasQuestionReport(reportId: string): Promise<boolean> {
    const data = await throwOnError(
      this.client
        .from("question_reports")
        .select("id")
        .eq("id", reportId)
        .maybeSingle(),
    );

    return Boolean(data);
  }

  async appendQuestionReport(report: StoredQuestionReport): Promise<void> {
    await throwOnError(
      this.client.from("question_reports").insert({
        id: report.id,
        participant_id: report.participantId,
        question_id: report.questionId,
        comment: report.comment,
        reported_at: report.reportedAt,
        source_id: report.snapshot.sourceId,
        source_label: report.snapshot.sourceLabel,
        series_id: report.snapshot.seriesId,
        series_label: report.snapshot.seriesLabel,
        topic: report.snapshot.topic,
        prompt: report.snapshot.prompt,
      }),
    );
  }
}

export class InMemoryQuizDataStore implements QuizDataStore {
  private participants = new Map<string, StoredParticipant>();
  private questionRatings = new Map<string, StoredQuestionRating>();
  private answerAttempts = new Map<string, StoredAnswerAttempt>();
  private questionReports = new Map<string, StoredQuestionReport>();

  async getParticipant(
    participantId: string,
  ): Promise<StoredParticipant | null> {
    return this.participants.get(participantId) ?? null;
  }

  async upsertParticipant(participant: StoredParticipant): Promise<void> {
    this.participants.set(participant.id, { ...participant });
  }

  async listQuestionRatings(): Promise<StoredQuestionRating[]> {
    return Array.from(this.questionRatings.values()).map((value) => ({
      ...value,
    }));
  }

  async getQuestionRating(
    questionId: string,
  ): Promise<StoredQuestionRating | null> {
    return this.questionRatings.get(questionId) ?? null;
  }

  async upsertQuestionRating(question: StoredQuestionRating): Promise<void> {
    this.questionRatings.set(question.questionId, { ...question });
  }

  async hasAnswerAttempt(attemptId: string): Promise<boolean> {
    return this.answerAttempts.has(attemptId);
  }

  async appendAnswerAttempt(attempt: StoredAnswerAttempt): Promise<void> {
    this.answerAttempts.set(attempt.attemptId, { ...attempt });
  }

  async listQuestionReports(): Promise<StoredQuestionReport[]> {
    return Array.from(this.questionReports.values()).map((value) => ({
      ...value,
    }));
  }

  async hasQuestionReport(reportId: string): Promise<boolean> {
    return this.questionReports.has(reportId);
  }

  async appendQuestionReport(report: StoredQuestionReport): Promise<void> {
    this.questionReports.set(report.id, { ...report });
  }
}

let storeOverride: QuizDataStore | null = null;
let cachedStore: QuizDataStore | null = null;

export function setQuizDataStoreForTests(store: QuizDataStore | null): void {
  storeOverride = store;
}

export function getQuizDataStore(): QuizDataStore {
  if (storeOverride) return storeOverride;
  if (!cachedStore) {
    try {
      cachedStore = new ResilientQuizDataStore(
        new SupabaseQuizDataStore(),
        new InMemoryQuizDataStore(),
      );
    } catch (err) {
      if (!isTransientSupabaseError(err)) {
        throw err;
      }

      console.warn(
        "[quiz-data] Supabase client could not be created; using in-memory fallback for this session.",
      );
      cachedStore = new InMemoryQuizDataStore();
    }
  }
  return cachedStore;
}

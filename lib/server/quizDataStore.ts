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
    const data = await throwOnError(
      this.client
        .from("question_ratings")
        .select(
          "question_id, rating, rd, sigma, last_updated_at, games_played, legacy_correct, legacy_wrong, label",
        ),
    );

    return (data as QuestionRatingRow[]).map(mapQuestionRatingRow);
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

  async listQuestionReports(): Promise<StoredQuestionReport[]> {
    const data = await throwOnError(
      this.client
        .from("question_reports")
        .select(
          "id, participant_id, question_id, comment, reported_at, source_id, source_label, series_id, series_label, topic, prompt",
        ),
    );

    return (data as QuestionReportRow[]).map(mapQuestionReportRow);
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
    cachedStore = new SupabaseQuizDataStore();
  }
  return cachedStore;
}

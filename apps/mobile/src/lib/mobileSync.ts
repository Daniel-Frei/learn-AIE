import type { SupabaseClient } from "@supabase/supabase-js";
import {
  createDefaultRatingState,
  type QuestionRating,
  type RatingEntity,
  type RatingStateV2,
} from "../../../../lib/ratingEngine";
import type { Difficulty } from "../../../../lib/quiz";
import type { ReportSummary } from "../../../../lib/quizSync";
import type {
  MobilePersistedQuizState,
  QueuedAnswerAttempt,
  QueuedQuestionReport,
} from "./mobileLocalStore";
import { getMobileSupabaseClient } from "./mobileSupabase";

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
  question_id: string;
  status?: string | null;
};

function participantToRow(
  participantId: string,
  user: RatingEntity,
): ParticipantRow {
  return {
    participant_id: participantId,
    rating: user.rating,
    rd: user.rd,
    sigma: user.sigma,
    last_updated_at: user.lastUpdatedAt,
    games_played: user.gamesPlayed,
    legacy_migrated_at: null,
  };
}

function questionToRow(
  questionId: string,
  question: QuestionRating,
): QuestionRatingRow {
  return {
    question_id: questionId,
    rating: question.rating,
    rd: question.rd,
    sigma: question.sigma,
    last_updated_at: question.lastUpdatedAt,
    games_played: question.gamesPlayed,
    legacy_correct: question.legacyCorrect,
    legacy_wrong: question.legacyWrong,
    label: question.label ?? null,
  };
}

function rowToQuestion(row: QuestionRatingRow): QuestionRating {
  return {
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

function rowToUser(row: ParticipantRow | null): RatingEntity {
  const base = createDefaultRatingState().user;
  if (!row) return base;

  return {
    rating: row.rating,
    rd: row.rd,
    sigma: row.sigma,
    lastUpdatedAt: row.last_updated_at,
    gamesPlayed: row.games_played,
  };
}

function buildReportSummary(reports: QuestionReportRow[]): ReportSummary {
  const countsByQuestion: Record<string, number> = {};
  const openReports = reports.filter((report) => report.status !== "resolved");
  for (const report of openReports) {
    countsByQuestion[report.question_id] =
      (countsByQuestion[report.question_id] ?? 0) + 1;
  }

  return {
    totalReportCount: openReports.length,
    countsByQuestion,
  };
}

async function throwIfError<T>(
  result: PromiseLike<{
    data: T;
    error: { message: string; code?: string } | null;
  }>,
): Promise<T> {
  const { data, error } = await result;
  if (error) {
    throw new Error(error.message);
  }
  return data;
}

async function fetchAllRows<T>(
  supabase: SupabaseClient,
  table: string,
  columns: string,
  orderColumn: string,
): Promise<T[]> {
  const pageSize = 1000;
  const rows: T[] = [];

  for (let offset = 0; ; offset += pageSize) {
    const page = await throwIfError(
      supabase
        .from(table)
        .select(columns)
        .order(orderColumn, { ascending: true })
        .range(offset, offset + pageSize - 1),
    );
    const pageRows = (page ?? []) as T[];
    rows.push(...pageRows);

    if (pageRows.length < pageSize) break;
  }

  return rows;
}

async function ensureParticipant(
  supabase: SupabaseClient,
  participantId: string,
  user: RatingEntity,
): Promise<void> {
  await throwIfError(
    supabase
      .from("participants")
      .upsert(participantToRow(participantId, user), {
        onConflict: "participant_id",
      })
      .select("participant_id")
      .single(),
  );
}

async function flushAnswers(
  supabase: SupabaseClient,
  answers: QueuedAnswerAttempt[],
): Promise<void> {
  for (const answer of answers) {
    await throwIfError(
      supabase
        .from("answer_attempts")
        .upsert(
          {
            attempt_id: answer.attemptId,
            participant_id: answer.participantId,
            question_id: answer.questionId,
            label: answer.label ?? null,
            is_correct: answer.isCorrect,
            elapsed_ms: answer.elapsedMs,
            mistake_count: answer.mistakeCount,
            answered_at: answer.answeredAt,
            source: answer.source,
          },
          {
            onConflict: "attempt_id",
            ignoreDuplicates: true,
          },
        )
        .select("attempt_id"),
    );
  }
}

async function flushQuestionRatings(
  supabase: SupabaseClient,
  ratingState: RatingStateV2,
  questionIds: string[],
): Promise<void> {
  const rows = Array.from(new Set(questionIds))
    .map((questionId) => {
      const question = ratingState.questions[questionId];
      return question ? questionToRow(questionId, question) : null;
    })
    .filter((row): row is QuestionRatingRow => Boolean(row));

  if (rows.length === 0) return;

  await throwIfError(
    supabase
      .from("question_ratings")
      .upsert(rows, { onConflict: "question_id" })
      .select("question_id"),
  );
}

async function flushReports(
  supabase: SupabaseClient,
  reports: QueuedQuestionReport[],
): Promise<void> {
  for (const report of reports) {
    await throwIfError(
      supabase
        .from("question_reports")
        .upsert(
          {
            id: report.reportId,
            participant_id: report.participantId,
            question_id: report.draft.questionId,
            comment: report.draft.comment,
            reported_at: report.reportedAt,
            status: "open",
            resolved_at: null,
            resolution_note: null,
            source_id: report.draft.snapshot.sourceId,
            source_label: report.draft.snapshot.sourceLabel,
            series_id: report.draft.snapshot.seriesId,
            series_label: report.draft.snapshot.seriesLabel,
            topic: report.draft.snapshot.topic,
            prompt: report.draft.snapshot.prompt,
          },
          {
            onConflict: "id",
            ignoreDuplicates: true,
          },
        )
        .select("id"),
    );
  }
}

export async function pullRemoteMobileState(participantId: string): Promise<{
  ratingState: RatingStateV2;
  reportSummary: ReportSummary;
}> {
  const supabase = getMobileSupabaseClient();
  if (!supabase) {
    throw new Error("Supabase is not configured.");
  }

  const [participant, questions, reports] = await Promise.all([
    throwIfError(
      supabase
        .from("participants")
        .select(
          "participant_id, rating, rd, sigma, last_updated_at, games_played, legacy_migrated_at",
        )
        .eq("participant_id", participantId)
        .maybeSingle(),
    ) as Promise<ParticipantRow | null>,
    fetchAllRows<QuestionRatingRow>(
      supabase,
      "question_ratings",
      "question_id, rating, rd, sigma, last_updated_at, games_played, legacy_correct, legacy_wrong, label",
      "question_id",
    ),
    fetchAllRows<QuestionReportRow>(
      supabase,
      "question_reports",
      "id, question_id, status",
      "id",
    ),
  ]);

  const base = createDefaultRatingState();
  return {
    ratingState: {
      ...base,
      user: rowToUser(participant),
      questions: Object.fromEntries(
        questions.map((question) => [
          question.question_id,
          rowToQuestion(question),
        ]),
      ),
    },
    reportSummary: buildReportSummary(reports),
  };
}

export async function syncMobileQuizState(
  state: MobilePersistedQuizState,
): Promise<MobilePersistedQuizState> {
  const supabase = getMobileSupabaseClient();
  if (!supabase) {
    throw new Error("Supabase is not configured.");
  }

  await ensureParticipant(supabase, state.profileId, state.ratingState.user);
  await flushAnswers(supabase, state.queuedAnswers);
  await flushQuestionRatings(
    supabase,
    state.ratingState,
    state.queuedAnswers.map((answer) => answer.questionId),
  );
  await flushReports(supabase, state.queuedReports);

  const remote = await pullRemoteMobileState(state.profileId);

  return {
    ...state,
    ratingState: remote.ratingState,
    reportCountsByQuestion: remote.reportSummary.countsByQuestion,
    totalReportCount: remote.reportSummary.totalReportCount,
    queuedAnswers: [],
    queuedReports: [],
  };
}

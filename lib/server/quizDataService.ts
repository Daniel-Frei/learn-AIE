import { randomUUID } from "node:crypto";
import {
  createQuestionReport,
  exportQuestionReportsJson,
  sanitizeQuestionReportsState,
  type QuestionReportDraft,
  type QuestionReportExportV1,
} from "../questionReportsStore";
import {
  createDefaultRatingState,
  exportRatingsJson,
  importRatingsJson,
  recordAnswer,
  type LegacyDifficultyMap,
  type QuestionMetadataMap,
  type QuestionRating,
  type RatingEntity,
  type RatingStateV2,
} from "../ratingEngine";
import type { Difficulty } from "../quiz";
import type {
  ReportSummary,
  QuizStateResponse,
  RecordAnswerResponse,
} from "../quizSync";
import {
  getQuizDataStore,
  type QuizDataStore,
  type StoredParticipant,
  type StoredQuestionRating,
  type StoredQuestionReport,
} from "./quizDataStore";

function makeDefaultUser(nowTimestamp: number): RatingEntity {
  const base = createDefaultRatingState();
  return {
    ...base.user,
    lastUpdatedAt: nowTimestamp,
  };
}

function toQuestionMap(
  questions: StoredQuestionRating[],
): Record<string, QuestionRating> {
  return Object.fromEntries(
    questions.map((question) => [
      question.questionId,
      {
        rating: question.rating,
        rd: question.rd,
        sigma: question.sigma,
        lastUpdatedAt: question.lastUpdatedAt,
        gamesPlayed: question.gamesPlayed,
        legacyCorrect: question.legacyCorrect,
        legacyWrong: question.legacyWrong,
        label: question.label,
      },
    ]),
  );
}

function buildRatingState(
  participant: StoredParticipant | null,
  questions: StoredQuestionRating[],
): RatingStateV2 {
  const base = createDefaultRatingState();
  return {
    ...base,
    user: participant
      ? {
          rating: participant.rating,
          rd: participant.rd,
          sigma: participant.sigma,
          lastUpdatedAt: participant.lastUpdatedAt,
          gamesPlayed: participant.gamesPlayed,
        }
      : base.user,
    questions: toQuestionMap(questions),
  };
}

function buildReportSummary(reports: StoredQuestionReport[]): ReportSummary {
  const countsByQuestion: Record<string, number> = {};
  for (const report of reports) {
    countsByQuestion[report.questionId] =
      (countsByQuestion[report.questionId] ?? 0) + 1;
  }

  return {
    totalReportCount: reports.length,
    countsByQuestion,
  };
}

function toStoredQuestionRating(
  questionId: string,
  rating: QuestionRating,
): StoredQuestionRating {
  return {
    questionId,
    rating: rating.rating,
    rd: rating.rd,
    sigma: rating.sigma,
    lastUpdatedAt: rating.lastUpdatedAt,
    gamesPlayed: rating.gamesPlayed,
    legacyCorrect: rating.legacyCorrect,
    legacyWrong: rating.legacyWrong,
    label: rating.label,
  };
}

function toStoredParticipant(
  participantId: string,
  user: RatingEntity,
  legacyMigratedAt: string | null,
): StoredParticipant {
  return {
    id: participantId,
    rating: user.rating,
    rd: user.rd,
    sigma: user.sigma,
    lastUpdatedAt: user.lastUpdatedAt,
    gamesPlayed: user.gamesPlayed,
    legacyMigratedAt,
  };
}

function makeAttemptId(): string {
  return randomUUID();
}

function buildSyntheticOutcomes(correct: number, wrong: number): boolean[] {
  const total = correct + wrong;
  if (total <= 0) return [];

  const targetRatio = correct / total;
  const outcomes: boolean[] = [];
  let placedCorrect = 0;
  let placedWrong = 0;

  for (let i = 0; i < total; i += 1) {
    const remainingCorrect = correct - placedCorrect;
    const remainingWrong = wrong - placedWrong;

    if (remainingCorrect <= 0) {
      outcomes.push(false);
      placedWrong += 1;
      continue;
    }
    if (remainingWrong <= 0) {
      outcomes.push(true);
      placedCorrect += 1;
      continue;
    }

    const nextIndex = i + 1;
    const errorIfCorrect = Math.abs(
      (placedCorrect + 1) / nextIndex - targetRatio,
    );
    const errorIfWrong = Math.abs(placedCorrect / nextIndex - targetRatio);

    if (errorIfCorrect <= errorIfWrong) {
      outcomes.push(true);
      placedCorrect += 1;
    } else {
      outcomes.push(false);
      placedWrong += 1;
    }
  }

  return outcomes;
}

function buildLegacyCounts(state: RatingStateV2): LegacyDifficultyMap {
  return Object.fromEntries(
    Object.entries(state.questions).map(([questionId, question]) => [
      questionId,
      {
        correct: question.legacyCorrect,
        wrong: question.legacyWrong,
      },
    ]),
  );
}

export async function getQuizState(
  participantId: string,
  store: QuizDataStore = getQuizDataStore(),
): Promise<QuizStateResponse> {
  const [participant, questions, reports] = await Promise.all([
    store.getParticipant(participantId),
    store.listQuestionRatings(),
    store.listQuestionReports(),
  ]);

  return {
    participantId,
    ratingState: buildRatingState(participant, questions),
    reportSummary: buildReportSummary(reports),
    legacyMigrationCompleted: Boolean(participant?.legacyMigratedAt),
  };
}

export async function recordQuizAnswerForParticipant(
  params: {
    participantId: string;
    questionId: string;
    label?: Difficulty;
    isCorrect: boolean;
    attemptId?: string;
    answeredAt?: string;
    source?: "live" | "migration";
  },
  store: QuizDataStore = getQuizDataStore(),
): Promise<RecordAnswerResponse> {
  const attemptId = params.attemptId ?? makeAttemptId();
  if (await store.hasAnswerAttempt(attemptId)) {
    const [participant, question] = await Promise.all([
      store.getParticipant(params.participantId),
      store.getQuestionRating(params.questionId),
    ]);

    if (!participant || !question) {
      throw new Error("Duplicate answer attempt was missing persisted state.");
    }

    return {
      participantId: params.participantId,
      user: {
        rating: participant.rating,
        rd: participant.rd,
        sigma: participant.sigma,
        lastUpdatedAt: participant.lastUpdatedAt,
        gamesPlayed: participant.gamesPlayed,
      },
      questionId: params.questionId,
      question,
    };
  }

  const nowTimestamp = params.answeredAt
    ? Date.parse(params.answeredAt)
    : Date.now();
  const [participant, existingQuestion] = await Promise.all([
    store.getParticipant(params.participantId),
    store.getQuestionRating(params.questionId),
  ]);

  const state: RatingStateV2 = {
    ...createDefaultRatingState(),
    user: participant
      ? {
          rating: participant.rating,
          rd: participant.rd,
          sigma: participant.sigma,
          lastUpdatedAt: participant.lastUpdatedAt,
          gamesPlayed: participant.gamesPlayed,
        }
      : makeDefaultUser(nowTimestamp),
    questions: existingQuestion
      ? {
          [params.questionId]: {
            rating: existingQuestion.rating,
            rd: existingQuestion.rd,
            sigma: existingQuestion.sigma,
            lastUpdatedAt: existingQuestion.lastUpdatedAt,
            gamesPlayed: existingQuestion.gamesPlayed,
            legacyCorrect: existingQuestion.legacyCorrect,
            legacyWrong: existingQuestion.legacyWrong,
            label: existingQuestion.label,
          },
        }
      : {},
  };

  const updated = recordAnswer(
    state,
    params.questionId,
    params.label,
    params.isCorrect,
    nowTimestamp,
  );
  const nextQuestion = updated.questions[params.questionId];
  if (!nextQuestion) {
    throw new Error("Question rating update did not produce a question row.");
  }

  await store.upsertParticipant(
    toStoredParticipant(
      params.participantId,
      updated.user,
      participant?.legacyMigratedAt ?? null,
    ),
  );
  await store.upsertQuestionRating(
    toStoredQuestionRating(params.questionId, nextQuestion),
  );
  await store.appendAnswerAttempt({
    attemptId,
    participantId: params.participantId,
    questionId: params.questionId,
    label: params.label,
    isCorrect: params.isCorrect,
    answeredAt: params.answeredAt ?? new Date(nowTimestamp).toISOString(),
    source: params.source ?? "live",
  });

  return {
    participantId: params.participantId,
    user: updated.user,
    questionId: params.questionId,
    question: nextQuestion,
  };
}

export async function submitQuestionReportForParticipant(
  params: {
    participantId: string;
    draft: QuestionReportDraft;
    reportId?: string;
    reportedAt?: string;
  },
  store: QuizDataStore = getQuizDataStore(),
): Promise<ReportSummary> {
  const report = createQuestionReport(params.draft, {
    reportId: params.reportId,
    reportedAt: params.reportedAt,
  });
  if (!report) {
    throw new Error("Invalid report payload.");
  }

  if (!(await store.hasQuestionReport(report.id))) {
    await store.appendQuestionReport({
      ...report,
      participantId: params.participantId,
    });
  }

  return buildReportSummary(await store.listQuestionReports());
}

export async function exportQuestionReportsFromStore(
  store: QuizDataStore = getQuizDataStore(),
): Promise<QuestionReportExportV1> {
  const reports = await store.listQuestionReports();
  const json = exportQuestionReportsJson(
    {
      version: 1,
      reports: reports.map((report) => ({
        id: report.id,
        questionId: report.questionId,
        comment: report.comment,
        reportedAt: report.reportedAt,
        snapshot: report.snapshot,
      })),
    },
    new Date().toISOString(),
  );

  return JSON.parse(json) as QuestionReportExportV1;
}

export async function migrateLocalStateForParticipant(
  params: {
    participantId: string;
    localRatingState?: unknown;
    localReportState?: unknown;
    questionMetadata: QuestionMetadataMap;
  },
  store: QuizDataStore = getQuizDataStore(),
): Promise<QuizStateResponse> {
  const participant = await store.getParticipant(params.participantId);
  if (participant?.legacyMigratedAt) {
    return getQuizState(params.participantId, store);
  }

  const ratingState = params.localRatingState
    ? importRatingsJson(
        JSON.stringify(params.localRatingState),
        params.questionMetadata,
      )
    : null;

  if (ratingState) {
    const legacyCounts = buildLegacyCounts(ratingState);
    const questionIds = Object.keys(legacyCounts).sort();
    const queues = Object.fromEntries(
      questionIds.map((questionId) => {
        const stats = legacyCounts[questionId];
        return [
          questionId,
          buildSyntheticOutcomes(stats.correct, stats.wrong),
        ] as const;
      }),
    );
    const maxQueueLength = Object.values(queues).reduce(
      (max, current) => Math.max(max, current.length),
      0,
    );
    const replayStart = Date.now();
    let tick = 0;

    for (let round = 0; round < maxQueueLength; round += 1) {
      for (const questionId of questionIds) {
        const outcome = queues[questionId]?.[round];
        if (typeof outcome !== "boolean") continue;

        const question = ratingState.questions[questionId];
        await recordQuizAnswerForParticipant(
          {
            participantId: params.participantId,
            questionId,
            label: question?.label,
            isCorrect: outcome,
            attemptId: `migration:${params.participantId}:${questionId}:${round}`,
            answeredAt: new Date(replayStart + tick).toISOString(),
            source: "migration",
          },
          store,
        );
        tick += 1;
      }
    }
  }

  const localReports = params.localReportState
    ? sanitizeQuestionReportsState(params.localReportState)
    : null;
  if (localReports) {
    for (const report of localReports.reports) {
      await submitQuestionReportForParticipant(
        {
          participantId: params.participantId,
          draft: {
            questionId: report.questionId,
            comment: report.comment,
            snapshot: report.snapshot,
          },
          reportId: report.id,
          reportedAt: report.reportedAt,
        },
        store,
      );
    }
  }

  const latestParticipant = await store.getParticipant(params.participantId);
  const migratedUser = latestParticipant
    ? toStoredParticipant(
        params.participantId,
        latestParticipant,
        new Date().toISOString(),
      )
    : toStoredParticipant(
        params.participantId,
        makeDefaultUser(Date.now()),
        new Date().toISOString(),
      );
  await store.upsertParticipant(migratedUser);

  return getQuizState(params.participantId, store);
}

export function exportCurrentRatingState(state: RatingStateV2): string {
  return exportRatingsJson(state);
}

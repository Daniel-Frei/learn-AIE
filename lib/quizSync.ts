import type { Difficulty } from "./quiz";
import type { QuestionReportDraft } from "./questionReportsStore";
import type {
  QuestionRating,
  RatingEntity,
  RatingStateV2,
} from "./ratingEngine";

export const QUIZ_PERSISTENCE_WARNING_GRACE_MS = 60_000;
export const QUIZ_PERSISTENCE_HEALTH_POLL_INTERVAL_MS = 10_000;

export type PersistenceHealthResponse = {
  mode: "supabase" | "memory";
  unavailableSince: string | null;
};

export type ReportSummary = {
  totalReportCount: number;
  countsByQuestion: Record<string, number>;
};

export type QuizStateResponse = {
  participantId: string;
  ratingState: RatingStateV2;
  reportSummary: ReportSummary;
  legacyMigrationCompleted: boolean;
};

export type RecordAnswerRequest = {
  participantId: string;
  questionId: string;
  label?: Difficulty;
  isCorrect: boolean;
  elapsedMs?: number;
  mistakeCount?: number;
};

export type RecordAnswerResponse = {
  participantId: string;
  user: RatingEntity;
  questionId: string;
  question: QuestionRating;
};

export type SubmitQuestionReportRequest = {
  participantId: string;
  draft: QuestionReportDraft;
};

export type SubmitQuestionReportResponse = {
  totalReportCount: number;
  questionReportCount: number;
};

export type LocalMigrationRequest = {
  participantId: string;
  localRatingState?: unknown;
};

export type LocalMigrationResponse = QuizStateResponse;

export type ResetParticipantRatingRequest = {
  participantId: string;
};

export type ResetParticipantRatingResponse = QuizStateResponse;

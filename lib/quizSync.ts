import type { Difficulty } from "./quiz";
import type {
  QuestionReportDraft,
  QuestionReportExportV1,
} from "./questionReportsStore";
import type {
  QuestionRating,
  RatingEntity,
  RatingStateV2,
} from "./ratingEngine";

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
  localReportState?: unknown;
};

export type LocalMigrationResponse = QuizStateResponse;

export type ReportsExportResponse = QuestionReportExportV1;

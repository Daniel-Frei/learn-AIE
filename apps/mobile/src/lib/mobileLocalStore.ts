import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  createDefaultRatingState,
  type RatingStateV2,
} from "../../../../lib/ratingEngine";
import type { Difficulty } from "../../../../lib/quiz";
import type { QuestionReportDraft } from "../../../../lib/questionReportsStore";

const STATE_KEY_PREFIX = "book-quiz-mobile-state-v1:";
export const GUEST_PROFILE_ID = "guest";

export type QueuedAnswerAttempt = {
  attemptId: string;
  participantId: string;
  questionId: string;
  label?: Difficulty;
  isCorrect: boolean;
  elapsedMs: number;
  mistakeCount: number;
  answeredAt: string;
  source: "live";
};

export type QueuedQuestionReport = {
  reportId: string;
  participantId: string;
  draft: QuestionReportDraft;
  reportedAt: string;
};

export type MobilePersistedQuizState = {
  version: 1;
  profileId: string;
  ratingState: RatingStateV2;
  reportCountsByQuestion: Record<string, number>;
  totalReportCount: number;
  queuedAnswers: QueuedAnswerAttempt[];
  queuedReports: QueuedQuestionReport[];
};

function storageKey(profileId: string): string {
  return `${STATE_KEY_PREFIX}${profileId || GUEST_PROFILE_ID}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function sanitizeQueuedAnswers(value: unknown): QueuedAnswerAttempt[] {
  if (!Array.isArray(value)) return [];

  return value.filter((item): item is QueuedAnswerAttempt => {
    return (
      isRecord(item) &&
      typeof item.attemptId === "string" &&
      typeof item.participantId === "string" &&
      typeof item.questionId === "string" &&
      typeof item.isCorrect === "boolean" &&
      typeof item.elapsedMs === "number" &&
      typeof item.mistakeCount === "number" &&
      typeof item.answeredAt === "string" &&
      item.source === "live"
    );
  });
}

function sanitizeQueuedReports(value: unknown): QueuedQuestionReport[] {
  if (!Array.isArray(value)) return [];

  return value.filter((item): item is QueuedQuestionReport => {
    return (
      isRecord(item) &&
      typeof item.reportId === "string" &&
      typeof item.participantId === "string" &&
      typeof item.reportedAt === "string" &&
      isRecord(item.draft)
    );
  });
}

export function createDefaultMobileQuizState(
  profileId = GUEST_PROFILE_ID,
): MobilePersistedQuizState {
  return {
    version: 1,
    profileId,
    ratingState: createDefaultRatingState(),
    reportCountsByQuestion: {},
    totalReportCount: 0,
    queuedAnswers: [],
    queuedReports: [],
  };
}

export async function loadMobileQuizState(
  profileId = GUEST_PROFILE_ID,
): Promise<MobilePersistedQuizState> {
  try {
    const raw = await AsyncStorage.getItem(storageKey(profileId));
    if (!raw) return createDefaultMobileQuizState(profileId);

    const parsed = JSON.parse(raw) as unknown;
    if (!isRecord(parsed) || parsed.version !== 1) {
      return createDefaultMobileQuizState(profileId);
    }

    return {
      version: 1,
      profileId,
      ratingState: isRecord(parsed.ratingState)
        ? (parsed.ratingState as RatingStateV2)
        : createDefaultRatingState(),
      reportCountsByQuestion: isRecord(parsed.reportCountsByQuestion)
        ? (parsed.reportCountsByQuestion as Record<string, number>)
        : {},
      totalReportCount:
        typeof parsed.totalReportCount === "number"
          ? parsed.totalReportCount
          : 0,
      queuedAnswers: sanitizeQueuedAnswers(parsed.queuedAnswers),
      queuedReports: sanitizeQueuedReports(parsed.queuedReports),
    };
  } catch (error) {
    console.error("Failed to load mobile quiz state:", error);
    return createDefaultMobileQuizState(profileId);
  }
}

export async function saveMobileQuizState(
  state: MobilePersistedQuizState,
): Promise<void> {
  await AsyncStorage.setItem(
    storageKey(state.profileId),
    JSON.stringify(state),
  );
}

export async function clearGuestMobileQuizState(): Promise<void> {
  await AsyncStorage.removeItem(storageKey(GUEST_PROFILE_ID));
}

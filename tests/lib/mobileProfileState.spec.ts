import { describe, expect, it } from "vitest";
import { createDefaultRatingState } from "@/lib/ratingEngine";
import {
  rewriteQueuedParticipantIds,
  shouldMigrateGuestState,
} from "@/apps/mobile/src/lib/mobileProfileState";
import type { MobilePersistedQuizState } from "@/apps/mobile/src/lib/mobileLocalStore";

function makeState(
  profileId: string,
  overrides: Partial<MobilePersistedQuizState> = {},
): MobilePersistedQuizState {
  return {
    version: 1,
    profileId,
    ratingState: createDefaultRatingState(),
    reportCountsByQuestion: {},
    totalReportCount: 0,
    queuedAnswers: [],
    queuedReports: [],
    ...overrides,
  };
}

describe("mobile profile state helpers", () => {
  it("migrates guest progress only into an empty profile", () => {
    const guest = makeState("guest", {
      ratingState: {
        ...createDefaultRatingState(),
        user: {
          ...createDefaultRatingState().user,
          gamesPlayed: 1,
        },
      },
    });

    expect(shouldMigrateGuestState(makeState("profile-a"), guest)).toBe(true);
    expect(
      shouldMigrateGuestState(
        makeState("profile-a", { queuedAnswers: [{} as never] }),
        guest,
      ),
    ).toBe(false);
  });

  it("rewrites queued writes to the signed-in profile id", () => {
    const state = makeState("guest", {
      queuedAnswers: [
        {
          attemptId: "attempt-1",
          participantId: "guest",
          questionId: "q1",
          isCorrect: true,
          elapsedMs: 1000,
          mistakeCount: 0,
          answeredAt: "2026-05-11T00:00:00.000Z",
          source: "live",
        },
      ],
      queuedReports: [
        {
          reportId: "report-1",
          participantId: "guest",
          reportedAt: "2026-05-11T00:00:00.000Z",
          draft: {
            questionId: "q1",
            comment: "Needs review",
            snapshot: {
              sourceId: "chapter-1",
              sourceLabel: "Chapter 1 only",
              seriesId: "aie-foundations",
              seriesLabel: "AIE Foundations Book",
              topic: "NLP",
              prompt: "Prompt",
            },
          },
        },
      ],
    });

    const migrated = rewriteQueuedParticipantIds(state, "profile-a");

    expect(migrated.profileId).toBe("profile-a");
    expect(migrated.queuedAnswers[0].participantId).toBe("profile-a");
    expect(migrated.queuedReports[0].participantId).toBe("profile-a");
  });
});

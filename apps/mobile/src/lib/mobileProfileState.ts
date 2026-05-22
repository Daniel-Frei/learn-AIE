import type { MobilePersistedQuizState } from "./mobileLocalStore";

const GUEST_PROFILE_ID = "guest";

export function rewriteQueuedParticipantIds(
  state: MobilePersistedQuizState,
  profileId: string,
): MobilePersistedQuizState {
  return {
    ...state,
    profileId,
    queuedAnswers: state.queuedAnswers.map((answer) => ({
      ...answer,
      participantId: profileId,
    })),
    queuedReports: [],
  };
}

export function shouldMigrateGuestState(
  profileState: MobilePersistedQuizState,
  guestState: MobilePersistedQuizState,
): boolean {
  const profileHasProgress =
    profileState.ratingState.user.gamesPlayed > 0 ||
    profileState.queuedAnswers.length > 0;
  const guestHasProgress =
    guestState.profileId === GUEST_PROFILE_ID &&
    (guestState.ratingState.user.gamesPlayed > 0 ||
      guestState.queuedAnswers.length > 0);

  return !profileHasProgress && guestHasProgress;
}

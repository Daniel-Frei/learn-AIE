import AsyncStorage from "@react-native-async-storage/async-storage";

const PARTICIPANT_KEY = "aie-quiz-participant-id-v1";

function makeParticipantId(): string {
  if (
    typeof globalThis.crypto !== "undefined" &&
    typeof globalThis.crypto.randomUUID === "function"
  ) {
    return globalThis.crypto.randomUUID();
  }

  return `participant-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export async function getOrCreateMobileParticipantId(): Promise<string> {
  const existing = (await AsyncStorage.getItem(PARTICIPANT_KEY))?.trim();
  if (existing) return existing;

  const created = makeParticipantId();
  await AsyncStorage.setItem(PARTICIPANT_KEY, created);
  return created;
}

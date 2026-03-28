"use client";

const PARTICIPANT_KEY = "aie-quiz-participant-id-v1";
const LEGACY_MIGRATION_KEY_PREFIX = "aie-quiz-legacy-migration-v1:";

function canUseStorage(): boolean {
  return typeof window !== "undefined" && Boolean(window.localStorage);
}

function makeParticipantId(): string {
  if (
    typeof globalThis.crypto !== "undefined" &&
    typeof globalThis.crypto.randomUUID === "function"
  ) {
    return globalThis.crypto.randomUUID();
  }

  return `participant-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function getOrCreateParticipantId(): string | null {
  if (!canUseStorage()) return null;

  const existing = window.localStorage.getItem(PARTICIPANT_KEY)?.trim();
  if (existing) return existing;

  const created = makeParticipantId();
  window.localStorage.setItem(PARTICIPANT_KEY, created);
  return created;
}

function getLegacyMigrationKey(participantId: string): string {
  return `${LEGACY_MIGRATION_KEY_PREFIX}${participantId}`;
}

export function hasCompletedLegacyMigration(participantId: string): boolean {
  if (!canUseStorage()) return false;
  return (
    window.localStorage.getItem(getLegacyMigrationKey(participantId)) === "1"
  );
}

export function markLegacyMigrationCompleted(participantId: string): void {
  if (!canUseStorage()) return;
  window.localStorage.setItem(getLegacyMigrationKey(participantId), "1");
}

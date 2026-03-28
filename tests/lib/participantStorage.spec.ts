import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getOrCreateParticipantId,
  hasCompletedLegacyMigration,
  markLegacyMigrationCompleted,
} from "@/lib/client/participantStorage";

type StorageMock = {
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
  removeItem: (key: string) => void;
};

function createLocalStorageMock(
  seed: Record<string, string> = {},
): StorageMock {
  const store = new Map(Object.entries(seed));

  return {
    getItem(key) {
      return store.get(key) ?? null;
    },
    setItem(key, value) {
      store.set(key, value);
    },
    removeItem(key) {
      store.delete(key);
    },
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("participant storage", () => {
  it("creates and reuses an anonymous participant id", () => {
    vi.stubGlobal("window", { localStorage: createLocalStorageMock() });

    const first = getOrCreateParticipantId();
    const second = getOrCreateParticipantId();

    expect(first).toBeTruthy();
    expect(second).toBe(first);
  });

  it("tracks local legacy migration completion per participant", () => {
    vi.stubGlobal("window", { localStorage: createLocalStorageMock() });

    expect(hasCompletedLegacyMigration("participant-a")).toBe(false);
    markLegacyMigrationCompleted("participant-a");
    expect(hasCompletedLegacyMigration("participant-a")).toBe(true);
    expect(hasCompletedLegacyMigration("participant-b")).toBe(false);
  });
});

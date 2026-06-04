import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createDefaultRatingState,
  computeQuestionDifficultyScore,
  exportRatingsJson,
  getQuestionRatingEstimate,
  importRatingsJson,
  loadRatingState,
  QUESTION_TIME_LIMIT_MS,
  recordAnswer,
  saveRatingState,
} from "@/lib/difficultyStore";

type StorageMock = {
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
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
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("difficulty store core behavior", () => {
  it("updates user and question counters when recording answers", () => {
    const base = loadRatingState();
    const afterCorrect = recordAnswer(base, "q-counter", "medium", true, 1000);
    const afterWrong = recordAnswer(
      afterCorrect,
      "q-counter",
      "medium",
      false,
      1001,
    );

    expect(afterWrong.user.gamesPlayed).toBe(base.user.gamesPlayed + 2);
    expect(afterWrong.questions["q-counter"]?.legacyCorrect).toBe(1);
    expect(afterWrong.questions["q-counter"]?.legacyWrong).toBe(1);
  });

  it("caps the slow-answer rating movement at 60% of a fast clean answer", () => {
    const base = createDefaultRatingState();
    const fast = recordAnswer(base, "q-fast", "medium", true, 1000, {
      elapsedMs: 1000,
      mistakeCount: 0,
    });
    const middle = recordAnswer(base, "q-middle", "medium", true, 1000, {
      elapsedMs: 100000,
      mistakeCount: 0,
    });
    const slow = recordAnswer(base, "q-slow", "medium", true, 1000, {
      elapsedMs: QUESTION_TIME_LIMIT_MS,
      mistakeCount: 0,
    });
    const fastUserGain = fast.user.rating - base.user.rating;
    const slowUserGain = slow.user.rating - base.user.rating;
    const fastQuestionLoss =
      base.config.defaultRating - fast.questions["q-fast"]!.rating;
    const slowQuestionLoss =
      base.config.defaultRating - slow.questions["q-slow"]!.rating;

    expect(fast.questions["q-fast"]).toBeDefined();
    expect(middle.questions["q-middle"]).toBeDefined();
    expect(slow.questions["q-slow"]).toBeDefined();
    expect(fast.user.rating).toBeGreaterThan(slow.user.rating);
    expect(middle.user.rating).toBeLessThan(fast.user.rating);
    expect(middle.user.rating).toBeGreaterThan(slow.user.rating);
    expect(fast.questions["q-fast"]!.rating).toBeLessThan(
      slow.questions["q-slow"]!.rating,
    );
    expect(slowUserGain / fastUserGain).toBeCloseTo(0.6, 2);
    expect(slowQuestionLoss / fastQuestionLoss).toBeCloseTo(0.6, 2);
  });

  it("caps extreme rating movement and scales the cap by answer weight", () => {
    const base = createDefaultRatingState();
    const mismatched = {
      ...base,
      user: {
        ...base.user,
        rating: 2600,
        rd: 30,
        gamesPlayed: 50,
      },
      questions: {
        "q-extreme": {
          rating: 1300,
          rd: 300,
          sigma: 0.06,
          lastUpdatedAt: 0,
          gamesPlayed: 0,
          legacyCorrect: 0,
          legacyWrong: 0,
          label: "easy" as const,
        },
      },
    };

    const fastWrong = recordAnswer(
      mismatched,
      "q-extreme",
      "easy",
      false,
      1000,
      {
        elapsedMs: 1000,
        mistakeCount: 0,
      },
    );
    const slowWrong = recordAnswer(
      mismatched,
      "q-extreme",
      "easy",
      false,
      1000,
      {
        elapsedMs: QUESTION_TIME_LIMIT_MS,
        mistakeCount: 0,
      },
    );

    expect(2600 - fastWrong.user.rating).toBeLessThanOrEqual(100);
    expect(fastWrong.questions["q-extreme"]!.rating - 1300).toBeCloseTo(100, 5);
    expect(2600 - slowWrong.user.rating).toBeLessThanOrEqual(60);
    expect(slowWrong.questions["q-extreme"]!.rating - 1300).toBeCloseTo(60, 5);
  });

  it("keeps a minimum rating exchange for expected outcomes", () => {
    const base = createDefaultRatingState();
    const expectedCorrect = {
      ...base,
      user: {
        ...base.user,
        rating: 2600,
        rd: 30,
        gamesPlayed: 50,
      },
      questions: {
        "q-expected": {
          rating: 1300,
          rd: 30,
          sigma: 0.06,
          lastUpdatedAt: 0,
          gamesPlayed: 50,
          legacyCorrect: 0,
          legacyWrong: 0,
          label: "easy" as const,
        },
      },
    };

    const updated = recordAnswer(
      expectedCorrect,
      "q-expected",
      "easy",
      true,
      1000,
      {
        elapsedMs: 1000,
        mistakeCount: 0,
      },
    );

    expect(updated.user.rating - 2600).toBeCloseTo(2, 5);
    expect(1300 - updated.questions["q-expected"]!.rating).toBeCloseTo(2, 5);
  });

  it("reduces the penalty for a single mistake compared with multiple mistakes", () => {
    const base = createDefaultRatingState();
    const oneMistake = recordAnswer(base, "q-one", "medium", false, 1000, {
      elapsedMs: 1000,
      mistakeCount: 1,
    });
    const twoMistakes = recordAnswer(base, "q-two", "medium", false, 1000, {
      elapsedMs: 1000,
      mistakeCount: 2,
    });

    expect(oneMistake.questions["q-one"]).toBeDefined();
    expect(twoMistakes.questions["q-two"]).toBeDefined();
    expect(oneMistake.user.rating).toBeGreaterThan(twoMistakes.user.rating);
    expect(oneMistake.questions["q-one"]!.rating).toBeLessThan(
      twoMistakes.questions["q-two"]!.rating,
    );
  });

  it("keeps old binary callers on full weight when timing metadata is omitted", () => {
    const base = createDefaultRatingState();
    const legacy = recordAnswer(base, "q-legacy-weight", "medium", true, 1000);
    const explicit = recordAnswer(
      base,
      "q-explicit-weight",
      "medium",
      true,
      1000,
      {
        elapsedMs: 0,
        mistakeCount: 0,
      },
    );

    expect(legacy.questions["q-legacy-weight"]).toBeDefined();
    expect(explicit.questions["q-explicit-weight"]).toBeDefined();
    expect(legacy.user.rating).toBeCloseTo(explicit.user.rating, 10);
    expect(legacy.questions["q-legacy-weight"]!.rating).toBeCloseTo(
      explicit.questions["q-explicit-weight"]!.rating,
      10,
    );
  });

  it("adds a label hint to existing unlabeled question ratings", () => {
    const base = createDefaultRatingState();
    const updated = recordAnswer(
      {
        ...base,
        questions: {
          "q-unlabeled": {
            rating: 1500,
            rd: 300,
            sigma: 0.06,
            lastUpdatedAt: 0,
            gamesPlayed: 0,
            legacyCorrect: 0,
            legacyWrong: 0,
          },
        },
      },
      "q-unlabeled",
      "hard",
      true,
      1000,
    );

    expect(updated.questions["q-unlabeled"]?.label).toBe("hard");
  });

  it("assigns harder prior score to hard labels than easy labels", () => {
    const state = loadRatingState();
    const easyScore = computeQuestionDifficultyScore("q-easy", "easy", state);
    const hardScore = computeQuestionDifficultyScore("q-hard", "hard", state);

    expect(hardScore).toBeGreaterThan(easyScore);
    expect(easyScore).toBeGreaterThanOrEqual(0);
    expect(hardScore).toBeLessThanOrEqual(1);
  });

  it("seeds new question ratings close to 1500 by difficulty label", () => {
    const state = createDefaultRatingState();

    expect(getQuestionRatingEstimate("q-easy", "easy", state).rating).toBe(
      1400,
    );
    expect(getQuestionRatingEstimate("q-medium", "medium", state).rating).toBe(
      1500,
    );
    expect(getQuestionRatingEstimate("q-hard", "hard", state).rating).toBe(
      1600,
    );
  });

  it("round-trips ratings through export/import json", () => {
    const base = loadRatingState();
    const updated = recordAnswer(base, "q-roundtrip", "hard", true, 2000);
    const json = exportRatingsJson(updated);
    const imported = importRatingsJson(json);

    expect(imported).not.toBeNull();
    expect(imported?.questions["q-roundtrip"]?.legacyCorrect).toBe(1);
    expect(imported?.questions["q-roundtrip"]?.label).toBe("hard");
  });

  it("imports legacy v1 map shape", () => {
    const legacyJson = JSON.stringify({
      "q-legacy": { correct: 2, wrong: 1 },
    });
    const imported = importRatingsJson(legacyJson, {
      "q-legacy": { label: "easy" },
    });

    expect(imported).not.toBeNull();
    expect(imported?.questions["q-legacy"]?.legacyCorrect).toBe(2);
    expect(imported?.questions["q-legacy"]?.legacyWrong).toBe(1);
    expect(imported?.questions["q-legacy"]?.label).toBe("easy");
  });

  it("loads and sanitizes persisted v2 rating state from local storage", () => {
    vi.stubGlobal("window", {
      localStorage: createLocalStorageMock({
        "aie-quiz-ratings-v2": JSON.stringify({
          version: 2,
          algorithm: "glicko-2",
          config: {
            minRd: 500,
            maxRd: 100,
            defaultSigma: -1,
            tau: 0,
            epsilon: 0,
            periodDays: 0,
            difficultyScale: 0,
          },
          user: {
            rating: "not-a-number",
            rd: 9999,
            sigma: -1,
            lastUpdatedAt: -10,
            gamesPlayed: 2.8,
          },
          questions: {
            "q-sanitize": {
              rating: 1600,
              rd: 1,
              sigma: 0,
              lastUpdatedAt: 12.8,
              gamesPlayed: "bad",
              legacyCorrect: 2.8,
              legacyWrong: -5,
              label: "not-a-label",
            },
          },
        }),
      }),
    });

    const loaded = loadRatingState({
      "q-sanitize": { label: "hard" },
    });

    expect(loaded.user.rating).toBe(1500);
    expect(loaded.user.rd).toBe(100);
    expect(loaded.user.sigma).toBeGreaterThan(0);
    expect(loaded.user.lastUpdatedAt).toBe(0);
    expect(loaded.user.gamesPlayed).toBe(2);
    expect(loaded.questions["q-sanitize"]).toMatchObject({
      rating: 1600,
      rd: 100,
      gamesPlayed: 0,
      legacyCorrect: 2,
      legacyWrong: 0,
      label: "hard",
    });

    vi.stubGlobal("window", {
      localStorage: createLocalStorageMock({
        "aie-quiz-ratings-v2": JSON.stringify({
          version: 2,
          algorithm: "glicko-2",
          config: null,
          user: null,
          questions: {
            "q-null": null,
          },
        }),
      }),
    });

    const loadedFromNulls = loadRatingState({
      "q-null": { label: "medium" },
    });
    expect(loadedFromNulls.questions["q-null"]).toMatchObject({
      rating: 1500,
      label: "medium",
    });
  });

  it("migrates legacy local rating maps and saves the v2 state", () => {
    const setItem = vi.fn();
    const storage = createLocalStorageMock({
      "aie-quiz-question-stats-v1": JSON.stringify({
        "q-legacy-load": { correct: 2, wrong: 1 },
        "q-empty": { correct: -1, wrong: "bad" },
      }),
    });
    vi.stubGlobal("window", {
      localStorage: {
        getItem: storage.getItem,
        setItem,
      },
    });

    const loaded = loadRatingState({
      "q-legacy-load": { label: "easy" },
    });

    expect(loaded.questions["q-legacy-load"]?.legacyCorrect).toBe(2);
    expect(loaded.questions["q-legacy-load"]?.legacyWrong).toBe(1);
    expect(loaded.questions["q-legacy-load"]?.label).toBe("easy");
    expect(loaded.questions["q-empty"]).toBeUndefined();
    expect(setItem).toHaveBeenCalledWith(
      "aie-quiz-ratings-v2",
      expect.stringContaining('"version":2'),
    );
  });

  it("logs and falls back when persisted rating JSON cannot be parsed or saved", () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal("window", {
      localStorage: {
        getItem: () => "{",
        setItem: () => {
          throw new Error("quota exceeded");
        },
      },
    });

    expect(loadRatingState()).toMatchObject({ version: 2, questions: {} });
    saveRatingState(createDefaultRatingState());

    expect(error).toHaveBeenCalledWith(
      "Failed to load rating state:",
      expect.any(SyntaxError),
    );
    expect(error).toHaveBeenCalledWith(
      "Failed to save rating state:",
      expect.any(Error),
    );

    vi.unstubAllGlobals();
    expect(() => saveRatingState(createDefaultRatingState())).not.toThrow();
  });

  it("imports nested legacy counts and rejects invalid rating JSON", () => {
    const nestedLegacy = importRatingsJson(
      JSON.stringify({
        exportedAt: "2026-05-01T00:00:00.000Z",
        legacyCounts: {
          "q-nested": { correct: 1, wrong: 2 },
        },
      }),
      {
        "q-nested": { label: "hard" },
      },
    );
    const error = vi.spyOn(console, "error").mockImplementation(() => {});

    expect(nestedLegacy?.questions["q-nested"]).toMatchObject({
      legacyCorrect: 1,
      legacyWrong: 2,
      label: "hard",
    });
    expect(importRatingsJson("{")).toBeNull();
    expect(importRatingsJson("null")).toBeNull();
    expect(importRatingsJson("5")).toBeNull();
    expect(importRatingsJson("{}")).toMatchObject({
      version: 2,
      questions: {},
    });
    expect(
      importRatingsJson(
        JSON.stringify({
          version: 2,
          algorithm: "glicko-2",
          questions: null,
        }),
      ),
    ).toMatchObject({ version: 2, questions: {} });
    expect(error).toHaveBeenCalledWith(
      "Failed to import ratings JSON:",
      expect.any(SyntaxError),
    );
  });

  it("falls back while exporting malformed rating state and recording unlabeled questions", () => {
    const exported = JSON.parse(exportRatingsJson({} as never)) as {
      version: number;
      legacyCounts: Record<string, unknown>;
    };
    const updated = recordAnswer(
      createDefaultRatingState(),
      "q-no-label",
      undefined,
      true,
      1000,
    );

    expect(exported).toMatchObject({ version: 2, legacyCounts: {} });
    expect(updated.questions["q-no-label"]?.label).toBeUndefined();
  });
});

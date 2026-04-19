import { describe, expect, it } from "vitest";
import {
  createDefaultRatingState,
  computeQuestionDifficultyScore,
  exportRatingsJson,
  importRatingsJson,
  loadRatingState,
  recordAnswer,
} from "@/lib/difficultyStore";

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

  it("gives a faster correct answer a larger rating gain than a slow one", () => {
    const base = createDefaultRatingState();
    const fast = recordAnswer(base, "q-fast", "medium", true, 1000, {
      elapsedMs: 1000,
      mistakeCount: 0,
    });
    const slow = recordAnswer(base, "q-slow", "medium", true, 1000, {
      elapsedMs: 180000,
      mistakeCount: 0,
    });

    expect(fast.questions["q-fast"]).toBeDefined();
    expect(slow.questions["q-slow"]).toBeDefined();
    expect(fast.user.rating).toBeGreaterThan(slow.user.rating);
    expect(fast.questions["q-fast"]!.rating).toBeLessThan(
      slow.questions["q-slow"]!.rating,
    );
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

  it("assigns harder prior score to hard labels than easy labels", () => {
    const state = loadRatingState();
    const easyScore = computeQuestionDifficultyScore("q-easy", "easy", state);
    const hardScore = computeQuestionDifficultyScore("q-hard", "hard", state);

    expect(hardScore).toBeGreaterThan(easyScore);
    expect(easyScore).toBeGreaterThanOrEqual(0);
    expect(hardScore).toBeLessThanOrEqual(1);
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
});

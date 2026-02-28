import { describe, expect, it } from "vitest";
import {
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

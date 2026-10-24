import { describe, expect, it } from "vitest";
import { referenceQuestion, uniqueQuestions } from "@/lib/questionReferences";
import {
  allQuestions,
  QUESTION_SOURCES,
  getQuestionSourceContext,
  getQuestionSourceMemberships,
  getQuestionSourceMetadata,
  getQuestionsForFilters,
} from "@/lib/quiz";
import {
  createDefaultRatingState,
  getQuestionRatingEstimate,
  recordAnswer,
} from "@/lib/ratingEngine";
import { getEligibleQuestionIds } from "@/lib/quizSession";

const oldId = "cme295-lect4-q104";
const oldSource = QUESTION_SOURCES.find((s) => s.id === "cme295-lect4")!;
const newSource = QUESTION_SOURCES.find((s) => s.id === "cme295-2026-lect2")!;
const canonical = oldSource.questions.find((q) => q.id === oldId)!;
const expectedCanonicalIds = [
  [],
  [
    "cme295-lect4-q104",
    "cme295-lect4-q105",
    "cme295-lect4-q107",
    "cme295-lect3-q188",
    "cme295-lect3-q189",
    "cme295-lect3-q190",
    "cme295-lect3-q191",
    "cme295-lect3-q194",
    "cme295-lect3-q195",
    "cme295-lect2-q77",
    "cme295-lect2-q02",
    "cme295-lect2-q17",
    "cme295-lect2-q05",
    "cme295-lect2-q06",
    "cme295-lect2-q07",
    "cme295-lect2-q13",
    "cme295-lect2-q19",
    "cme295-lect2-q18",
    "cme295-lect2-q09",
    "cme295-lect2-q20",
    "cme295-lect3-q197",
    "cme295-lect3-q199",
    "cme295-lect3-q202",
    "cme295-lect3-q203",
    "cme295-lect3-q204",
    "cme295-lect3-q206",
    "cme295-lect3-q209",
    "cme295-lect3-q218",
  ],
  [
    "cme295-lect4-q109",
    "cme295-lect4-q149",
    "cme295-lect4-q168",
    "cme295-lect5-q05",
    "cme295-lect5-q09",
    "cme295-lect5-q10",
    "cme295-lect5-q11",
    "cme295-lect5-q42",
    "cme295-lect5-q14",
    "cme295-lect5-q25",
    "cme295-lect6-q04",
    "cme295-lect6-q11",
    "cme295-lect6-q12",
  ],
  [
    "cme295-lect5-q07",
    "cme295-lect5-q47",
    "cme295-lect5-q46",
    "cme295-lect6-q45",
    "cme295-lect6-q49",
  ],
];

describe("canonical question references", () => {
  it("keeps all 46 intended canonical memberships in the correct new lecture", () => {
    const sources = QUESTION_SOURCES.filter(
      (s) => s.seriesId === "stanford-cme295-2026",
    );
    for (const [i, source] of sources.entries()) {
      expect(
        source.questions
          .filter((q) => q.id.startsWith("cme295-lect"))
          .map((q) => q.id),
      ).toEqual(expectedCanonicalIds[i]);
    }
  });
  it("returns the original object and rejects missing or ambiguous targets", () => {
    expect(referenceQuestion(oldSource.questions, oldId)).toBe(canonical);
    expect(newSource.questions.find((q) => q.id === oldId)).toBe(canonical);
    expect(() => referenceQuestion(oldSource.questions, "missing")).toThrow(
      "Question reference must resolve exactly once: missing",
    );
    expect(() => referenceQuestion([canonical, canonical], oldId)).toThrow(
      "Question reference must resolve exactly once: " + oldId,
    );
  });

  it("keeps the first occurrence and refuses conflicting definitions", () => {
    const other = oldSource.questions.find((q) => q.id !== oldId)!;
    expect(uniqueQuestions([canonical, other, canonical])).toEqual([
      canonical,
      other,
    ]);
    expect(uniqueQuestions([])).toEqual([]);
    expect(() => uniqueQuestions([canonical, { ...canonical }])).toThrow(
      "Conflicting question definitions for ID: " + oldId,
    );
  });

  it("offers a shared question once when selecting both editions or a whole topic", () => {
    const combined = getQuestionsForFilters([oldSource.id, newSource.id], []);
    const expected = new Set(
      [...oldSource.questions, ...newSource.questions].map((q) => q.id),
    );
    expect(combined).toHaveLength(expected.size);
    expect(combined.filter((q) => q.id === oldId)).toEqual([canonical]);
    for (const pool of [allQuestions, getQuestionsForFilters([], ["NLP"])]) {
      expect(new Set(pool.map((q) => q.id)).size).toBe(pool.length);
      expect(pool.filter((q) => q.id === oldId)).toEqual([canonical]);
    }
  });

  it("tracks both memberships and resolves tooltip/report metadata from selection", () => {
    expect(getQuestionSourceMemberships(oldId).map((m) => m.sourceId)).toEqual([
      oldSource.id,
      newSource.id,
    ]);
    expect(getQuestionSourceMetadata(oldId)?.sourceId).toBe(oldSource.id);
    expect(
      getQuestionSourceMetadata(oldId, [newSource.id], ["NLP"])?.sourceId,
    ).toBe(newSource.id);
    expect(getQuestionSourceMetadata(oldId, [oldSource.id])?.sourceId).toBe(
      oldSource.id,
    );
    expect(
      getQuestionSourceMetadata(oldId, [newSource.id, oldSource.id])?.sourceId,
    ).toBe(oldSource.id);
    expect(getQuestionSourceContext(oldId, [newSource.id])).toContain("2026");
    expect(getQuestionSourceMetadata("missing")).toBeNull();
    expect(getQuestionSourceContext("missing")).toBeNull();
    expect(getQuestionSourceMemberships("missing")).toEqual([]);
  });

  it("uses the existing old-ID Glicko record for selection and subsequent answers", () => {
    const now = Date.now();
    const state = createDefaultRatingState();
    state.questions[oldId] = {
      rating: 2200,
      rd: 80,
      sigma: 0.06,
      lastUpdatedAt: now,
      gamesPlayed: 12,
      legacyCorrect: 7,
      legacyWrong: 5,
      label: canonical.difficulty,
    };
    expect(
      getEligibleQuestionIds({
        sources: [newSource.id],
        topics: [],
        difficultyRange: { min: 2100, max: 2300 },
        ratingState: state,
      }),
    ).toEqual([oldId]);
    const from2026 = newSource.questions.find((q) => q.id === oldId)!;
    expect(
      getQuestionRatingEstimate(from2026.id, from2026.difficulty, state).rating,
    ).toBe(2200);
    const updated = recordAnswer(
      state,
      from2026.id,
      from2026.difficulty,
      true,
      now,
    );
    expect(updated.questions[oldId].gamesPlayed).toBe(13);
    expect(updated.questions[oldId].legacyCorrect).toBe(8);
    expect(updated.questions[oldId].legacyWrong).toBe(5);
    expect(Object.keys(updated.questions)).toEqual([oldId]);
    expect(updated.questions["cme295-2026-lect2-q08"]).toBeUndefined();
    expect(state.questions[oldId].gamesPlayed).toBe(12);
  });

  it("has one definition per identity and no duplicate membership within any source", () => {
    const definitions = new Map<string, object>();
    for (const source of QUESTION_SOURCES) {
      expect(new Set(source.questions.map((q) => q.id)).size).toBe(
        source.questions.length,
      );
      for (const question of source.questions) {
        if (definitions.has(question.id))
          expect(question).toBe(definitions.get(question.id));
        definitions.set(question.id, question);
      }
    }
    expect(allQuestions).toHaveLength(definitions.size);
  });
});

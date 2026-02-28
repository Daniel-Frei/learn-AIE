import { describe, expect, it } from "vitest";
import {
  ALL_SOURCE_IDS,
  QUESTION_SOURCES,
  allQuestions,
  getQuestionsForMode,
  getQuestionsForSources,
  getTitleForSelection,
} from "@/lib/quiz";

describe("quiz source registry helpers", () => {
  it("returns all questions when mode is all", () => {
    expect(getQuestionsForMode("all")).toEqual(allQuestions);
  });

  it("returns only source questions for an explicit source mode", () => {
    const first = QUESTION_SOURCES[0];
    expect(getQuestionsForMode(first.id)).toEqual(first.questions);
  });

  it("uses all sources when source selection is empty", () => {
    expect(getQuestionsForSources([])).toEqual(allQuestions);
  });

  it("returns the combined set for selected sources only", () => {
    const [first, second] = QUESTION_SOURCES;
    const selected = getQuestionsForSources([first.id, second.id]);
    const expectedLength = first.questions.length + second.questions.length;
    const allowedIds = new Set([
      ...first.questions.map((q) => q.id),
      ...second.questions.map((q) => q.id),
    ]);

    expect(selected).toHaveLength(expectedLength);
    expect(selected.every((q) => allowedIds.has(q.id))).toBe(true);
  });

  it("builds an all-sources title for empty selection", () => {
    expect(getTitleForSelection([])).toContain("All Chapters Quiz");
  });

  it("builds a source-specific title for a single source", () => {
    const source = QUESTION_SOURCES[0];
    expect(getTitleForSelection([source.id])).toBe(source.title);
  });

  it("builds a custom title summary for multiple sources", () => {
    const title = getTitleForSelection(ALL_SOURCE_IDS.slice(0, 3));
    expect(title).toContain("Custom Quiz");
    expect(title).toContain("+1 more");
  });
});

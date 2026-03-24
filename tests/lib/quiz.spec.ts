import { describe, expect, it } from "vitest";
import {
  ALL_SOURCE_IDS,
  SOURCE_SERIES,
  QUESTION_SOURCES,
  allQuestions,
  getQuestionsForFilters,
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
    expect(getQuestionsForSources([])).toEqual([]);
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

  it("filters questions by topic", () => {
    const rlQuestions = getQuestionsForFilters([], ["RL"]);
    expect(rlQuestions.length).toBeGreaterThan(0);

    const rlSourceIds = new Set(
      QUESTION_SOURCES.filter((s) => s.topic === "RL").map((s) => s.id),
    );
    const rlQuestionIds = new Set(
      QUESTION_SOURCES.filter((s) => s.topic === "RL").flatMap((s) =>
        s.questions.map((q) => q.id),
      ),
    );

    expect(rlSourceIds.size).toBeGreaterThan(0);
    expect(rlQuestions.every((q) => rlQuestionIds.has(q.id))).toBe(true);
  });

  it("returns no questions when both source and topic filters are empty", () => {
    expect(getQuestionsForFilters([], [])).toEqual([]);
  });

  it("combines source and topic filters using OR semantics", () => {
    const rlSource = QUESTION_SOURCES.find((s) => s.topic === "RL");
    const nlpSourceQuestionIds = new Set(
      QUESTION_SOURCES.filter((s) => s.topic === "NLP").flatMap((s) =>
        s.questions.map((q) => q.id),
      ),
    );

    expect(rlSource).toBeDefined();

    const selected = getQuestionsForFilters([rlSource!.id], ["NLP"]);
    const rlQuestionIds = new Set(rlSource!.questions.map((q) => q.id));

    expect(selected.some((q) => rlQuestionIds.has(q.id))).toBe(true);
    expect(selected.some((q) => nlpSourceQuestionIds.has(q.id))).toBe(true);
  });

  it("builds non-empty series groups for selecting books/lecture series", () => {
    expect(SOURCE_SERIES.length).toBeGreaterThan(0);
    expect(SOURCE_SERIES.every((series) => series.sourceIds.length > 0)).toBe(
      true,
    );
  });

  it("builds an explicit empty-filter title when nothing is selected", () => {
    expect(getTitleForSelection([])).toContain("Select sources");
  });

  it("builds a source-specific title for a single source", () => {
    const source = QUESTION_SOURCES[0];
    expect(getTitleForSelection([source.id])).toBe(source.title);
  });

  it("builds a topic-based title when only topics are selected", () => {
    expect(getTitleForSelection([], ["NLP"])).toContain("Topic Quiz");
  });

  it("builds a custom title summary for multiple sources", () => {
    const title = getTitleForSelection(ALL_SOURCE_IDS.slice(0, 3));
    expect(title).toContain("Custom Quiz");
    expect(title).toContain("+1 more");
  });

  it("registers newly added MIT 6.S191, linear algebra, and LangChain sources", () => {
    const ids = new Set(QUESTION_SOURCES.map((s) => s.id));
    expect(ids.has("mit6s191-l3")).toBe(true);
    expect(ids.has("mit6s191-l4")).toBe(true);
    expect(ids.has("mit6s191-l6")).toBe(true);
    expect(ids.has("crash-linalg-l1")).toBe(true);
    expect(ids.has("langchain-deepagents")).toBe(true);
  });
});

import { afterEach, describe, expect, it, vi } from "vitest";
import {
  appendQuestionReport,
  createDefaultQuestionReportsState,
  createQuestionReport,
  sanitizeQuestionReportsState,
} from "@/lib/questionReportsStore";

const sampleSnapshot = {
  sourceId: "mit15773-l4" as const,
  sourceLabel: "MIT 15.773 L4",
  seriesId: "mit-15773-2024" as const,
  seriesLabel: "MIT 15.773 Hands-On Deep Learning 2024",
  topic: "DL" as const,
  prompt: "Why might transfer learning help when data is limited?",
};

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("question report helpers", () => {
  it("appends separate entries for repeated reports on the same question", () => {
    const base = createDefaultQuestionReportsState();

    const afterFirst = appendQuestionReport(
      base,
      {
        questionId: "mit15773-l4-q1",
        comment: "First report",
        snapshot: sampleSnapshot,
      },
      {
        reportId: "r-1",
        reportedAt: "2026-03-16T10:00:00.000Z",
      },
    );

    const afterSecond = appendQuestionReport(
      afterFirst,
      {
        questionId: "mit15773-l4-q1",
        comment: "Second report",
        snapshot: sampleSnapshot,
      },
      {
        reportId: "r-2",
        reportedAt: "2026-03-16T11:00:00.000Z",
      },
    );

    expect(afterSecond.reports).toHaveLength(2);
    expect(afterSecond.reports.map((report) => report.id)).toEqual([
      "r-1",
      "r-2",
    ]);
    expect(
      afterSecond.reports.every(
        (report) => report.questionId === "mit15773-l4-q1",
      ),
    ).toBe(true);
  });

  it("drops malformed persisted reports while keeping valid entries", () => {
    const sanitized = sanitizeQuestionReportsState({
      version: 1,
      reports: [
        {
          id: "r-valid",
          questionId: "q-valid",
          comment: "  Keep me  ",
          reportedAt: "2026-03-16T13:00:00.000Z",
          snapshot: sampleSnapshot,
        },
        null,
        {
          id: "r-missing-snapshot",
          questionId: "q-invalid",
          comment: "Missing snapshot",
          reportedAt: "2026-03-16T13:00:00.000Z",
        },
        {
          id: "r-empty-comment",
          questionId: "q-invalid",
          comment: "   ",
          reportedAt: "2026-03-16T13:00:00.000Z",
          snapshot: sampleSnapshot,
        },
        {
          id: "r-empty-snapshot-field",
          questionId: "q-invalid",
          comment: "Missing prompt",
          reportedAt: "2026-03-16T13:00:00.000Z",
          snapshot: { ...sampleSnapshot, prompt: "", sourceId: 123 },
        },
      ],
    });

    expect(sanitized).toEqual({
      version: 1,
      reports: [
        expect.objectContaining({
          id: "r-valid",
          comment: "Keep me",
        }),
      ],
    });
    expect(sanitizeQuestionReportsState({ version: 1 })).toBeNull();
    expect(sanitizeQuestionReportsState(null)).toBeNull();
  });

  it("ignores invalid drafts and can generate ids without crypto", () => {
    vi.stubGlobal("crypto", undefined);
    vi.spyOn(Date, "now").mockReturnValue(12345);
    vi.spyOn(Math, "random").mockReturnValue(0.5);

    const unchanged = appendQuestionReport(
      createDefaultQuestionReportsState(),
      {
        questionId: " ",
        comment: " ",
        snapshot: sampleSnapshot,
      },
    );
    const generated = createQuestionReport({
      questionId: "q-generated",
      comment: " Generated report ",
      snapshot: sampleSnapshot,
    });

    expect(unchanged.reports).toEqual([]);
    expect(generated).toMatchObject({
      id: expect.stringMatching(/^report-12345-/),
      questionId: "q-generated",
      comment: "Generated report",
    });
    expect(
      createQuestionReport({
        questionId: " ",
        comment: "No question",
        snapshot: sampleSnapshot,
      }),
    ).toBeNull();
  });

  it("falls back for malformed report state inputs", () => {
    const appended = appendQuestionReport(
      { version: 2 } as never,
      {
        questionId: "q-valid",
        comment: "Valid comment",
        snapshot: sampleSnapshot,
      },
      {
        reportId: "r-valid",
        reportedAt: "2026-05-01T00:00:00.000Z",
      },
    );

    expect(appended.reports).toHaveLength(1);
    expect(appended.reports[0]).toMatchObject({
      id: "r-valid",
      questionId: "q-valid",
      comment: "Valid comment",
    });
  });
});

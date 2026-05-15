import { afterEach, describe, expect, it, vi } from "vitest";
import {
  appendQuestionReport,
  createQuestionReport,
  createDefaultQuestionReportsState,
  exportQuestionReportsJson,
  loadQuestionReports,
  saveQuestionReports,
  sanitizeQuestionReportsState,
} from "@/lib/questionReportsStore";

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

describe("question report store", () => {
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

  it("exports versioned json with export timestamp and snapshot fields", () => {
    const state = appendQuestionReport(
      createDefaultQuestionReportsState(),
      {
        questionId: "mit15773-l4-q2",
        comment: "Prompt is too vague.",
        snapshot: sampleSnapshot,
      },
      {
        reportId: "r-export",
        reportedAt: "2026-03-16T12:00:00.000Z",
      },
    );

    const json = exportQuestionReportsJson(state, "2026-03-16T12:30:00.000Z");
    const parsed = JSON.parse(json) as {
      version: number;
      exportedAt: string;
      reports: Array<{
        id: string;
        questionId: string;
        comment: string;
        reportedAt: string;
        snapshot: {
          sourceId: string;
          sourceLabel: string;
          seriesId: string;
          seriesLabel: string;
          topic: string;
          prompt: string;
        };
      }>;
    };

    expect(parsed.version).toBe(1);
    expect(parsed.exportedAt).toBe("2026-03-16T12:30:00.000Z");
    expect(parsed.reports).toHaveLength(1);
    expect(parsed.reports[0]?.snapshot.prompt).toBe(sampleSnapshot.prompt);
    expect(parsed.reports[0]?.snapshot.sourceId).toBe(sampleSnapshot.sourceId);
  });

  it("loads safely from empty or invalid local storage data", () => {
    expect(loadQuestionReports()).toEqual(createDefaultQuestionReportsState());

    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal("window", {
      localStorage: createLocalStorageMock(),
    });

    expect(loadQuestionReports()).toEqual(createDefaultQuestionReportsState());

    vi.stubGlobal("window", {
      localStorage: createLocalStorageMock({
        "aie-quiz-question-reports-v1": "{not-valid-json",
      }),
    });

    expect(loadQuestionReports()).toEqual(createDefaultQuestionReportsState());

    vi.stubGlobal("window", {
      localStorage: createLocalStorageMock({
        "aie-quiz-question-reports-v1": JSON.stringify({
          version: 1,
          reports: "not-an-array",
        }),
      }),
    });

    expect(loadQuestionReports()).toEqual(createDefaultQuestionReportsState());
  });

  it("round-trips saved reports through local storage", () => {
    const localStorage = createLocalStorageMock();
    vi.stubGlobal("window", { localStorage });

    const state = appendQuestionReport(
      createDefaultQuestionReportsState(),
      {
        questionId: "mit15773-l4-q3",
        comment: "Explanation contradicts the marked answer.",
        snapshot: sampleSnapshot,
      },
      {
        reportId: "r-save",
        reportedAt: "2026-03-16T13:00:00.000Z",
      },
    );

    saveQuestionReports(state);

    const loaded = loadQuestionReports();
    expect(loaded.reports).toHaveLength(1);
    expect(loaded.reports[0]?.id).toBe("r-save");
    expect(loaded.reports[0]?.comment).toBe(
      "Explanation contradicts the marked answer.",
    );
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

  it("logs when report saving fails", () => {
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    vi.stubGlobal("window", {
      localStorage: {
        getItem: () => null,
        setItem: () => {
          throw new Error("quota exceeded");
        },
        removeItem: () => {},
      },
    });

    saveQuestionReports(createDefaultQuestionReportsState());

    expect(error).toHaveBeenCalledWith(
      "Failed to save question reports:",
      expect.any(Error),
    );
  });

  it("falls back for server-side or malformed report save/export inputs", () => {
    expect(() =>
      saveQuestionReports(createDefaultQuestionReportsState()),
    ).not.toThrow();

    const writes: string[] = [];
    vi.stubGlobal("window", {
      localStorage: {
        getItem: () => null,
        setItem: (_key: string, value: string) => {
          writes.push(value);
        },
        removeItem: () => {},
      },
    });

    saveQuestionReports({ version: 2 } as never);
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
    const exported = JSON.parse(
      exportQuestionReportsJson({ version: 2 } as never),
    ) as { version: number; reports: unknown[] };

    expect(JSON.parse(writes[0])).toEqual(createDefaultQuestionReportsState());
    expect(appended.reports).toHaveLength(1);
    expect(exported).toMatchObject({ version: 1, reports: [] });
  });
});

import type { Topic, SourceId, SourceSeriesId } from "./quiz";

export type QuestionReportSnapshot = {
  sourceId: SourceId;
  sourceLabel: string;
  seriesId: SourceSeriesId;
  seriesLabel: string;
  topic: Topic;
  prompt: string;
};

export type QuestionReport = {
  id: string;
  questionId: string;
  comment: string;
  reportedAt: string;
  snapshot: QuestionReportSnapshot;
};

export type QuestionReportDraft = {
  questionId: string;
  comment: string;
  snapshot: QuestionReportSnapshot;
};

export type QuestionReportsStateV1 = {
  version: 1;
  reports: QuestionReport[];
};

export type QuestionReportExportV1 = QuestionReportsStateV1 & {
  exportedAt: string;
};

const STORAGE_KEY = "aie-quiz-question-reports-v1";

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function sanitizeString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function sanitizeSnapshot(value: unknown): QuestionReportSnapshot | null {
  if (!isRecord(value)) return null;

  const sourceId = sanitizeString(value.sourceId);
  const sourceLabel = sanitizeString(value.sourceLabel);
  const seriesId = sanitizeString(value.seriesId);
  const seriesLabel = sanitizeString(value.seriesLabel);
  const topic = sanitizeString(value.topic);
  const prompt = sanitizeString(value.prompt);

  if (
    !sourceId ||
    !sourceLabel ||
    !seriesId ||
    !seriesLabel ||
    !topic ||
    !prompt
  ) {
    return null;
  }

  return {
    sourceId: sourceId as SourceId,
    sourceLabel,
    seriesId: seriesId as SourceSeriesId,
    seriesLabel,
    topic: topic as Topic,
    prompt,
  };
}

function sanitizeReport(value: unknown): QuestionReport | null {
  if (!isRecord(value)) return null;

  const id = sanitizeString(value.id);
  const questionId = sanitizeString(value.questionId);
  const comment = sanitizeString(value.comment).trim();
  const reportedAt = sanitizeString(value.reportedAt);
  const snapshot = sanitizeSnapshot(value.snapshot);

  if (!id || !questionId || !comment || !reportedAt || !snapshot) {
    return null;
  }

  return {
    id,
    questionId,
    comment,
    reportedAt,
    snapshot,
  };
}

export function sanitizeQuestionReportsState(
  value: unknown,
): QuestionReportsStateV1 | null {
  if (
    !isRecord(value) ||
    value.version !== 1 ||
    !Array.isArray(value.reports)
  ) {
    return null;
  }

  return {
    version: 1,
    reports: value.reports
      .map((report) => sanitizeReport(report))
      .filter((report): report is QuestionReport => Boolean(report)),
  };
}

function makeReportId(): string {
  if (
    typeof globalThis.crypto !== "undefined" &&
    typeof globalThis.crypto.randomUUID === "function"
  ) {
    return globalThis.crypto.randomUUID();
  }

  return `report-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

export function createDefaultQuestionReportsState(): QuestionReportsStateV1 {
  return {
    version: 1,
    reports: [],
  };
}

export function loadQuestionReports(): QuestionReportsStateV1 {
  if (typeof window === "undefined") {
    return createDefaultQuestionReportsState();
  }

  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return createDefaultQuestionReportsState();

    const parsed = JSON.parse(raw) as unknown;
    return (
      sanitizeQuestionReportsState(parsed) ??
      createDefaultQuestionReportsState()
    );
  } catch (err) {
    console.error("Failed to load question reports:", err);
    return createDefaultQuestionReportsState();
  }
}

export function saveQuestionReports(state: QuestionReportsStateV1): void {
  if (typeof window === "undefined") return;

  try {
    const normalized =
      sanitizeQuestionReportsState(state) ??
      createDefaultQuestionReportsState();
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalized));
  } catch (err) {
    console.error("Failed to save question reports:", err);
  }
}

export function appendQuestionReport(
  state: QuestionReportsStateV1,
  draft: QuestionReportDraft,
  options?: {
    reportId?: string;
    reportedAt?: string;
  },
): QuestionReportsStateV1 {
  const normalized =
    sanitizeQuestionReportsState(state) ?? createDefaultQuestionReportsState();
  const comment = draft.comment.trim();

  if (!draft.questionId.trim() || !comment) {
    return normalized;
  }

  const nextReport: QuestionReport = {
    id: options?.reportId ?? makeReportId(),
    questionId: draft.questionId,
    comment,
    reportedAt: options?.reportedAt ?? new Date().toISOString(),
    snapshot: draft.snapshot,
  };

  return {
    version: 1,
    reports: [...normalized.reports, nextReport],
  };
}

export function exportQuestionReportsJson(
  state: QuestionReportsStateV1,
  exportedAt: string = new Date().toISOString(),
): string {
  const normalized =
    sanitizeQuestionReportsState(state) ?? createDefaultQuestionReportsState();
  const envelope: QuestionReportExportV1 = {
    version: 1,
    exportedAt,
    reports: normalized.reports,
  };

  return JSON.stringify(envelope, null, 2);
}

export function createQuestionReport(
  draft: QuestionReportDraft,
  options?: {
    reportId?: string;
    reportedAt?: string;
  },
): QuestionReport | null {
  const nextState = appendQuestionReport(
    createDefaultQuestionReportsState(),
    draft,
    options,
  );
  return nextState.reports[0] ?? null;
}
